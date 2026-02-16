"""
Core Utilities for Pitch Anomaly Detection

Contains data loading, normalization, and general utility functions.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Data Loading
# =============================================================================

def load_processed_trials(processed_dir: str) -> list:
    """
    Load all preprocessed CSV files from a directory.

    Args:
        processed_dir: Path to directory containing CSV files

    Returns:
        List of pandas DataFrames, one per trial
    """
    dfs = []
    for entry_name in sorted(os.listdir(processed_dir)):
        if entry_name.lower().endswith(".csv"):
            full_path = os.path.join(processed_dir, entry_name)
            df = pd.read_csv(full_path)
            dfs.append(df)
            print(f"Loaded: {entry_name}")
    return dfs


def load_and_correct_trials(trial_dir: str, sensor_columns: list = None,
                            skip_trials: set = None,
                            global_stats: dict = None) -> list:
    """
    Load raw trials, apply spike correction and optional normalization.

    Requires preprocess_pitch module to be importable.

    Args:
        trial_dir: Path to directory containing raw CSV files
        sensor_columns: List of column names for sensor data
        skip_trials: Set of filenames to skip
        global_stats: Dict of per-channel {mean, std} for z-score normalization.
                      If None, no normalization is applied.
                      Use compute_channel_stats() to generate this from training data.

    Returns:
        list of corrected (and optionally normalized) DataFrames
    """
    # Import here to avoid circular dependency
    from preprocess_pitch import load_raw_trial, detect_and_fix_spikes, SENSOR_COLUMNS, SKIP_TRIALS

    if sensor_columns is None:
        sensor_columns = SENSOR_COLUMNS
    if skip_trials is None:
        skip_trials = SKIP_TRIALS

    corrected_dfs = []
    trial_path = Path(trial_dir)

    for csv_file in sorted(trial_path.glob('*.csv')):
        if csv_file.name in skip_trials:
            continue

        df = load_raw_trial(str(csv_file))

        # Apply spike correction to each channel
        for col in sensor_columns:
            if col in df.columns:
                signal = df[col].values.astype(np.float64)
                fixed_signal, _ = detect_and_fix_spikes(signal)
                df[col] = fixed_signal

        # Apply global normalization if stats provided
        if global_stats is not None:
            for col in sensor_columns:
                if col in df.columns and col in global_stats:
                    mean = global_stats[col]["mean"]
                    std = global_stats[col]["std"]
                    df[col] = (df[col] - mean) / std

        # Add time column and reorder
        n = len(df)
        df['time'] = np.arange(n) / 240.0
        df = df[['time'] + sensor_columns]

        corrected_dfs.append(df)

    return corrected_dfs


# =============================================================================
# Normalization
# =============================================================================

def compute_channel_stats(trials: list, sensor_columns: list) -> dict:
    """
    Compute per-channel mean/std from a list of trial DataFrames.

    Use this to compute normalization statistics from TRAINING data only,
    then apply those stats to normalize all data (train + test).

    Args:
        trials: List of DataFrames, each containing sensor columns
        sensor_columns: List of column names to compute stats for

    Returns:
        Dict mapping column name to {"mean": float, "std": float}
    """
    all_values = {col: [] for col in sensor_columns}

    for df in trials:
        for col in sensor_columns:
            if col in df.columns:
                all_values[col].extend(df[col].values.tolist())

    stats = {}
    for col in sensor_columns:
        if all_values[col]:
            arr = np.array(all_values[col])
            std_val = float(np.std(arr))
            stats[col] = {
                "mean": float(np.mean(arr)),
                "std": std_val if std_val > 1e-10 else 1.0
            }
    return stats


def normalize_trials(trials: list, stats: dict, sensor_columns: list) -> list:
    """
    Apply z-score normalization to a list of trial DataFrames using global stats.

    Args:
        trials: List of DataFrames to normalize
        stats: Dict from compute_channel_stats with per-channel mean/std
        sensor_columns: List of column names to normalize

    Returns:
        List of normalized DataFrames (copies, originals unchanged)
    """
    normalized = []
    for df in trials:
        df_norm = df.copy()
        for col in sensor_columns:
            if col in df.columns and col in stats:
                df_norm[col] = (df[col] - stats[col]["mean"]) / stats[col]["std"]
        normalized.append(df_norm)
    return normalized


def normalize_trials_per_trial(trials: list, sensor_columns: list) -> list:
    """
    Apply per-trial z-score normalization.

    Each trial is normalized using its own mean/std per channel.
    This removes between-trial baseline differences (sensor placement, etc.)
    while preserving within-trial dynamics.

    Use this when:
    - Sensor placement varies between recording sessions
    - Motion quality (shape/timing) matters more than absolute amplitude
    - You want the model to learn dynamics, not baseline offsets

    Args:
        trials: List of DataFrames to normalize
        sensor_columns: List of column names to normalize

    Returns:
        List of normalized DataFrames (copies, originals unchanged)
    """
    normalized = []
    for df in trials:
        df_norm = df.copy()
        for col in sensor_columns:
            if col in df.columns:
                values = df[col].values
                mean = values.mean()
                std = values.std()
                if std < 1e-10:
                    std = 1.0
                df_norm[col] = (values - mean) / std
        normalized.append(df_norm)
    return normalized


def normalize_batch(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of time series using channel-wise statistics.

    Args:
        x: Input tensor of shape (B, C, T)
        mean: Per-channel means of shape (C,)
        std: Per-channel standard deviations of shape (C,)

    Returns:
        Normalized tensor of shape (B, C, T)
    """
    return (x - mean.view(1, -1, 1)) / std.view(1, -1, 1)


def denorm(xn: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a batch of time series back to original units.

    Args:
        xn: Normalized tensor of shape (B, C, T)
        mean: Per-channel means of shape (C,)
        std: Per-channel standard deviations of shape (C,)

    Returns:
        Denormalized tensor of shape (B, C, T)
    """
    return xn * std.view(1, -1, 1) + mean.view(1, -1, 1)


def fit_channel_stats_windows(loader, max_batches: int = 100) -> tuple:
    """
    Estimate per-channel mean and standard deviation from windowed training data.

    Uses numerically stable one-pass algorithm.

    Args:
        loader: PyTorch DataLoader yielding (x, trial_ids) batches
        max_batches: Maximum number of batches to process

    Returns:
        Tuple of (mean, std) tensors, each of shape (C,)
    """
    s1, s2, N = None, None, None
    for i, (x, _) in enumerate(loader):
        x = x.float()
        s1_batch = x.sum(dim=(0, 2))
        s2_batch = (x ** 2).sum(dim=(0, 2))
        N_batch = torch.ones_like(s1_batch) * x.shape[0] * x.shape[2]
        if s1 is None:
            s1, s2, N = s1_batch, s2_batch, N_batch
        else:
            s1 += s1_batch
            s2 += s2_batch
            N += N_batch
        if i + 1 >= max_batches:
            break
    mean = s1 / N
    std = ((s2 / N) - mean ** 2).clamp_min(1e-12).sqrt()
    return mean, std


# =============================================================================
# Model Utilities
# =============================================================================

def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model: PyTorch nn.Module

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# LOF Anomaly Detection
# =============================================================================

def extract_lof_features(residual: torch.Tensor) -> np.ndarray:
    """
    Extract 24 features from reconstruction residual for LOF input.

    Features (per channel):
    - Mean error: 8 values
    - Std error: 8 values
    - Max absolute error: 8 values

    Args:
        residual: Tensor of shape (C, T) or (B, C, T)

    Returns:
        features: array of shape (24,) for single input, (B, 24) for batch
    """
    if residual.dim() == 2:
        residual = residual.unsqueeze(0)

    # Per-channel statistics (across time dimension)
    mean_err = residual.mean(dim=2)         # (B, C)
    std_err = residual.std(dim=2)           # (B, C)
    max_err = residual.abs().max(dim=2)[0]  # (B, C)

    features = torch.cat([mean_err, std_err, max_err], dim=1)  # (B, 24)
    return features.squeeze().cpu().numpy()


class LOFAnomalyDetector:
    """
    LOF-based anomaly detector using reconstruction error features.

    Trains on "good" data and detects outliers based on local density.

    Example:
        detector = LOFAnomalyDetector(n_neighbors=20, contamination=0.1)
        detector.fit(model, train_loader, ch_mean, ch_std, device)
        predictions = detector.predict(model, test_loader, ch_mean, ch_std, device)
        # predictions: 1 = normal, -1 = anomaly
    """

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1):
        """
        Args:
            n_neighbors: Number of neighbors for LOF density estimation
            contamination: Expected proportion of outliers (0.0 to 0.5)
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.lof = None
        self.scaler = StandardScaler()

    def _extract_features(self, model, data_loader, ch_mean, ch_std, device) -> np.ndarray:
        """Extract LOF features from all windows in a data loader."""
        features = []
        model.eval()

        with torch.no_grad():
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(device)
                x_norm = normalize_batch(x_batch, ch_mean, ch_std)
                x_hat, _ = model(x_norm)
                residual = x_norm - x_hat

                for i in range(residual.size(0)):
                    features.append(extract_lof_features(residual[i]))

        return np.array(features)

    def fit(self, model, train_loader, ch_mean, ch_std, device):
        """
        Fit LOF on reconstruction errors from good training data.

        Args:
            model: Trained autoencoder model
            train_loader: DataLoader with good/normal training windows
            ch_mean: Per-channel mean for normalization
            ch_std: Per-channel std for normalization
            device: torch device (cuda/cpu)

        Returns:
            self (for method chaining)
        """
        X = self._extract_features(model, train_loader, ch_mean, ch_std, device)
        X_scaled = self.scaler.fit_transform(X)

        self.lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True  # Enable predict() on new data
        )
        self.lof.fit(X_scaled)
        return self

    def predict(self, model, data_loader, ch_mean, ch_std, device) -> np.ndarray:
        """
        Predict anomaly labels for new data.

        Args:
            model: Trained autoencoder model
            data_loader: DataLoader with windows to classify
            ch_mean: Per-channel mean for normalization
            ch_std: Per-channel std for normalization
            device: torch device (cuda/cpu)

        Returns:
            Array of labels: 1 = normal (inlier), -1 = anomaly (outlier)
        """
        X = self._extract_features(model, data_loader, ch_mean, ch_std, device)
        X_scaled = self.scaler.transform(X)
        return self.lof.predict(X_scaled)

    def score_samples(self, model, data_loader, ch_mean, ch_std, device) -> np.ndarray:
        """
        Return anomaly scores for new data.

        Higher scores (closer to 0) indicate more normal samples.
        Lower scores (more negative) indicate anomalies.

        Args:
            model: Trained autoencoder model
            data_loader: DataLoader with windows to score
            ch_mean: Per-channel mean for normalization
            ch_std: Per-channel std for normalization
            device: torch device (cuda/cpu)

        Returns:
            Array of negative LOF scores (higher = more normal)
        """
        X = self._extract_features(model, data_loader, ch_mean, ch_std, device)
        X_scaled = self.scaler.transform(X)
        return self.lof.score_samples(X_scaled)

    def evaluate(self, model, test_good_loader, test_bad_loader,
                 ch_mean, ch_std, device) -> dict:
        """
        Evaluate anomaly detection on held-out test set.

        Args:
            model: Trained autoencoder model
            test_good_loader: DataLoader with good/normal test windows
            test_bad_loader: DataLoader with bad/anomaly test windows
            ch_mean: Per-channel mean for normalization
            ch_std: Per-channel std for normalization
            device: torch device (cuda/cpu)

        Returns:
            Dictionary with evaluation metrics:
                - good_accuracy: Fraction of good samples correctly classified as normal
                - bad_detection_rate: Fraction of bad samples correctly detected as anomalies
                - good_predictions: Array of predictions for good test data
                - bad_predictions: Array of predictions for bad test data
                - good_scores: Array of scores for good test data
                - bad_scores: Array of scores for bad test data
        """
        # Get predictions
        good_preds = self.predict(model, test_good_loader, ch_mean, ch_std, device)
        bad_preds = self.predict(model, test_bad_loader, ch_mean, ch_std, device)

        # Get scores
        good_scores = self.score_samples(model, test_good_loader, ch_mean, ch_std, device)
        bad_scores = self.score_samples(model, test_bad_loader, ch_mean, ch_std, device)

        # Calculate metrics
        good_accuracy = (good_preds == 1).mean()  # Should be classified as normal (1)
        bad_detection_rate = (bad_preds == -1).mean()  # Should be classified as anomaly (-1)

        return {
            'good_accuracy': good_accuracy,
            'bad_detection_rate': bad_detection_rate,
            'good_predictions': good_preds,
            'bad_predictions': bad_preds,
            'good_scores': good_scores,
            'bad_scores': bad_scores
        }


class IsolationForestAnomalyDetector:
    """
    Isolation Forest anomaly detector using reconstruction error features.

    More robust than LOF for cases with inter-trial variability, as it
    isolates anomalies using random tree splits rather than local density.

    Example:
        detector = IsolationForestAnomalyDetector(n_estimators=100)
        detector.fit(model, train_loader, ch_mean, ch_std, device)
        predictions = detector.predict(model, test_loader, ch_mean, ch_std, device)
        # predictions: 1 = normal, -1 = anomaly
    """

    def __init__(self, n_estimators: int = 100, contamination: float = "auto",
                 random_state: int = 42):
        """
        Args:
            n_estimators: Number of trees in the forest
            contamination: Expected proportion of outliers ("auto" or 0.0 to 0.5)
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.iforest = None
        self.scaler = StandardScaler()

    def _extract_features(self, model, data_loader, ch_mean, ch_std, device) -> np.ndarray:
        """Extract features from all windows in a data loader."""
        features = []
        model.eval()

        with torch.no_grad():
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(device)
                x_norm = normalize_batch(x_batch, ch_mean, ch_std)
                x_hat, _ = model(x_norm)
                residual = x_norm - x_hat

                for i in range(residual.size(0)):
                    features.append(extract_lof_features(residual[i]))

        return np.array(features)

    def fit(self, model, train_loader, ch_mean, ch_std, device):
        """
        Fit Isolation Forest on reconstruction errors from good training data.

        Args:
            model: Trained autoencoder model
            train_loader: DataLoader with good/normal training windows
            ch_mean: Per-channel mean for normalization
            ch_std: Per-channel std for normalization
            device: torch device (cuda/cpu)

        Returns:
            self (for method chaining)
        """
        X = self._extract_features(model, train_loader, ch_mean, ch_std, device)
        X_scaled = self.scaler.fit_transform(X)

        self.iforest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.iforest.fit(X_scaled)
        return self

    def predict(self, model, data_loader, ch_mean, ch_std, device) -> np.ndarray:
        """
        Predict anomaly labels for new data.

        Args:
            model: Trained autoencoder model
            data_loader: DataLoader with windows to classify
            ch_mean: Per-channel mean for normalization
            ch_std: Per-channel std for normalization
            device: torch device (cuda/cpu)

        Returns:
            Array of labels: 1 = normal (inlier), -1 = anomaly (outlier)
        """
        X = self._extract_features(model, data_loader, ch_mean, ch_std, device)
        X_scaled = self.scaler.transform(X)
        return self.iforest.predict(X_scaled)

    def score_samples(self, model, data_loader, ch_mean, ch_std, device) -> np.ndarray:
        """
        Return anomaly scores for new data.

        Higher scores (closer to 0) indicate anomalies.
        Lower scores (more negative) indicate normal samples.
        (Note: opposite sign convention from LOF)

        Args:
            model: Trained autoencoder model
            data_loader: DataLoader with windows to score
            ch_mean: Per-channel mean for normalization
            ch_std: Per-channel std for normalization
            device: torch device (cuda/cpu)

        Returns:
            Array of anomaly scores
        """
        X = self._extract_features(model, data_loader, ch_mean, ch_std, device)
        X_scaled = self.scaler.transform(X)
        return self.iforest.score_samples(X_scaled)

    def evaluate(self, model, test_good_loader, test_bad_loader,
                 ch_mean, ch_std, device) -> dict:
        """
        Evaluate anomaly detection on held-out test set.

        Args:
            model: Trained autoencoder model
            test_good_loader: DataLoader with good/normal test windows
            test_bad_loader: DataLoader with bad/anomaly test windows
            ch_mean: Per-channel mean for normalization
            ch_std: Per-channel std for normalization
            device: torch device (cuda/cpu)

        Returns:
            Dictionary with evaluation metrics:
                - good_accuracy: Fraction of good samples correctly classified as normal
                - bad_detection_rate: Fraction of bad samples correctly detected as anomalies
                - good_predictions: Array of predictions for good test data
                - bad_predictions: Array of predictions for bad test data
                - good_scores: Array of scores for good test data
                - bad_scores: Array of scores for bad test data
        """
        # Get predictions
        good_preds = self.predict(model, test_good_loader, ch_mean, ch_std, device)
        bad_preds = self.predict(model, test_bad_loader, ch_mean, ch_std, device)

        # Get scores
        good_scores = self.score_samples(model, test_good_loader, ch_mean, ch_std, device)
        bad_scores = self.score_samples(model, test_bad_loader, ch_mean, ch_std, device)

        # Calculate metrics
        good_accuracy = (good_preds == 1).mean()  # Should be classified as normal (1)
        bad_detection_rate = (bad_preds == -1).mean()  # Should be classified as anomaly (-1)

        return {
            'good_accuracy': good_accuracy,
            'bad_detection_rate': bad_detection_rate,
            'good_predictions': good_preds,
            'bad_predictions': bad_preds,
            'good_scores': good_scores,
            'bad_scores': bad_scores
        }

"""
PyTorch Dataset Classes for Pitch Data

Contains custom Dataset implementations for time-series pitch data:
- WindowedTimeSeriesDataset: Extracts overlapping sliding windows
- TrialDataset: Full trial dataset for evaluation
- window_collate: Custom collate function for windowed data
- create_train_val_test_split: Split data for anomaly detection
"""

from typing import Dict, List, Union
import numpy as np
import torch
from torch.utils.data import Dataset


def create_train_val_test_split(
    good_trials: List[np.ndarray],
    bad_trials: List[np.ndarray],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, List[np.ndarray]]:
    """
    Split data for anomaly detection at the trial level.

    For autoencoder-based anomaly detection:
    - Good data is split into train/val/test
    - Bad data goes entirely to test (never seen during training)

    Args:
        good_trials: List of numpy arrays (normal/good trials)
        bad_trials: List of numpy arrays (anomalous/bad trials)
        train_ratio: Fraction of good data for training (default 0.6)
        val_ratio: Fraction of good data for validation (default 0.2)
        test_ratio: Fraction of good data for testing (default 0.2)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with keys:
            'train': Good trials for training autoencoder
            'val': Good trials for validation/early stopping
            'test_good': Good trials for final evaluation
            'test_bad': All bad trials for anomaly detection evaluation
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rng = np.random.default_rng(seed)
    n_good = len(good_trials)

    # Shuffle indices
    indices = rng.permutation(n_good)

    # Calculate split points
    n_train = int(n_good * train_ratio)
    n_val = int(n_good * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {
        'train': [good_trials[i] for i in train_idx],
        'val': [good_trials[i] for i in val_idx],
        'test_good': [good_trials[i] for i in test_idx],
        'test_bad': list(bad_trials)  # All bad data goes to test
    }

    print(f"Split summary:")
    print(f"  Train:     {len(splits['train'])} good trials")
    print(f"  Val:       {len(splits['val'])} good trials")
    print(f"  Test good: {len(splits['test_good'])} good trials")
    print(f"  Test bad:  {len(splits['test_bad'])} bad trials")

    return splits


class WindowedTimeSeriesDataset(Dataset):
    """
    Dataset that extracts overlapping windows from variable-length trials.

    Args:
        data: List of DataFrames or numpy arrays (each is one trial)
        window_size: Size of each window in timesteps
        stride: Step size between consecutive windows
    """
    def __init__(self, data, window_size: int = 256, stride: int = 64):
        self.windows = []
        self.trial_ids = []  # Track which trial each window came from

        for trial_idx, item in enumerate(data):
            if hasattr(item, "to_numpy"):  # pandas.DataFrame
                df = item.drop(columns=["time"]) if "time" in item.columns else item
                x = df.to_numpy(dtype=np.float32)
            else:
                x = np.asarray(item, dtype=np.float32)
                if x.ndim == 1:
                    x = x[:, None]

            # Extract overlapping windows
            T = x.shape[0]
            for start in range(0, T - window_size + 1, stride):
                window = x[start:start + window_size]
                self.windows.append(torch.from_numpy(window))
                self.trial_ids.append(trial_idx)

        print(f"Created {len(self.windows)} windows from {len(data)} trials")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]  # (T, C)
        return x.T, self.trial_ids[idx]  # (C, T), trial_id


class TrialDataset(Dataset):
    """
    Dataset for full trials (for evaluation).

    Unlike WindowedTimeSeriesDataset, this returns complete trials
    without windowing, useful for inference and evaluation.

    Args:
        data: List of DataFrames or numpy arrays (each is one trial)
    """
    def __init__(self, data):
        self.items = []
        for item in data:
            if hasattr(item, "to_numpy"):
                df = item.drop(columns=["time"]) if "time" in item.columns else item
                x = df.to_numpy(dtype=np.float32)
            else:
                x = np.asarray(item, dtype=np.float32)
                if x.ndim == 1:
                    x = x[:, None]
            self.items.append(torch.from_numpy(x))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx], idx


def window_collate(batch):
    """
    Collate function for fixed-size windows.

    Args:
        batch: List of (window, trial_id) tuples

    Returns:
        x_batch: Tensor of shape (B, C, T)
        trial_ids: Tensor of trial indices
    """
    xs, trial_ids = zip(*batch)
    x_batch = torch.stack(xs, dim=0)  # (B, C, T)
    return x_batch, torch.tensor(trial_ids)

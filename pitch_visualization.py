"""
Visualization Utilities for Pitch Anomaly Detection

Contains plotting functions for trial visualization, spike analysis,
and anomaly detection results.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


SAMPLING_RATE = 240  # Hz


def plot_trial(df, title: str = "Trial", ax=None, fs: int = SAMPLING_RATE):
    """
    Plot all sensor channels for a single trial.

    Args:
        df: DataFrame with 'time' column and sensor columns
        title: Plot title
        ax: Matplotlib axes (creates new figure if None)
        fs: Sampling rate in Hz

    Returns:
        Matplotlib axes object
    """
    time = df["time"].values if "time" in df.columns else np.arange(len(df)) / fs
    sensor_cols = [c for c in df.columns if c != "time"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    for col in sensor_cols:
        ax.plot(time, df[col].values, label=col, alpha=0.8, linewidth=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Scaled Voltage")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    return ax


def plot_trial_with_spikes(trial_path: str, figsize: tuple = (16, 10),
                           sensor_columns: list = None):
    """
    Plot all channels of a trial, highlighting detected spikes.

    Requires preprocess_pitch module to be importable.

    Args:
        trial_path: Path to trial CSV file
        figsize: Figure size
        sensor_columns: List of sensor column names

    Returns:
        Tuple of (figure, total_spikes_detected)
    """
    from preprocess_pitch import load_raw_trial, detect_and_fix_spikes, SENSOR_COLUMNS

    if sensor_columns is None:
        sensor_columns = SENSOR_COLUMNS

    df = load_raw_trial(trial_path)
    n = len(df)
    time = np.arange(n) / SAMPLING_RATE

    fig, axes = plt.subplots(4, 2, figsize=figsize, sharex=True)
    axes = axes.flatten()

    total_spikes = 0
    for i, col in enumerate(sensor_columns):
        if col not in df.columns:
            continue

        signal = df[col].values.astype(np.float64)
        fixed, info = detect_and_fix_spikes(signal)
        n_spikes = info['n_spikes_detected']
        total_spikes += n_spikes

        ax = axes[i]

        # Plot before (faded) and after
        if n_spikes > 0:
            ax.plot(time, signal, 'b-', alpha=0.4, linewidth=1, label='Before')
            ax.plot(time, fixed, 'g-', linewidth=1, label='After')

            # Mark spike locations
            spike_indices = info['spike_indices']
            if spike_indices:
                spike_times = time[spike_indices]
                spike_vals = signal[spike_indices]
                ax.scatter(spike_times, spike_vals, c='red', s=60, zorder=5,
                          marker='o', label=f'{n_spikes} spike(s)')
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.plot(time, signal, 'b-', linewidth=1)

        ax.set_ylabel(f'Ch {col}', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-2].set_xlabel('Time (seconds)')
    axes[-1].set_xlabel('Time (seconds)')

    trial_name = Path(trial_path).name
    fig.suptitle(f'{trial_name} - {total_spikes} total spikes detected',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig, total_spikes


def analyze_all_trials_for_spikes(sensor_columns: list = None, skip_trials: set = None):
    """
    Analyze all trials and return spike statistics.

    Requires preprocess_pitch module to be importable.

    Args:
        sensor_columns: List of sensor column names
        skip_trials: Set of trial filenames to skip

    Returns:
        List of dicts with spike info per trial
    """
    from preprocess_pitch import load_raw_trial, detect_and_fix_spikes, SENSOR_COLUMNS, SKIP_TRIALS

    if sensor_columns is None:
        sensor_columns = SENSOR_COLUMNS
    if skip_trials is None:
        skip_trials = SKIP_TRIALS

    results = []

    for category in ['good', 'bad']:
        trial_dir = Path(f'pitch/{category}')
        for csv_file in sorted(trial_dir.glob('*.csv')):
            if csv_file.name in skip_trials:
                continue

            try:
                df = load_raw_trial(str(csv_file))
                trial_info = {
                    'path': str(csv_file),
                    'name': csv_file.name,
                    'category': category,
                    'n_samples': len(df),
                    'channels': {},
                    'total_spikes': 0
                }

                for col in sensor_columns:
                    if col in df.columns:
                        signal = df[col].values.astype(np.float64)
                        _, info = detect_and_fix_spikes(signal)
                        trial_info['channels'][col] = info
                        trial_info['total_spikes'] += info['n_spikes_detected']

                results.append(trial_info)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")

    return results


def plot_trial_anomaly_scores(residuals, scores, trial_name: str, threshold: float,
                              ax, window_length: int = 8, fs: int = SAMPLING_RATE):
    """
    Plot anomaly scores over time with threshold and anomaly regions.

    Args:
        residuals: Residual array of shape (C, T)
        scores: Mahalanobis scores array
        trial_name: Name for title
        threshold: Anomaly threshold tau
        ax: Matplotlib axes
        window_length: Window length L used in detector
        fs: Sampling rate
    """
    L = window_length
    T = residuals.shape[1]

    # Time axis for scores (offset by L-1 due to windowing)
    t_scores = np.arange(L - 1, T) / fs

    # Plot scores
    ax.plot(t_scores, scores, 'b-', linewidth=1, alpha=0.8)
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold ({threshold:.1f})')

    # Highlight anomaly regions
    anomaly_mask = scores > threshold
    if anomaly_mask.any():
        ax.fill_between(t_scores, 0, scores, where=anomaly_mask,
                        color='red', alpha=0.3, label='Anomaly')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('M[n]')
    ax.set_title(f'{trial_name}\nMax={scores.max():.1f}, Mean={scores.mean():.1f}, '
                 f'Anomalies={anomaly_mask.sum()}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t_scores[0], t_scores[-1])


def plot_training_curves(histories: dict, save_path: str = None):
    """
    Plot training and validation loss curves for multiple models.

    Args:
        histories: Dict mapping model_name -> {'train_loss': [...], 'val_loss': [...]}
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Training loss
    ax1 = axes[0]
    for (name, hist), color in zip(histories.items(), colors):
        ax1.plot(hist['train_loss'], label=name, color=color, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training MSE')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Validation loss
    ax2 = axes[1]
    for (name, hist), color in zip(histories.items(), colors):
        ax2.plot(hist['val_loss'], label=name, color=color, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation MSE')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig

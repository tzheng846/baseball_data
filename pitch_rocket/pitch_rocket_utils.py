"""
MiniRocket Utilities for Pitch Classification

Functions for training and evaluating MiniRocket classifiers on pitch data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from sktime.transformations.panel.rocket import MiniRocketMultivariateVariable
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pitch_dataset import WindowedTimeSeriesDataset

SENSOR_COLS = ['25', '26', '27', '28', '29', '30', '31', '32']


def list_of_dfs_to_sktime_nested(
    X_input,
    sensor_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert data to sktime nested panel format.

    Args:
        X_input: Either:
            - list of DataFrames (T_i x C each)
            - WindowedTimeSeriesDataset instance
        sensor_order: Column names for sensors

    Returns:
        Nested DataFrame (n_samples x C) where each cell is a pd.Series
    """
    if hasattr(X_input, 'windows'):
        windows = X_input.windows

        if sensor_order is None:
            n_channels = windows[0].shape[1]
            sensor_order = [str(i) for i in range(n_channels)]

        nested = pd.DataFrame(index=range(len(windows)), columns=sensor_order, dtype=object)
        for i, window in enumerate(windows):
            arr = window.numpy()
            for c_idx, col in enumerate(sensor_order):
                nested.at[i, col] = pd.Series(arr[:, c_idx])

        return nested

    X_list = X_input
    if len(X_list) == 0:
        raise ValueError("X_list is empty.")

    if sensor_order is None:
        sensor_order = [c for c in X_list[0].columns if c != 'time']

    nested = pd.DataFrame(index=range(len(X_list)), columns=sensor_order, dtype=object)
    for i, df in enumerate(X_list):
        for c in sensor_order:
            nested.at[i, c] = pd.Series(df[c].to_numpy(dtype=float))

    return nested


def create_minirocket_pipeline():
    """Create a MiniRocket + RidgeClassifier pipeline."""
    return Pipeline([
        ("rocket", MiniRocketMultivariateVariable(pad_value_short_series=0.0)),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", RidgeClassifierCV(alphas=np.logspace(-3, 3, 13))),
    ])


def prepare_windowed_data(
    good_trials: list,
    bad_trials: list,
    sensor_cols: list = None,
    window_size: int = 256,
    stride: int = 64,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Prepare windowed data for MiniRocket training.

    Splits trials first, then creates windows to prevent leakage.

    Returns:
        dict with keys: X_train, y_train, X_test, y_test,
                       good_train, good_test, bad_train, bad_test
    """
    if sensor_cols is None:
        sensor_cols = SENSOR_COLS

    # Split trials first
    good_train, good_test = train_test_split(good_trials, test_size=test_size, random_state=random_state)
    bad_train, bad_test = train_test_split(bad_trials, test_size=test_size, random_state=random_state)

    # Create windows
    train_good_win = WindowedTimeSeriesDataset(good_train, window_size=window_size, stride=stride)
    train_bad_win = WindowedTimeSeriesDataset(bad_train, window_size=window_size, stride=stride)
    test_good_win = WindowedTimeSeriesDataset(good_test, window_size=window_size, stride=stride)
    test_bad_win = WindowedTimeSeriesDataset(bad_test, window_size=window_size, stride=stride)

    # Convert to sktime format
    X_train_good = list_of_dfs_to_sktime_nested(train_good_win, sensor_order=sensor_cols)
    X_train_bad = list_of_dfs_to_sktime_nested(train_bad_win, sensor_order=sensor_cols)
    X_test_good = list_of_dfs_to_sktime_nested(test_good_win, sensor_order=sensor_cols)
    X_test_bad = list_of_dfs_to_sktime_nested(test_bad_win, sensor_order=sensor_cols)

    # Combine
    X_train = pd.concat([X_train_good, X_train_bad], ignore_index=True)
    y_train = np.array([1]*len(X_train_good) + [0]*len(X_train_bad))

    X_test = pd.concat([X_test_good, X_test_bad], ignore_index=True)
    y_test = np.array([1]*len(X_test_good) + [0]*len(X_test_bad))

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'good_train': good_train, 'good_test': good_test,
        'bad_train': bad_train, 'bad_test': bad_test,
    }


def get_rolling_predictions(
    trials: list,
    pipe,
    sensor_cols: list = None,
    window_size: int = 256,
    stride: int = 64,
):
    """
    Get rolling window predictions for each trial.

    Args:
        trials: List of trial DataFrames
        pipe: Trained sklearn pipeline
        sensor_cols: Sensor column names
        window_size: Window size in samples
        stride: Stride between windows

    Returns:
        List of dicts with positions, predictions, scores per trial
    """
    if sensor_cols is None:
        sensor_cols = SENSOR_COLS

    results = []
    for trial_idx, trial in enumerate(trials):
        ds = WindowedTimeSeriesDataset([trial], window_size=window_size, stride=stride)
        if len(ds) == 0:
            continue

        X_nested = list_of_dfs_to_sktime_nested(ds, sensor_order=sensor_cols)
        preds = pipe.predict(X_nested)

        try:
            scores = pipe.decision_function(X_nested)
        except:
            scores = preds.astype(float)

        T = len(trial)
        positions = [(i * stride + window_size / 2) / 240.0 for i in range(len(ds))]

        results.append({
            'trial_idx': trial_idx,
            'positions': positions,
            'predictions': preds,
            'scores': scores,
            'trial_length': T / 240.0
        })
    return results


def plot_rolling_classification(
    good_trials: list,
    bad_trials: list,
    good_results: list,
    bad_results: list,
    sensor_cols: list = None,
    window_size: int = 256,
    figsize_per_trial: float = 2.5,
):
    """
    Plot rolling classification heatmap with sensor signals overlaid.

    Args:
        good_trials: List of good trial DataFrames (normalized)
        bad_trials: List of bad trial DataFrames (normalized)
        good_results: Output from get_rolling_predictions for good trials
        bad_results: Output from get_rolling_predictions for bad trials
        sensor_cols: Sensor column names
        window_size: Window size (for rectangle width)
        figsize_per_trial: Figure height per trial
    """
    if sensor_cols is None:
        sensor_cols = SENSOR_COLS

    all_scores = np.concatenate([r['scores'] for r in good_results + bad_results])
    vmin, vmax = all_scores.min(), all_scores.max()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    n_good, n_bad = len(good_results), len(bad_results)
    n_total = n_good + n_bad

    fig, axes = plt.subplots(n_total, 1, figsize=(14, figsize_per_trial * n_total))
    if n_total == 1:
        axes = [axes]

    window_width = window_size / 240.0

    # Plot good trials
    for i, (r, trial) in enumerate(zip(good_results, good_trials)):
        ax = axes[i]
        _plot_trial_with_heatmap(ax, trial, r, sensor_cols, window_width, norm)
        ax.set_title(f'Good Trial {i+1} (True Label: Good)', fontsize=10, fontweight='bold', color='green')

    # Plot bad trials
    for i, (r, trial) in enumerate(zip(bad_results, bad_trials)):
        ax = axes[n_good + i]
        _plot_trial_with_heatmap(ax, trial, r, sensor_cols, window_width, norm)
        ax.set_title(f'Bad Trial {i+1} (True Label: Bad)', fontsize=10, fontweight='bold', color='red')

    axes[-1].set_xlabel('Time (s)')

    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.015, pad=0.02)
    cbar.set_label('Prediction Score\n(Red=Bad, Green=Good)')

    plt.tight_layout()
    return fig, axes


def _plot_trial_with_heatmap(ax, trial, result, sensor_cols, window_width, norm):
    """Helper to plot a single trial with classification heatmap."""
    positions = np.array(result['positions'])
    scores = np.array(result['scores'])

    time = trial['time'].values
    for col in sensor_cols:
        ax.plot(time, trial[col].values, linewidth=0.8, alpha=0.8)

    ylim = ax.get_ylim()

    for pos, score in zip(positions, scores):
        color = plt.cm.RdYlGn(norm(score))
        rect = Rectangle((pos - window_width/2, ylim[0]),
                         window_width, ylim[1] - ylim[0], color=color, alpha=0.3, zorder=0)
        ax.add_patch(rect)

    ax.set_ylim(ylim)
    ax.set_ylabel('Sensor')
    ax.set_xlim(0, result['trial_length'])

"""
Preprocess pitch sensor data for CAE-based anomaly detection.

This script reads raw sensor CSV files from pitch/good and pitch/bad directories,
applies preprocessing, and saves normalized data to processed_data subdirectories.

IMPORTANT: Global Z-Score Normalization
------------------------------------------
We use GLOBAL normalization (statistics computed from good trials only) rather than
per-trial min-max scaling. This is critical because:

1. Per-trial scaling destroys amplitude information - a weak pitch scaled to [-1,1]
   looks identical to a strong pitch after scaling.

2. By computing mean/std from good trials only, bad trials that deviate from normal
   amplitude patterns will have noticeably different normalized values.

3. This preserves the discriminative signal that the autoencoder needs to learn
   what "normal" looks like and detect anomalies.

Pipeline:
1. Pass 1: Compute global mean/std per channel from all GOOD trials
2. Pass 2: Apply z-score normalization to ALL trials using good-trial statistics

Note: Trial43 is skipped (different sensor schema with 34 channels).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
SAMPLING_RATE = 240  # Hz
SKIP_TRIALS = {"Trial43.csv"}  # Different schema (34 channels, V/s units)
SENSOR_COLUMNS = ["25", "26", "27", "28", "29", "30", "31", "32"]

# Spike detection parameters (tuned for 240 Hz motion capture)
SPIKE_DETECTION_DEFAULTS = {
    "window_size": 21,           # ~87ms at 240 Hz
    "derivative_threshold": 3.0,  # std deviations of derivative
    "zscore_threshold": 4.0,      # std deviations from local mean
    "max_consecutive": 5,         # max spike length to fix (~21ms)
}


# =============================================================================
# IMPULSE SPIKE DETECTION AND CORRECTION
# =============================================================================

def compute_local_zscore(signal: np.ndarray, window_size: int = 21) -> np.ndarray:
    """
    Compute robust rolling z-score for each point using median/MAD.

    Improvements over basic mean/std approach:
    1. Uses median instead of mean (robust to outliers)
    2. Uses MAD (Median Absolute Deviation) instead of std
    3. Excludes the center point from window so spike doesn't "explain itself"

    Args:
        signal: 1D array of sensor values
        window_size: Size of rolling window (should be odd)

    Returns:
        zscore: Array of robust local z-scores (same length as signal)
    """
    n = len(signal)
    half_win = window_size // 2
    zscore = np.zeros(n)

    for i in range(n):
        # Define window bounds (handle edges)
        start = max(0, i - half_win)
        end = min(n, i + half_win + 1)

        # Exclude center point so spike doesn't inflate its own statistics
        window = np.concatenate([signal[start:i], signal[i+1:end]])

        if len(window) < 3:
            zscore[i] = 0.0
            continue

        # Use robust statistics: median and MAD
        median_val = np.median(window)
        mad = np.median(np.abs(window - median_val))
        robust_std = 1.4826 * mad  # Scale factor to match normal std

        if robust_std < 1e-10:
            zscore[i] = 0.0
        else:
            zscore[i] = (signal[i] - median_val) / robust_std

    return zscore


def detect_spikes(signal: np.ndarray,
                  window_size: int = 21,
                  derivative_threshold: float = 3.0,
                  zscore_threshold: float = 4.0,
                  require_sign_flip: bool = True) -> np.ndarray:
    """
    Detect impulse spikes using robust statistics and impulse shape detection.

    A point is flagged as a spike if ALL conditions are met:
    1. Its derivative magnitude exceeds threshold (using robust MAD-based threshold)
    2. It is a local statistical outlier (robust z-score with median/MAD)
    3. (Optional) It shows impulse shape: derivative sign flips (spike up then down)

    Improvements over basic approach:
    - Uses MAD instead of std for derivative threshold (robust to outliers)
    - Spikes can't inflate threshold to hide themselves
    - Sign-flip detection rejects legitimate fast movements

    Args:
        signal: 1D array of sensor values
        window_size: Window for local statistics
        derivative_threshold: Threshold in MAD-scaled units of derivative
        zscore_threshold: Threshold for robust local z-score
        require_sign_flip: If True, require derivative sign reversal (impulse shape)

    Returns:
        spike_mask: Boolean array, True where spikes detected
    """
    n = len(signal)
    if n < 3:
        return np.zeros(n, dtype=bool)

    # Step 1: Compute first derivative with ROBUST threshold (MAD)
    deriv = np.diff(signal, prepend=signal[0])
    deriv_median = np.median(deriv)
    deriv_mad = np.median(np.abs(deriv - deriv_median))
    deriv_robust_std = 1.4826 * deriv_mad  # Scale to match normal std

    if deriv_robust_std < 1e-10:
        return np.zeros(n, dtype=bool)

    deriv_outliers = np.abs(deriv) > derivative_threshold * deriv_robust_std

    # Step 2: Compute robust local z-score (using median/MAD, excluding center)
    local_z = compute_local_zscore(signal, window_size)
    zscore_outliers = np.abs(local_z) > zscore_threshold

    # Step 3: Check for impulse shape (sign flip in consecutive derivatives)
    # True impulses spike up then down (or down then up) - derivative reverses sign
    # Legitimate fast movements maintain direction
    if require_sign_flip:
        # Check if derivative at position i has opposite sign from derivative at i+1
        # This detects the "spike shape": rapid change in one direction, then back
        sign_product = deriv[:-1] * deriv[1:]  # negative if signs differ
        sign_flip = np.zeros(n, dtype=bool)
        # A point at index i has sign flip if deriv[i-1] and deriv[i] have opposite signs
        sign_flip[1:] = sign_product < 0

        # Combine all three conditions
        spike_mask = deriv_outliers & zscore_outliers & sign_flip
    else:
        spike_mask = deriv_outliers & zscore_outliers

    return spike_mask


def interpolate_spikes(signal: np.ndarray,
                       spike_mask: np.ndarray,
                       max_consecutive: int = 5) -> tuple:
    """
    Fix detected spikes using linear interpolation.

    Args:
        signal: 1D array of sensor values
        spike_mask: Boolean array marking spike locations
        max_consecutive: Maximum consecutive spikes to fix

    Returns:
        fixed_signal: Signal with spikes interpolated
        fix_info: Dict with statistics about fixes applied
    """
    fixed = signal.copy()
    n = len(signal)

    fix_info = {
        "n_spikes_detected": int(np.sum(spike_mask)),
        "n_spikes_fixed": 0,
        "n_spikes_skipped": 0,
        "spike_indices": [],
    }

    if fix_info["n_spikes_detected"] == 0:
        return fixed, fix_info

    # Find groups of consecutive spikes
    spike_indices = np.where(spike_mask)[0]
    if len(spike_indices) == 0:
        return fixed, fix_info

    # Group consecutive indices
    groups = []
    current_group = [spike_indices[0]]

    for i in range(1, len(spike_indices)):
        if spike_indices[i] == spike_indices[i-1] + 1:
            current_group.append(spike_indices[i])
        else:
            groups.append(current_group)
            current_group = [spike_indices[i]]
    groups.append(current_group)

    # Fix each group
    for group in groups:
        fix_info["spike_indices"].extend(group)

        if len(group) > max_consecutive:
            # Too large a gap - skip
            fix_info["n_spikes_skipped"] += len(group)
            continue

        start_idx = group[0]
        end_idx = group[-1]

        # Find nearest valid neighbors
        left_idx = start_idx - 1 if start_idx > 0 else None
        right_idx = end_idx + 1 if end_idx < n - 1 else None

        if left_idx is not None and right_idx is not None:
            # Linear interpolation
            left_val = signal[left_idx]
            right_val = signal[right_idx]
            n_points = len(group)
            for i, idx in enumerate(group):
                t = (i + 1) / (n_points + 1)
                fixed[idx] = left_val + t * (right_val - left_val)
            fix_info["n_spikes_fixed"] += len(group)
        elif left_idx is not None:
            # Boundary case: extend from left
            for idx in group:
                fixed[idx] = signal[left_idx]
            fix_info["n_spikes_fixed"] += len(group)
        elif right_idx is not None:
            # Boundary case: extend from right
            for idx in group:
                fixed[idx] = signal[right_idx]
            fix_info["n_spikes_fixed"] += len(group)
        else:
            # Signal too short, skip
            fix_info["n_spikes_skipped"] += len(group)

    return fixed, fix_info


def detect_and_fix_spikes(signal: np.ndarray, **kwargs) -> tuple:
    """
    Main entry point: detect and fix impulse spikes in a signal.

    Uses robust statistics (MAD) and impulse shape detection (sign-flip).

    Args:
        signal: 1D array of sensor values
        **kwargs: Override default detection parameters:
            - window_size: Rolling window size (default: 21)
            - derivative_threshold: Derivative outlier threshold (default: 3.0)
            - zscore_threshold: Z-score outlier threshold (default: 4.0)
            - max_consecutive: Max spikes to fix in a row (default: 5)
            - require_sign_flip: Require impulse shape (default: True)

    Returns:
        fixed_signal: Signal with spikes corrected
        info: Dict with detection/fix statistics
    """
    params = {**SPIKE_DETECTION_DEFAULTS, **kwargs}

    spike_mask = detect_spikes(
        signal,
        window_size=params["window_size"],
        derivative_threshold=params["derivative_threshold"],
        zscore_threshold=params["zscore_threshold"],
        require_sign_flip=params.get("require_sign_flip", True)
    )

    fixed_signal, info = interpolate_spikes(
        signal, spike_mask,
        max_consecutive=params["max_consecutive"]
    )

    return fixed_signal, info


def load_raw_trial(filepath: str) -> pd.DataFrame:
    """
    Load a raw pitch trial CSV file.

    The raw files have TWO sections concatenated:
    Section 1 (voltage data - what we want):
    - Row 0: Device type ("Devices")
    - Row 1: Sampling rate (240)
    - Row 2: Metadata
    - Row 3: Column headers (Frame, Sub Frame, 25-32)
    - Row 4: Units (V)
    - Row 5+: Data

    Section 2 (derivative data - ignore):
    - Starts with another "Devices" line mid-file
    - Has 34 columns with V/s units

    We only read Section 1.
    """
    # First, read the file line by line to find where Section 2 starts
    with open(filepath, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Find the second occurrence of "Devices" (start of Section 2)
    devices_indices = []
    for i, line in enumerate(lines):
        if line.strip().startswith("Devices"):
            devices_indices.append(i)

    # Determine how many lines to read (stop before Section 2)
    if len(devices_indices) >= 2:
        nrows = devices_indices[1] - 5  # Subtract header rows (0-4)
    else:
        nrows = None  # Read all if only one section

    # Read with header at row 3 (0-indexed), skip rows 0-2 and row 4 (units)
    df = pd.read_csv(filepath, header=3, skiprows=[4], nrows=nrows, encoding="utf-8-sig")

    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()

    # Keep only Frame, Sub Frame, and sensor columns
    keep_cols = ["Frame", "Sub Frame"] + SENSOR_COLUMNS
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()

    # Drop rows with NaN values (from empty lines in raw data)
    df = df.dropna()

    return df


def compute_time(df: pd.DataFrame, fs: float = SAMPLING_RATE) -> np.ndarray:
    """
    Convert Frame/SubFrame to time in seconds.

    Each frame contains multiple subframes. Time is computed as:
    time = (frame_index) / fs

    Where frame_index increments for each row.
    """
    n_samples = len(df)
    return np.arange(n_samples) / fs


def compute_global_stats(filepaths: list, fix_spikes: bool = True) -> dict:
    """
    Compute global mean and std for each sensor channel from a list of trials.

    This preserves amplitude differences between trials by using
    consistent normalization across all data.

    Args:
        filepaths: List of trial file paths
        fix_spikes: Whether to fix spikes before computing stats (recommended)
    """
    all_values = {col: [] for col in SENSOR_COLUMNS}

    for filepath in filepaths:
        try:
            df = load_raw_trial(filepath)

            # Fix spikes before computing stats (so stats aren't polluted)
            if fix_spikes:
                for col in SENSOR_COLUMNS:
                    if col in df.columns:
                        fixed_values, _ = detect_and_fix_spikes(
                            df[col].values.astype(np.float64)
                        )
                        df[col] = fixed_values

            for col in SENSOR_COLUMNS:
                if col in df.columns:
                    values = df[col].values.astype(np.float64)
                    all_values[col].extend(values.tolist())
        except Exception as e:
            print(f"  Warning: Could not read {filepath} for stats: {e}")

    stats = {}
    for col in SENSOR_COLUMNS:
        if all_values[col]:
            arr = np.array(all_values[col])
            stats[col] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr))
            }
            # Prevent division by zero
            if stats[col]["std"] < 1e-10:
                stats[col]["std"] = 1.0

    return stats


def z_score_normalize(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Apply z-score normalization using provided statistics."""
    return (arr - mean) / std


def preprocess_trial(filepath: str, global_stats: dict = None,
                     fix_spikes: bool = True, spike_params: dict = None) -> tuple:
    """
    Preprocess a single trial.

    Returns DataFrame with columns: time, 25, 26, ..., 32
    Also returns spike_info dict with detection statistics per channel.

    Args:
        filepath: Path to raw CSV file
        global_stats: Dict of per-channel mean/std for normalization
        fix_spikes: Whether to detect and fix impulse spikes (default: True)
        spike_params: Override default spike detection parameters

    If global_stats is provided, applies z-score normalization using those stats.
    Otherwise, falls back to per-trial min-max scaling (not recommended).
    """
    df = load_raw_trial(filepath)

    # Fix spikes BEFORE normalization (on raw voltage data)
    spike_info = {}
    if fix_spikes:
        for col in SENSOR_COLUMNS:
            if col in df.columns:
                fixed_values, info = detect_and_fix_spikes(
                    df[col].values.astype(np.float64),
                    **(spike_params or {})
                )
                df[col] = fixed_values
                spike_info[col] = info

    # Compute time
    time = compute_time(df)

    # Extract and scale sensor data
    result = {"time": time}
    for col in SENSOR_COLUMNS:
        if col in df.columns:
            values = df[col].values.astype(np.float64)
            if global_stats and col in global_stats:
                # Use global z-score normalization (preserves amplitude info)
                result[col] = z_score_normalize(
                    values,
                    global_stats[col]["mean"],
                    global_stats[col]["std"]
                )
            else:
                # Fallback: per-trial scaling (loses amplitude info)
                arr_min, arr_max = values.min(), values.max()
                if arr_max - arr_min < 1e-10:
                    result[col] = np.zeros_like(values)
                else:
                    result[col] = 2 * (values - arr_min) / (arr_max - arr_min) - 1

    return pd.DataFrame(result), spike_info


def get_valid_csv_files(input_dir: str) -> list:
    """Get list of valid CSV files (excluding SKIP_TRIALS)."""
    input_path = Path(input_dir)
    csv_files = sorted(input_path.glob("*.csv"))
    return [f for f in csv_files if f.name not in SKIP_TRIALS]


def process_directory(input_dir: str, output_dir: str, global_stats: dict = None,
                      fix_spikes: bool = True):
    """
    Process all CSV files in a directory.

    Skips files in SKIP_TRIALS set.
    Uses global_stats for normalization if provided.

    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Directory for processed output files
        global_stats: Dict of per-channel mean/std for normalization
        fix_spikes: Whether to detect and fix impulse spikes
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_path.glob("*.csv"))
    processed_count = 0
    skipped_count = 0
    total_spikes_fixed = 0

    for csv_file in csv_files:
        if csv_file.name in SKIP_TRIALS:
            print(f"  Skipping {csv_file.name} (different schema)")
            skipped_count += 1
            continue

        try:
            df, spike_info = preprocess_trial(
                str(csv_file),
                global_stats=global_stats,
                fix_spikes=fix_spikes
            )
            output_file = output_path / csv_file.name
            df.to_csv(output_file, index=False)

            # Count spikes fixed
            trial_spikes = sum(info["n_spikes_fixed"] for info in spike_info.values())
            total_spikes_fixed += trial_spikes

            spike_str = f", {trial_spikes} spikes fixed" if trial_spikes > 0 else ""
            print(f"  Processed: {csv_file.name} -> {len(df)} samples{spike_str}")
            processed_count += 1
        except Exception as e:
            print(f"  ERROR processing {csv_file.name}: {e}")

    return processed_count, skipped_count, total_spikes_fixed


def main():
    # Get script directory
    script_dir = Path(__file__).parent

    print("=" * 60)
    print("Pitch Data Preprocessing")
    print("(Spike Detection + Global Z-Score Normalization)")
    print("=" * 60)

    good_input = script_dir / "good"
    bad_input = script_dir / "bad"
    good_output = script_dir / "good" / "processed_data"
    bad_output = script_dir / "bad" / "processed_data"

    # PASS 1: Compute global statistics from GOOD trials only
    # This ensures bad trials are normalized relative to what "normal" looks like
    # Spikes are fixed before computing stats so they don't pollute the statistics
    print("\nPass 1: Computing global statistics from GOOD trials...")
    print("  (Spike detection enabled - fixing spikes before computing stats)")
    good_files = get_valid_csv_files(str(good_input))
    global_stats = compute_global_stats([str(f) for f in good_files], fix_spikes=True)

    print(f"  Computed stats from {len(good_files)} good trials")
    for col in SENSOR_COLUMNS[:3]:  # Show first 3 channels
        if col in global_stats:
            s = global_stats[col]
            print(f"  Channel {col}: mean={s['mean']:.4f}, std={s['std']:.4f}")
    print("  ...")

    # PASS 2: Apply global normalization to ALL trials (with spike fixing)
    print("\nPass 2: Processing GOOD trials (spike fix + global normalization):")
    good_processed, good_skipped, good_spikes = process_directory(
        good_input, good_output, global_stats, fix_spikes=True
    )

    print("\nPass 2: Processing BAD trials (spike fix + global normalization):")
    bad_processed, bad_skipped, bad_spikes = process_directory(
        bad_input, bad_output, global_stats, fix_spikes=True
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Good trials: {good_processed} processed, {good_skipped} skipped")
    print(f"Bad trials:  {bad_processed} processed, {bad_skipped} skipped")
    print(f"Total:       {good_processed + bad_processed} processed")
    print(f"\nSpike correction: {good_spikes + bad_spikes} total spikes fixed")
    print(f"  Good trials: {good_spikes} spikes")
    print(f"  Bad trials:  {bad_spikes} spikes")
    print(f"\nNormalization: Global z-score (stats from good trials)")
    print(f"\nOutput directories:")
    print(f"  Good: {good_output}")
    print(f"  Bad:  {bad_output}")


if __name__ == "__main__":
    main()

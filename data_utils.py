"""
Shared data utilities for baseball swing sensor processing.

Pulled out of the notebook to make preprocessing reusable, testable, and
configurable. Follows a simple parse -> clean -> save pipeline with
deterministic ordering and lightweight logging.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class CleanConfig:
    fs: float = 220.0
    R1: float = 5000.0
    stop_at_next_devices: bool = True
    v_full: float = 5.0
    near_full_delta: float = 0.02
    v2r_mode: str = "nan"  # 'nan' or 'clip'
    saturation_v: float = 4.5
    interp_limit: int = 5  # max gap (samples) to interpolate
    baseline_len: int = 100
    scale_p_hi: float = 99.0


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def _find_header_index(lines: List[str]) -> int:
    header_pat = re.compile(r"^\s*Frame\s*,\s*Sub\s*Frame\b", re.IGNORECASE)
    for i, line in enumerate(lines):
        if header_pat.search(line):
            return i
    raise ValueError("Header line starting with 'Frame, Sub Frame' was not found.")


def _truncate_at_next_devices(lines: List[str], start: int) -> List[str]:
    dev_pat = re.compile(r"^\s*['\"]?\s*Devices\b", re.IGNORECASE)
    for j in range(start + 1, len(lines)):
        if dev_pat.search(lines[j]):
            return lines[start:j]
    return lines[start:]


def _read_block_to_df(block_lines: List[str]) -> pd.DataFrame:
    df = pd.read_csv(StringIO("\n".join(block_lines)), header=0)
    if df.empty:
        return df

    first_row = df.iloc[0].astype(str).str.strip().tolist()
    unit_like = re.compile(r"^[A-Za-zµμ/%°ΩohmVvAakKHz\s\-]+$")
    non_empty = [v for v in first_row if v not in ("", "nan", "None")]
    if non_empty and all(unit_like.match(v) for v in non_empty):
        df = df.iloc[1:].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: str(c).strip() for c in df.columns})


def _drop_frame_cols(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in df.columns if str(c).lower().replace(" ", "") in ("frame", "subframe")]
    return df.drop(columns=to_drop, errors="ignore")


def _coerce_numeric_inplace(df: pd.DataFrame) -> None:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")


def _insert_time_column(df: pd.DataFrame, fs: float) -> None:
    n = len(df)
    t = np.arange(n, dtype=float) / float(fs)
    df.insert(0, "time", t)


def _detect_voltage_scale(col: pd.Series, v_full: float = 5.0) -> tuple[str, float]:
    s = col.dropna()
    if s.empty:
        return ("V", 1.0)
    q99 = float(s.quantile(0.99))
    if 600 < q99 <= 6000:
        scale = 1023.0 if abs(q99 - 1023) < abs(q99 - 4095) else 4095.0
        return ("counts", scale)
    if 6.0 < q99 <= 6000.0 and s.median() > 1.0:
        return ("mV", 1000.0)
    return ("V", 1.0)


def _normalize_to_volts_inplace(df: pd.DataFrame, v_full: float = 5.0) -> None:
    ycols = [c for c in df.columns if c != "time" and pd.api.types.is_numeric_dtype(df[c])]
    for c in ycols:
        mode, scale = _detect_voltage_scale(df[c], v_full=v_full)
        if mode == "counts":
            df[c] = (df[c] / scale) * v_full
        elif mode == "mV":
            df[c] = df[c] / 1000.0


def _V_to_R_array(V: np.ndarray, R1: float, v_full: float = 5.0,
                  delta: float = 0.02, mode: str = "nan") -> np.ndarray:
    V = V.astype(float)
    if mode == "clip":
        Vc = np.clip(V, 0.0, v_full - delta)
        return R1 * (Vc / (v_full - Vc))
    invalid = (~np.isfinite(V)) | (V < 0) | (V >= v_full - delta)
    denom = v_full - V
    R = R1 * (V / denom)
    R[invalid | (denom <= 0)] = np.nan
    return R


def _convert_all_channels_V_to_R_inplace(df: pd.DataFrame, R1: float,
                                         v_full: float = 5.0,
                                         delta: float = 0.02,
                                         mode: str = "nan") -> None:
    for c in [c for c in df.columns if c != "time" and pd.api.types.is_numeric_dtype(df[c])]:
        df[c] = _V_to_R_array(df[c].to_numpy(), R1=R1, v_full=v_full, delta=delta, mode=mode)


def apply_baseline(df: pd.DataFrame, columns: Iterable[str], baseline_len: int = 100, robust: bool = True):
    for col in columns:
        data = df[col].to_numpy(dtype=float)
        n0 = min(baseline_len, len(data))
        base = np.nanmedian(data[:n0]) if robust else np.nanmean(data[:n0])
        df[col] = data - base


def _scale_symmetric(df: pd.DataFrame, columns: Iterable[str], p_hi: float = 99.0, eps: float = 1e-12):
    for c in columns:
        x = df[c].astype(float).to_numpy()
        lim = np.nanpercentile(np.abs(x), p_hi)
        if not np.isfinite(lim) or lim <= eps:
            df[c] = 0.0
        else:
            df[c] = np.clip(x / (lim + eps), -1.0, 1.0)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_trim_convert(filepath: Path, cfg: Optional[CleanConfig] = None) -> pd.DataFrame:
    """
    Load a logger CSV-like export and return a cleaned DataFrame with:
      - 'time' (s) from 0 with step 1/fs
      - sensor columns converted to resistance, baseline-subtracted, scaled to ~[-1,1]
    """
    cfg = cfg or CleanConfig()
    lines = _read_lines(filepath)
    header_idx = _find_header_index(lines)
    block_lines = _truncate_at_next_devices(lines, header_idx) if cfg.stop_at_next_devices else lines[header_idx:]

    df = _read_block_to_df(block_lines)
    df = _normalize_colnames(df)
    _coerce_numeric_inplace(df)
    df = _drop_frame_cols(df)

    _insert_time_column(df, fs=cfg.fs)
    ycols = [c for c in df.columns if c != "time"]

    _normalize_to_volts_inplace(df, v_full=cfg.v_full)

    # Mask near-saturation, interpolate short gaps
    for c in ycols:
        df.loc[df[c] >= cfg.saturation_v, c] = np.nan
    if cfg.interp_limit and cfg.interp_limit > 0:
        df[ycols] = df[ycols].interpolate(limit=cfg.interp_limit, limit_direction="both")

    _convert_all_channels_V_to_R_inplace(df, R1=cfg.R1, v_full=cfg.v_full,
                                         delta=cfg.near_full_delta, mode=cfg.v2r_mode)
    apply_baseline(df, ycols, baseline_len=cfg.baseline_len)
    _scale_symmetric(df, ycols, p_hi=cfg.scale_p_hi)
    return df


def clean_folder(input_dir: Path, output_dir: Optional[Path] = None,
                 cfg: Optional[CleanConfig] = None, verbose: bool = True) -> list[Path]:
    """
    Clean every CSV in a folder (deterministic sorted order) and write to output_dir.
    Returns list of written file paths.
    """
    cfg = cfg or CleanConfig()
    input_dir = Path(input_dir)
    output_dir = output_dir or input_dir / "processed_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for entry in sorted(input_dir.glob("*.csv")):
        df = load_trim_convert(entry, cfg)
        out_path = output_dir / entry.name
        df.to_csv(out_path, index=False)
        written.append(out_path)
        if verbose:
            sat_ratio = float(np.mean([df[c].isna().mean() for c in df.columns if c != "time"]))
            print(f"[clean] {entry.name} -> {out_path.name} (nan_ratio≈{sat_ratio:.3f})")
    return written


def load_cleaned_trials(folder: Path, pattern: str = "*.csv") -> list[pd.DataFrame]:
    """
    Load already-cleaned CSVs (e.g., from processed_data) into DataFrames.
    """
    dfs: list[pd.DataFrame] = []
    for entry in sorted(Path(folder).glob(pattern)):
        dfs.append(pd.read_csv(entry))
    return dfs


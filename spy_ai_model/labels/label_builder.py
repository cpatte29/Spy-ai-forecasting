"""
label_builder.py
────────────────
Constructs forward-looking labels for each bar.

Labels
──────
y_dir   : int   1 if close[t+horizon_dir] > close[t], else 0
y_range : float (max(high[t+1 : t+horizon_range]) - min(low[t+1 : t+horizon_range])) / close[t]

Each label is NaN for the last max(horizon_dir, horizon_range) rows.

All computations are performed *after* the current bar closes, so there
is strictly no look-ahead on the current bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import HORIZON_DIR, HORIZON_RANGE


def build_labels(
    df: pd.DataFrame,
    horizon_dir: int | None = None,
    horizon_range: int | None = None,
    # legacy single-horizon shim
    horizon: int | None = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV bars, DatetimeIndex sorted ascending.
    horizon_dir : int or None
        Bars ahead for the direction label.  Defaults to config.HORIZON_DIR.
    horizon_range : int or None
        Bars ahead for the range label.  Defaults to config.HORIZON_RANGE.
    horizon : int or None
        Legacy shorthand – sets both horizons when provided.

    Returns
    -------
    pd.DataFrame with columns y_dir and y_range, same index as df.
    """
    if horizon is not None:
        horizon_dir   = horizon_dir   or horizon
        horizon_range = horizon_range or horizon

    h_dir = HORIZON_DIR   if horizon_dir   is None else horizon_dir
    h_rng = HORIZON_RANGE if horizon_range is None else horizon_range

    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n     = len(df)

    y_dir   = np.full(n, np.nan)
    y_range = np.full(n, np.nan)

    for t in range(n - h_dir):
        c_t = close[t]
        y_dir[t] = 1.0 if close[t + h_dir] > c_t else 0.0

    for t in range(n - h_rng):
        c_t = close[t]
        future_high = high[t + 1 : t + h_rng + 1].max()
        future_low  = low[t  + 1 : t + h_rng + 1].min()
        y_range[t]  = (future_high - future_low) / c_t

    return pd.DataFrame(
        {"y_dir": y_dir, "y_range": y_range},
        index=df.index,
    )

"""
label_builder.py
────────────────
Constructs forward-looking labels for each 1-minute bar.

Labels
──────
y_dir   : int   1 if close[t+HORIZON] > close[t], else 0
y_range : float (max(high[t+1 : t+HORIZON]) - min(low[t+1 : t+HORIZON])) / close[t]

Both labels are NaN for the last HORIZON rows of each trading day
(no complete future window available).

All computations are performed *after* the current bar closes, so there
is strictly no look-ahead on the current bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import HORIZON


def build_labels(df: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        1-minute OHLCV bars, DatetimeIndex sorted ascending.
    horizon : int or None
        Prediction horizon in bars.  Defaults to config.HORIZON when None.

    Returns
    -------
    pd.DataFrame with columns y_dir and y_range, same index as df.
    Rows that lack a complete horizon-bar future window are NaN.
    """
    h = HORIZON if horizon is None else horizon

    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n     = len(df)

    y_dir   = np.full(n, np.nan)
    y_range = np.full(n, np.nan)

    for t in range(n - h):
        c_t      = close[t]
        c_future = close[t + h]

        # direction: 1 if close h bars ahead is higher
        y_dir[t] = 1.0 if c_future > c_t else 0.0

        # range: max-high minus min-low over the NEXT h bars
        future_high = high[t + 1 : t + h + 1].max()
        future_low  = low[t  + 1 : t + h + 1].min()
        y_range[t]  = (future_high - future_low) / c_t

    result = pd.DataFrame(
        {"y_dir": y_dir, "y_range": y_range},
        index=df.index,
    )
    return result

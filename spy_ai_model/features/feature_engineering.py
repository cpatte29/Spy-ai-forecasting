"""
feature_engineering.py
───────────────────────
All rolling features are computed causally (look-back only) so there is
zero look-ahead / data-leakage.

Feature groups
──────────────
Momentum      ret_1, ret_3, ret_5, ret_15, ret_30
Trend         dist_ema_9/20/50/200, slope_ema_9_5, slope_ema_20_5
VWAP          dist_vwap, vwap_slope_5
Volatility    rv_5/15/30, range_mean_5/15
Candlestick   body_to_range, upper_wick_to_range, lower_wick_to_range, close_location
Volume        vol_rel_5, vol_rel_15
Structure     rolling_high_15_dist, rolling_low_15_dist
Time          minutes_since_open, tod_sin, tod_cos
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    EMA_WINDOWS, SLOPE_LOOKBACK,
    RV_WINDOWS, RANGE_WINDOWS,
    VOL_REL_WINDOWS, ROLLING_HL_WINDOW,
    MARKET_OPEN,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _log_ret(close: pd.Series, lag: int) -> pd.Series:
    return np.log(close / close.shift(lag))


def _realised_vol(log_rets: pd.Series, window: int) -> pd.Series:
    """Rolling std of log returns (causal)."""
    return log_rets.rolling(window, min_periods=window).std()


# ── VWAP (resets each day) ────────────────────────────────────────────────────

def _daily_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute intraday VWAP that resets at each market open.
    Uses typical price = (H+L+C)/3 * volume, cumulative within the day.
    """
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    tpv = tp * df["volume"]

    date_key = df.index.normalize()
    cum_tpv  = tpv.groupby(date_key).cumsum()
    cum_vol  = df["volume"].groupby(date_key).cumsum()

    return cum_tpv / cum_vol.replace(0, np.nan)


# ── main builder ──────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        1-minute OHLCV bars, DatetimeIndex, tz-naive, sorted ascending.

    Returns
    -------
    pd.DataFrame with one feature column per feature.  Rows at the start
    of the series that can't be computed will be NaN (handled by caller).
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"].astype(float)

    feat: dict[str, pd.Series] = {}

    # ── Momentum ──────────────────────────────────────────────────────────────
    for lag in [1, 3, 5, 15, 30]:
        feat[f"ret_{lag}"] = _log_ret(close, lag)

    # ── Trend: EMA distances and slopes ──────────────────────────────────────
    ema_series: dict[int, pd.Series] = {}
    for w in EMA_WINDOWS:
        e = _ema(close, w)
        ema_series[w] = e
        feat[f"dist_ema_{w}"] = (close - e) / close

    for w in [9, 20]:
        e = ema_series[w]
        feat[f"slope_ema_{w}_{SLOPE_LOOKBACK}"] = (e - e.shift(SLOPE_LOOKBACK)) / close

    # ── VWAP ──────────────────────────────────────────────────────────────────
    vwap = _daily_vwap(df)
    feat["dist_vwap"]    = (close - vwap) / close
    feat["vwap_slope_5"] = (vwap - vwap.shift(5)) / close

    # ── Volatility ────────────────────────────────────────────────────────────
    log_ret_1 = _log_ret(close, 1)
    for w in RV_WINDOWS:
        feat[f"rv_{w}"] = _realised_vol(log_ret_1, w)

    bar_range = high - low
    for w in RANGE_WINDOWS:
        feat[f"range_mean_{w}"] = bar_range.rolling(w, min_periods=w).mean() / close

    # ── Candlestick geometry ──────────────────────────────────────────────────
    rng    = (high - low).replace(0, np.nan)
    body   = (df["close"] - df["open"]).abs()

    feat["body_to_range"]        = body / rng
    feat["upper_wick_to_range"]  = (high - pd.concat([df["close"], df["open"]], axis=1).max(axis=1)) / rng
    feat["lower_wick_to_range"]  = (pd.concat([df["close"], df["open"]], axis=1).min(axis=1) - low) / rng
    # close_location: 0 = bottom of bar, 1 = top
    feat["close_location"]       = (close - low) / rng

    # ── Volume ────────────────────────────────────────────────────────────────
    for w in VOL_REL_WINDOWS:
        avg_vol = volume.rolling(w, min_periods=w).mean().replace(0, np.nan)
        feat[f"vol_rel_{w}"] = volume / avg_vol

    # ── Structure: rolling high/low distances ─────────────────────────────────
    w = ROLLING_HL_WINDOW
    roll_high = high.rolling(w, min_periods=w).max()
    roll_low  = low.rolling(w,  min_periods=w).min()
    feat[f"rolling_high_{w}_dist"] = (roll_high - close) / close
    feat[f"rolling_low_{w}_dist"]  = (close - roll_low)  / close

    # ── Time features ─────────────────────────────────────────────────────────
    open_minutes = int(MARKET_OPEN.split(":")[0]) * 60 + int(MARKET_OPEN.split(":")[1])
    bar_minutes  = df.index.hour * 60 + df.index.minute
    mins_since   = bar_minutes - open_minutes

    feat["minutes_since_open"] = pd.Series(mins_since, index=df.index)

    total_day_minutes = 390.0   # 6.5 trading hours
    frac = mins_since / total_day_minutes
    feat["tod_sin"] = pd.Series(np.sin(2 * np.pi * frac), index=df.index)
    feat["tod_cos"] = pd.Series(np.cos(2 * np.pi * frac), index=df.index)

    # ── Assemble ──────────────────────────────────────────────────────────────
    result = pd.DataFrame(feat, index=df.index)
    return result

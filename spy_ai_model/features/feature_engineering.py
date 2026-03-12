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
Session       minutes_to_close, opening_range_high, opening_range_low,
              dist_orb_high, dist_orb_low, orb_break_flag, orb_reject_flag,
              power_hour_flag, lunch_hour_flag, day_of_week
VWAP Regime   vwap_reclaim_flag, vwap_loss_flag, vwap_trend_strength,
              vwap_distance_percentile
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
    MARKET_OPEN, MARKET_CLOSE,
    OPENING_RANGE_BARS,
)

# ── public constant: new features added in this module extension ───────────────
# Used by compare_session_features.py to split baseline vs. enhanced sets.
NEW_SESSION_FEATURES: list[str] = [
    "minutes_to_close",
    "opening_range_high",
    "opening_range_low",
    "dist_orb_high",
    "dist_orb_low",
    "orb_break_flag",
    "orb_reject_flag",
    "power_hour_flag",
    "lunch_hour_flag",
    "day_of_week",
]

# VWAP regime features added on top of the 39-feature session-enhanced set.
# Used by compare_vwap_persistence.py.
NEW_VWAP_FEATURES: list[str] = [
    "vwap_reclaim_flag",        # 1 on bars where price crosses back above VWAP
    "vwap_loss_flag",           # 1 on bars where price crosses below VWAP
    "vwap_trend_strength",      # rolling mean of sign(close-vwap), ranges −1…+1
    "vwap_distance_percentile", # rolling %-rank of |dist_vwap| vs. recent history
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rolling_percentile(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """
    Rolling percentile rank of the current value within its own look-back window.

    For each bar, returns the fraction of the previous (window-1) values that are
    strictly less than the current value.  Fully causal – uses only past data.

    Returns values in [0, 1].  NaN when fewer than min_periods observations exist.
    """
    mp = min_periods if min_periods is not None else max(2, window // 2)

    def _pct_rank(x: np.ndarray) -> float:
        # x[-1] is the current bar; x[:-1] are the look-back values
        if len(x) < 2:
            return np.nan
        return float((x[:-1] < x[-1]).mean())

    return series.rolling(window, min_periods=mp).apply(_pct_rank, raw=True)


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


# ── Opening range (resets each day) ──────────────────────────────────────────

def _opening_range(df: pd.DataFrame, n_bars: int) -> tuple[pd.Series, pd.Series]:
    """
    Compute the daily opening range high and low.

    The opening range is defined by the first n_bars bars of each session.
    This function is fully causal: bars within the first n_bars return NaN
    (the ORB is still forming); bars from position n_bars onward receive the
    completed ORB value.

    Implementation uses a vectorised groupby + cummax/cummin approach:
      1. Mask out bars outside the ORB window to NaN.
      2. groupby(date).cummax() forward-fills the running extreme through NaN,
         so post-ORB bars retain the final ORB value.
      3. Mask out the ORB-window bars so the first n_bars per day are NaN.

    Returns
    -------
    (orb_high, orb_low) : pd.Series, same index as df
    """
    date_key = df.index.normalize()
    bar_num  = df.groupby(date_key).cumcount()   # 0-indexed position within day

    # Within-window values; everything after → NaN
    orb_h_raw = df["high"].where(bar_num < n_bars, np.nan)
    orb_l_raw = df["low"].where(bar_num < n_bars, np.nan)

    # cummax / cummin gives the running extreme through bar n_bars-1, but
    # pandas cummax/cummin does NOT forward-fill through NaN on its own.
    # A subsequent groupby ffill() carries the last in-window value across all
    # post-ORB bars within the same session.
    orb_h_cummax = orb_h_raw.groupby(date_key).cummax()
    orb_l_cummin = orb_l_raw.groupby(date_key).cummin()

    orb_h_cummax = orb_h_cummax.groupby(date_key).ffill()
    orb_l_cummin = orb_l_cummin.groupby(date_key).ffill()

    # Expose only post-ORB bars (causal)
    orb_h = orb_h_cummax.where(bar_num >= n_bars, np.nan)
    orb_l = orb_l_cummin.where(bar_num >= n_bars, np.nan)

    return orb_h, orb_l


# ── main builder ──────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV bars with DatetimeIndex, tz-naive ET, sorted ascending.

    Returns
    -------
    pd.DataFrame with one column per feature.  Rows at the start of the
    series that cannot be computed will be NaN (handled by the caller via
    dropna in dataset_builder).
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

    # ── VWAP regime features ──────────────────────────────────────────────────
    # Binary flag: price was below VWAP last bar and is above VWAP now (reclaim).
    # Computed from the close-vs-vwap sign; NaN propagated when vwap is NaN.
    above_vwap = (close > vwap).astype(float).where(vwap.notna(), np.nan)
    feat["vwap_reclaim_flag"] = (
        (above_vwap == 1.0) & (above_vwap.shift(1) == 0.0)
    ).astype(float).where(vwap.notna() & vwap.shift(1).notna(), np.nan)

    # Binary flag: price was above VWAP last bar and is below VWAP now (loss).
    feat["vwap_loss_flag"] = (
        (above_vwap == 0.0) & (above_vwap.shift(1) == 1.0)
    ).astype(float).where(vwap.notna() & vwap.shift(1).notna(), np.nan)

    # Trend strength: rolling mean of sign(close − vwap) over 10 bars.
    # +1.0 = consistently above VWAP; −1.0 = consistently below.
    vwap_sign = pd.Series(
        np.where(close > vwap, 1.0, np.where(close < vwap, -1.0, 0.0)),
        index=df.index,
    ).where(vwap.notna(), np.nan)
    feat["vwap_trend_strength"] = vwap_sign.rolling(10, min_periods=5).mean()

    # Distance percentile: how extreme is the current |dist_vwap| vs. past 20 bars?
    dist_vwap_abs = (close - vwap).abs() / close.where(close != 0, np.nan)
    feat["vwap_distance_percentile"] = _rolling_percentile(dist_vwap_abs, window=20, min_periods=10)

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
    open_minutes  = int(MARKET_OPEN.split(":")[0])  * 60 + int(MARKET_OPEN.split(":")[1])
    close_minutes = int(MARKET_CLOSE.split(":")[0]) * 60 + int(MARKET_CLOSE.split(":")[1])
    bar_minutes   = df.index.hour * 60 + df.index.minute
    mins_since    = bar_minutes - open_minutes

    feat["minutes_since_open"] = pd.Series(mins_since, index=df.index)

    total_day_minutes = 390.0   # 6.5 trading hours
    frac = mins_since / total_day_minutes
    feat["tod_sin"] = pd.Series(np.sin(2 * np.pi * frac), index=df.index)
    feat["tod_cos"] = pd.Series(np.cos(2 * np.pi * frac), index=df.index)

    # ── Session-aware features ────────────────────────────────────────────────

    # Minutes remaining in the session
    feat["minutes_to_close"] = pd.Series(
        close_minutes - bar_minutes, index=df.index, dtype=float
    )

    # Opening range (first OPENING_RANGE_BARS bars per session = 30 min for 5m)
    orb_h, orb_l = _opening_range(df, OPENING_RANGE_BARS)

    # Normalized ORB levels (ratio to current close)
    # opening_range_high > 1.0 means ORB high is above current price
    feat["opening_range_high"] = orb_h / close
    feat["opening_range_low"]  = orb_l / close

    # Signed distances: positive = price is below ORB high / above ORB low
    feat["dist_orb_high"] = (orb_h - close) / close
    feat["dist_orb_low"]  = (close - orb_l) / close

    # Break flag: +1 close above ORB high, -1 close below ORB low, 0 inside
    orb_break = np.where(
        orb_h.isna(), np.nan,
        np.where(close > orb_h,  1.0,
        np.where(close < orb_l, -1.0,
                 0.0))
    )
    feat["orb_break_flag"] = pd.Series(orb_break, index=df.index)

    # Reject flag: wick tested an ORB level but close stayed inside the range
    #   upper reject: high exceeded ORB high but close <= ORB high (false breakout up)
    #   lower reject: low fell below ORB low but close >= ORB low (false breakout down)
    upper_reject = (df["high"] > orb_h) & (close <= orb_h)
    lower_reject = (df["low"]  < orb_l) & (close >= orb_l)
    orb_reject   = np.where(
        orb_h.isna(), np.nan,
        (upper_reject | lower_reject).astype(float)
    )
    feat["orb_reject_flag"] = pd.Series(orb_reject, index=df.index)

    # Power hour: last 60 min of the session (3:00–4:00 PM ET = 330–390 min since open)
    feat["power_hour_flag"] = pd.Series(
        (mins_since >= 330).astype(float), index=df.index
    )

    # Lunch lull: 11:30 AM – 1:00 PM ET (120–210 min since open)
    feat["lunch_hour_flag"] = pd.Series(
        ((mins_since >= 120) & (mins_since < 210)).astype(float), index=df.index
    )

    # Day of week: Monday=0, Tuesday=1, …, Friday=4
    feat["day_of_week"] = pd.Series(
        df.index.dayofweek.astype(float), index=df.index
    )

    # ── Assemble ──────────────────────────────────────────────────────────────
    result = pd.DataFrame(feat, index=df.index)
    return result

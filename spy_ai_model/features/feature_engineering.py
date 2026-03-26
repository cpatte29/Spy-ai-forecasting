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
Gap / Prior   overnight_gap_pct, dist_prior_high, dist_prior_low,
              dist_prior_close, prior_day_range, prior_day_trend,
              opened_in_prior_range, prior_close_reclaimed, gap_fill_pct,
              prior_day_return_pct, opened_above_prior_high_flag,
              opened_below_prior_low_flag, prior_day_vwap_distance
Vol Regime    rv_regime_pct, vol_trend_5, signal_noise_ratio
              These three features give the model vocabulary to detect
              choppy / high-uncertainty market regimes (e.g. macro shock,
              news-driven volatility) vs. clean trending conditions.
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

# Volatility-regime features — help the model detect choppy / macro-shock conditions.
# Add these to a retrain to improve edge in high-uncertainty environments.
NEW_VOL_REGIME_FEATURES: list[str] = [
    "rv_regime_pct",       # rolling %-rank of rv_5 over last 60 bars: 0=quiet, 1=crisis
    "vol_trend_5",         # (rv_5 - rv_5[5]) / rv_5[5]: positive=expanding, negative=contracting
    "signal_noise_ratio",  # |ret_5| / rv_5: near 0 in chop, high in clean trend
]

# VWAP regime features added on top of the 39-feature session-enhanced set.
# Used by compare_vwap_persistence.py.
NEW_VWAP_FEATURES: list[str] = [
    "vwap_reclaim_flag",        # 1 on bars where price crosses back above VWAP
    "vwap_loss_flag",           # 1 on bars where price crosses below VWAP
    "vwap_trend_strength",      # rolling mean of sign(close-vwap), ranges −1…+1
    "vwap_distance_percentile", # rolling %-rank of |dist_vwap| vs. recent history
]

# Overnight-gap and prior-day context features.
# Used by compare_gap_features.py to isolate the lift from this feature group.
NEW_GAP_FEATURES: list[str] = [
    # ── original 9 ──────────────────────────────────────────────────────────
    "overnight_gap_pct",             # (session_open − prior_close) / prior_close; constant/session
    "dist_prior_high",               # (close − prior_high) / close; positive = above prior high
    "dist_prior_low",                # (close − prior_low)  / close; positive = above prior low
    "dist_prior_close",              # (close − prior_close) / close; signed distance from prior close
    "prior_day_range",               # (prior_high − prior_low) / prior_close; constant/session
    "prior_day_trend",               # +1 prior day bullish, −1 bearish, 0 flat; constant/session
    "opened_in_prior_range",         # 1 if session open ∈ [prior_low, prior_high]; constant/session
    "prior_close_reclaimed",         # 1 if current close ≥ prior close; dynamic/bar
    "gap_fill_pct",                  # fraction of overnight gap recovered (0=none, 1=full); dynamic
    # ── extension 4 ─────────────────────────────────────────────────────────
    "prior_day_return_pct",          # (prior_close − prior_open) / prior_open; magnitude+direction
    "opened_above_prior_high_flag",  # 1 if session open > prior high (gap-up breakout); constant/session
    "opened_below_prior_low_flag",   # 1 if session open < prior low  (gap-down breakdown); constant/session
    "prior_day_vwap_distance",       # (close − prior_day_vwap) / close; VWAP computed from OHLCV
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


# ── Prior day context ─────────────────────────────────────────────────────────

def _prior_day_context(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Compute 9 overnight-gap and prior-day reference-level features.

    All features are strictly causal: they use only data from the prior
    calendar session (which closed before today's open).  The first calendar
    day in df will have NaN values for all 9 outputs — downstream dropna()
    removes those rows automatically.

    Per-session constants (same value for every bar in a given day)
    ───────────────────────────────────────────────────────────────
    overnight_gap_pct     (session_open − prior_close) / prior_close
                          Positive = gap up; negative = gap down.

    prior_day_range       (prior_high − prior_low) / prior_close
                          Normalised prior-session price range.

    prior_day_trend       +1 prior day bullish (close > open)
                          −1 prior day bearish
                           0 flat

    opened_in_prior_range 1 if today's session open ∈ [prior_low, prior_high]
                          0 if the session opened with a true gap outside the
                          prior range.

    Per-bar variables (change each bar as price moves)
    ───────────────────────────────────────────────────
    dist_prior_high       (close − prior_high) / close
                          Positive = above prior high (breakout).
                          Negative = below prior high (resistance overhead).

    dist_prior_low        (close − prior_low) / close
                          Positive = above prior low (support intact).
                          Negative = below prior low (breakdown).

    dist_prior_close      (close − prior_close) / close
                          Signed distance from the prior close reference.

    prior_close_reclaimed 1 if current close ≥ prior close; 0 otherwise.
                          Turns on once price reclaims that level intraday.

    gap_fill_pct          Fraction of the overnight gap recovered so far.
                          Formula: (close − session_open) / (prior_close − session_open)
                            0 = price is still at the session open (no fill)
                            1 = price has returned to prior close (fully filled)
                          Positive values mean filling a gap-up (price drifted down).
                          Negative values mean the gap is extending.
                          Clamped to [−2, 2]; set to 0 when gap ≈ 0.

    prior_day_vwap_distance (close − prior_vwap) / close
                          Distance from prior session's cumulative VWAP (at close).
                          Computed entirely from OHLCV; no external column needed.

    Extension constants (one value per session)
    ─────────────────────────────────────────────
    prior_day_return_pct  (prior_close − prior_open) / prior_open
                          Signed magnitude of the prior session's return.
                          Positive = prior day up; negative = prior day down.
                          More informative than prior_day_trend (±1 only).

    opened_above_prior_high_flag  1 if session_open > prior_high (gap-up breakout)
    opened_below_prior_low_flag   1 if session_open < prior_low  (gap-down breakdown)
                          Together with opened_in_prior_range these three flags
                          partition every session into exactly one gap category.
    """
    close    = df["close"]
    date_key = df.index.normalize()   # DatetimeIndex of each bar's calendar date

    # ── Daily aggregates (indexed by calendar date) ───────────────────────────
    grp          = df.groupby(date_key)
    daily_high   = grp["high"].max()
    daily_low    = grp["low"].min()
    daily_close  = grp["close"].last()
    daily_open   = grp["open"].first()   # session open price

    # ── Prior-day shift (index = current date, value = prior date's stat) ─────
    prior_high   = daily_high.shift(1)
    prior_low    = daily_low.shift(1)
    prior_close  = daily_close.shift(1)
    prior_open   = daily_open.shift(1)

    # ── Broadcast daily → bar level ───────────────────────────────────────────
    def _to_bar(daily: pd.Series) -> pd.Series:
        """Map a date-indexed daily series onto the bar-level DatetimeIndex."""
        return pd.Series(daily.reindex(date_key).values, index=df.index)

    pc  = _to_bar(prior_close)   # prior close at bar level
    ph  = _to_bar(prior_high)    # prior high  at bar level
    pl  = _to_bar(prior_low)     # prior low   at bar level
    po  = _to_bar(prior_open)    # prior open  at bar level
    so  = _to_bar(daily_open)    # today's session open at bar level

    # Avoid division by zero
    c_safe  = close.replace(0, np.nan)
    pc_safe = pc.replace(0, np.nan)

    feat: dict[str, pd.Series] = {}

    # 1. overnight_gap_pct – fraction of prior close                  [constant/session]
    feat["overnight_gap_pct"] = (so - pc) / pc_safe

    # 2. dist_prior_high – signed distance to prior high              [dynamic/bar]
    feat["dist_prior_high"] = (close - ph) / c_safe

    # 3. dist_prior_low – signed distance to prior low                [dynamic/bar]
    feat["dist_prior_low"] = (close - pl) / c_safe

    # 4. dist_prior_close – signed distance to prior close            [dynamic/bar]
    feat["dist_prior_close"] = (close - pc) / c_safe

    # 5. prior_day_range – normalised prior session range             [constant/session]
    feat["prior_day_range"] = (ph - pl) / pc_safe

    # 6. prior_day_trend – directional bias of prior session          [constant/session]
    #    Re-index onto date_key so we can broadcast through _to_bar.
    prior_trend_daily = pd.Series(
        np.sign((prior_close - prior_open).values).astype(float),
        index=daily_close.index,
    )
    feat["prior_day_trend"] = _to_bar(prior_trend_daily)

    # 7. opened_in_prior_range – gap flag                             [constant/session]
    in_range = ((so >= pl) & (so <= ph)).astype(float)
    feat["opened_in_prior_range"] = in_range.where(ph.notna() & pl.notna(), np.nan)

    # 8. prior_close_reclaimed – has price retaken the prior close?   [dynamic/bar]
    feat["prior_close_reclaimed"] = (close >= pc).astype(float).where(pc.notna(), np.nan)

    # 9. gap_fill_pct – fraction of overnight gap recovered           [dynamic/bar]
    #    denom = prior_close − session_open  (= −gap_size)
    #    → 0 when close == session_open (no move from open)
    #    → 1 when close == prior_close  (full fill)
    #    fillna(0) when gap ≈ 0 (avoid division by ~0 noise)
    denom = (pc - so).replace(0, np.nan)
    feat["gap_fill_pct"] = ((close - so) / denom).clip(-2.0, 2.0).fillna(0.0)

    # ── Extension features ────────────────────────────────────────────────────

    # 10. prior_day_return_pct – signed magnitude of prior session move [constant/session]
    #     Positive = prior day closed higher than it opened (bullish).
    #     Negative = prior day closed lower (bearish).
    #     Carries more information than prior_day_trend (which is just ±1/0).
    po_safe = po.replace(0, np.nan)
    prior_return_daily = pd.Series(
        ((prior_close - prior_open) / prior_open.replace(0, np.nan)).values,
        index=daily_close.index,
    )
    feat["prior_day_return_pct"] = _to_bar(prior_return_daily)

    # 11. opened_above_prior_high_flag – gap-up breakout              [constant/session]
    #     1 if today's session open price is strictly above the prior day high.
    #     Combines with opened_in_prior_range (which would be 0 in this case)
    #     to fully characterise the three gap categories:
    #       opened_above_prior_high = 1 → gap-up breakout
    #       opened_in_prior_range   = 1 → opened inside prior range (no gap)
    #       opened_below_prior_low  = 1 → gap-down breakdown
    above_ph = (so > ph).astype(float)
    feat["opened_above_prior_high_flag"] = above_ph.where(ph.notna(), np.nan)

    # 12. opened_below_prior_low_flag – gap-down breakdown            [constant/session]
    below_pl = (so < pl).astype(float)
    feat["opened_below_prior_low_flag"] = below_pl.where(pl.notna(), np.nan)

    # 13. prior_day_vwap_distance – distance from prior session's VWAP [dynamic/bar]
    #     Uses the cumulative intraday VWAP (typical price * volume, reset daily)
    #     and takes the LAST value of each session as that day's closing VWAP.
    #     This is fully causal: prior_vwap only uses data up to yesterday's close.
    #     Computed entirely from OHLCV — no dependency on the Polygon vwap column.
    intraday_vwap     = _daily_vwap(df)
    daily_vwap_close  = intraday_vwap.groupby(date_key).last()   # session-close VWAP
    prior_vwap_daily  = daily_vwap_close.shift(1)
    pv                = _to_bar(prior_vwap_daily)
    pv_safe           = pv.replace(0, np.nan)
    feat["prior_day_vwap_distance"] = (close - pv) / c_safe

    return feat


# ── main builder ──────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, include_gap_features: bool = True) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV bars with DatetimeIndex, tz-naive ET, sorted ascending.
    include_gap_features : bool, default True
        When True (default), appends the 9 overnight-gap / prior-day context
        features from NEW_GAP_FEATURES.  Pass False to reproduce the baseline
        43-feature set for A/B comparisons.

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

    # ── Volatility regime ─────────────────────────────────────────────────
    # rv_regime_pct: where does current rv_5 sit in its own 60-bar history?
    # 0.0 = quietest, 1.0 = most volatile.  Captures macro / shock regimes.
    rv5 = feat["rv_5"]
    feat["rv_regime_pct"] = _rolling_percentile(rv5, window=60, min_periods=20)

    # vol_trend_5: is volatility expanding (+) or contracting (-)?
    # Normalised by the lagged value so it's scale-invariant.
    rv5_lag5 = rv5.shift(5).replace(0, np.nan)
    feat["vol_trend_5"] = ((rv5 - rv5_lag5) / rv5_lag5).clip(-2.0, 2.0)

    # signal_noise_ratio: how strong is the directional signal vs. noise?
    # High = clean trending move; near 0 = choppy / indecisive market.
    feat["signal_noise_ratio"] = (feat["ret_5"].abs() / rv5.replace(0, np.nan)).clip(0.0, 5.0)

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

    # ── Overnight gap / prior-day context features (optional) ────────────────
    if include_gap_features:
        feat.update(_prior_day_context(df))

    # ── Assemble ──────────────────────────────────────────────────────────────
    result = pd.DataFrame(feat, index=df.index)
    return result

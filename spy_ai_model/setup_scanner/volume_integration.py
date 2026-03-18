"""
volume_integration.py
─────────────────────
Volume confirmation layer for the setup scanner.

Computes three independent volume signals and combines them into a single
volume_score (-10 to +15) that is added to the setup scorer's total.

Signals
───────
1. Relative Volume (RVOL)
   Current bar volume vs the average volume at the same time-of-day across
   the most recent sessions in df_bars.  RVOL > 1 means above-average
   participation; RVOL < 1 means thin / low-conviction tape.

2. Short-term Volume Imbalance
   Compare cumulative up-bar volume vs down-bar volume over the last N bars
   (default 5).  An imbalance aligned with the setup direction adds
   conviction; opposing imbalance is a warning sign.

3. Breakout / Rejection Confirmation
   For BREAKOUT and BREAKDOWN setups: RVOL >= 1.3 on the decisive bar.
   For REJECTION and BOUNCE setups:   RVOL >= 1.3 on the reaction bar AND
   opposing wick present.

Volume Regimes
──────────────
  HIGH    RVOL >= 1.3
  NORMAL  0.7 <= RVOL < 1.3
  LOW     RVOL < 0.7

Volume Score (added to setup total, capped at final [0, 100] clamp)
────────────────────────────────────────────────────────────────────
  +15   RVOL >= 1.5, imbalance strongly aligned with setup
  +12   RVOL >= 1.3, imbalance aligned
  +8    RVOL >= 1.3, imbalance neutral
  +4    RVOL >= 1.0, mild alignment
   0    RVOL 0.7-1.0, neutral or no data
  -5    RVOL < 0.7 (low participation)
  -10   RVOL < 0.7 AND imbalance opposing setup direction

The score is directionally aware: "aligned" means bullish imbalance for
LONG setups and bearish imbalance for SHORT setups.

Usage
─────
  from setup_scanner.volume_integration import compute_volume_context, score_volume

  vol_ctx = compute_volume_context(df_bars)
  pts     = score_volume(vol_ctx, setup_type, direction)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

from setup_scanner.setup_definitions import SetupDirection, SetupType

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────

_TOD_LOOKBACK_SESSIONS = 20    # max sessions used for TOD average
_IMBALANCE_BARS        = 5     # bars for up/down volume comparison
_RVOL_HIGH             = 1.30
_RVOL_VERY_HIGH        = 1.50
_RVOL_LOW              = 0.70
_IMBALANCE_STRONG      = 0.65  # up/(up+down) >= 0.65 → strong up bias
_IMBALANCE_WEAK        = 0.35  # up/(up+down) <= 0.35 → strong down bias

# Setup types that benefit from high volume on the decisive bar
_BREAKOUT_SETUPS = {
    SetupType.BREAKOUT_ABOVE_SUPPLY,
    SetupType.BREAKDOWN_BELOW_DEMAND,
}
_REACTION_SETUPS = {
    SetupType.DEMAND_BOUNCE,
    SetupType.SUPPLY_REJECTION,
    SetupType.TREND_PULLBACK_CONTINUATION,
}


# ── time-of-day RVOL computation ───────────────────────────────────────────────

def _tod_average_volume(
    df: pd.DataFrame,
    bar_ts: pd.Timestamp,
    lookback_sessions: int = _TOD_LOOKBACK_SESSIONS,
) -> float | None:
    """
    Return the mean volume at the same time-of-day (hour + minute) across the
    most recent `lookback_sessions` trading sessions in df.

    df.index is assumed to be tz-naive ET DatetimeIndex sorted ascending.
    Returns None if fewer than 3 historical matches are found.
    """
    target_time = (bar_ts.hour, bar_ts.minute)

    # Filter to same time-of-day, excluding the current bar itself
    mask = (
        (df.index.hour   == target_time[0]) &
        (df.index.minute == target_time[1]) &
        (df.index        <  bar_ts)
    )
    tod_bars = df.loc[mask]

    if len(tod_bars) < 3:
        return None

    # Use only the most recent N sessions
    recent = tod_bars.iloc[-lookback_sessions:]
    return float(recent["volume"].mean())


def _session_average_volume(df: pd.DataFrame, bar_ts: pd.Timestamp) -> float | None:
    """
    Fallback: mean volume across the current session only (all bars today).
    Used when there is insufficient TOD history.
    """
    today = bar_ts.date()
    session = df[df.index.date == today]
    if len(session) < 2:
        return None
    # Exclude the current bar
    before = session[session.index < bar_ts]
    if len(before) < 1:
        return None
    return float(before["volume"].mean())


# ── volume imbalance ──────────────────────────────────────────────────────────

def _volume_imbalance(df: pd.DataFrame, n_bars: int = _IMBALANCE_BARS) -> float | None:
    """
    Return up-volume fraction = up_vol / (up_vol + down_vol) over the last
    n_bars (inclusive of the current bar).

    Returns None if there are fewer than 2 bars or total volume is zero.

    Values:
      > 0.65  → buyers clearly dominating
      < 0.35  → sellers clearly dominating
      ~0.50   → balanced
    """
    if len(df) < 2:
        return None

    window = df.iloc[-n_bars:]
    up_mask   = window["close"] >= window["open"]
    down_mask = window["close"] <  window["open"]

    up_vol   = float(window.loc[up_mask,   "volume"].sum())
    down_vol = float(window.loc[down_mask, "volume"].sum())
    total    = up_vol + down_vol

    if total <= 0:
        return None

    return up_vol / total


# ── public API ────────────────────────────────────────────────────────────────

def compute_volume_context(
    df_bars:            pd.DataFrame,
    bar_ts:             pd.Timestamp | None = None,
    tod_lookback:       int                 = _TOD_LOOKBACK_SESSIONS,
    imbalance_bars:     int                 = _IMBALANCE_BARS,
) -> dict:
    """
    Compute all volume signals for the scored bar.

    Parameters
    ──────────
    df_bars         OHLCV DataFrame up to and including the scored bar.
                    Index is tz-naive ET DatetimeIndex.
    bar_ts          Timestamp of the bar to score.  Defaults to last bar.
    tod_lookback    Sessions of history for TOD average (default 20).
    imbalance_bars  Bars for up/down volume comparison (default 5).

    Returns
    ───────
    dict with keys:
        rvol                  float | None   current bar vol / TOD avg vol
        tod_avg_volume        float | None   TOD average volume
        current_volume        float | None   volume of the scored bar
        volume_regime         str            "LOW" | "NORMAL" | "HIGH"
        vol_imbalance         float | None   up_vol / (up+down), last N bars
        imbalance_label       str            "BULLISH" | "BEARISH" | "NEUTRAL"
        breakout_confirmed    bool           RVOL >= 1.3 on decisive bar
        rejection_confirmed   bool           RVOL >= 1.3 on reaction bar
        low_participation     bool           RVOL < 0.70
    """
    if df_bars.empty:
        return _empty_context()

    if bar_ts is None:
        bar_ts = df_bars.index[-1]

    # ── Current bar volume ────────────────────────────────────────────────
    try:
        current_vol = float(df_bars.loc[bar_ts, "volume"])
    except KeyError:
        current_vol = float(df_bars.iloc[-1]["volume"])

    # ── TOD average ───────────────────────────────────────────────────────
    tod_avg = _tod_average_volume(df_bars, bar_ts, tod_lookback)

    # Fallback to session average if TOD history is thin
    if tod_avg is None:
        tod_avg = _session_average_volume(df_bars, bar_ts)

    # ── RVOL ──────────────────────────────────────────────────────────────
    rvol: float | None = None
    if tod_avg is not None and tod_avg > 0:
        rvol = current_vol / tod_avg

    # ── Volume regime ─────────────────────────────────────────────────────
    if rvol is None:
        regime = "NORMAL"   # unknown → assume normal
    elif rvol >= _RVOL_HIGH:
        regime = "HIGH"
    elif rvol < _RVOL_LOW:
        regime = "LOW"
    else:
        regime = "NORMAL"

    # ── Volume imbalance (last N bars) ────────────────────────────────────
    imbalance = _volume_imbalance(df_bars, imbalance_bars)
    if imbalance is None:
        imb_label = "NEUTRAL"
    elif imbalance >= _IMBALANCE_STRONG:
        imb_label = "BULLISH"
    elif imbalance <= _IMBALANCE_WEAK:
        imb_label = "BEARISH"
    else:
        imb_label = "NEUTRAL"

    # ── Confirmation flags ────────────────────────────────────────────────
    high_vol           = (rvol is not None and rvol >= _RVOL_HIGH)
    breakout_confirmed = high_vol                # for breakout/breakdown setups
    rejection_confirmed = high_vol               # for rejection/bounce setups
    low_participation  = (rvol is not None and rvol < _RVOL_LOW)

    return {
        "rvol":                 rvol,
        "tod_avg_volume":       tod_avg,
        "current_volume":       current_vol,
        "volume_regime":        regime,
        "vol_imbalance":        imbalance,
        "imbalance_label":      imb_label,
        "breakout_confirmed":   breakout_confirmed,
        "rejection_confirmed":  rejection_confirmed,
        "low_participation":    low_participation,
    }


def _empty_context() -> dict:
    return {
        "rvol":                 None,
        "tod_avg_volume":       None,
        "current_volume":       None,
        "volume_regime":        "NORMAL",
        "vol_imbalance":        None,
        "imbalance_label":      "NEUTRAL",
        "breakout_confirmed":   False,
        "rejection_confirmed":  False,
        "low_participation":    False,
    }


# ── scoring ───────────────────────────────────────────────────────────────────

def score_volume(
    volume_ctx: dict,
    setup_type: SetupType,
    direction:  SetupDirection,
) -> int:
    """
    Convert volume context into a signed score: -10 to +15.

    Parameters
    ──────────
    volume_ctx   Output of compute_volume_context().
    setup_type   The setup being scored.
    direction    LONG / SHORT / NEUTRAL for this setup.

    Returns
    ───────
    int in [-10, +15].  The caller adds this to the running total before
    the final [0, 100] clamp — no additional clamping is applied here.
    """
    if setup_type == SetupType.NO_SETUP:
        return 0

    rvol      = volume_ctx.get("rvol")
    imb       = volume_ctx.get("vol_imbalance")
    imb_label = volume_ctx.get("imbalance_label", "NEUTRAL")
    regime    = volume_ctx.get("volume_regime", "NORMAL")

    # ── No RVOL data → neutral ────────────────────────────────────────────
    if rvol is None:
        return 0

    # ── Imbalance alignment ───────────────────────────────────────────────
    # "aligned" = imbalance supports the setup direction
    if direction == SetupDirection.LONG:
        imb_aligned  = imb_label == "BULLISH"
        imb_opposing = imb_label == "BEARISH"
    elif direction == SetupDirection.SHORT:
        imb_aligned  = imb_label == "BEARISH"
        imb_opposing = imb_label == "BULLISH"
    else:
        imb_aligned  = False
        imb_opposing = False

    # ── Low participation penalty ─────────────────────────────────────────
    if regime == "LOW":
        if imb_opposing:
            return -10   # thin tape AND opposing flow
        return -5        # thin tape alone

    # ── Normal volume ─────────────────────────────────────────────────────
    if regime == "NORMAL":
        if rvol >= 1.0 and imb_aligned:
            return 4     # above-average within normal range, aligned
        return 0         # plain neutral

    # ── High volume ───────────────────────────────────────────────────────
    # regime == "HIGH" (rvol >= 1.3)
    if rvol >= _RVOL_VERY_HIGH:           # rvol >= 1.5
        if imb_aligned:
            return 15    # very high volume + aligned flow → strongest signal
        elif imb_opposing:
            return 4     # high vol but opposing → caution, partial credit
        else:
            return 8     # high vol, neutral imbalance
    else:                                 # 1.3 <= rvol < 1.5
        if imb_aligned:
            return 12
        elif imb_opposing:
            return 3
        else:
            return 8     # matches spec: RVOL >= 1.3, neutral → +8


def volume_regime_label(volume_ctx: dict) -> str:
    """
    Return a compact one-line label for report display.

    Examples:
      "RVOL 1.42  │  HIGH  │  ✓ CONFIRMED"
      "RVOL 0.61  │  LOW   │  ⚠ LOW PARTICIPATION"
      "RVOL 1.05  │  NORMAL│  — neutral"
      "RVOL n/a   │  NORMAL│  — no TOD data"
    """
    rvol   = volume_ctx.get("rvol")
    regime = volume_ctx.get("volume_regime", "NORMAL")
    brk_c  = volume_ctx.get("breakout_confirmed",  False)
    rej_c  = volume_ctx.get("rejection_confirmed", False)
    low_p  = volume_ctx.get("low_participation",   False)
    imb    = volume_ctx.get("imbalance_label", "NEUTRAL")

    rvol_str = f"{rvol:.2f}" if rvol is not None else "n/a"

    if brk_c or rej_c:
        conf_str = "✓ VOLUME CONFIRMED"
    elif low_p:
        conf_str = "⚠ LOW PARTICIPATION"
    else:
        conf_str = f"─ {imb.lower()} imbalance"

    return f"RVOL {rvol_str:5s}  │  {regime:<6s}  │  {conf_str}"

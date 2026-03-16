"""
zone_scorer.py
──────────────
Score the strength of a supply or demand zone on a 0–1 scale.

Five factors are combined into a single zone_strength score:

  1. Displacement size   (weight 0.25)
       How far price moved away from the zone after formation.
       Anchor: 1.5% displacement → full score.

  2. Volume spike        (weight 0.20)
       How much above-average volume was present at zone formation.
       Anchor: 3× average volume → full score.

  3. Prior rejections    (weight 0.25)
       How many times price returned to the zone and was turned away
       without closing through the zone boundary.
       Anchor: 3 rejections → full score.

  4. Freshness           (weight 0.15)
       Exponential decay from zone creation — fresher zones more likely
       to still have unfilled institutional orders.
       Half-life: 100 bars (~8.3 hours of 5-min bars).

  5. Major trend move    (weight 0.15)
       Whether the zone caused a significant directional impulse (large
       displacement × fast speed).

Usage
─────
  from confluence_engine.zone_scorer import score_zone_strength

  strength = score_zone_strength(zone_row, df_bars, zone_type="supply",
                                  bars_since=bars_since_created)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── normalization anchors ─────────────────────────────────────────────────────
_DISP_FULL        = 0.015   # 1.5 % displacement → full displacement score
_VOL_SPIKE_FULL   = 3.0     # 3× average volume  → full volume score
_REJ_FULL         = 3       # 3 prior rejections  → full rejection score
_DECAY_HALF_BARS  = 100     # freshness half-life in bars (~8.3 h at 5-min)
_MAJOR_DISP_MIN   = 0.008   # 0.8 % – threshold for "major" move displacement
_MAJOR_SPEED_MIN  = 0.001   # speed threshold for major move classification

# ── factor weights (must sum to 1.0) ─────────────────────────────────────────
_W_DISP       = 0.25
_W_VOLUME     = 0.20
_W_REJECTION  = 0.25
_W_FRESHNESS  = 0.15
_W_MAJOR_MOVE = 0.15


# ── public API ────────────────────────────────────────────────────────────────

def score_zone_strength(
    zone:       "pd.Series",
    df:         "pd.DataFrame",
    zone_type:  str        = "supply",
    bars_since: int | None = None,
) -> float:
    """
    Return a zone strength score in [0.0, 1.0].

    Parameters
    ──────────
    zone        One row from a supply_zones or demand_zones DataFrame.
    df          Full bar history (used for rejection counting + freshness).
    zone_type   "supply" or "demand".
    bars_since  Pre-computed bar distance from creation to last bar in df.
                If None, computed from df internally.

    Returns
    ───────
    float in [0.0, 1.0] — higher means stronger / more institutional zone.
    """
    # ── factor 1: displacement size ───────────────────────────────────────
    disp = float(zone.get("displacement_size", 0.0))
    f_disp = min(disp / max(_DISP_FULL, 1e-9), 1.0)

    # ── factor 2: volume spike ────────────────────────────────────────────
    vsr = float(zone.get("volume_spike_ratio", 1.0))
    # Scale: 1.0× (no spike) → 0.0 … 3.0× spike → 1.0
    f_vol = max(0.0, min((vsr - 1.0) / max(_VOL_SPIKE_FULL - 1.0, 1e-9), 1.0))

    # ── factor 3: prior rejections ────────────────────────────────────────
    n_rej = _count_rejections(zone, df, zone_type)
    f_rej = min(n_rej / max(_REJ_FULL, 1), 1.0)

    # ── factor 4: freshness (exponential decay) ───────────────────────────
    if bars_since is None:
        bars_since = _bars_since_creation(zone, df)
    bars_since = max(0, bars_since or 0)
    f_fresh = float(np.exp(-bars_since / max(_DECAY_HALF_BARS, 1)))

    # ── factor 5: major trend move ────────────────────────────────────────
    speed = float(zone.get("displacement_speed", 0.0))
    major_disp  = min(disp  / max(_MAJOR_DISP_MIN,  1e-9), 1.0)
    major_speed = min(speed / max(_MAJOR_SPEED_MIN, 1e-9), 1.0)
    f_major = min(major_disp * major_speed, 1.0)

    # ── weighted combination ──────────────────────────────────────────────
    strength = (
        _W_DISP       * f_disp
        + _W_VOLUME     * f_vol
        + _W_REJECTION  * f_rej
        + _W_FRESHNESS  * f_fresh
        + _W_MAJOR_MOVE * f_major
    )
    return round(float(np.clip(strength, 0.0, 1.0)), 4)


# ── internal helpers ──────────────────────────────────────────────────────────

def _count_rejections(
    zone:      "pd.Series",
    df:        "pd.DataFrame",
    zone_type: str,
) -> int:
    """
    Count how many times price visited the zone after its creation and was
    rejected (did not close through the zone boundary).

    A "visit" is a run of consecutive bars where price overlaps the zone
    body (high >= zone_low for supply; low <= zone_high for demand).
    The visit is a rejection if no bar in the run closed through the
    opposite edge of the zone.
    """
    if df.empty:
        return 0

    creation_ts = zone.get("creation_timestamp")
    if creation_ts is None:
        return 0

    try:
        df_after = df[df.index > creation_ts]
    except Exception:
        return 0

    if df_after.empty:
        return 0

    zone_high = float(zone.get("zone_high", 0.0))
    zone_low  = float(zone.get("zone_low",  0.0))

    highs  = df_after["high"].to_numpy(dtype=float)
    lows   = df_after["low"].to_numpy(dtype=float)
    closes = df_after["close"].to_numpy(dtype=float)

    rejections    = 0
    in_touch      = False
    episode_broke = False

    if zone_type == "supply":
        # Touch: bar enters zone (high >= zone_low)
        # Break: close > zone_high (closed through zone top)
        # Rejection: episode ended without a breakout close
        breakout_threshold = zone_high * 1.001
        for i in range(len(highs)):
            touches = highs[i] >= zone_low
            if touches:
                if not in_touch:
                    in_touch      = True
                    episode_broke = False
                if closes[i] > breakout_threshold:
                    episode_broke = True
            else:
                if in_touch:
                    if not episode_broke:
                        rejections += 1
                    in_touch      = False
                    episode_broke = False
    else:
        # Demand zone
        # Touch: bar enters zone (low <= zone_high)
        # Break: close < zone_low (closed through zone bottom)
        # Rejection: episode ended without a breakdown close
        breakdown_threshold = zone_low * 0.999
        for i in range(len(lows)):
            touches = lows[i] <= zone_high
            if touches:
                if not in_touch:
                    in_touch      = True
                    episode_broke = False
                if closes[i] < breakdown_threshold:
                    episode_broke = True
            else:
                if in_touch:
                    if not episode_broke:
                        rejections += 1
                    in_touch      = False
                    episode_broke = False

    return rejections


def _bars_since_creation(zone: "pd.Series", df: "pd.DataFrame") -> int | None:
    """Return bars between zone creation timestamp and the last bar in df."""
    creation_ts = zone.get("creation_timestamp")
    if creation_ts is None or df.empty:
        return None
    try:
        idx = df.index.searchsorted(creation_ts)
        return max(0, len(df) - int(idx) - 1)
    except Exception:
        return None

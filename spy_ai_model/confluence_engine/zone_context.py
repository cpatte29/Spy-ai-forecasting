"""
zone_context.py
───────────────
Detect the nearest supply and demand zones relative to the current price
and return a structured context dict consumed by the confluence scorer.

Supply zone (resistance overhead)
──────────────────────────────────
  Formed by a pivot HIGH followed by significant downward displacement.
  Delegates to analog_engine.zone_detector.detect_supply_zones().
  Zone boundaries: zone_high = pivot bar high; zone_low = body base.

Demand zone (support below)
────────────────────────────
  Mirror image of a supply zone: pivot LOW followed by significant upward
  displacement.  Implemented here directly (zone_detector only covers supply).
  Zone boundaries: zone_low = pivot bar low; zone_high = body top.

Output dict structure
─────────────────────
  {
    "current_price": float,
    "supply": {
        "zone_high":             float | None
        "zone_low":              float | None
        "distance_to_zone_pct":  float | None   # (zone_low - price) / price
        "inside_zone_flag":      bool
        "bars_since_created":    int  | None
        "displacement_size":     float | None
        "volume_spike_ratio":    float | None
        "n_zones_found":         int
    },
    "demand": { same keys, distance = (price - zone_high) / price },
    "bias":  "SUPPLY_OVERHEAD" | "AT_SUPPLY" | "DEMAND_BELOW" | "AT_DEMAND" |
             "BETWEEN_ZONES"  | "NEUTRAL"
  }

Usage
─────
  from confluence_engine.zone_context import get_zone_context

  ctx = get_zone_context(df_bars, current_price=580.42, bar_interval="5m")
  print(ctx["bias"])
  print(ctx["supply"]["distance_to_zone_pct"])
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
# A zone counts as "overhead" if zone_low > current_price.
# A zone counts as "at" (inside) if zone_low <= current_price <= zone_high.
# A zone counts as "below" if zone_high < current_price.

# Bar-count threshold: only flag "close to zone" within this relative distance.
_CLOSE_TO_ZONE_PCT = 0.005   # 0.5 % from the nearest zone edge


# ── demand zone detection ─────────────────────────────────────────────────────

def _find_pivot_lows(df: pd.DataFrame, left: int = 5, right: int = 5) -> pd.Series:
    """
    Return a boolean Series marking pivot-low bars.

    A bar at position i is a pivot low when its low is the minimum in the
    (left + 1 + right) bar window centred on i.  Tie-breaking: the rightmost
    equal value wins (consistent with zone_detector.find_pivot_highs()).
    """
    if len(df) < left + right + 1:
        return pd.Series(False, index=df.index)

    lows     = df["low"].to_numpy(dtype=float)
    n        = len(lows)
    is_pivot = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        window_min = lows[i - left : i + right + 1].min()
        if lows[i] == window_min:
            if right == 0 or lows[i] < lows[i + 1 : i + right + 1].min():
                is_pivot[i] = True

    return pd.Series(is_pivot, index=df.index)


def detect_demand_zones(
    df:                    pd.DataFrame,
    pivot_left:            int   = 5,
    pivot_right:           int   = 5,
    min_displacement_pct:  float = 0.003,
    max_displacement_bars: int   = 20,
    volume_spike_factor:   float = 1.5,
    require_volume_spike:  bool  = False,
) -> pd.DataFrame:
    """
    Detect demand zones from historical bars — the mirror of supply zones.

    A demand zone is formed when:
      1. A pivot LOW exists (bar's low is lowest in the symmetric window).
      2. Within `max_displacement_bars`, price rises at least
         `min_displacement_pct` from the pivot low (upward displacement).
      3. Optional volume spike at the pivot bar.

    Zone boundaries
    ───────────────
      zone_low  = pivot bar low         (base of the demand zone)
      zone_high = max(open, close)      (body top of the pivot candle)

    Returns
    ───────
    pd.DataFrame – one row per demand zone, same column schema as
    detect_supply_zones(), sorted by creation_timestamp ascending.
    """
    if df.empty or len(df) < pivot_left + pivot_right + 2:
        return _empty_demand_zones()

    vol_ma  = df["volume"].rolling(20, min_periods=1).mean()
    pos_map = {ts: i for i, ts in enumerate(df.index)}

    pivot_mask       = _find_pivot_lows(df, left=pivot_left, right=pivot_right)
    pivot_timestamps = df.index[pivot_mask].tolist()

    records: list[dict] = []

    for ts in pivot_timestamps:
        pos       = pos_map[ts]
        row       = df.iloc[pos]
        piv_low   = float(row["low"])
        piv_open  = float(row["open"])
        piv_close = float(row["close"])
        piv_vol   = float(row["volume"])
        avg_vol   = float(vol_ma.iloc[pos])

        if require_volume_spike and avg_vol > 0:
            if piv_vol < volume_spike_factor * avg_vol:
                continue

        # Upward displacement window
        disp_slice = df.iloc[pos + 1 : pos + 1 + max_displacement_bars]
        if disp_slice.empty:
            continue

        max_high_idx = int(disp_slice["high"].argmax())
        max_high_val = float(disp_slice["high"].iloc[max_high_idx])
        max_high_ts  = disp_slice.index[max_high_idx]

        displacement_size = (max_high_val - piv_low) / max(piv_low, 1e-8)
        if displacement_size < min_displacement_pct:
            continue

        bars_to_high       = pos_map[max_high_ts] - pos
        displacement_speed = displacement_size / max(bars_to_high, 1)

        zone_low  = piv_low
        zone_high = max(piv_open, piv_close)   # body top

        records.append({
            "creation_timestamp":        ts,
            "zone_type":                 "demand",
            "zone_high":                 round(zone_high, 4),
            "zone_low":                  round(zone_low,  4),
            "displacement_size":         round(displacement_size, 6),
            "displacement_speed":        round(displacement_speed, 9),
            "pivot_bar_volume":          int(piv_vol),
            "avg_volume_20":             round(avg_vol, 1),
            "volume_spike_ratio":        round(piv_vol / avg_vol if avg_vol > 0 else 1.0, 4),
            "bars_to_displacement_high": int(bars_to_high),
        })

    if not records:
        return _empty_demand_zones()

    return (
        pd.DataFrame(records)
        .sort_values("creation_timestamp")
        .reset_index(drop=True)
    )


def _empty_demand_zones() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "creation_timestamp", "zone_type", "zone_high", "zone_low",
        "displacement_size", "displacement_speed",
        "pivot_bar_volume", "avg_volume_20", "volume_spike_ratio",
        "bars_to_displacement_high",
    ])


# ── public API ────────────────────────────────────────────────────────────────

def get_zone_context(
    df:                 pd.DataFrame,
    current_price:      float,
    bar_interval:       str = "5m",
    pivot_left:         int = 5,
    pivot_right:        int = 5,
    min_disp_pct:       float = 0.003,
    zone_lookback_bars: int | None = None,
) -> dict:
    """
    Detect the nearest supply and demand zones and classify the zone bias.

    Parameters
    ──────────
    df                 Bar DataFrame (project-standard OHLCV, tz-naive ET).
    current_price      The bar close price to measure distances from.
    bar_interval       Used only to compute bars_since_created.
    pivot_left/right   Bars either side for pivot detection.
    min_disp_pct       Minimum displacement fraction (default 0.3 %).
    zone_lookback_bars If set, only scan the last N bars for zones.

    Returns
    ───────
    dict – see module docstring for schema.
    """
    from analog_engine.zone_detector import detect_supply_zones

    if zone_lookback_bars is not None and len(df) > zone_lookback_bars:
        df = df.iloc[-zone_lookback_bars:]

    # ── 1. Detect supply + demand zones ───────────────────────────────────
    try:
        supply_zones = detect_supply_zones(
            df,
            pivot_left=pivot_left,
            pivot_right=pivot_right,
            min_displacement_pct=min_disp_pct,
        )
    except Exception as e:
        logger.warning("Supply zone detection failed: %s", e)
        supply_zones = pd.DataFrame()

    try:
        demand_zones = detect_demand_zones(
            df,
            pivot_left=pivot_left,
            pivot_right=pivot_right,
            min_displacement_pct=min_disp_pct,
        )
    except Exception as e:
        logger.warning("Demand zone detection failed: %s", e)
        demand_zones = pd.DataFrame()

    # ── 2. Find nearest supply zone (overhead or inside) ──────────────────
    supply_ctx = _nearest_supply_context(supply_zones, current_price, df)

    # ── 3. Find nearest demand zone (below or inside) ─────────────────────
    demand_ctx = _nearest_demand_context(demand_zones, current_price, df)

    # ── 4. Bias classification ─────────────────────────────────────────────
    bias = _classify_bias(supply_ctx, demand_ctx)

    return {
        "current_price": current_price,
        "supply":        supply_ctx,
        "demand":        demand_ctx,
        "bias":          bias,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _nearest_supply_context(
    supply_zones:  pd.DataFrame,
    current_price: float,
    df:            pd.DataFrame,
) -> dict:
    """Find the nearest supply zone at or above current_price."""
    base: dict[str, Any] = {
        "zone_high":            None,
        "zone_low":             None,
        "distance_to_zone_pct": None,
        "inside_zone_flag":     False,
        "bars_since_created":   None,
        "displacement_size":    None,
        "volume_spike_ratio":   None,
        "n_zones_found":        0,
    }

    if supply_zones.empty:
        return base

    base["n_zones_found"] = len(supply_zones)

    # Classify each zone relative to current price
    inside  = supply_zones[
        (supply_zones["zone_low"]  <= current_price) &
        (supply_zones["zone_high"] >= current_price)
    ]
    overhead = supply_zones[supply_zones["zone_low"] > current_price]

    if not inside.empty:
        # Price is inside a supply zone — use the most recent one
        zone = inside.sort_values("creation_timestamp").iloc[-1]
        base["inside_zone_flag"] = True
        base["distance_to_zone_pct"] = 0.0
    elif not overhead.empty:
        # Nearest overhead zone
        zone = overhead.sort_values("zone_low").iloc[0]
        dist = (float(zone["zone_low"]) - current_price) / current_price
        base["distance_to_zone_pct"] = round(dist, 6)
    else:
        # All supply zones are below price (broken through)
        return base

    base["zone_high"]          = float(zone["zone_high"])
    base["zone_low"]           = float(zone["zone_low"])
    base["displacement_size"]  = float(zone.get("displacement_size", 0))
    base["volume_spike_ratio"] = float(zone.get("volume_spike_ratio", 1.0))
    base["bars_since_created"] = _bars_since(zone["creation_timestamp"], df)

    return base


def _nearest_demand_context(
    demand_zones:  pd.DataFrame,
    current_price: float,
    df:            pd.DataFrame,
) -> dict:
    """Find the nearest demand zone at or below current_price."""
    base: dict[str, Any] = {
        "zone_high":            None,
        "zone_low":             None,
        "distance_to_zone_pct": None,
        "inside_zone_flag":     False,
        "bars_since_created":   None,
        "displacement_size":    None,
        "volume_spike_ratio":   None,
        "n_zones_found":        0,
    }

    if demand_zones.empty:
        return base

    base["n_zones_found"] = len(demand_zones)

    inside = demand_zones[
        (demand_zones["zone_low"]  <= current_price) &
        (demand_zones["zone_high"] >= current_price)
    ]
    below = demand_zones[demand_zones["zone_high"] < current_price]

    if not inside.empty:
        zone = inside.sort_values("creation_timestamp").iloc[-1]
        base["inside_zone_flag"] = True
        base["distance_to_zone_pct"] = 0.0
    elif not below.empty:
        # Nearest below: highest zone_high below current_price
        zone = below.sort_values("zone_high", ascending=False).iloc[0]
        dist = (current_price - float(zone["zone_high"])) / current_price
        base["distance_to_zone_pct"] = round(dist, 6)
    else:
        return base

    base["zone_high"]          = float(zone["zone_high"])
    base["zone_low"]           = float(zone["zone_low"])
    base["displacement_size"]  = float(zone.get("displacement_size", 0))
    base["volume_spike_ratio"] = float(zone.get("volume_spike_ratio", 1.0))
    base["bars_since_created"] = _bars_since(zone["creation_timestamp"], df)

    return base


def _classify_bias(supply: dict, demand: dict) -> str:
    """
    Classify the directional zone bias from supply + demand context.

    Returns
    ───────
    "AT_SUPPLY"       – price is inside a supply zone
    "SUPPLY_OVERHEAD" – supply zone is close overhead (< 0.5 %)
    "AT_DEMAND"       – price is inside a demand zone
    "DEMAND_BELOW"    – demand zone is close below (< 0.5 %)
    "BETWEEN_ZONES"   – both zones exist but neither is within 0.5 %
    "NEUTRAL"         – not enough zone data to classify
    """
    if supply.get("inside_zone_flag"):
        return "AT_SUPPLY"
    if demand.get("inside_zone_flag"):
        return "AT_DEMAND"

    sup_dist = supply.get("distance_to_zone_pct")
    dem_dist = demand.get("distance_to_zone_pct")

    if sup_dist is not None and sup_dist <= _CLOSE_TO_ZONE_PCT:
        return "SUPPLY_OVERHEAD"
    if dem_dist is not None and dem_dist <= _CLOSE_TO_ZONE_PCT:
        return "DEMAND_BELOW"

    if sup_dist is not None and dem_dist is not None:
        return "BETWEEN_ZONES"
    if sup_dist is not None:
        return "SUPPLY_OVERHEAD"   # only supply, further away
    if dem_dist is not None:
        return "DEMAND_BELOW"      # only demand, further away

    return "NEUTRAL"


def _bars_since(creation_ts: pd.Timestamp, df: pd.DataFrame) -> int | None:
    """Return number of bars between zone creation and the last bar in df."""
    try:
        idx = df.index.searchsorted(creation_ts)
        return max(0, len(df) - int(idx) - 1)
    except Exception:
        return None

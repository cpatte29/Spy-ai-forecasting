"""
pattern_features.py
───────────────────
Compute a fixed-length feature vector that describes the candlestick
structure and market context of a zone-test event window.

The vector is split into three logical groups:

  A. APPROACH features (bars labelled "before")
     – how price arrived at the zone

  B. ZONE-CONTACT features (bars labelled "zone")
     – what happened while price overlapped the zone

  C. ZONE METADATA features (from the zone itself)
     – zone size, displacement severity, volume context

Feature naming convention
─────────────────────────
  approach__<name>      computed from the "before" bars
  zone__<name>          computed from the "zone" bars
  meta__<name>          from zone-creation metadata (event row)

Usage
─────
    from analog_engine.pattern_features import compute_event_features

    # single event
    fvec = compute_event_features(window_df, event_row)

    # full dataset (vectorised over windows dict)
    features_df = build_feature_matrix(events_df, windows)

Output
──────
    pd.Series   (single event)  – index = feature names
    pd.DataFrame (full dataset) – index = event_id, columns = feature names
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


# ── constants ──────────────────────────────────────────────────────────────────

# Number of "before" bars used in the approach features.  If a window has
# fewer (near the start of history), values are NaN-safe.
_APPROACH_BARS = 10

# Realized-vol window for the approach
_RV_WINDOW = 5


# ── single-event feature vector ───────────────────────────────────────────────

def compute_event_features(
    window:    pd.DataFrame,
    event_row: pd.Series,
) -> pd.Series:
    """
    Compute a fixed-length feature vector for one zone-test event.

    Parameters
    ──────────
    window     : the raw bar slice for this event, with a "bar_role" column
                 ("before" | "zone" | "after").
    event_row  : a row from the events DataFrame produced by event_extractor.

    Returns
    ───────
    pd.Series with float values; index = feature names.
    All values are finite floats (NaN filled with 0 where unavailable).
    """
    before = window[window["bar_role"] == "before"]
    zone   = window[window["bar_role"] == "zone"]

    feats: dict[str, float] = {}

    # ── A. Approach features ───────────────────────────────────────────────
    feats.update(_approach_features(before, event_row))

    # ── B. Zone-contact features ───────────────────────────────────────────
    feats.update(_zone_contact_features(zone, before, event_row))

    # ── C. Zone metadata features ─────────────────────────────────────────
    feats.update(_meta_features(event_row))

    # Guarantee finite floats
    result = pd.Series(feats, dtype=float)
    result = result.fillna(0.0)
    return result


def _approach_features(before: pd.DataFrame, event_row: pd.Series) -> dict:
    """Features computed from the bars leading up to the zone touch."""
    f: dict[str, float] = {}

    if before.empty:
        return {k: 0.0 for k in _approach_feature_names()}

    closes = before["close"].to_numpy(dtype=float)
    highs  = before["high"].to_numpy(dtype=float)
    lows   = before["low"].to_numpy(dtype=float)
    opens  = before["open"].to_numpy(dtype=float)
    vols   = before["volume"].to_numpy(dtype=float)
    n      = len(closes)

    # ── momentum ─────────────────────────────────────────────────────────
    # Total return over the approach window
    f["approach__total_return"] = float(
        (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0.0
    )
    # Per-bar return (speed of approach)
    f["approach__return_per_bar"] = f["approach__total_return"] / max(n, 1)

    # Last 3 bars momentum
    if n >= 3:
        f["approach__ret_last3"] = float((closes[-1] - closes[-3]) / closes[-3])
    else:
        f["approach__ret_last3"] = f["approach__total_return"]

    # Consecutive up bars in approach (bullish pressure into zone)
    up_bars = int(np.sum(np.diff(closes) > 0)) if n > 1 else 0
    f["approach__up_bar_fraction"] = up_bars / max(n - 1, 1)

    # ── candlestick structure ────────────────────────────────────────────
    ranges = highs - lows
    bodies = np.abs(closes - opens)
    upper_wicks = highs - np.maximum(closes, opens)
    lower_wicks = np.minimum(closes, opens) - lows

    safe_ranges = np.where(ranges > 0, ranges, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f["approach__avg_body_to_range"]       = float(np.nanmean(bodies / safe_ranges))
        f["approach__avg_upper_wick_to_range"] = float(np.nanmean(upper_wicks / safe_ranges))
        f["approach__avg_lower_wick_to_range"] = float(np.nanmean(lower_wicks / safe_ranges))
        # Close location: 0=at low, 1=at high
        f["approach__avg_close_location"]      = float(
            np.nanmean((closes - lows) / safe_ranges)
        )

    # Last bar structure (most recent before zone touch)
    last_range = highs[-1] - lows[-1]
    if last_range > 0:
        f["approach__last_body_to_range"]       = float(abs(closes[-1] - opens[-1]) / last_range)
        f["approach__last_upper_wick_to_range"] = float((highs[-1] - max(closes[-1], opens[-1])) / last_range)
        f["approach__last_lower_wick_to_range"] = float((min(closes[-1], opens[-1]) - lows[-1]) / last_range)
        f["approach__last_close_location"]      = float((closes[-1] - lows[-1]) / last_range)
    else:
        f["approach__last_body_to_range"]       = 0.5
        f["approach__last_upper_wick_to_range"] = 0.0
        f["approach__last_lower_wick_to_range"] = 0.0
        f["approach__last_close_location"]      = 0.5

    # ── volatility ────────────────────────────────────────────────────────
    log_rets = np.diff(np.log(np.maximum(closes, 1e-8)))
    f["approach__realized_vol"] = float(
        np.std(log_rets) * np.sqrt(len(log_rets)) if len(log_rets) >= 2 else 0.0
    )
    f["approach__avg_range_pct"] = float(
        np.nanmean(safe_ranges / np.maximum(lows, 1e-8))
    )

    # ── volume ────────────────────────────────────────────────────────────
    avg_vol = float(np.mean(vols)) if n > 0 else 1.0
    if avg_vol > 0:
        f["approach__last_vol_rel"]   = float(vols[-1] / avg_vol)
        f["approach__max_vol_rel"]    = float(vols.max() / avg_vol)
        f["approach__vol_trend"]      = float(
            np.polyfit(np.arange(n), vols / avg_vol, 1)[0] if n >= 2 else 0.0
        )
    else:
        f["approach__last_vol_rel"] = 1.0
        f["approach__max_vol_rel"]  = 1.0
        f["approach__vol_trend"]    = 0.0

    # ── distance from zone_high at touch ─────────────────────────────────
    zone_high = float(event_row["zone_high"])
    if closes[-1] > 0 and zone_high > 0:
        f["approach__dist_to_zone_high"] = float(
            (zone_high - closes[-1]) / closes[-1]
        )
    else:
        f["approach__dist_to_zone_high"] = 0.0

    # ── number of approach bars available ─────────────────────────────────
    f["approach__bar_count"] = float(n)

    # ── EMA deviation at approach end ─────────────────────────────────────
    if n >= 9:
        ema9  = _ema(closes, 9)
        f["approach__dist_ema9"] = float((closes[-1] - ema9[-1]) / closes[-1]) if closes[-1] != 0 else 0.0
    else:
        f["approach__dist_ema9"] = 0.0

    if n >= 20:
        ema20 = _ema(closes, 20)
        f["approach__dist_ema20"] = float((closes[-1] - ema20[-1]) / closes[-1]) if closes[-1] != 0 else 0.0
    else:
        f["approach__dist_ema20"] = 0.0

    return f


def _zone_contact_features(
    zone: pd.DataFrame,
    before: pd.DataFrame,
    event_row: pd.Series,
) -> dict:
    """Features computed from the bars while price overlaps the zone."""
    f: dict[str, float] = {}

    if zone.empty:
        return {k: 0.0 for k in _zone_feature_names()}

    closes = zone["close"].to_numpy(dtype=float)
    highs  = zone["high"].to_numpy(dtype=float)
    lows   = zone["low"].to_numpy(dtype=float)
    opens  = zone["open"].to_numpy(dtype=float)
    vols   = zone["volume"].to_numpy(dtype=float)
    n      = len(closes)

    zone_high = float(event_row["zone_high"])
    zone_low  = float(event_row["zone_low"])
    zone_mid  = (zone_high + zone_low) / 2.0

    ranges     = highs - lows
    bodies     = np.abs(closes - opens)
    safe_ranges = np.where(ranges > 0, ranges, np.nan)

    # ── how high price penetrated the zone ────────────────────────────────
    zone_width = max(zone_high - zone_low, 1e-6)
    max_penet  = float(np.maximum(0, highs - zone_low).max()) / zone_width
    f["zone__max_penetration_ratio"] = min(max_penet, 2.0)  # cap at 2× zone width

    # ── bars spent inside zone ────────────────────────────────────────────
    f["zone__bar_count"] = float(n)

    # ── close location within zone at exit ───────────────────────────────
    last_close = closes[-1]
    f["zone__exit_close_vs_zone_mid"] = float(
        (last_close - zone_mid) / zone_width
    )  # positive = above mid (bullish), negative = below mid (bearish)

    f["zone__exit_close_vs_zone_low"] = float(
        (last_close - zone_low) / zone_width
    )  # 0 = at low, 1 = at high, >1 = above zone

    # ── wick rejection signals ────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f["zone__avg_upper_wick"]  = float(np.nanmean(
            (highs - np.maximum(closes, opens)) / safe_ranges
        ))
        f["zone__avg_lower_wick"]  = float(np.nanmean(
            (np.minimum(closes, opens) - lows) / safe_ranges
        ))
        f["zone__avg_body_to_range"] = float(np.nanmean(bodies / safe_ranges))
        f["zone__avg_close_location"] = float(
            np.nanmean((closes - lows) / safe_ranges)
        )

    # ── bearish bar fraction ──────────────────────────────────────────────
    f["zone__bearish_fraction"] = float(
        np.sum(closes < opens) / max(n, 1)
    )

    # ── momentum inside zone ─────────────────────────────────────────────
    f["zone__internal_return"] = float(
        (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0.0
    )

    # ── volume inside zone vs approach ───────────────────────────────────
    zone_avg_vol = float(np.mean(vols)) if n > 0 else 0.0
    if not before.empty:
        before_avg_vol = float(before["volume"].mean())
        f["zone__vol_vs_approach"] = (
            zone_avg_vol / before_avg_vol if before_avg_vol > 0 else 1.0
        )
    else:
        f["zone__vol_vs_approach"] = 1.0

    f["zone__max_vol_rel"] = float(
        vols.max() / zone_avg_vol if zone_avg_vol > 0 else 1.0
    )

    # ── range volatility inside zone ─────────────────────────────────────
    f["zone__avg_range_pct"] = float(
        np.nanmean(safe_ranges / np.maximum(lows, 1e-8))
    )

    return f


def _meta_features(event_row: pd.Series) -> dict:
    """Features from zone-creation metadata."""
    return {
        "meta__zone_width_pct":       float(event_row.get("zone_width_pct", 0.0)),
        "meta__displacement_size":    float(event_row.get("displacement_size", 0.0)),
        "meta__displacement_speed":   float(event_row.get("displacement_speed", 0.0)),
        "meta__volume_spike_ratio":   float(event_row.get("volume_spike_ratio", 1.0)),
        "meta__bars_since_creation":  float(event_row.get("bars_since_creation", 0)),
        "meta__test_number":          float(event_row.get("test_number", 1)),
        "meta__approach_return":      float(event_row.get("approach_return", 0.0)),
        "meta__approach_speed":       float(event_row.get("approach_speed", 0.0)),
    }


# ── batch processing ───────────────────────────────────────────────────────────

def build_feature_matrix(
    events:  pd.DataFrame,
    windows: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute feature vectors for all events and return a DataFrame.

    Parameters
    ──────────
    events  : events DataFrame from event_extractor.detect_zone_tests().
    windows : windows dict from event_extractor.detect_zone_tests().

    Returns
    ───────
    pd.DataFrame – index = event_id, columns = feature names.
    """
    rows: list[pd.Series] = []
    ids:  list[int]       = []

    for _, event_row in events.iterrows():
        eid    = int(event_row["event_id"])
        window = windows.get(eid)
        if window is None or window.empty:
            continue
        fvec = compute_event_features(window, event_row)
        rows.append(fvec)
        ids.append(eid)

    if not rows:
        return pd.DataFrame()

    feat_df = pd.DataFrame(rows, index=ids)
    feat_df.index.name = "event_id"
    return feat_df


# ── EMA helper (self-contained, no dependency on feature_engineering) ──────────

def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (same formula as feature_engineering.py)."""
    alpha  = 2.0 / (span + 1)
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return result


# ── feature-name helpers (for zero-init in missing-data cases) ─────────────────

def _approach_feature_names() -> list[str]:
    return [
        "approach__total_return", "approach__return_per_bar", "approach__ret_last3",
        "approach__up_bar_fraction", "approach__avg_body_to_range",
        "approach__avg_upper_wick_to_range", "approach__avg_lower_wick_to_range",
        "approach__avg_close_location", "approach__last_body_to_range",
        "approach__last_upper_wick_to_range", "approach__last_lower_wick_to_range",
        "approach__last_close_location", "approach__realized_vol",
        "approach__avg_range_pct", "approach__last_vol_rel", "approach__max_vol_rel",
        "approach__vol_trend", "approach__dist_to_zone_high", "approach__bar_count",
        "approach__dist_ema9", "approach__dist_ema20",
    ]


def _zone_feature_names() -> list[str]:
    return [
        "zone__max_penetration_ratio", "zone__bar_count",
        "zone__exit_close_vs_zone_mid", "zone__exit_close_vs_zone_low",
        "zone__avg_upper_wick", "zone__avg_lower_wick", "zone__avg_body_to_range",
        "zone__avg_close_location", "zone__bearish_fraction", "zone__internal_return",
        "zone__vol_vs_approach", "zone__max_vol_rel", "zone__avg_range_pct",
    ]


def feature_names() -> list[str]:
    """Return the full ordered list of feature names in the vector."""
    return _approach_feature_names() + _zone_feature_names() + [
        "meta__zone_width_pct", "meta__displacement_size", "meta__displacement_speed",
        "meta__volume_spike_ratio", "meta__bars_since_creation", "meta__test_number",
        "meta__approach_return", "meta__approach_speed",
    ]

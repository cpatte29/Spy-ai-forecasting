"""
setup_scorer.py
───────────────
Score each detected setup on a 0–100 scale and assign a letter grade.

Scoring philosophy
──────────────────
Each setup is evaluated independently — the scorer does not know which
setup is "top", it just grades the raw evidence for *that* setup type.
The caller ranks setups by score to find the top one.

Component weights
─────────────────
  forecast_prob      25 pts  – direction probability edge from 0.50
  range_forecast     10 pts  – predicted volatility (range) level
  zone_proximity     20 pts  – how close / inside the relevant zone
  zone_strength      10 pts  – institutional quality of the zone (0–1 field)
  analog_alignment   15 pts  – historical analog rejection/breakout rates
  structure          15 pts  – candlestick + price structure signals
  volume_score    −10..+15   – RVOL + imbalance + participation confirmation
                    ───────
  subtotal          (95 pts max from fixed components, + volume adjustment)

  + confirmation bonus  up to +5 pts for extra confirming signals
  − chop penalty        up to −15 pts for chop conditions

Final score is clamped to [0, 100].

Letter grades
─────────────
  A+  85–100
  A   70–84
  B   55–69
  C   40–54
  IGNORE  0–39

Usage
─────
  from setup_scanner.setup_scorer import score_setup, grade_setup

  result = score_setup(setup_type, snapshot, detection)
  grade  = grade_setup(result.score)
"""

from __future__ import annotations

import logging
from typing import Optional

from setup_scanner.setup_definitions import (
    AlertState,
    ALERT_GRADE_MAP,
    GRADE_THRESHOLDS,
    MAX_CHOP_PENALTY,
    SetupDirection,
    SetupGrade,
    SetupResult,
    SetupType,
    SETUP_DIRECTION,
)
from setup_scanner.volume_integration import score_volume

logger = logging.getLogger(__name__)

# ── thresholds ────────────────────────────────────────────────────────────────

# Forecast probability points (25 max)
# Edge = |prob - 0.50|
_PROB_TIERS = [
    (0.10, 25),   # |edge| >= 10pp → full points
    (0.07, 20),
    (0.05, 15),
    (0.03, 10),
    (0.01,  5),
    (0.00,  0),
]

# Range forecast points (10 max)
_RANGE_TIERS = [
    (0.010, 10),
    (0.007,  8),
    (0.004,  6),
    (0.002,  3),
    (0.000,  0),
]

# Zone proximity points (20 max)
# Format: (max_distance, points) — sorted ascending; first matching entry wins.
# i.e. if dist <= threshold → award pts.  Lower distance = more points.
_ZONE_PROXIMITY_TIERS = [
    (0.001, 18),   # within 0.1% (nearly touching / just through)
    (0.003, 14),   # within 0.3%
    (0.007, 10),   # within 0.7%
    (0.015,  6),   # within 1.5%
    (0.030,  3),   # within 3.0%
]

# Analog alignment (15 max) – based on net alignment with setup direction
_ANALOG_TIERS = [
    (0.70, 15),   # dominant rate >= 70%
    (0.60, 12),
    (0.50,  9),
    (0.40,  6),
    (0.00,  3),   # analog active but inconclusive → small credit
]

_ANALOG_MIN_MATCHES = 5

# Chop tiers (deduction)
_CHOP_TIERS = [
    (0.8, 15),
    (0.6, 10),
    (0.4,  5),
    (0.0,  0),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _tier_lookup(value: float, tiers: list[tuple[float, int]]) -> int:
    """Return points from the first tier whose threshold is <= value."""
    for threshold, pts in sorted(tiers, reverse=True):
        if value >= threshold:
            return pts
    return 0


def _score_forecast_prob(dir_prob: float, direction: SetupDirection) -> int:
    """
    Points for how well the model's probability aligns with the setup direction.
    """
    if direction == SetupDirection.LONG:
        edge = dir_prob - 0.50
    elif direction == SetupDirection.SHORT:
        edge = 0.50 - dir_prob
    else:
        edge = abs(dir_prob - 0.50)
    # Penalise opposing signal: if edge < 0 the model disagrees
    if edge <= -0.02:
        return 0
    return _tier_lookup(edge, _PROB_TIERS)


def _score_range_forecast(pred_range: float) -> int:
    return _tier_lookup(pred_range, _RANGE_TIERS)


def _score_zone_proximity(
    setup_type: SetupType,
    zone_ctx: dict,
) -> int:
    """
    Points based on proximity to the *relevant* zone for this setup type.
    DEMAND setups use demand zone; SUPPLY setups use supply zone.
    TREND_PULLBACK uses whichever is closer.
    NO_SETUP gets 0.

    For BREAKOUT / BREAKDOWN setups the price has already moved through the
    zone, so we award credit based on how recently (how close) the break was.
    """
    if setup_type == SetupType.NO_SETUP:
        return 0

    supply  = zone_ctx.get("supply", {})
    demand  = zone_ctx.get("demand", {})
    price   = zone_ctx.get("current_price", 0.0)

    if setup_type in (SetupType.DEMAND_BOUNCE, SetupType.BREAKDOWN_BELOW_DEMAND):
        zone = demand
    elif setup_type in (SetupType.SUPPLY_REJECTION, SetupType.BREAKOUT_ABOVE_SUPPLY):
        zone = supply
    else:
        # TREND_PULLBACK: use whichever zone is closer
        sd = supply.get("distance_to_zone_pct")
        dd = demand.get("distance_to_zone_pct")
        if sd is None and dd is None:
            return 0
        if sd is None:
            zone = demand
        elif dd is None:
            zone = supply
        else:
            zone = demand if float(dd) <= float(sd) else supply

    if zone.get("inside_zone_flag"):
        return 20

    dist = zone.get("distance_to_zone_pct")

    # For BREAKOUT / BREAKDOWN: if price already crossed the zone, compute
    # how far below (demand) or above (supply) the zone edge we are.
    if (dist is None or dist == 0.0) and price > 0:
        if setup_type == SetupType.BREAKDOWN_BELOW_DEMAND:
            dz_low = zone.get("zone_low")
            if dz_low is not None and price < float(dz_low):
                dist = (float(dz_low) - price) / price
        elif setup_type == SetupType.BREAKOUT_ABOVE_SUPPLY:
            sz_high = zone.get("zone_high")
            if sz_high is not None and price > float(sz_high):
                dist = (price - float(sz_high)) / price

    if dist is None:
        return 0
    # Proximity uses "lower distance = more points" logic
    d = abs(float(dist))
    for max_dist, pts in _ZONE_PROXIMITY_TIERS:
        if d <= max_dist:
            return pts
    return 0


def _score_zone_strength(
    setup_type: SetupType,
    zone_ctx: dict,
) -> int:
    """
    Points for zone institutional quality (0–1 zone_strength field).
    """
    if setup_type == SetupType.NO_SETUP:
        return 0

    supply = zone_ctx.get("supply", {})
    demand = zone_ctx.get("demand", {})

    if setup_type in (SetupType.DEMAND_BOUNCE, SetupType.BREAKDOWN_BELOW_DEMAND):
        strength = demand.get("zone_strength")
    elif setup_type in (SetupType.SUPPLY_REJECTION, SetupType.BREAKOUT_ABOVE_SUPPLY):
        strength = supply.get("zone_strength")
    else:
        # use whichever zone we're closer to
        ss = supply.get("zone_strength")
        ds = demand.get("zone_strength")
        strength = ss if ss is not None else ds

    if strength is None:
        return 5   # unknown strength → partial credit
    s = float(strength)
    if s >= 0.80:
        return 10
    if s >= 0.60:
        return 8
    if s >= 0.40:
        return 6
    if s >= 0.20:
        return 3
    return 1


def _score_analog(
    setup_type: SetupType,
    analog: dict,
) -> int:
    """
    Points for historical analog alignment.

    For LONG setups:  breakout_rate is positive signal
    For SHORT setups: rejection_rate is positive signal
    TREND_PULLBACK:   direction determined by setup direction
    NO_SETUP:         0
    """
    if setup_type == SetupType.NO_SETUP:
        return 0

    n_matches = int(analog.get("n_matches", 0))
    if n_matches < _ANALOG_MIN_MATCHES:
        return 5   # unavailable → small neutral credit

    rej_rate = float(analog.get("rejection_rate", 0.5))
    brk_rate = float(analog.get("breakout_rate",  0.25))

    direction = SETUP_DIRECTION.get(setup_type, SetupDirection.NEUTRAL)
    # For TREND_PULLBACK direction is set per context — check the detection
    if setup_type == SetupType.TREND_PULLBACK_CONTINUATION:
        # default to long; caller may override
        direction = SetupDirection.LONG

    if direction == SetupDirection.LONG:
        dominant = brk_rate
    elif direction == SetupDirection.SHORT:
        dominant = rej_rate
    else:
        dominant = max(brk_rate, rej_rate)

    pts = _tier_lookup(dominant, _ANALOG_TIERS)

    # Penalise strong opposing signal
    if direction == SetupDirection.LONG and rej_rate >= 0.65:
        pts = max(0, pts - 6)
    if direction == SetupDirection.SHORT and brk_rate >= 0.65:
        pts = max(0, pts - 6)

    return pts


def _score_structure(snapshot: dict, setup_type: SetupType, detection: dict) -> int:
    """
    Points for price/candlestick structure signals.
    """
    if setup_type == SetupType.NO_SETUP:
        return 0

    pts = 0
    direction = detection.get("direction", SetupDirection.NEUTRAL)

    # Bullish structure signals
    price = snapshot["current_price"]
    open_ = snapshot["current_open"]
    high  = snapshot["current_high"]
    low   = snapshot["current_low"]
    bar_range = high - low if high > low else 1e-8

    bullish_candle = price > open_
    bearish_candle = price < open_

    body    = abs(price - open_) / bar_range
    uw      = (high - max(price, open_)) / bar_range
    lw      = (min(price, open_) - low)  / bar_range
    cl      = (price - low) / bar_range

    # Use stored feature values if available (more accurate)
    feat_body = snapshot.get("body_to_range")
    feat_uw   = snapshot.get("upper_wick_to_range")
    feat_lw   = snapshot.get("lower_wick_to_range")
    feat_cl   = snapshot.get("close_location")
    if feat_body is not None:
        body = float(feat_body)
    if feat_uw is not None:
        uw = float(feat_uw)
    if feat_lw is not None:
        lw = float(feat_lw)
    if feat_cl is not None:
        cl = float(feat_cl)

    if direction == SetupDirection.LONG:
        # Bullish body
        if bullish_candle and body >= 0.4:
            pts += 4
        elif bullish_candle:
            pts += 2
        # Lower wick (hammer / demand rejection)
        if lw >= 0.35:
            pts += 4
        elif lw >= 0.20:
            pts += 2
        # Close location high
        if cl >= 0.65:
            pts += 4
        elif cl >= 0.50:
            pts += 2
        # VWAP reclaim
        if snapshot.get("vwap_reclaim_flag") == 1:
            pts += 3
        # Volume
        vr = snapshot.get("vol_rel_5")
        if vr is not None and float(vr) >= 1.5:
            pts += 2
        elif vr is not None and float(vr) >= 1.2:
            pts += 1

    elif direction == SetupDirection.SHORT:
        # Bearish body
        if bearish_candle and body >= 0.4:
            pts += 4
        elif bearish_candle:
            pts += 2
        # Upper wick (supply rejection)
        if uw >= 0.35:
            pts += 4
        elif uw >= 0.20:
            pts += 2
        # Close location low
        if cl <= 0.35:
            pts += 4
        elif cl <= 0.50:
            pts += 2
        # VWAP loss
        if snapshot.get("vwap_loss_flag") == 1:
            pts += 3
        # Volume
        vr = snapshot.get("vol_rel_5")
        if vr is not None and float(vr) >= 1.5:
            pts += 2
        elif vr is not None and float(vr) >= 1.2:
            pts += 1

    else:
        pts += 5   # neutral / no-setup gets small structural credit

    return min(pts, 15)


def _score_chop_penalty(snapshot: dict) -> int:
    """
    Return points to DEDUCT for chop conditions. Always non-negative.
    """
    if not snapshot.get("chop_detected"):
        return 0
    chop_s = float(snapshot.get("chop_score", 0.5))
    for threshold, penalty in sorted(_CHOP_TIERS, reverse=True):
        if chop_s >= threshold:
            return min(penalty, MAX_CHOP_PENALTY)
    return 0


def _confirmation_bonus(snapshot: dict, setup_type: SetupType, detection: dict) -> int:
    """
    Small bonus for extra confirming signals (cap = 5).
    """
    if setup_type == SetupType.NO_SETUP:
        return 0

    pts = 0
    direction = detection.get("direction", SetupDirection.NEUTRAL)

    # ORB break in direction
    orb = snapshot.get("orb_break_flag")
    if orb is not None:
        if direction == SetupDirection.LONG and float(orb) > 0:
            pts += 2
        elif direction == SetupDirection.SHORT and float(orb) < 0:
            pts += 2

    # Confluence score alignment
    conf = snapshot.get("confluence") or {}
    conf_label = conf.get("label", "")
    conf_score = float(conf.get("score", 0.0))
    if direction == SetupDirection.LONG and conf_score >= 0.25:
        pts += 2
    elif direction == SetupDirection.SHORT and conf_score <= -0.25:
        pts += 2

    # VWAP trend strength aligned
    vts = snapshot.get("vwap_trend_strength")
    if vts is not None:
        if direction == SetupDirection.LONG  and float(vts) >= 0.4:
            pts += 1
        elif direction == SetupDirection.SHORT and float(vts) <= -0.4:
            pts += 1

    return min(pts, 5)


# ── main score function ───────────────────────────────────────────────────────

def score_setup(
    setup_type: SetupType,
    snapshot:   dict,
    detection:  dict,
) -> SetupResult:
    """
    Compute a 0–100 score for a given setup type and return a SetupResult.

    Parameters
    ──────────
    setup_type  The SetupType being scored.
    snapshot    Normalised snapshot dict (from build_snapshot()).
    detection   Detection dict for this setup (from detect_setups()[setup_type]).

    Returns
    ───────
    SetupResult with score, grade, conditions, invalidation, etc.
    """
    dir_prob    = snapshot["dir_prob"]
    pred_range  = snapshot["pred_range"]
    zone_ctx    = snapshot["zone_ctx"]
    analog      = snapshot["analog"]
    direction   = detection["direction"]

    # ── 0. Invalid setup → score 0 ────────────────────────────────────────
    if not detection["valid"] and setup_type != SetupType.NO_SETUP:
        breakdown = {
            "forecast_prob":    0,
            "range_forecast":   0,
            "zone_proximity":   0,
            "zone_strength":    0,
            "analog_alignment": 0,
            "structure":        0,
            "volume_score":     0,
            "chop_penalty":     0,
            "confirmation":     0,
            "total":            0,
        }
        grade = grade_setup(0)
        return SetupResult(
            setup_type          = setup_type,
            direction           = direction,
            score               = 0,
            grade               = grade,
            conditions_met      = detection["conditions_met"],
            missing_confirms    = detection["missing_confirms"],
            invalidation_level  = detection["invalidation_level"],
            target_zone         = detection["target_zone"],
            guidance            = detection["guidance"],
            score_breakdown     = breakdown,
            raw                 = _build_raw(snapshot),
        )

    # ── 1. Component scores ───────────────────────────────────────────────
    prob_pts     = _score_forecast_prob(dir_prob, direction)
    range_pts    = _score_range_forecast(pred_range)
    prox_pts     = _score_zone_proximity(setup_type, zone_ctx)
    strength_pts = _score_zone_strength(setup_type, zone_ctx)
    analog_pts   = _score_analog(setup_type, analog)
    struct_pts   = _score_structure(snapshot, setup_type, detection)

    # ── 2. Volume score (−10 to +15) ─────────────────────────────────────
    vol_ctx  = snapshot.get("volume_ctx") or {}
    vol_pts  = score_volume(vol_ctx, setup_type, direction)

    # ── 3. Chop penalty and confirmation bonus ────────────────────────────
    chop_ded  = _score_chop_penalty(snapshot)
    conf_bon  = _confirmation_bonus(snapshot, setup_type, detection)

    # ── 4. Raw total ──────────────────────────────────────────────────────
    raw_total = (
        prob_pts + range_pts + prox_pts + strength_pts
        + analog_pts + struct_pts + vol_pts + conf_bon - chop_ded
    )

    # For NO_SETUP: cap at 45 (it can never grade above C)
    if setup_type == SetupType.NO_SETUP:
        raw_total = min(raw_total, 45)

    score = max(0, min(100, raw_total))

    breakdown = {
        "forecast_prob":    prob_pts,
        "range_forecast":   range_pts,
        "zone_proximity":   prox_pts,
        "zone_strength":    strength_pts,
        "analog_alignment": analog_pts,
        "structure":        struct_pts,
        "volume_score":     vol_pts,
        "chop_penalty":    -chop_ded,
        "confirmation":     conf_bon,
        "total":            score,
    }

    grade = grade_setup(score)

    return SetupResult(
        setup_type          = setup_type,
        direction           = direction,
        score               = score,
        grade               = grade,
        conditions_met      = detection["conditions_met"],
        missing_confirms    = detection["missing_confirms"],
        invalidation_level  = detection["invalidation_level"],
        target_zone         = detection["target_zone"],
        guidance            = detection["guidance"],
        score_breakdown     = breakdown,
        raw                 = _build_raw(snapshot),
    )


def grade_setup(score: int) -> SetupGrade:
    """Map a 0–100 integer score to a SetupGrade."""
    for threshold, grade in sorted(GRADE_THRESHOLDS, reverse=True, key=lambda x: x[0]):
        if score >= threshold:
            return grade
    return SetupGrade.IGNORE


def _build_raw(snapshot: dict) -> dict:
    """Extract a compact raw-values dict for logging."""
    vol = snapshot.get("volume_ctx") or {}
    return {
        "dir_prob":            snapshot["dir_prob"],
        "pred_range":          snapshot["pred_range"],
        "zone_bias":           snapshot["zone_ctx"].get("bias", "NEUTRAL"),
        "supply_inside":       snapshot["zone_ctx"].get("supply", {}).get("inside_zone_flag"),
        "supply_dist":         snapshot["zone_ctx"].get("supply", {}).get("distance_to_zone_pct"),
        "supply_strength":     snapshot["zone_ctx"].get("supply", {}).get("zone_strength"),
        "demand_inside":       snapshot["zone_ctx"].get("demand", {}).get("inside_zone_flag"),
        "demand_dist":         snapshot["zone_ctx"].get("demand", {}).get("distance_to_zone_pct"),
        "demand_strength":     snapshot["zone_ctx"].get("demand", {}).get("zone_strength"),
        "n_analog_matches":    snapshot["analog"].get("n_matches"),
        "analog_rej_rate":     snapshot["analog"].get("rejection_rate"),
        "analog_brk_rate":     snapshot["analog"].get("breakout_rate"),
        "confluence_label":    snapshot.get("confluence", {}).get("label"),
        "confluence_score":    snapshot.get("confluence", {}).get("score"),
        "chop_detected":       snapshot["chop_detected"],
        "chop_score":          snapshot["chop_score"],
        "body_to_range":       snapshot.get("body_to_range"),
        "upper_wick":          snapshot.get("upper_wick_to_range"),
        "lower_wick":          snapshot.get("lower_wick_to_range"),
        "close_location":      snapshot.get("close_location"),
        "slope_ema_20_5":      snapshot.get("slope_ema_20_5"),
        "dist_vwap":           snapshot.get("dist_vwap"),
        "vwap_reclaim_flag":   snapshot.get("vwap_reclaim_flag"),
        "vwap_loss_flag":      snapshot.get("vwap_loss_flag"),
        # volume fields
        "rvol":                vol.get("rvol"),
        "volume_regime":       vol.get("volume_regime", "NORMAL"),
        "vol_imbalance":       vol.get("vol_imbalance"),
        "imbalance_label":     vol.get("imbalance_label", "NEUTRAL"),
        "breakout_confirmed":  vol.get("breakout_confirmed", False),
        "rejection_confirmed": vol.get("rejection_confirmed", False),
        "low_participation":   vol.get("low_participation", False),
        "current_volume":      vol.get("current_volume"),
        "tod_avg_volume":      vol.get("tod_avg_volume"),
    }


# ── rank all setups ───────────────────────────────────────────────────────────

def rank_all_setups(
    snapshot:   dict,
    detections: dict[SetupType, dict],
) -> list[SetupResult]:
    """
    Score all detected setups and return them sorted by score (descending).

    Returns a list of SetupResult, highest score first.
    """
    results: list[SetupResult] = []
    for setup_type, detection in detections.items():
        result = score_setup(setup_type, snapshot, detection)
        results.append(result)

    results.sort(key=lambda r: r.score, reverse=True)
    return results

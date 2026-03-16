"""
confluence_scorer.py
────────────────────
Combine three signal sources into a weighted confluence score and label.

Weighting
─────────
  Model     50 %  – direction probability from LightGBM
  Zone      30 %  – proximity and direction relative to nearest supply/demand zone
  Analog    20 %  – historical zone-test outcome statistics

Score range
───────────
  +1.0 = maximum bullish confidence
  −1.0 = maximum bearish confidence

Labels (default thresholds)
────────────────────────────
  score ≥ +0.35  →  STRONG_LONG
  score ≥ +0.12  →  MODERATE_LONG
  |score| < 0.12 →  NEUTRAL
  score ≤ −0.12  →  MODERATE_SHORT
  score ≤ −0.35  →  STRONG_SHORT

Usage
─────
  from confluence_engine.confluence_scorer import compute_confluence

  result = compute_confluence(
      dir_prob   = 0.61,
      pred_range = 0.012,
      zone_ctx   = ctx,      # from get_zone_context()
      analog     = report,   # from run_analog_analysis()
  )
  print(result["label"])    # e.g. "STRONG_SHORT"
  print(result["score"])    # e.g. -0.41
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── label thresholds ──────────────────────────────────────────────────────────
_STRONG_SCORE   = 0.35
_MODERATE_SCORE = 0.12

# ── zone score constants ───────────────────────────────────────────────────────
_INSIDE_ZONE_SCORE = 0.60    # magnitude when price is inside a zone
_CLOSE_ZONE_SCORE  = 0.35    # within 0.3 %
_NEAR_ZONE_SCORE   = 0.15    # within 1.0 %
_FAR_ZONE_SCORE    = 0.08    # within 3.0 %  (detected but not nearby)

_CLOSE_DIST_THRESHOLD = 0.003   # 0.3 %
_NEAR_DIST_THRESHOLD  = 0.010   # 1.0 %
_FAR_DIST_THRESHOLD   = 0.030   # 3.0 %

# ── analog constants ──────────────────────────────────────────────────────────
_MIN_ANALOG_MATCHES = 5     # require at least this many matches to trust analog


def _score_to_signal(score: float, strong: float, moderate: float) -> str:
    """Map a component score ∈ [−1, +1] to a directional signal label."""
    if score >= strong:
        return "STRONG_LONG"
    elif score >= moderate:
        return "MODERATE_LONG"
    elif score <= -strong:
        return "STRONG_SHORT"
    elif score <= -moderate:
        return "MODERATE_SHORT"
    else:
        return "NEUTRAL"


def _classify_alignment(signals: list[str]) -> str:
    """
    Classify how well the individual signal labels agree.

    Returns one of:
      STRONG_ALIGNMENT  – all active signals point same direction
      MODERATE_ALIGNMENT – majority point same direction
      WEAK_CONFLICT     – at least two active signals disagree
      STRONG_CONFLICT   – active signals point opposite extremes
    """
    if not signals:
        return "STRONG_ALIGNMENT"

    bullish = sum(1 for s in signals if "LONG"  in s)
    bearish = sum(1 for s in signals if "SHORT" in s)
    neutral = sum(1 for s in signals if s == "NEUTRAL")
    active  = bullish + bearish

    if active == 0:
        return "STRONG_ALIGNMENT"  # all neutral

    if bullish > 0 and bearish > 0:
        # At least one signal points each way
        if (bullish >= 2 and "STRONG" in "".join(s for s in signals if "SHORT" in s)) or \
           (bearish >= 2 and "STRONG" in "".join(s for s in signals if "LONG"  in s)):
            return "STRONG_CONFLICT"
        return "WEAK_CONFLICT"

    # All active signals agree
    if neutral == 0:
        return "STRONG_ALIGNMENT"
    return "MODERATE_ALIGNMENT"


def compute_confluence(
    dir_prob:       float,
    pred_range:     float,
    zone_ctx:       dict,
    analog:         dict,
    strong_score:   float = _STRONG_SCORE,
    moderate_score: float = _MODERATE_SCORE,
    model_weight:   float | None = None,
    zone_weight:    float | None = None,
    analog_weight:  float | None = None,
) -> dict:
    """
    Compute a weighted confluence score and map it to a label.

    Parameters
    ──────────
    dir_prob     Probability that price will be higher N bars from now (0–1).
                 Values > 0.5 are bullish; < 0.5 are bearish.
    pred_range   Predicted high-low range fraction (e.g. 0.012 = 1.2 %).
                 Currently used only for the "range context" note; future
                 versions may weight it into the score.
    zone_ctx     Output of confluence_engine.zone_context.get_zone_context().
    analog       Output of analog_engine.analog_report.run_analog_analysis()
                 (or a dict with rejection_rate, breakout_rate, n_matches).
    strong_score   Score threshold for STRONG labels (default 0.35).
    moderate_score Score threshold for MODERATE labels (default 0.12).
    model_weight   Override model weight (default 0.50).  Will be normalised
    zone_weight    Override zone  weight (default 0.30).  with the other two
    analog_weight  Override analog weight (default 0.20). so they sum to 1.

    Returns
    ───────
    dict with keys:
        label          str    – confluence label
        score          float  – weighted score in [−1, +1]
        components     dict   – individual component scores
        signals        dict   – individual signal labels (model/zone/analog)
        alignment      str    – STRONG_ALIGNMENT / MODERATE_ALIGNMENT /
                                WEAK_CONFLICT / STRONG_CONFLICT
        weights        dict   – normalised weights used
        detail         dict   – raw inputs for logging / report
    """
    # ── 0. Resolve weights ────────────────────────────────────────────────
    mw = model_weight  if model_weight  is not None else 0.50
    zw = zone_weight   if zone_weight   is not None else 0.30
    aw = analog_weight if analog_weight is not None else 0.20
    total_w = mw + zw + aw
    if total_w <= 0:
        mw, zw, aw, total_w = 0.50, 0.30, 0.20, 1.0
    mw /= total_w
    zw /= total_w
    aw /= total_w

    # ── 1. Model component ────────────────────────────────────────────────
    # Transform P(up) ∈ [0, 1] → model_score ∈ [−1, +1]
    # 0.5 → 0.0,  1.0 → +1.0,  0.0 → −1.0
    model_score = (dir_prob - 0.5) * 2.0

    # ── 2. Zone component ─────────────────────────────────────────────────
    # zone_score = proximity_score × direction × zone_strength
    # zone_strength ∈ [0, 1] scales each zone's contribution so that weak
    # zones contribute little and strong institutional zones contribute fully.
    supply = zone_ctx.get("supply", {})
    demand = zone_ctx.get("demand", {})
    zone_score = 0.0

    _s = supply.get("zone_strength")
    _d = demand.get("zone_strength")
    supply_strength = float(_s) if _s is not None else 1.0
    demand_strength = float(_d) if _d is not None else 1.0

    # Supply zone contribution (bearish pressure)
    if supply.get("inside_zone_flag"):
        zone_score -= _INSIDE_ZONE_SCORE * supply_strength
    elif supply.get("distance_to_zone_pct") is not None:
        dist = float(supply["distance_to_zone_pct"])
        if dist <= _CLOSE_DIST_THRESHOLD:
            zone_score -= _CLOSE_ZONE_SCORE * supply_strength
        elif dist <= _NEAR_DIST_THRESHOLD:
            zone_score -= _NEAR_ZONE_SCORE * supply_strength
        elif dist <= _FAR_DIST_THRESHOLD:
            zone_score -= _FAR_ZONE_SCORE * supply_strength

    # Demand zone contribution (bullish support)
    if demand.get("inside_zone_flag"):
        zone_score += _INSIDE_ZONE_SCORE * demand_strength
    elif demand.get("distance_to_zone_pct") is not None:
        dist = float(demand["distance_to_zone_pct"])
        if dist <= _CLOSE_DIST_THRESHOLD:
            zone_score += _CLOSE_ZONE_SCORE * demand_strength
        elif dist <= _NEAR_DIST_THRESHOLD:
            zone_score += _NEAR_ZONE_SCORE * demand_strength
        elif dist <= _FAR_DIST_THRESHOLD:
            zone_score += _FAR_ZONE_SCORE * demand_strength

    zone_score = max(-1.0, min(1.0, zone_score))

    # ── 3. Analog component ────────────────────────────────────────────────
    analog_score  = 0.0
    analog_active = False
    n_matches     = int(analog.get("n_matches", 0)) if analog else 0

    if n_matches >= _MIN_ANALOG_MATCHES and not analog.get("error"):
        rej = float(analog.get("rejection_rate", 0.5))
        brk = float(analog.get("breakout_rate",  0.25))
        # rejection_rate high → bearish; breakout_rate high → bullish
        # net analog score: brk − rej ∈ [−1, +1]
        analog_score  = brk - rej
        analog_active = True

    # ── 4. Weighted combination ───────────────────────────────────────────
    score = (mw * model_score
           + zw * zone_score
           + aw * analog_score)
    score = max(-1.0, min(1.0, score))

    # ── 5. Label ──────────────────────────────────────────────────────────
    label = _score_to_signal(score, strong_score, moderate_score)

    # ── 6. Individual signal labels ───────────────────────────────────────
    model_signal  = _score_to_signal(model_score,  strong_score, moderate_score)
    zone_signal   = _score_to_signal(zone_score,   strong_score, moderate_score)
    analog_signal = (
        _score_to_signal(analog_score, strong_score, moderate_score)
        if analog_active else "N/A"
    )

    active_signals = [model_signal, zone_signal]
    if analog_active:
        active_signals.append(analog_signal)
    alignment = _classify_alignment(active_signals)

    return {
        "label": label,
        "score": round(score, 4),
        "components": {
            "model_score":  round(model_score,  4),
            "zone_score":   round(zone_score,   4),
            "analog_score": round(analog_score, 4),
        },
        "signals": {
            "model_signal":  model_signal,
            "zone_signal":   zone_signal,
            "analog_signal": analog_signal,
        },
        "alignment": alignment,
        "weights": {
            "model":  round(mw, 4),
            "zone":   round(zw, 4),
            "analog": round(aw, 4),
        },
        "detail": {
            "dir_prob":             dir_prob,
            "pred_range":           pred_range,
            "zone_bias":            zone_ctx.get("bias", "NEUTRAL"),
            "supply_zone_strength": supply_strength,
            "demand_zone_strength": demand_strength,
            "n_matches":            n_matches,
            "analog_active":        analog_active,
            "rejection_rate":       analog.get("rejection_rate", None) if analog else None,
            "breakout_rate":        analog.get("breakout_rate",  None) if analog else None,
        },
    }


def apply_zone_tiebreaker(
    signal:           str,
    zone_ctx:         dict,
    require_strength: float = 0.30,
) -> str:
    """
    When the direction model produces NO_TRADE, use zone bias as a tiebreaker.

    Rules
    ─────
    AT_SUPPLY / SUPPLY_OVERHEAD  +  zone_strength >= require_strength  →  SHORT_BIAS
    AT_DEMAND / DEMAND_BELOW     +  zone_strength >= require_strength  →  LONG_BIAS
    All other cases leave the signal unchanged.

    Parameters
    ──────────
    signal           Current signal string (LONG_BIAS / SHORT_BIAS / NO_TRADE).
    zone_ctx         Output of get_zone_context().
    require_strength Minimum zone_strength score to allow the override (0–1).
                     Default 0.30 — weak zones do not break the tie.

    Returns
    ───────
    Possibly-overridden signal string.
    """
    if signal != "NO_TRADE":
        return signal

    bias = zone_ctx.get("bias", "NEUTRAL")

    if bias in ("AT_SUPPLY", "SUPPLY_OVERHEAD"):
        strength = zone_ctx.get("supply", {}).get("zone_strength")
        if strength is not None and float(strength) >= require_strength:
            logger.debug(
                "Zone tiebreaker: %s (strength=%.2f) → SHORT_BIAS", bias, strength
            )
            return "SHORT_BIAS"

    if bias in ("AT_DEMAND", "DEMAND_BELOW"):
        strength = zone_ctx.get("demand", {}).get("zone_strength")
        if strength is not None and float(strength) >= require_strength:
            logger.debug(
                "Zone tiebreaker: %s (strength=%.2f) → LONG_BIAS", bias, strength
            )
            return "LONG_BIAS"

    return signal

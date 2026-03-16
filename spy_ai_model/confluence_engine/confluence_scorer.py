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

_CLOSE_DIST_THRESHOLD = 0.003   # 0.3 %
_NEAR_DIST_THRESHOLD  = 0.010   # 1.0 %

# ── analog constants ──────────────────────────────────────────────────────────
_MIN_ANALOG_MATCHES = 5     # require at least this many matches to trust analog


def compute_confluence(
    dir_prob:    float,
    pred_range:  float,
    zone_ctx:    dict,
    analog:      dict,
    strong_score:   float = _STRONG_SCORE,
    moderate_score: float = _MODERATE_SCORE,
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

    Returns
    ───────
    dict with keys:
        label        str    – confluence label
        score        float  – weighted score in [−1, +1]
        components   dict   – individual component scores
        detail       dict   – raw inputs for logging / report
    """
    # ── 1. Model component ────────────────────────────────────────────────
    # Transform P(up) ∈ [0, 1] → model_score ∈ [−1, +1]
    # 0.5 → 0.0,  1.0 → +1.0,  0.0 → −1.0
    model_score = (dir_prob - 0.5) * 2.0

    # ── 2. Zone component ─────────────────────────────────────────────────
    supply = zone_ctx.get("supply", {})
    demand = zone_ctx.get("demand", {})
    zone_score = 0.0

    # Supply zone contribution (bearish pressure)
    if supply.get("inside_zone_flag"):
        zone_score -= _INSIDE_ZONE_SCORE
    elif supply.get("distance_to_zone_pct") is not None:
        dist = float(supply["distance_to_zone_pct"])
        if dist <= _CLOSE_DIST_THRESHOLD:
            zone_score -= _CLOSE_ZONE_SCORE
        elif dist <= _NEAR_DIST_THRESHOLD:
            zone_score -= _NEAR_ZONE_SCORE

    # Demand zone contribution (bullish support)
    if demand.get("inside_zone_flag"):
        zone_score += _INSIDE_ZONE_SCORE
    elif demand.get("distance_to_zone_pct") is not None:
        dist = float(demand["distance_to_zone_pct"])
        if dist <= _CLOSE_DIST_THRESHOLD:
            zone_score += _CLOSE_ZONE_SCORE
        elif dist <= _NEAR_DIST_THRESHOLD:
            zone_score += _NEAR_ZONE_SCORE

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
    score = (0.50 * model_score
           + 0.30 * zone_score
           + 0.20 * analog_score)
    score = max(-1.0, min(1.0, score))

    # ── 5. Label ──────────────────────────────────────────────────────────
    if score >= strong_score:
        label = "STRONG_LONG"
    elif score >= moderate_score:
        label = "MODERATE_LONG"
    elif score <= -strong_score:
        label = "STRONG_SHORT"
    elif score <= -moderate_score:
        label = "MODERATE_SHORT"
    else:
        label = "NEUTRAL"

    return {
        "label": label,
        "score": round(score, 4),
        "components": {
            "model_score":  round(model_score,  4),
            "zone_score":   round(zone_score,   4),
            "analog_score": round(analog_score, 4),
        },
        "weights": {
            "model":  0.50,
            "zone":   0.30,
            "analog": 0.20,
        },
        "detail": {
            "dir_prob":       dir_prob,
            "pred_range":     pred_range,
            "zone_bias":      zone_ctx.get("bias", "NEUTRAL"),
            "n_matches":      n_matches,
            "analog_active":  analog_active,
            "rejection_rate": analog.get("rejection_rate", None) if analog else None,
            "breakout_rate":  analog.get("breakout_rate",  None) if analog else None,
        },
    }

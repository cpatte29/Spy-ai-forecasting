"""
flow_scorer.py
──────────────
Score unusual options flow and assign alert grades.

Scoring philosophy
──────────────────
Each component contributes points toward a 0–100 total.

  flow_intensity    30 pts   vol/OI ratio of the dominant side
  premium_size      20 pts   total premium at risk (size filter)
  directional_skew  20 pts   how skewed call vs put activity is
  contract_quality  15 pts   liquidity: spread quality, OI depth
  concentration     10 pts   strike/expiry focus (conviction signal)
  stock_confirm      5 pts   price action roughly aligns with flow bias

  Total            100 pts

Deductions
──────────
  - far_otm_penalty  −10 pts   if >50% of volume is lotto/far-OTM
  - mixed_penalty    − 8 pts   if MIXED_FLOW detected (ambiguous)
  - low_prem_penalty − 5 pts   if total premium < $100K (noise filter)

Letter grades  (matches setup_scanner thresholds)
──────────────
  A+  85–100   HIGH_CONVICTION alert
  A   70–84    ACTIONABLE alert
  B   55–69    WATCHLIST alert
  C   40–54    logged, not alerted
  IGNORE  <40  suppressed

Bias determination
──────────────────
  BULLISH   directional_score >= 0.20  (and bullish detections dominate)
  BEARISH   directional_score <= −0.20 (and bearish detections dominate)
  MIXED     significant signals on both sides
  NEUTRAL   insufficient directional evidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── enums ─────────────────────────────────────────────────────────────────────

class FlowBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    MIXED   = "MIXED"
    NEUTRAL = "NEUTRAL"


class FlowGrade(str, Enum):
    A_PLUS = "A+"
    A      = "A"
    B      = "B"
    C      = "C"
    IGNORE = "IGNORE"


class FlowAlertState(str, Enum):
    HIGH_CONVICTION = "HIGH_CONVICTION"
    ACTIONABLE      = "ACTIONABLE"
    WATCHLIST       = "WATCHLIST"
    SILENT          = "SILENT"


# ── thresholds ────────────────────────────────────────────────────────────────

FLOW_GRADE_THRESHOLDS: list[tuple[int, FlowGrade]] = [
    (85, FlowGrade.A_PLUS),
    (70, FlowGrade.A),
    (55, FlowGrade.B),
    (40, FlowGrade.C),
    (0,  FlowGrade.IGNORE),
]

FLOW_ALERT_MAP: dict[FlowGrade, FlowAlertState] = {
    FlowGrade.A_PLUS: FlowAlertState.HIGH_CONVICTION,
    FlowGrade.A:      FlowAlertState.ACTIONABLE,
    FlowGrade.B:      FlowAlertState.WATCHLIST,
    FlowGrade.C:      FlowAlertState.SILENT,
    FlowGrade.IGNORE: FlowAlertState.SILENT,
}

FLOW_ALERT_MIN_GRADE = FlowGrade.B


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FlowResult:
    ticker:           str
    bias:             FlowBias
    score:            int
    grade:            FlowGrade
    alert_state:      FlowAlertState
    detections:       list[dict]        = field(default_factory=list)
    conditions:       list[str]         = field(default_factory=list)
    caveats:          list[str]         = field(default_factory=list)
    top_contract:     dict              = field(default_factory=dict)
    score_breakdown:  dict              = field(default_factory=dict)
    features:         dict              = field(default_factory=dict)
    stock_price:      float             = 0.0
    guidance:         str               = ""

    def to_dict(self) -> dict:
        return {
            "ticker":         self.ticker,
            "bias":           self.bias.value,
            "score":          self.score,
            "grade":          self.grade.value,
            "alert_state":    self.alert_state.value,
            "stock_price":    self.stock_price,
            "guidance":       self.guidance,
            "top_contract":   self.top_contract,
            "conditions":     self.conditions,
            "caveats":        self.caveats,
            "score_breakdown": self.score_breakdown,
            "features":       self.features,
        }


# ── scoring helpers ───────────────────────────────────────────────────────────

def _tier_lookup(value: float, tiers: list[tuple[float, int]]) -> int:
    """Return pts from the first tier where value >= threshold (descending)."""
    for threshold, pts in sorted(tiers, key=lambda x: x[0], reverse=True):
        if value >= threshold:
            return pts
    return 0


_VOL_OI_TIERS = [
    (3.0, 30), (2.0, 24), (1.0, 18), (0.5, 12), (0.2, 6), (0.0, 0),
]

_PREMIUM_TIERS = [
    (2_000_000, 20), (1_000_000, 17), (500_000, 14), (200_000, 10),
    (100_000,    6), (50_000,     3), (0,           0),
]

_CP_RATIO_TIERS = [
    (4.0, 20), (2.5, 16), (1.5, 10), (1.2, 5), (0.0, 0),
]

_PC_RATIO_TIERS = [          # for bearish skew (put/call)
    (4.0, 20), (2.5, 16), (1.5, 10), (1.2, 5), (0.0, 0),
]

_OI_DEPTH_TIERS = [
    (50_000, 15), (20_000, 12), (10_000, 9), (5_000, 6),
    (2_000,   3), (0,         0),
]

_CONC_TIERS = [
    (0.8, 10), (0.6, 8), (0.4, 5), (0.2, 2), (0.0, 0),
]


def _score_flow_intensity(features: dict, bias: FlowBias) -> int:
    """30 pts — vol/OI ratio of dominant side."""
    if bias in (FlowBias.BULLISH, FlowBias.MIXED):
        voi = features.get("call_vol_oi", 0.0)
    elif bias == FlowBias.BEARISH:
        voi = features.get("put_vol_oi", 0.0)
    else:
        voi = max(features.get("call_vol_oi", 0.0), features.get("put_vol_oi", 0.0))
    return _tier_lookup(voi, _VOL_OI_TIERS)


def _score_premium(features: dict) -> int:
    """20 pts — total premium at risk (size filter)."""
    return _tier_lookup(features.get("total_premium", 0.0), _PREMIUM_TIERS)


def _score_directional_skew(features: dict, bias: FlowBias) -> int:
    """20 pts — how skewed the activity is toward one direction."""
    if bias == FlowBias.BULLISH:
        return _tier_lookup(features.get("cp_ratio", 1.0), _CP_RATIO_TIERS)
    if bias == FlowBias.BEARISH:
        return _tier_lookup(features.get("pc_ratio", 1.0), _PC_RATIO_TIERS)
    # MIXED / NEUTRAL — partial credit for having some skew
    max_ratio = max(features.get("cp_ratio", 1.0), features.get("pc_ratio", 1.0))
    return _tier_lookup(max_ratio, _CP_RATIO_TIERS) // 2


def _score_contract_quality(features: dict) -> int:
    """15 pts — OI depth as a liquidity proxy."""
    total_oi = features.get("total_oi", 0)
    return _tier_lookup(float(total_oi), _OI_DEPTH_TIERS)


def _score_concentration(features: dict, bias: FlowBias) -> int:
    """10 pts — how focused is the activity (strike + expiry concentration)."""
    if bias in (FlowBias.BULLISH, FlowBias.MIXED):
        conc = (
            features.get("call_strike_conc", 0.0)
            + features.get("call_expiry_conc", 0.0)
        ) / 2.0
    elif bias == FlowBias.BEARISH:
        conc = (
            features.get("put_strike_conc", 0.0)
            + features.get("put_expiry_conc", 0.0)
        ) / 2.0
    else:
        conc = max(
            features.get("call_strike_conc", 0.0),
            features.get("put_strike_conc", 0.0),
        )
    return _tier_lookup(conc, _CONC_TIERS)


def _score_stock_confirm(features: dict, bias: FlowBias) -> int:
    """5 pts — rough stock price alignment with the flow bias."""
    # Without a full bar DataFrame here, use the directional_score as a proxy.
    # A positive directional_score on bullish bias = mild confirmation.
    ds = features.get("directional_score", 0.0)
    if bias == FlowBias.BULLISH and ds > 0.1:
        return 5
    if bias == FlowBias.BEARISH and ds < -0.1:
        return 5
    if bias in (FlowBias.BULLISH, FlowBias.BEARISH) and abs(ds) < 0.1:
        return 2   # ambiguous stock action
    return 0


def _determine_bias(features: dict, detections: list[dict]) -> FlowBias:
    """Infer the flow bias from features and detection results."""
    bull_count = sum(1 for d in detections if d.get("bias") == "BULLISH")
    bear_count = sum(1 for d in detections if d.get("bias") == "BEARISH")

    ds = features.get("directional_score", 0.0)

    if bull_count > 0 and bear_count > 0:
        return FlowBias.MIXED

    if bull_count > 0 or ds >= 0.20:
        return FlowBias.BULLISH

    if bear_count > 0 or ds <= -0.20:
        return FlowBias.BEARISH

    return FlowBias.NEUTRAL


def _build_guidance(
    ticker: str,
    bias: FlowBias,
    grade: FlowGrade,
    features: dict,
    caveats: list[str],
) -> str:
    """Build a short human-readable guidance string."""
    if grade in (FlowGrade.IGNORE, FlowGrade.C):
        return "Flow below conviction threshold — monitor but no action warranted."

    if bias == FlowBias.NEUTRAL:
        return "No clear directional bias in options flow — avoid directional positioning."

    if bias == FlowBias.MIXED:
        return (
            "Mixed call and put activity. Could reflect straddle/strangle positioning "
            "or disagreement between participants. Research catalyst before acting."
        )

    direction = "upside" if bias == FlowBias.BULLISH else "downside"
    caveat_note = " Note: " + caveats[0] if caveats else ""
    prem = features.get("total_premium", 0.0)

    return (
        f"Options flow suggests {direction} positioning in {ticker} "
        f"(~${prem:,.0f} premium at risk). "
        f"Research catalyst and confirm chart setup before entry.{caveat_note}"
    )


# ── public API ────────────────────────────────────────────────────────────────

def grade_flow(score: int) -> FlowGrade:
    """Map a 0–100 score to a FlowGrade letter."""
    for threshold, g in FLOW_GRADE_THRESHOLDS:
        if score >= threshold:
            return g
    return FlowGrade.IGNORE


def score_flow(
    ticker:     str,
    features:   dict,
    detections: list[dict],
    stock_price: float = 0.0,
) -> FlowResult:
    """
    Score the options flow for a single ticker and return a FlowResult.

    Parameters
    ──────────
    ticker       Underlying symbol.
    features     Output of compute_flow_features().
    detections   Output of detect_unusual_flow().
    stock_price  Current underlying price (used for labelling).

    Returns
    ───────
    FlowResult (immutable dataclass).
    """
    bias = _determine_bias(features, detections)

    # ── component scores ──────────────────────────────────────────────────────
    intensity_pts  = _score_flow_intensity(features, bias)
    premium_pts    = _score_premium(features)
    skew_pts       = _score_directional_skew(features, bias)
    quality_pts    = _score_contract_quality(features)
    conc_pts       = _score_concentration(features, bias)
    confirm_pts    = _score_stock_confirm(features, bias)

    raw_total = intensity_pts + premium_pts + skew_pts + quality_pts + conc_pts + confirm_pts

    # ── deductions ────────────────────────────────────────────────────────────
    deductions = {}

    if features.get("far_otm_dominated", False):
        deductions["far_otm_penalty"] = -10
    if any(d.get("signal_type") == "MIXED_FLOW" for d in detections):
        deductions["mixed_penalty"] = -8
    if features.get("total_premium", 0.0) < 100_000:
        deductions["low_prem_penalty"] = -5

    total_ded = sum(deductions.values())
    final_score = max(0, min(100, raw_total + total_ded))

    grade = grade_flow(final_score)
    alert = FLOW_ALERT_MAP.get(grade, FlowAlertState.SILENT)

    # ── conditions and caveats ────────────────────────────────────────────────
    conditions: list[str] = []
    caveats:    list[str] = []

    for d in detections:
        if d.get("bias") not in ("NEUTRAL", "AMBIGUOUS"):
            conditions.append(d.get("reason", ""))
        else:
            caveats.append(d.get("reason", ""))

    if features.get("far_otm_dominated", False):
        caveats.append(
            "Majority of volume is in far-OTM contracts — lower predictive value."
        )

    if bias == FlowBias.BEARISH and features.get("put_lotto_pct", 0.0) > 0.3:
        caveats.append(
            "Elevated far-OTM put activity could be portfolio hedging, not directional."
        )

    # ── top contract ──────────────────────────────────────────────────────────
    if bias == FlowBias.BULLISH:
        top_contract = features.get("top_call", {})
    elif bias == FlowBias.BEARISH:
        top_contract = features.get("top_put", {})
    else:
        top_contract = features.get("top_contract", {})

    score_breakdown = {
        "flow_intensity":    intensity_pts,
        "premium_size":      premium_pts,
        "directional_skew":  skew_pts,
        "contract_quality":  quality_pts,
        "concentration":     conc_pts,
        "stock_confirm":     confirm_pts,
        **deductions,
        "total":             final_score,
    }

    guidance = _build_guidance(ticker, bias, grade, features, caveats)

    return FlowResult(
        ticker          = ticker,
        bias            = bias,
        score           = final_score,
        grade           = grade,
        alert_state     = alert,
        detections      = detections,
        conditions      = [c for c in conditions if c],
        caveats         = [c for c in caveats if c],
        top_contract    = top_contract,
        score_breakdown = score_breakdown,
        features        = features,
        stock_price     = stock_price,
        guidance        = guidance,
    )

"""
setup_definitions.py
────────────────────
Data types and constants for the setup scanner.

SetupType    – the six recognised intraday setups
SetupDirection – directional bias of each setup
SetupGrade   – quality grade (A+, A, B, C, IGNORE)
AlertState   – escalation tier for alert engine
SetupResult  – immutable result object for a single scanned bar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Setup types ───────────────────────────────────────────────────────────────

class SetupType(str, Enum):
    DEMAND_BOUNCE               = "DEMAND_BOUNCE"
    SUPPLY_REJECTION            = "SUPPLY_REJECTION"
    BREAKOUT_ABOVE_SUPPLY       = "BREAKOUT_ABOVE_SUPPLY"
    BREAKDOWN_BELOW_DEMAND      = "BREAKDOWN_BELOW_DEMAND"
    TREND_PULLBACK_CONTINUATION = "TREND_PULLBACK_CONTINUATION"
    NO_SETUP                    = "NO_SETUP"


# ── Directional bias ──────────────────────────────────────────────────────────

class SetupDirection(str, Enum):
    LONG    = "LONG"
    SHORT   = "SHORT"
    NEUTRAL = "NEUTRAL"


# ── Setup-to-direction mapping ────────────────────────────────────────────────

SETUP_DIRECTION: dict[SetupType, SetupDirection] = {
    SetupType.DEMAND_BOUNCE:               SetupDirection.LONG,
    SetupType.SUPPLY_REJECTION:            SetupDirection.SHORT,
    SetupType.BREAKOUT_ABOVE_SUPPLY:       SetupDirection.LONG,
    SetupType.BREAKDOWN_BELOW_DEMAND:      SetupDirection.SHORT,
    SetupType.TREND_PULLBACK_CONTINUATION: SetupDirection.LONG,   # overridden per context
    SetupType.NO_SETUP:                    SetupDirection.NEUTRAL,
}


# ── Quality grades ────────────────────────────────────────────────────────────

class SetupGrade(str, Enum):
    A_PLUS = "A+"
    A      = "A"
    B      = "B"
    C      = "C"
    IGNORE = "IGNORE"


# Grade score bands (inclusive lower bound)
GRADE_THRESHOLDS: list[tuple[int, SetupGrade]] = [
    (85, SetupGrade.A_PLUS),
    (70, SetupGrade.A),
    (55, SetupGrade.B),
    (40, SetupGrade.C),
    (0,  SetupGrade.IGNORE),
]


# ── Alert states ──────────────────────────────────────────────────────────────

class AlertState(str, Enum):
    HIGH_CONVICTION = "HIGH_CONVICTION"
    ACTIONABLE      = "ACTIONABLE"
    WATCHLIST       = "WATCHLIST"
    SILENT          = "SILENT"


# Alert grade thresholds
ALERT_GRADE_MAP: dict[SetupGrade, AlertState] = {
    SetupGrade.A_PLUS: AlertState.HIGH_CONVICTION,
    SetupGrade.A:      AlertState.ACTIONABLE,
    SetupGrade.B:      AlertState.WATCHLIST,
    SetupGrade.C:      AlertState.SILENT,
    SetupGrade.IGNORE: AlertState.SILENT,
}

# Minimum grade to actually emit an alert to the terminal / log
ALERT_MIN_GRADE = SetupGrade.B


# ── Scoring component weights (must sum to 100) ───────────────────────────────

SCORE_WEIGHTS = {
    "forecast_prob":     25,   # direction probability edge vs 0.50
    "range_forecast":    10,   # predicted price range (volatility)
    "zone_proximity":    20,   # how close / inside the relevant zone
    "zone_strength":     10,   # institutional quality of the zone
    "analog_alignment":  15,   # historical analog outcome rates
    "structure":         15,   # candlestick / price structure signals
    "chop_penalty":      -0,   # deducted when chop is detected (up to −15)
}
# Note: chop_penalty is applied as a deduction, not a weight.
MAX_CHOP_PENALTY = 15          # max points deducted for chop conditions


# ── SetupResult ───────────────────────────────────────────────────────────────

@dataclass
class SetupResult:
    """
    Immutable description of a single detected setup at a given bar.

    Fields
    ──────
    setup_type          SetupType enum value
    direction           LONG / SHORT / NEUTRAL
    score               0–100 integer quality score
    grade               A+, A, B, C, IGNORE
    conditions_met      human-readable list of satisfied conditions
    missing_confirms    human-readable list of unconfirmed signals
    invalidation_level  price level that would negate the setup (or None)
    target_zone         tuple (low, high) of next key level, or None
    guidance            one-line plain-English guidance note
    score_breakdown     dict mapping component name → points earned
    raw                 dict of raw signal values used for scoring (for logging)
    """
    setup_type:          SetupType
    direction:           SetupDirection
    score:               int
    grade:               SetupGrade
    conditions_met:      list[str]       = field(default_factory=list)
    missing_confirms:    list[str]       = field(default_factory=list)
    invalidation_level:  Optional[float] = None
    target_zone:         Optional[tuple[float, float]] = None
    guidance:            str             = ""
    score_breakdown:     dict            = field(default_factory=dict)
    raw:                 dict            = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "setup_type":         self.setup_type.value,
            "direction":          self.direction.value,
            "score":              self.score,
            "grade":              self.grade.value,
            "conditions_met":     self.conditions_met,
            "missing_confirms":   self.missing_confirms,
            "invalidation_level": self.invalidation_level,
            "target_zone_low":    self.target_zone[0] if self.target_zone else None,
            "target_zone_high":   self.target_zone[1] if self.target_zone else None,
            "guidance":           self.guidance,
            "score_breakdown":    self.score_breakdown,
        }

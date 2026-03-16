"""
confluence_engine
─────────────────
Independent analysis layer that combines three signal sources into a single
directional confluence score and label.

  from confluence_engine.zone_context      import get_zone_context
  from confluence_engine.confluence_scorer import compute_confluence
  from confluence_engine.report_formatter  import format_confluence_report

Signal sources
──────────────
  1. Forecast model   – LightGBM direction probability + predicted range
  2. Analog engine    – historical supply-zone pattern outcome statistics
  3. Zone context     – nearest supply / demand zone relative to current price

Confluence labels
─────────────────
  STRONG_LONG      weighted score ≥ +0.35
  MODERATE_LONG    weighted score ≥ +0.12
  NEUTRAL          |score| < 0.12
  MODERATE_SHORT   weighted score ≤ −0.12
  STRONG_SHORT     weighted score ≤ −0.35
"""

from .zone_context       import get_zone_context
from .confluence_scorer  import compute_confluence
from .report_formatter   import format_confluence_report
from .zone_scorer        import score_zone_strength

__all__ = [
    "get_zone_context",
    "compute_confluence",
    "format_confluence_report",
    "score_zone_strength",
]

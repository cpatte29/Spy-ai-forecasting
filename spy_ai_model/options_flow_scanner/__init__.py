"""
options_flow_scanner
────────────────────
Unusual options-flow detection and alert engine.

Scans a ticker universe every N minutes, detects unusual bullish/bearish
options activity, scores each ticker 0–100, and emits structured alerts.

This is a research/alert tool — no trade execution.

Public API
──────────
  from options_flow_scanner import (
      load_universe,
      load_chain,
      compute_flow_features,
      detect_unusual_flow,
      score_flow,
      grade_flow,
      FlowResult,
      FlowBias,
      FlowGrade,
      FlowAlertState,
      format_flow_report,
      format_flow_summary,
      log_flow_alert_csv,
  )
"""

from options_flow_scanner.ticker_universe import (
    load_universe,
    DEFAULT_UNIVERSE,
    EXTENDED_UNIVERSE,
)
from options_flow_scanner.option_chain_loader import load_chain
from options_flow_scanner.flow_features import compute_flow_features
from options_flow_scanner.unusual_flow_detector import detect_unusual_flow
from options_flow_scanner.flow_scorer import (
    score_flow,
    grade_flow,
    FlowResult,
    FlowBias,
    FlowGrade,
    FlowAlertState,
    FLOW_GRADE_THRESHOLDS,
    FLOW_ALERT_MAP,
)
from options_flow_scanner.flow_report import (
    format_flow_report,
    format_flow_summary,
    log_flow_alert_csv,
    LOG_FILE as FLOW_LOG_FILE,
)

__all__ = [
    "load_universe",
    "DEFAULT_UNIVERSE",
    "EXTENDED_UNIVERSE",
    "load_chain",
    "compute_flow_features",
    "detect_unusual_flow",
    "score_flow",
    "grade_flow",
    "FlowResult",
    "FlowBias",
    "FlowGrade",
    "FlowAlertState",
    "FLOW_GRADE_THRESHOLDS",
    "FLOW_ALERT_MAP",
    "format_flow_report",
    "format_flow_summary",
    "log_flow_alert_csv",
    "FLOW_LOG_FILE",
]

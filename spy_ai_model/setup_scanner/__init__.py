"""
setup_scanner
─────────────
Intraday setup detection and alert engine for SPY.

Detects and scores potential intraday trading setups using:
  • Forecast model (direction + range)
  • Supply/demand zones
  • Analog engine outcomes
  • Confluence engine signals
  • Price / candlestick structure

This module is read-only with respect to positions — it is a research and
alert tool, not an auto-trading system.

Public API
──────────
  from setup_scanner import scan_snapshot, scan_replay

  # Single-bar snapshot
  results = scan_snapshot(df_bars, dir_prob, pred_range, zone_ctx, analog)

  # Historical replay over a full bar DataFrame
  replay_df = scan_replay(df_bars, dir_model, range_model)
"""

from setup_scanner.setup_definitions import (
    SetupType,
    SetupDirection,
    SetupGrade,
    AlertState,
    SetupResult,
)
from setup_scanner.setup_detector import detect_setups
from setup_scanner.setup_scorer import score_setup, grade_setup
from setup_scanner.setup_report import format_setup_report
from setup_scanner.setup_alerts import evaluate_alert_state, emit_alert, log_alert_csv
from setup_scanner.volume_integration import compute_volume_context, score_volume, volume_regime_label

__all__ = [
    "SetupType",
    "SetupDirection",
    "SetupGrade",
    "AlertState",
    "SetupResult",
    "detect_setups",
    "score_setup",
    "grade_setup",
    "format_setup_report",
    "evaluate_alert_state",
    "emit_alert",
    "log_alert_csv",
    "compute_volume_context",
    "score_volume",
    "volume_regime_label",
]

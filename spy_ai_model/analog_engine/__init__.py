"""
analog_engine
─────────────
Historical analog / supply-zone pattern-matching engine for SPY.

This module is a standalone research tool.  It does NOT touch the
forecasting pipeline.  It reuses the same data-loading utilities
(data.data_loader) and the same bar format (tz-naive ET DatetimeIndex,
open/high/low/close/volume columns).

Public API
──────────
    from analog_engine import run_analog_analysis
    report = run_analog_analysis(df_current_window)

    from analog_engine.analog_dataset_builder import build_analog_dataset
    build_analog_dataset()

Pipeline stages
───────────────
  1. zone_detector        – detect supply zones from pivot highs + displacement
  2. event_extractor      – find zone re-tests and capture ±10-bar windows
  3. pattern_features     – compute a feature vector for each event window
  4. analog_dataset_builder – assemble and persist the parquet dataset
  5. similarity_search    – find nearest historical events for a live window
  6. analog_report        – summarise statistics over matched events
"""

from analog_engine.analog_report import run_analog_analysis

__all__ = ["run_analog_analysis"]

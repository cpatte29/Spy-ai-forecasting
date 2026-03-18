"""
setup_alerts.py
───────────────
Alert engine for the setup scanner.

Alert states
────────────
  HIGH_CONVICTION  Grade A+ setup — strong alignment across all signals
  ACTIONABLE       Grade A setup  — good alignment, worth watching closely
  WATCHLIST        Grade B setup  — conditions forming, not confirmed
  SILENT           Grade C / IGNORE — do not emit

Alert output
────────────
  1. Terminal print  (always for WATCHLIST+)
  2. CSV log file    spy_ai_model/setup_scanner/logs/setup_signals.csv

CSV columns
───────────
  timestamp_et, bar_ts, ticker, interval, setup_type, direction,
  score, grade, alert_state, invalidation_level, target_zone_low,
  target_zone_high, dir_prob, pred_range, zone_bias,
  supply_strength, demand_strength, n_analog_matches,
  analog_rej_rate, analog_brk_rate, confluence_label,
  conditions_count, missing_count, guidance

Usage
─────
  from setup_scanner.setup_alerts import evaluate_alert_state, emit_alert, log_alert_csv

  alert_state = evaluate_alert_state(top_result)
  if alert_state != AlertState.SILENT:
      emit_alert(top_result, alert_state, bar_ts=ts, price=667.19)
      log_alert_csv(top_result, alert_state, bar_ts=ts, price=667.19)
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from setup_scanner.setup_definitions import (
    ALERT_GRADE_MAP,
    ALERT_MIN_GRADE,
    AlertState,
    SetupGrade,
    SetupResult,
)

logger = logging.getLogger(__name__)

# ── log file path ─────────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
LOG_DIR     = _MODULE_DIR / "logs"
LOG_FILE    = LOG_DIR / "setup_signals.csv"

_CSV_FIELDS = [
    "timestamp_et",
    "bar_ts",
    "ticker",
    "interval",
    "setup_type",
    "direction",
    "score",
    "grade",
    "alert_state",
    "invalidation_level",
    "target_zone_low",
    "target_zone_high",
    "dir_prob",
    "pred_range",
    "zone_bias",
    "supply_strength",
    "demand_strength",
    "n_analog_matches",
    "analog_rej_rate",
    "analog_brk_rate",
    "confluence_label",
    "conditions_count",
    "missing_count",
    "guidance",
]

# ANSI codes for alert severity
_ANSI_BOLD   = "\033[1m"
_ANSI_RESET  = "\033[0m"
_ANSI_RED    = "\033[91m"
_ANSI_YELLOW = "\033[93m"
_ANSI_GREEN  = "\033[92m"
_ANSI_CYAN   = "\033[96m"

_ALERT_COLOUR = {
    AlertState.HIGH_CONVICTION: _ANSI_RED    + _ANSI_BOLD,
    AlertState.ACTIONABLE:      _ANSI_YELLOW + _ANSI_BOLD,
    AlertState.WATCHLIST:       _ANSI_GREEN,
    AlertState.SILENT:          "",
}

_ALERT_BANNER = {
    AlertState.HIGH_CONVICTION: "🔴  HIGH CONVICTION SETUP",
    AlertState.ACTIONABLE:      "🟡  ACTIONABLE SETUP",
    AlertState.WATCHLIST:       "🟢  WATCHLIST SETUP",
    AlertState.SILENT:          "   (silent)",
}

# Grade ordering for comparison
_GRADE_ORDER = {
    SetupGrade.IGNORE: 0,
    SetupGrade.C:      1,
    SetupGrade.B:      2,
    SetupGrade.A:      3,
    SetupGrade.A_PLUS: 4,
}

_MIN_GRADE_ORDER = _GRADE_ORDER[ALERT_MIN_GRADE]


# ── public functions ──────────────────────────────────────────────────────────

def evaluate_alert_state(result: SetupResult) -> AlertState:
    """
    Determine the alert state for a SetupResult.

    Returns SILENT if the grade is below ALERT_MIN_GRADE.
    """
    return ALERT_GRADE_MAP.get(result.grade, AlertState.SILENT)


def should_alert(result: SetupResult) -> bool:
    """Return True if this result should generate an alert (above minimum grade)."""
    return _GRADE_ORDER.get(result.grade, 0) >= _MIN_GRADE_ORDER


def emit_alert(
    result:      SetupResult,
    alert_state: AlertState,
    bar_ts:      Optional[pd.Timestamp] = None,
    price:       Optional[float]        = None,
    ticker:      str                    = "SPY",
    interval:    str                    = "5m",
    colour:      bool                   = True,
) -> None:
    """
    Print a concise alert to the terminal.

    This does NOT place trades. Output only.
    """
    if alert_state == AlertState.SILENT:
        return

    sep    = "━" * 52
    banner = _ALERT_BANNER.get(alert_state, "")
    code   = _ALERT_COLOUR.get(alert_state, "")
    reset  = _ANSI_RESET if colour else ""

    ts_str    = pd.Timestamp(bar_ts).strftime("%Y-%m-%d %H:%M ET") if bar_ts else "n/a"
    price_str = f"${price:.2f}" if price is not None else "n/a"

    if colour:
        print(f"{code}{sep}{reset}")
        print(f"{code}  {banner}{reset}")
    else:
        print(sep)
        print(f"  {banner}")

    print(f"  {ticker} | Bar: {ts_str} | Price: {price_str} | {interval}")
    print(f"  Setup   : {result.setup_type.value}")
    print(f"  Grade   : {result.grade.value}  ({result.score}/100)  Direction: {result.direction.value}")

    if result.conditions_met:
        print("  Signals :")
        for c in result.conditions_met[:4]:
            print(f"    • {c}")
        if len(result.conditions_met) > 4:
            print(f"    … (+{len(result.conditions_met) - 4} more)")

    if result.invalidation_level is not None:
        print(f"  Invalid : {result.invalidation_level:.2f}")

    if result.target_zone is not None:
        lo, hi = result.target_zone
        print(f"  Target  : {lo:.2f} – {hi:.2f}")

    print(f"  Note    : {result.guidance[:100]}" if len(result.guidance) > 100
          else f"  Note    : {result.guidance}")

    if colour:
        print(f"{code}{sep}{reset}")
    else:
        print(sep)


def log_alert_csv(
    result:      SetupResult,
    alert_state: AlertState,
    bar_ts:      Optional[pd.Timestamp] = None,
    price:       Optional[float]        = None,
    ticker:      str                    = "SPY",
    interval:    str                    = "5m",
    log_file:    Path | None            = None,
) -> bool:
    """
    Append a single alert row to the CSV log file.

    Returns True if the row was written, False on error.
    """
    log_path = Path(log_file) if log_file else LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)

    raw = result.raw or {}

    try:
        from zoneinfo import ZoneInfo
        tz_et = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        tz_et = pytz.timezone("America/New_York")

    now_et = datetime.now(tz_et).strftime("%Y-%m-%d %H:%M:%S")
    bar_str = pd.Timestamp(bar_ts).isoformat() if bar_ts is not None else ""

    tz_low  = result.target_zone[0] if result.target_zone else ""
    tz_high = result.target_zone[1] if result.target_zone else ""

    row = {
        "timestamp_et":       now_et,
        "bar_ts":             bar_str,
        "ticker":             ticker,
        "interval":           interval,
        "setup_type":         result.setup_type.value,
        "direction":          result.direction.value,
        "score":              result.score,
        "grade":              result.grade.value,
        "alert_state":        alert_state.value,
        "invalidation_level": result.invalidation_level if result.invalidation_level is not None else "",
        "target_zone_low":    tz_low,
        "target_zone_high":   tz_high,
        "dir_prob":           raw.get("dir_prob", ""),
        "pred_range":         raw.get("pred_range", ""),
        "zone_bias":          raw.get("zone_bias", ""),
        "supply_strength":    raw.get("supply_strength", ""),
        "demand_strength":    raw.get("demand_strength", ""),
        "n_analog_matches":   raw.get("n_analog_matches", ""),
        "analog_rej_rate":    raw.get("analog_rej_rate", ""),
        "analog_brk_rate":    raw.get("analog_brk_rate", ""),
        "confluence_label":   raw.get("confluence_label", ""),
        "conditions_count":   len(result.conditions_met),
        "missing_count":      len(result.missing_confirms),
        "guidance":           result.guidance.replace("\n", " ")[:200],
    }

    write_header = not log_path.exists() or log_path.stat().st_size == 0
    try:
        with log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        logger.debug("Alert logged → %s", log_path)
        return True
    except Exception as exc:
        logger.error("Failed to write alert CSV: %s", exc)
        return False


def process_alerts(
    ranked_results: list[SetupResult],
    bar_ts:         Optional[pd.Timestamp] = None,
    price:          Optional[float]        = None,
    ticker:         str                    = "SPY",
    interval:       str                    = "5m",
    colour:         bool                   = True,
    log_file:       Path | None            = None,
    min_alert_grade: SetupGrade            = ALERT_MIN_GRADE,
) -> list[tuple[SetupResult, AlertState]]:
    """
    Evaluate, emit, and log alerts for all qualifying results.

    Returns a list of (result, alert_state) pairs that were emitted.
    """
    emitted: list[tuple[SetupResult, AlertState]] = []
    min_ord = _GRADE_ORDER.get(min_alert_grade, 0)

    for result in ranked_results:
        if _GRADE_ORDER.get(result.grade, 0) < min_ord:
            continue

        alert_state = evaluate_alert_state(result)
        if alert_state == AlertState.SILENT:
            continue

        emit_alert(result, alert_state, bar_ts=bar_ts, price=price,
                   ticker=ticker, interval=interval, colour=colour)
        log_alert_csv(result, alert_state, bar_ts=bar_ts, price=price,
                      ticker=ticker, interval=interval, log_file=log_file)
        emitted.append((result, alert_state))

    return emitted

"""
live_loop.py
────────────
Run run_live_prediction every 5 minutes during regular market hours
(08:30–15:00 Central Time = 09:30–16:00 Eastern Time).

All predictions are appended to live_predictions/prediction_log.csv.
Duplicate bar timestamps are silently skipped.  Any prediction failure
is logged and the loop continues rather than aborting.

Usage
-----
  python live_loop.py
  python live_loop.py --threshold 0.528
  python live_loop.py --poll-seconds 300 --log-level INFO
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DIR_PROB_THRESHOLD
from run_live_prediction import (
    DEFAULT_MIN_SESSION_BARS,
    LIVE_DIR,
    run_live_prediction,
)

logger = logging.getLogger(__name__)

PREDICTION_LOG = LIVE_DIR / "prediction_log.csv"

# Regular market hours in Central Time (Chicago).
# 08:30 CT = 09:30 ET  (NYSE open)
# 15:00 CT = 16:00 ET  (NYSE close)
_OPEN_CT  = (8,  30)
_CLOSE_CT = (15,  0)

_CSV_FIELDS = [
    "generated_at_utc",
    "now_et",
    "bar_timestamp",
    "ticker",
    "interval",
    "current_close",
    "dir_probability",
    "predicted_range",
    "range_pts",
    "threshold",
    "min_range",
    "confidence_bucket",
    "range_bucket",
    "signal",
    "session_bars",
]


# ── market-hours gate ─────────────────────────────────────────────────────────

def _is_market_hours_ct() -> bool:
    """Return True when the current CT wall-clock time is inside market hours."""
    try:
        from zoneinfo import ZoneInfo          # Python 3.9+
        tz = ZoneInfo("America/Chicago")
    except ImportError:
        try:
            import pytz
            tz = pytz.timezone("America/Chicago")
        except ImportError:
            logger.warning(
                "Neither zoneinfo nor pytz is installed. "
                "Market-hours gate disabled – predictions will run every poll cycle."
            )
            return True

    now_ct = datetime.now(tz)
    if now_ct.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    t = (now_ct.hour, now_ct.minute)
    return _OPEN_CT <= t <= _CLOSE_CT


# ── CSV log helpers ───────────────────────────────────────────────────────────

def _load_seen_timestamps() -> set[str]:
    """Return all bar_timestamps already present in the prediction log."""
    if not PREDICTION_LOG.exists():
        return set()
    seen: set[str] = set()
    with PREDICTION_LOG.open(newline="") as f:
        for row in csv.DictReader(f):
            seen.add(row.get("bar_timestamp", ""))
    return seen


def _append_to_log(result: dict) -> bool:
    """
    Append a prediction result row to PREDICTION_LOG.

    Returns True if written, False if the bar_timestamp was already logged.
    """
    if result["bar_timestamp"] in _load_seen_timestamps():
        logger.debug(
            "Duplicate bar_timestamp %s – skipping log entry.",
            result["bar_timestamp"],
        )
        return False

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not PREDICTION_LOG.exists()
    with PREDICTION_LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    return True


# ── loop ──────────────────────────────────────────────────────────────────────

def run_loop(
    interval:         str   = "5m",
    lookback_days:    int   = 7,
    threshold:        float = DIR_PROB_THRESHOLD,
    min_range:        float = 0.0,
    min_session_bars: int   = DEFAULT_MIN_SESSION_BARS,
    poll_seconds:     int   = 300,
) -> None:
    """
    Block indefinitely, running live predictions on each market-hours poll cycle.

    Outside market hours the loop sleeps 60 s between checks so it wakes up
    promptly at open without spinning.
    """
    logger.info(
        "SPY AI Live Loop started  |  interval=%s  threshold=%.3f  "
        "min_range=%.4f  min_session_bars=%d  poll=%ds",
        interval, threshold, min_range, min_session_bars, poll_seconds,
    )
    logger.info("Prediction log → %s", PREDICTION_LOG)

    while True:
        if not _is_market_hours_ct():
            logger.debug(
                "[%s] Outside CT market hours (08:30–15:00 Mon–Fri). Sleeping 60s.",
                datetime.now().strftime("%H:%M:%S"),
            )
            time.sleep(60)
            continue

        try:
            result  = run_live_prediction(
                interval=interval,
                lookback_days=lookback_days,
                threshold=threshold,
                min_range=min_range,
                min_session_bars=min_session_bars,
            )
            written = _append_to_log(result)
            if written:
                logger.info(
                    "Logged  bar=%-30s  prob=%.4f  signal=%s",
                    result["bar_timestamp"],
                    result["dir_probability"],
                    result["signal"],
                )
            else:
                logger.debug("Bar already logged – waiting for next bar.")

        except Exception as exc:
            logger.error(
                "Prediction attempt failed: %s", exc, exc_info=True,
            )

        time.sleep(poll_seconds)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPY AI continuous 5-minute live loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--interval",          default="5m",
                   help="Bar interval (must match model training interval)")
    p.add_argument("--lookback-days",     default=7, type=int,
                   help="Calendar days of history fetched per prediction")
    p.add_argument("--threshold",         default=DIR_PROB_THRESHOLD, type=float,
                   help="Direction probability threshold")
    p.add_argument("--min-range",         default=0.0, type=float,
                   help="Minimum predicted range for signal (0 = disabled)")
    p.add_argument("--min-session-bars",  default=DEFAULT_MIN_SESSION_BARS, type=int,
                   help="Minimum session bars before prediction is trusted")
    p.add_argument("--poll-seconds",      default=300, type=int,
                   help="Seconds between prediction attempts (300 = 5 min)")
    p.add_argument("--log-level",         default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_loop(
        interval=args.interval,
        lookback_days=args.lookback_days,
        threshold=args.threshold,
        min_range=args.min_range,
        min_session_bars=args.min_session_bars,
        poll_seconds=args.poll_seconds,
    )

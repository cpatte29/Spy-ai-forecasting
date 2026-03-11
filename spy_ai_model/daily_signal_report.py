"""
daily_signal_report.py
──────────────────────
Read the prediction log and print a human-readable daily summary.

Summary includes
----------------
  • Total bars logged for the day
  • Number of bullish / bearish / no-trade signals
  • Highest and lowest direction probability readings
  • Average predicted range (normalised and in dollar points)
  • All timestamps where a threshold-crossing signal was issued

Usage
-----
  python daily_signal_report.py               # today
  python daily_signal_report.py --date 2026-03-10
  python daily_signal_report.py --threshold 0.528
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DIR_PROB_THRESHOLD

logger = logging.getLogger(__name__)

# Canonical log path defined in live_loop; import it if available, otherwise
# fall back to the same hard-coded location so this script is self-contained.
try:
    from live_loop import PREDICTION_LOG
except ImportError:
    PREDICTION_LOG = ROOT / "live_predictions" / "prediction_log.csv"


# ── data helpers ──────────────────────────────────────────────────────────────

def _load_day_rows(target: date) -> list[dict]:
    """Return all prediction rows whose bar_timestamp falls on target."""
    if not PREDICTION_LOG.exists():
        return []
    rows: list[dict] = []
    with PREDICTION_LOG.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                row_date = date.fromisoformat(row["bar_timestamp"][:10])
            except (KeyError, ValueError):
                continue
            if row_date == target:
                rows.append(row)
    return rows


# ── report ────────────────────────────────────────────────────────────────────

def generate_report(target: date, threshold: float = DIR_PROB_THRESHOLD) -> None:
    rows = _load_day_rows(target)
    sep  = "=" * 62
    dash = "-" * 62

    print(sep)
    print(f"  SPY AI Daily Signal Report  ─  {target.isoformat()}")
    print(sep)

    if not rows:
        print(f"  No predictions found in:")
        print(f"    {PREDICTION_LOG}")
        print(f"  for date {target.isoformat()}.")
        print(f"  Run live_loop.py during market hours to populate the log.")
        print(sep)
        return

    # Parse columns ───────────────────────────────────────────────────────────
    timestamps  = [r["bar_timestamp"]          for r in rows]
    probs       = [float(r["dir_probability"]) for r in rows]
    ranges_norm = [float(r["predicted_range"]) for r in rows]
    signals     = [r["signal"]                 for r in rows]
    closes      = [float(r["current_close"])   for r in rows]

    n_total    = len(rows)
    n_bullish  = signals.count("LONG_BIAS")
    n_bearish  = signals.count("SHORT_BIAS")
    n_no_trade = signals.count("NO_TRADE")

    max_prob   = max(probs)
    min_prob   = min(probs)
    avg_range  = sum(ranges_norm) / len(ranges_norm)
    ref_close  = closes[-1]    # last close of the session as reference

    # Threshold-crossing rows ─────────────────────────────────────────────────
    signal_rows = [
        (ts, sig, prob, rng)
        for ts, sig, prob, rng in zip(timestamps, signals, probs, ranges_norm)
        if sig in ("LONG_BIAS", "SHORT_BIAS")
    ]

    # Print ───────────────────────────────────────────────────────────────────
    print(f"  Total bars logged       : {n_total}")
    print(f"  Bullish (LONG_BIAS)     : {n_bullish}")
    print(f"  Bearish (SHORT_BIAS)    : {n_bearish}")
    print(f"  No-trade                : {n_no_trade}")
    print(f"  Threshold used          : {threshold}")
    print(dash)
    print(f"  Highest dir probability : {max_prob:.4f}")
    print(f"  Lowest  dir probability : {min_prob:.4f}")
    print(
        f"  Avg predicted range     : {avg_range:.4f}"
        f"  ({avg_range * ref_close:.2f} pts @ ${ref_close:.2f})"
    )
    print(dash)

    if signal_rows:
        print(f"  Threshold-crossing bars ({len(signal_rows)}):")
        print(f"    {'Timestamp':<32}  {'Signal':<12}  {'Prob':>6}  {'Range':>6}")
        print(f"    {'-'*30}  {'-'*12}  {'-'*6}  {'-'*6}")
        for ts, sig, prob, rng in signal_rows:
            print(f"    {ts:<32}  {sig:<12}  {prob:.4f}  {rng:.4f}")
    else:
        print("  No threshold-crossing signals today.")

    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPY AI daily signal summary report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--date", default=None,
        help="Target date YYYY-MM-DD (default: today)",
    )
    p.add_argument(
        "--threshold", default=DIR_PROB_THRESHOLD, type=float,
        help="Probability threshold used to classify signals",
    )
    p.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s  %(message)s")
    target = (
        date.fromisoformat(args.date)
        if args.date
        else date.today()
    )
    generate_report(target=target, threshold=args.threshold)

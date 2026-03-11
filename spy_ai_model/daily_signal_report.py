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

# Canonical paths defined in their respective modules; fall back gracefully
# so this script is self-contained if imports are unavailable.
try:
    from live_loop import PREDICTION_LOG
except ImportError:
    PREDICTION_LOG = ROOT / "live_predictions" / "prediction_log.csv"

try:
    from run_live_prediction import PAPER_TRADE_LOG
except ImportError:
    PAPER_TRADE_LOG = ROOT / "live_predictions" / "paper_trade_log.csv"


# ── data helpers ──────────────────────────────────────────────────────────────

def _load_day_rows(target: date) -> list[dict]:
    """Return all prediction-log rows whose bar_timestamp falls on target."""
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


def _load_paper_trade_rows(target: date) -> list[dict]:
    """Return all paper-trade rows whose signal_timestamp falls on target."""
    if not PAPER_TRADE_LOG.exists():
        return []
    rows: list[dict] = []
    with PAPER_TRADE_LOG.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                row_date = date.fromisoformat(row["signal_timestamp"][:10])
            except (KeyError, ValueError):
                continue
            if row_date == target:
                rows.append(row)
    return rows


# ── report ────────────────────────────────────────────────────────────────────

def generate_report(target: date, threshold: float = DIR_PROB_THRESHOLD) -> None:
    rows   = _load_day_rows(target)
    pt_rows = _load_paper_trade_rows(target)
    sep    = "=" * 64
    dash   = "─" * 64

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

    # ── Prediction-log summary ────────────────────────────────────────────────
    timestamps   = [r["bar_timestamp"]          for r in rows]
    probs        = [float(r["dir_probability"]) for r in rows]
    ranges_norm  = [float(r["predicted_range"]) for r in rows]
    signals      = [r["signal"]                 for r in rows]
    closes       = [float(r["current_close"])   for r in rows]
    conf_buckets = [r.get("confidence_bucket", "") for r in rows]
    rng_buckets  = [r.get("range_bucket", "")   for r in rows]

    n_total    = len(rows)
    n_bullish  = signals.count("LONG_BIAS")
    n_bearish  = signals.count("SHORT_BIAS")
    n_no_trade = signals.count("NO_TRADE")
    max_prob   = max(probs)
    min_prob   = min(probs)
    avg_range  = sum(ranges_norm) / len(ranges_norm)
    ref_close  = closes[-1]

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

    # Confidence bucket breakdown
    if any(conf_buckets):
        print(dash)
        print(f"  Confidence buckets:")
        for bucket in ("STRONG", "MODERATE", "WEAK"):
            count = conf_buckets.count(bucket)
            print(f"    {bucket:<10} : {count}")

    # Range bucket breakdown
    if any(rng_buckets):
        print(f"  Range buckets:")
        for bucket in ("VERY_WIDE", "WIDE", "MODERATE", "TIGHT"):
            count = rng_buckets.count(bucket)
            print(f"    {bucket:<10} : {count}")

    # Threshold-crossing bars ─────────────────────────────────────────────────
    signal_rows = [
        (ts, sig, prob, rng)
        for ts, sig, prob, rng in zip(timestamps, signals, probs, ranges_norm)
        if sig in ("LONG_BIAS", "SHORT_BIAS")
    ]
    print(dash)
    if signal_rows:
        print(f"  Threshold-crossing bars ({len(signal_rows)}):")
        print(f"    {'Timestamp':<28}  {'Signal':<12}  {'Prob':>6}  {'Range':>6}")
        print(f"    {'-'*26}  {'-'*12}  {'-'*6}  {'-'*6}")
        for ts, sig, prob, rng in signal_rows:
            print(f"    {ts:<28}  {sig:<12}  {prob:.4f}  {rng:.4f}")
    else:
        print("  No threshold-crossing signals today.")

    # ── Paper-trade backfill summary ──────────────────────────────────────────
    if pt_rows:
        backfilled = [r for r in pt_rows if r.get("realized_dir") != ""]
        pending    = len(pt_rows) - len(backfilled)

        print(sep)
        print(f"  Paper Trade Log  ─  {PAPER_TRADE_LOG.name}")
        print(dash)
        print(f"  Rows logged today       : {len(pt_rows)}")
        print(f"  Backfilled (realized)   : {len(backfilled)}")
        print(f"  Pending backfill        : {pending}")

        if backfilled:
            # Win-rate on LONG signals
            long_bf = [r for r in backfilled if r["signal"] == "LONG_BIAS"]
            if long_bf:
                long_wins = sum(1 for r in long_bf if str(r["realized_dir"]) == "1")
                print(f"  LONG win rate           : {long_wins}/{len(long_bf)}"
                      f"  ({100*long_wins/len(long_bf):.0f}%)")

            # Win-rate on SHORT signals
            short_bf = [r for r in backfilled if r["signal"] == "SHORT_BIAS"]
            if short_bf:
                # SHORT wins when realized_dir = 0 (price went down)
                short_wins = sum(1 for r in short_bf if str(r["realized_dir"]) == "0")
                print(f"  SHORT win rate          : {short_wins}/{len(short_bf)}"
                      f"  ({100*short_wins/len(short_bf):.0f}%)")

            # Range accuracy: mean(realized / predicted)
            ratios = []
            for r in backfilled:
                try:
                    pred = float(r["predicted_range"])
                    real = float(r["realized_range_actual"])
                    if pred > 0:
                        ratios.append(real / pred)
                except (ValueError, ZeroDivisionError):
                    continue
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                print(f"  Actual/pred range ratio : {avg_ratio:.2f}x"
                      f"  ({'under' if avg_ratio < 1 else 'over'}-predicted)")

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

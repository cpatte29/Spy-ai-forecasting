#!/usr/bin/env python3
"""
download_polygon_history.py
───────────────────────────
Fetch historical aggregate bars from Polygon.io and save them locally as a
Parquet file suitable for offline training with the SPY AI forecasting pipeline.

Timestamp storage
─────────────────
  Polygon sends Unix-millisecond timestamps in UTC.
  This script converts them to Eastern Time (ET / America/New_York) and stores
  them as a tz-naive DatetimeIndex.  Every row timestamp represents the bar
  open time in ET.  For example, 2024-01-02 09:30:00 means the 09:30 ET bar.

  Summary: stored as ET, tz-naive (America/New_York, tzinfo stripped).

Output schema
─────────────
  Index   : timestamp  (tz-naive ET, bar open time)
  Columns :
    open          float64  – bar open price
    high          float64  – bar high price
    low           float64  – bar low price
    close         float64  – bar close price
    volume        float64  – share volume
    vwap          float64  – volume-weighted average price (when available)
    transactions  int64    – number of trades in the bar (when available)

  Only regular-session bars are kept: 09:30 – 16:00 ET.
  Pre-market and after-hours bars are dropped automatically.

Built-in validation
───────────────────
  After saving, the script prints a data quality summary:
    • duplicate timestamp count
    • bars per trading day (min / median / max)
    • sessions with fewer bars than expected
    • overall row count and date range

Usage
─────
  # Full history from 2016-01-01 to today:
  python data/download_polygon_history.py \\
      --symbol SPY --interval 5m \\
      --start 2016-01-01 --end 2026-03-15

  # Last 90 calendar days (quick test):
  python data/download_polygon_history.py --symbol SPY --interval 5m --period 90d

  # Custom output path:
  python data/download_polygon_history.py \\
      --symbol SPY --interval 5m --start 2020-01-01 \\
      --output /data/spy_5m_custom.parquet

Environment variable required
─────────────────────────────
  export POLYGON_API_KEY=<your_key>

  ⚠️  Never hardcode or commit your API key.

After downloading, train the model:
────────────────────────────────────
  python main_pipeline.py \\
      --mode file \\
      --file-path data/raw/spy_5m_polygon.parquet \\
      --interval 5m \\
      --horizon-dir 12 --horizon-range 12

Run a live prediction using Polygon:
─────────────────────────────────────
  export POLYGON_API_KEY=<your_key>
  python run_live_prediction.py --provider polygon --interval 5m
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── make project root importable ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("download_polygon_history")

# Expected bars per regular session by interval
_EXPECTED_BARS: dict[str, int] = {
    "1m":  390,
    "5m":   78,
    "15m":  26,
    "30m":  13,
    "1h":    7,
    "60m":   7,
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch SPY bars from Polygon.io and save as Parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data/download_polygon_history.py"
            " --symbol SPY --interval 5m --start 2016-01-01 --end 2026-03-15\n"
            "  python data/download_polygon_history.py"
            " --symbol SPY --interval 5m --period 1y\n"
            "  python data/download_polygon_history.py"
            " --symbol SPY --interval 1m --start 2024-01-01\n"
        ),
    )
    parser.add_argument(
        "--symbol",
        default="SPY",
        metavar="TICKER",
        help="Ticker symbol to download (default: SPY).",
    )
    parser.add_argument(
        "--interval",
        default="5m",
        metavar="INTERVAL",
        help=(
            "Bar size: 1m, 5m, 15m, 30m, 1h.  Default: 5m.\n"
            "Polygon supports arbitrary history for all intervals."
        ),
    )
    parser.add_argument(
        "--start",
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date (inclusive).  Required unless --period is given.",
    )
    parser.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "End date (exclusive, i.e. fetch up to but not including this date).\n"
            "Defaults to tomorrow so today's intraday bars are included."
        ),
    )
    parser.add_argument(
        "--period",
        default=None,
        metavar="PERIOD",
        help=(
            "Calendar period shorthand: 7d, 30d, 90d, 1y, 2y, etc.\n"
            "Used only when --start is not given."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Output Parquet file path.\n"
            "Default: data/raw/<symbol_lower>_<interval>_polygon.parquet\n"
            "Example default for SPY 5m: data/raw/spy_5m_polygon.parquet"
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip the post-download validation report (faster for large downloads).",
    )
    return parser.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _default_output(symbol: str, interval: str) -> Path:
    """Return the default parquet output path for a given symbol + interval."""
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{symbol.lower()}_{interval}_polygon.parquet"
    return raw_dir / fname


def _fetch(symbol: str, interval: str, start: str | None,
           end: str | None, period: str | None) -> pd.DataFrame:
    """Call PolygonProvider and return the raw extended DataFrame."""
    from data.providers.polygon_provider import PolygonProvider

    provider = PolygonProvider()
    logger.info(
        "Fetching %s %s bars from Polygon  [start=%s  end=%s  period=%s]",
        symbol, interval, start or "–", end or "–", period or "–",
    )
    df = provider.get_stock_bars(
        symbol   = symbol,
        interval = interval,
        start    = start,
        end      = end,
        period   = period,
    )
    return df


def _save(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to Parquet, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    size_mb = path.stat().st_size / 1_048_576
    logger.info("Saved %d bars → %s  (%.2f MB)", len(df), path, size_mb)


# ── inline validation report ──────────────────────────────────────────────────

def _print_validation_report(df: pd.DataFrame, interval: str) -> None:
    """
    Print a concise data quality summary directly to stdout.

    Checks:
      1. Duplicate timestamps
      2. Sort order
      3. Date range and total row count
      4. Bars per trading day (min/median/max)
      5. Sessions with fewer bars than expected (flagged as missing bars)
      6. Row counts by month (top-level session coverage view)
    """
    dash = "─" * 62

    print()
    print(dash)
    print("  Polygon Download — Data Quality Report")
    print(dash)

    if df.empty:
        print("  ⚠  DataFrame is empty — nothing to validate.")
        print(dash)
        return

    # ── 1. Basic stats ─────────────────────────────────────────────────────
    total_rows = len(df)
    min_ts     = df.index.min()
    max_ts     = df.index.max()

    print(f"  Rows             : {total_rows:,}")
    print(f"  Min timestamp    : {min_ts}  (ET, tz-naive)")
    print(f"  Max timestamp    : {max_ts}  (ET, tz-naive)")
    print(f"  Columns          : {', '.join(df.columns.tolist())}")

    # ── 2. Duplicate timestamps ────────────────────────────────────────────
    n_dupes = df.index.duplicated().sum()
    dupe_str = f"NONE ✓" if n_dupes == 0 else f"{n_dupes}  ← ⚠  review required"
    print(f"  Duplicates       : {dupe_str}")

    # ── 3. Sort order ──────────────────────────────────────────────────────
    sorted_ok = df.index.is_monotonic_increasing
    sort_str  = "ascending ✓" if sorted_ok else "NOT SORTED  ← ⚠  run sort_index()"
    print(f"  Sort order       : {sort_str}")

    # ── 4. Bars per trading day ─────────────────────────────────────────────
    date_key     = df.index.normalize()
    bars_per_day = df.groupby(date_key).size()
    n_sessions   = len(bars_per_day)

    print()
    print(f"  Trading sessions : {n_sessions}")
    print(f"  Bars/session     : "
          f"min={bars_per_day.min()}  "
          f"median={int(bars_per_day.median())}  "
          f"max={bars_per_day.max()}")

    # ── 5. Sessions with fewer bars than expected ──────────────────────────
    expected = _EXPECTED_BARS.get(interval)
    if expected:
        threshold_warn     = int(expected * 0.70)   # < 70% → warn
        threshold_critical = int(expected * 0.50)   # < 50% → critical
        warn_sessions      = bars_per_day[bars_per_day < threshold_warn]
        crit_sessions      = bars_per_day[bars_per_day < threshold_critical]

        print(f"  Expected bars/session: {expected}  "
              f"(warn < {threshold_warn}, critical < {threshold_critical})")
        if warn_sessions.empty:
            print("  Low-bar sessions : NONE ✓")
        else:
            print(f"  Low-bar sessions : {len(warn_sessions)} warning  "
                  f"({len(crit_sessions)} critical)")
            if len(warn_sessions) <= 20:
                for ts, cnt in warn_sessions.items():
                    severity = "CRITICAL" if cnt < threshold_critical else "WARN    "
                    print(f"    {severity}  {ts.date()}  {cnt} bars "
                          f"({cnt/expected:.0%} coverage)")
            else:
                # Show only critical when list is long
                print(f"  (showing critical only; {len(warn_sessions) - len(crit_sessions)}"
                      f" additional warn sessions omitted)")
                for ts, cnt in crit_sessions.items():
                    print(f"    CRITICAL  {ts.date()}  {cnt} bars "
                          f"({cnt/expected:.0%} coverage)")
    else:
        print(f"  (no expected-bar benchmark configured for interval={interval!r})")

    # ── 6. Rows by year ────────────────────────────────────────────────────
    print()
    print("  Rows by year:")
    rows_by_year = df.groupby(df.index.year).size()
    for yr, cnt in rows_by_year.items():
        bar_len = int(cnt / max(rows_by_year) * 30)
        bar     = "█" * bar_len
        print(f"    {yr}  {cnt:6,}  {bar}")

    print(dash)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not args.start and not args.period:
        logger.error(
            "Provide --start YYYY-MM-DD or --period (e.g. --period 1y).  "
            "Run with --help for usage."
        )
        sys.exit(1)

    # Determine output path
    out_path = Path(args.output) if args.output else _default_output(args.symbol, args.interval)

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║   Polygon Historical Downloader                   ║")
    logger.info("╠══════════════════════════════════════════════════╣")
    logger.info("║  symbol    : %-35s ║", args.symbol)
    logger.info("║  interval  : %-35s ║", args.interval)
    if args.start:
        logger.info("║  start     : %-35s ║", args.start)
    if args.end:
        logger.info("║  end       : %-35s ║", args.end)
    if args.period:
        logger.info("║  period    : %-35s ║", args.period)
    logger.info("║  output    : %-35s ║", str(out_path)[-35:])
    logger.info("╚══════════════════════════════════════════════════╝")

    # Fetch
    df = _fetch(
        symbol   = args.symbol,
        interval = args.interval,
        start    = args.start,
        end      = args.end,
        period   = args.period,
    )

    if df.empty:
        logger.error(
            "Polygon returned no bars for %s %s [%s → %s]. "
            "Check POLYGON_API_KEY and date range.",
            args.symbol, args.interval, args.start, args.end,
        )
        sys.exit(1)

    logger.info(
        "Fetched %d bars  |  %s → %s",
        len(df),
        df.index[0].strftime("%Y-%m-%d"),
        df.index[-1].strftime("%Y-%m-%d"),
    )

    # Save
    _save(df, out_path)

    # Validate
    if not args.no_validate:
        _print_validation_report(df, args.interval)

    logger.info("Download complete.")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  Train model:")
    logger.info("    python main_pipeline.py \\")
    logger.info("        --mode file \\")
    logger.info("        --file-path %s \\", out_path)
    logger.info("        --interval %s \\", args.interval)
    logger.info("        --horizon-dir 12 --horizon-range 12")
    logger.info("")
    logger.info("  Live prediction (requires POLYGON_API_KEY):")
    logger.info("    python run_live_prediction.py --provider polygon --interval %s", args.interval)


if __name__ == "__main__":
    main()

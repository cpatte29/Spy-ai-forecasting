#!/usr/bin/env python3
"""
validate_polygon_data.py
────────────────────────
Standalone data quality report for a locally saved Parquet (or CSV) file of
intraday bars downloaded from Polygon.io (or any compatible source).

Prints:
  • Total rows
  • Min / max timestamp (with timezone note)
  • Bars per trading day  (min / median / max / stddev)
  • Days with missing bars  (below expected count for the interval)
  • Duplicate timestamp count
  • NaN counts per column
  • Rows by year and by month

Usage
─────
  python data/validate_polygon_data.py \\
      --file-path data/raw/spy_5m_polygon.parquet \\
      --interval 5m

  python data/validate_polygon_data.py \\
      --file-path data/raw/spy_1m_polygon.parquet \\
      --interval 1m --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Expected regular-session bars per interval
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
    p = argparse.ArgumentParser(
        description="Print a data quality report for a saved bar file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data/validate_polygon_data.py"
            " --file-path data/raw/spy_5m_polygon.parquet --interval 5m\n"
            "  python data/validate_polygon_data.py"
            " --file-path data/raw/spy_1m_polygon.parquet --interval 1m --verbose\n"
        ),
    )
    p.add_argument(
        "--file-path",
        required=True,
        metavar="PATH",
        help="Path to the Parquet or CSV file to validate.",
    )
    p.add_argument(
        "--interval",
        default="5m",
        metavar="INTERVAL",
        help="Bar interval for expected-count calculations (default: 5m).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-month bar counts and all low-coverage sessions.",
    )
    return p.parse_args()


# ── loader ────────────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame:
    """Load bars from Parquet or CSV, normalising the index."""
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    if path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".txt"):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        print(f"ERROR: unsupported format {path.suffix!r}.", file=sys.stderr)
        sys.exit(1)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)

    # Strip timezone if present (project stores tz-naive ET)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)

    return df


# ── report ────────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, interval: str, verbose: bool = False) -> None:
    """Print a comprehensive validation report to stdout."""
    W  = 64
    dash  = "─" * W
    thick = "═" * W

    def _section(title: str) -> None:
        print()
        print(f"  ┌{'─' * (W - 2)}┐")
        print(f"  │  {title:<{W - 5}}│")
        print(f"  └{'─' * (W - 2)}┘")

    print()
    print(thick)
    print("  Polygon Bar File — Validation Report")
    print(thick)

    # ── 1. Overview ────────────────────────────────────────────────────────
    _section("1. Overview")

    total_rows = len(df)
    cols       = df.columns.tolist()
    size_info  = ""

    print(f"  Total rows       : {total_rows:,}")
    print(f"  Columns          : {', '.join(cols)}")

    if df.empty:
        print("  ⚠  DataFrame is empty — nothing more to report.")
        print(thick)
        return

    min_ts = df.index.min()
    max_ts = df.index.max()
    print(f"  Min timestamp    : {min_ts}")
    print(f"  Max timestamp    : {max_ts}")
    print(f"  Timezone         : tz-naive ET (America/New_York)")
    print(f"  Span             : {(max_ts - min_ts).days} calendar days")

    # ── 2. Duplicate timestamps ────────────────────────────────────────────
    _section("2. Duplicate Timestamps")

    n_dupes = int(df.index.duplicated().sum())
    if n_dupes == 0:
        print("  Duplicates       : NONE ✓")
    else:
        print(f"  Duplicates       : {n_dupes}  ← ⚠  de-duplicate before training")
        dupe_ts = df.index[df.index.duplicated(keep=False)].unique()
        for ts in dupe_ts[:10]:
            print(f"    {ts}")
        if len(dupe_ts) > 10:
            print(f"    ... ({len(dupe_ts) - 10} more)")

    # ── 3. Sort order ──────────────────────────────────────────────────────
    _section("3. Sort Order")

    sorted_ok = df.index.is_monotonic_increasing
    print(f"  Sorted ascending : {'YES ✓' if sorted_ok else 'NO  ← ⚠  sort_index() needed'}")

    # ── 4. NaN / missing values ────────────────────────────────────────────
    _section("4. NaN / Missing Values")

    nan_counts = df.isnull().sum()
    any_nan    = nan_counts.any()
    if not any_nan:
        print("  NaN values       : NONE ✓")
    else:
        for col, cnt in nan_counts[nan_counts > 0].items():
            pct = cnt / total_rows * 100
            print(f"  {col:<20} : {cnt:,} NaN  ({pct:.2f}%)")

    # ── 5. Bars per trading session ────────────────────────────────────────
    _section("5. Bars per Trading Session")

    date_key     = df.index.normalize()
    bars_per_day = df.groupby(date_key).size()
    n_sessions   = len(bars_per_day)

    print(f"  Trading sessions : {n_sessions}")
    print(f"  Bars/session  min: {bars_per_day.min()}")
    print(f"  Bars/session  med: {int(bars_per_day.median())}")
    print(f"  Bars/session  max: {bars_per_day.max()}")
    print(f"  Bars/session  std: {bars_per_day.std():.1f}")

    # ── 6. Sessions with missing bars ─────────────────────────────────────
    _section("6. Sessions with Missing Bars")

    expected = _EXPECTED_BARS.get(interval)
    if expected is None:
        print(f"  (no expected-bar threshold for interval={interval!r}; skipping)")
    else:
        thresh_warn = int(expected * 0.70)   # < 70% → warn
        thresh_crit = int(expected * 0.50)   # < 50% → critical

        perfect   = bars_per_day[bars_per_day == expected]
        warn_sess = bars_per_day[(bars_per_day < thresh_warn) & (bars_per_day >= thresh_crit)]
        crit_sess = bars_per_day[bars_per_day < thresh_crit]
        ok_sess   = bars_per_day[
            (bars_per_day >= thresh_warn) & (bars_per_day < expected)
        ]

        print(f"  Expected bars/session   : {expected}")
        print(f"  Warn threshold (<70%)   : {thresh_warn}")
        print(f"  Critical threshold(<50%): {thresh_crit}")
        print()
        print(f"  Full-coverage sessions  : {len(perfect)}")
        print(f"  Minor shortfall (<100%) : {len(ok_sess)}")
        print(f"  Warning  (<70%) sessions: {len(warn_sess)}")
        print(f"  Critical (<50%) sessions: {len(crit_sess)}")

        if verbose or len(crit_sess) > 0:
            show_all = warn_sess if verbose else crit_sess
            label    = "WARN/CRIT" if verbose else "CRITICAL"
            if not show_all.empty:
                print()
                print(f"  {label} sessions:")
                for ts, cnt in show_all.sort_index().items():
                    sev  = "CRITICAL" if cnt < thresh_crit else "WARN    "
                    pct  = cnt / expected * 100
                    print(f"    {sev}  {ts.date()}  {cnt:3d} bars  ({pct:5.1f}%)")

    # ── 7. Rows by year ────────────────────────────────────────────────────
    _section("7. Rows by Year")

    rows_by_year = df.groupby(df.index.year).size()
    max_yr_rows  = rows_by_year.max()
    for yr, cnt in rows_by_year.items():
        bar_len = int(cnt / max_yr_rows * 30)
        bar     = "█" * bar_len
        sessions_yr = len(bars_per_day[bars_per_day.index.year == yr])
        print(f"  {yr}  {cnt:7,} rows  {sessions_yr:4d} sessions  {bar}")

    # ── 8. Rows by month (verbose only) ───────────────────────────────────
    if verbose:
        _section("8. Rows by Month (verbose)")

        rows_by_month = df.groupby(df.index.to_period("M")).size()
        for period, cnt in rows_by_month.items():
            bar_len = int(cnt / rows_by_month.max() * 30)
            bar     = "█" * bar_len
            print(f"  {period}  {cnt:6,}  {bar}")

    # ── 9. Price sanity ────────────────────────────────────────────────────
    if "close" in df.columns:
        _section("9. Price Sanity")
        c = df["close"].dropna()
        print(f"  close  min: {c.min():.4f}")
        print(f"  close  max: {c.max():.4f}")
        print(f"  close  mean: {c.mean():.4f}")

        # Bar-level return (absolute)
        rets    = c.pct_change().abs().dropna()
        big_ret = rets[rets > 0.005]    # > 0.5% single-bar move
        print(f"  Single-bar moves > 0.5%: {len(big_ret)}")
        if verbose and not big_ret.empty:
            for ts, r in big_ret.nlargest(10).items():
                print(f"    {ts}  {r:.4%}")

    print()
    print(thick)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    path = Path(args.file_path)
    print(f"\nLoading: {path}")
    df = _load(path)
    print_report(df, interval=args.interval, verbose=args.verbose)


if __name__ == "__main__":
    main()

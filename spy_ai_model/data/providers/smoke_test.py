"""
smoke_test.py
─────────────
Quick verification that a provider can fetch live SPY bars and that the
resulting DataFrame matches training expectations.

Checks performed
────────────────
  1. Provider resolves without error.
  2. fetch() returns a non-empty DataFrame.
  3. Columns match: open / high / low / close / volume.
  4. Index is a sorted tz-naive DatetimeIndex.
  5. No NaN in OHLCV columns.
  6. All bars fall inside regular session (09:30 – 16:00 ET).
  7. Row count / min timestamp / max timestamp printed.
  8. Last 5 rows printed.
  9. Data freshness check: latest bar is reported as today or labelled stale.
     (Skipped for file provider – staleness is expected for historical files.)
 10. Feature pipeline can run on the fetched data without error.

Usage
─────
  # Auto provider (Polygon when POLYGON_API_KEY set, else yfinance):
  python data/providers/smoke_test.py

  # Explicit network provider:
  python data/providers/smoke_test.py --provider polygon
  python data/providers/smoke_test.py --provider yfinance

  # Local file provider:
  python data/providers/smoke_test.py --provider file --file-path /path/to/spy_5m.csv

  # Different symbol / interval / lookback:
  python data/providers/smoke_test.py --symbol SPY --interval 5m --days 5

  # Also run feature pipeline on the fetched data:
  python data/providers/smoke_test.py --run-features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import pandas as pd


# ── checks ────────────────────────────────────────────────────────────────────

REQUIRED_COLS   = ["open", "high", "low", "close", "volume"]
MARKET_OPEN_T   = pd.Timestamp("1970-01-01 09:30").time()
MARKET_CLOSE_T  = pd.Timestamp("1970-01-01 16:00").time()

PASS = "  [PASS]"
FAIL = "  [FAIL]"
INFO = "  [INFO]"
WARN = "  [WARN]"


def _check(condition: bool, msg: str) -> bool:
    tag = PASS if condition else FAIL
    print(f"{tag}  {msg}")
    return condition


def run_smoke_test(
    provider_name:  str  = "auto",
    symbol:         str  = "SPY",
    interval:       str  = "5m",
    lookback_days:  int  = 5,
    run_features:   bool = False,
    file_path:      str  = "",
) -> bool:
    """
    Run the full smoke test suite.

    Returns True if all checks pass, False if any fail.
    """
    sep  = "═" * 62
    dash = "─" * 62

    is_file_provider = (provider_name == "file")

    print(sep)
    print(f"  SPY AI  –  Provider Smoke Test")
    print(sep)
    print(f"  Provider     : {provider_name}")
    if is_file_provider:
        print(f"  File path    : {file_path or '(not set)'}")
    print(f"  Symbol       : {symbol}")
    print(f"  Interval     : {interval}")
    if not is_file_provider:
        print(f"  Lookback     : {lookback_days} days")
    print(dash)

    all_pass = True

    # ── 1. Resolve provider ────────────────────────────────────────────────
    print("[ 1 ] Provider resolution")
    try:
        from data.providers import get_provider
        kwargs = {}
        if is_file_provider:
            if not file_path:
                print(f"{FAIL}  --file-path is required when --provider=file")
                return False
            kwargs["file_path"] = file_path
        provider = get_provider(provider_name, **kwargs)
        print(f"{PASS}  Resolved provider: {provider.provider_name}")
    except Exception as exc:
        print(f"{FAIL}  Could not resolve provider: {exc}")
        return False

    # ── 2. Fetch bars ──────────────────────────────────────────────────────
    if is_file_provider:
        print(f"[ 2 ] Loading {symbol} {interval} bars from file …")
        try:
            df = provider.get_stock_bars(symbol, interval)
        except Exception as exc:
            print(f"{FAIL}  File load raised: {exc}")
            return False
    else:
        print(f"[ 2 ] Fetching {symbol} {interval} bars ({lookback_days} days) …")
        try:
            df = provider.get_latest_stock_bars(symbol, interval, lookback_days)
        except Exception as exc:
            print(f"{FAIL}  Bar fetch raised: {exc}")
            return False

    ok = _check(not df.empty, f"DataFrame is not empty ({len(df)} rows)")
    all_pass &= ok
    if not ok:
        print(f"{FAIL}  No bars returned – cannot continue checks.")
        return False

    # ── 3. Column check ────────────────────────────────────────────────────
    print("[ 3 ] Column schema")
    for col in REQUIRED_COLS:
        ok = _check(col in df.columns, f"Column '{col}' present")
        all_pass &= ok

    # ── 4. Index type ──────────────────────────────────────────────────────
    print("[ 4 ] Index type")
    is_dti = isinstance(df.index, pd.DatetimeIndex)
    ok = _check(is_dti, "Index is DatetimeIndex")
    all_pass &= ok
    if is_dti:
        ok = _check(df.index.tz is None, "Index is tz-naive (ET, no tz info)")
        all_pass &= ok

    # ── 5. Sort order ──────────────────────────────────────────────────────
    print("[ 5 ] Sort order")
    ok = _check(df.index.is_monotonic_increasing, "Index sorted ascending")
    all_pass &= ok

    # ── 6. NaN check ──────────────────────────────────────────────────────
    print("[ 6 ] NaN check")
    nan_rows = df[REQUIRED_COLS].isnull().any(axis=1).sum()
    ok = _check(nan_rows == 0, f"No NaN in OHLCV columns ({nan_rows} NaN rows)")
    all_pass &= ok

    # ── 7. Market-hours filter ─────────────────────────────────────────────
    print("[ 7 ] Market-hours filter")
    times   = df.index.time
    in_sess = ((times >= MARKET_OPEN_T) & (times <= MARKET_CLOSE_T)).sum()
    ok = _check(in_sess == len(df),
                f"All {len(df)} bars inside 09:30–16:00 ET ({len(df) - in_sess} outside)")
    all_pass &= ok

    # ── 8. Statistics ──────────────────────────────────────────────────────
    print("[ 8 ] Statistics")
    print(f"{INFO}  Row count        : {len(df)}")
    print(f"{INFO}  Min timestamp    : {df.index.min()}")
    print(f"{INFO}  Max timestamp    : {df.index.max()}")
    print(f"{INFO}  Trading days     : {df.index.normalize().nunique()}")
    print(f"{INFO}  Close range      : ${df['close'].min():.4f} – ${df['close'].max():.4f}")
    print(f"{INFO}  Avg volume/bar   : {df['volume'].mean():,.0f}")

    print(f"\n  Last 5 rows:")
    print(df.tail(5).to_string())
    print()

    # ── 9. Freshness ───────────────────────────────────────────────────────
    print("[ 9 ] Data freshness")
    if is_file_provider:
        print(f"{INFO}  Freshness check skipped for file provider "
              "(historical files are intentionally stale).")
        latest_ts = df.index.max()
        print(f"{INFO}  File latest bar  : {latest_ts}")
    else:
        try:
            health = provider.print_health(symbol, interval, df)
            all_pass &= not health["stale"]
        except Exception as exc:
            print(f"{WARN}  health_check raised: {exc}")

    # ── 10. Feature pipeline ───────────────────────────────────────────────
    if run_features:
        print("[10 ] Feature pipeline")
        try:
            from features.feature_engineering import build_features
            df_feat = build_features(df)
            ok = _check(not df_feat.empty, f"build_features() returned {len(df_feat)} rows")
            all_pass &= ok
            print(f"{INFO}  Feature columns  : {len(df_feat.columns)}")
            nan_feat = df_feat.isnull().any(axis=1).sum()
            print(f"{INFO}  Rows with NaN    : {nan_feat}")
            print(f"{INFO}  First features   : {df_feat.columns[:5].tolist()}")
        except Exception as exc:
            print(f"{FAIL}  build_features() raised: {exc}")
            all_pass = False

    # ── Final verdict ──────────────────────────────────────────────────────
    print(dash)
    verdict = "ALL CHECKS PASSED" if all_pass else "ONE OR MORE CHECKS FAILED"
    marker  = "✓" if all_pass else "✗"
    print(f"  {marker}  {verdict}")
    print(sep)

    return all_pass


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke-test a market data provider",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--provider",      default="auto",
                   choices=["auto", "polygon", "yfinance", "file"],
                   help=(
                       "Provider to test. "
                       "Use 'file' with --file-path to test a local CSV/Parquet file."
                   ))
    p.add_argument("--file-path",     default="",
                   help="Path to local CSV or Parquet file (required when --provider=file)")
    p.add_argument("--symbol",        default="SPY",
                   help="Ticker symbol to fetch")
    p.add_argument("--interval",      default="5m",
                   help="Bar interval: 1m, 5m, 15m")
    p.add_argument("--days",          default=5, type=int,
                   help="Calendar days to look back (ignored for file provider)")
    p.add_argument("--run-features",  action="store_true", default=False,
                   help="Also run build_features() on the fetched data")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    passed = run_smoke_test(
        provider_name=args.provider,
        symbol=args.symbol,
        interval=args.interval,
        lookback_days=args.days,
        run_features=args.run_features,
        file_path=args.file_path,
    )
    sys.exit(0 if passed else 1)

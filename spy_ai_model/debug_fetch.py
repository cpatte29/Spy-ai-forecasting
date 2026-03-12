"""
debug_fetch.py
──────────────
Diagnostic script: traces exactly where today's bars get lost (or not)
in the live fetch pipeline.

Prints at every stage:
  1. Raw yfinance response       – row count, min/max ts, last 10 timestamps
  2. After timezone → ET strip   – same
  3. After market-hours filter   – same + today-only bar count
  4. Closed-bar selection        – today's ET date, current ET time,
                                   last closed bar chosen, reason for
                                   any fallback to yesterday

Usage
─────
  python debug_fetch.py
  python debug_fetch.py --interval 5m --lookback-days 7
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import TICKER, MARKET_OPEN, MARKET_CLOSE

SEP  = "─" * 62
DSEP = "═" * 62

_BAR_MINS = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60}


def _et_now() -> pd.Timestamp:
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        tz = pytz.timezone("America/New_York")
    return pd.Timestamp(datetime.datetime.now(tz).replace(tzinfo=None))


def _show_df(label: str, df: pd.DataFrame, today_date=None) -> None:
    print(f"\n  [{label}]")
    print(f"    row count   : {len(df)}")
    if df.empty:
        print("    *** EMPTY ***")
        return
    print(f"    min ts      : {df.index[0]}")
    print(f"    max ts      : {df.index[-1]}")
    print(f"    last 10 ts  :")
    for ts in df.index[-10:]:
        print(f"                  {ts}")
    if today_date is not None:
        n_today = int((df.index.date == today_date).sum())
        print(f"    today bars  : {n_today}  (date={today_date})")


def main(interval: str = "5m", lookback_days: int = 7) -> None:
    print(DSEP)
    print(f"  SPY AI  –  Live Fetch Diagnostic")
    print(f"  ticker={TICKER}  interval={interval}  lookback={lookback_days}d")
    print(DSEP)

    # ── ET clock ──────────────────────────────────────────────────────────────
    now_et      = _et_now()
    today_date  = now_et.date()
    print(f"\n  Current ET time : {now_et.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Today ET date   : {today_date}")

    bar_mins = _BAR_MINS.get(interval, 5)
    bar_dur  = pd.Timedelta(minutes=bar_mins)

    # ── 1. Compute date range ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  STEP 1 – Date range calculation")
    print(SEP)

    end_dt_old   = pd.Timestamp.today().normalize()
    end_dt_fixed = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start_dt     = end_dt_fixed - pd.Timedelta(days=lookback_days)

    print(f"  end_dt (OLD – broken)  : {end_dt_old}  → yfinance fetches up to but NOT including this date")
    print(f"  end_dt (FIXED)         : {end_dt_fixed}  → yfinance now includes today")
    print(f"  start_dt               : {start_dt}")
    print(f"  yfinance call          : download('{TICKER}', start='{start_dt.date()}', end='{end_dt_fixed.date()}', interval='{interval}')")

    # ── 2. Raw yfinance fetch ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  STEP 2 – Raw yfinance response (with FIXED end_dt)")
    print(SEP)

    raw = yf.download(
        TICKER,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt_fixed.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    _show_df("raw yfinance", raw, today_date)

    if raw.empty:
        print("\n  *** yfinance returned EMPTY DataFrame – check network / ticker ***")
        return

    # ── 3. Timezone strip → ET ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  STEP 3 – After timezone conversion to tz-naive ET")
    print(SEP)
    print(f"    raw index tz before strip : {raw.index.tz}")

    df_et = raw.copy()
    if df_et.index.tz is not None:
        df_et.index = df_et.index.tz_convert("America/New_York").tz_localize(None)
    _show_df("tz-naive ET", df_et, today_date)

    # ── 4. Market-hours filter ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  STEP 4 – After market-hours filter [{MARKET_OPEN} – {MARKET_CLOSE}]")
    print(SEP)

    open_t  = pd.Timestamp(f"1970-01-01 {MARKET_OPEN}").time()
    close_t = pd.Timestamp(f"1970-01-01 {MARKET_CLOSE}").time()
    times   = df_et.index.time
    mask    = (times >= open_t) & (times <= close_t)
    df_mh   = df_et.loc[mask].copy()
    _show_df("market-hours", df_mh, today_date)

    # ── 5. Closed-bar selection ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  STEP 5 – Closed-bar selection")
    print(SEP)
    print(f"    now_et          : {now_et}")
    print(f"    bar duration    : {bar_mins} min")

    if df_mh.empty:
        print("    *** No bars after filtering – cannot select scored bar ***")
        return

    last_ts  = df_mh.index[-1]
    elapsed  = now_et - last_ts
    elapsed_min = elapsed.total_seconds() / 60

    print(f"    last bar ts     : {last_ts}")
    print(f"    elapsed since   : {elapsed_min:.1f} min")
    print(f"    bar_dur         : {bar_mins} min")

    if elapsed < bar_dur:
        reason = (
            f"last bar ({last_ts}) is still forming "
            f"({elapsed_min:.1f}/{bar_mins} min elapsed) → falling back to previous bar"
        )
        if len(df_mh) < 2:
            print(f"    FALLBACK REASON : {reason}")
            print(f"    *** ERROR: no previous bar to fall back to ***")
        else:
            prev_ts = df_mh.index[-2]
            print(f"    FALLBACK REASON : {reason}")
            print(f"    scored bar      : {prev_ts}  ← FALLBACK")
            scored_ts = prev_ts
    else:
        print(f"    bar is closed   : elapsed ({elapsed_min:.1f} min) >= bar_dur ({bar_mins} min)")
        print(f"    scored bar      : {last_ts}  ← SELECTED")
        scored_ts = last_ts

    scored_date = scored_ts.date()
    if scored_date < today_date:
        print(f"\n  *** STALE DATA WARNING ***")
        print(f"      scored bar date ({scored_date}) < today ({today_date})")
        print(f"      today's bars in df_mh: {int((df_mh.index.date == today_date).sum())}")
        print(f"      This means today's bars are missing from the fetch.")
    else:
        print(f"\n  ✓ scored bar is from today ({scored_date})")

    # ── 6. Session bar count ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  STEP 6 – Session bar count for scored bar's date")
    print(SEP)
    session_bars = int((df_mh.index.date == scored_ts.date()).sum())
    print(f"    session date    : {scored_ts.date()}")
    print(f"    session bars    : {session_bars}")
    print(f"    min_session_bars: 12  → {'OK' if session_bars >= 12 else 'BELOW THRESHOLD – will use yesterday fallback'}")

    print(f"\n{DSEP}")
    print(f"  DIAGNOSIS COMPLETE")
    print(f"  Root cause of stale data: end_dt = today().normalize() is EXCLUSIVE in yfinance,")
    print(f"  so today's bars are never requested. Fixed by using today+1 as end_dt.")
    print(DSEP)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Debug SPY live fetch pipeline")
    p.add_argument("--interval",      default="5m")
    p.add_argument("--lookback-days", default=7, type=int)
    args = p.parse_args()
    main(interval=args.interval, lookback_days=args.lookback_days)

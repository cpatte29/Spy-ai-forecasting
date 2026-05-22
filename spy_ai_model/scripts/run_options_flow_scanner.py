"""
run_options_flow_scanner.py
───────────────────────────
Options unusual-flow scanner CLI.  Three operating modes:

  snapshot   One-time scan of the universe (or a single ticker).
  live       Continuous loop, rescanning every N seconds.
  single     Deep scan of a single ticker with full breakdown.

Usage examples
──────────────
  # Scan the default universe (SPY QQQ AAPL NVDA TSLA META AMD MSFT AMZN):
  python3 scripts/run_options_flow_scanner.py snapshot

  # Scan a single ticker:
  python3 scripts/run_options_flow_scanner.py single --ticker NVDA

  # Scan a custom list of tickers:
  python3 scripts/run_options_flow_scanner.py snapshot --tickers NVDA AMD TSLA

  # Live mode — rescan every 5 minutes:
  python3 scripts/run_options_flow_scanner.py live --interval-sec 300

  # Only show tickers with grade B or better:
  python3 scripts/run_options_flow_scanner.py snapshot --min-grade B

  # Full score breakdown:
  python3 scripts/run_options_flow_scanner.py single --ticker AAPL --breakdown

  # Suppress ANSI colours (for logging to file):
  python3 scripts/run_options_flow_scanner.py snapshot --no-colour

Requirements
────────────
  export POLYGON_API_KEY=<your_key>
  (Options data requires a Polygon plan with options access.)

This script is read-only — no trades are placed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT        = _SCRIPTS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── imports ───────────────────────────────────────────────────────────────────
from options_flow_scanner.ticker_universe import load_universe
from options_flow_scanner.option_chain_loader import load_chain
from options_flow_scanner.flow_features import compute_flow_features
from options_flow_scanner.unusual_flow_detector import detect_unusual_flow
from options_flow_scanner.flow_scorer import (
    score_flow,
    FlowGrade,
    FlowAlertState,
    FLOW_ALERT_MIN_GRADE,
)
from options_flow_scanner.flow_report import (
    format_flow_report,
    format_flow_summary,
    log_flow_alert_csv,
    emit_flow_alert,
    LOG_FILE,
)

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SEC = 300   # 5 minutes
_DEFAULT_MAX_DTE      = 90
_DEFAULT_MIN_OI       = 10
_DEFAULT_MIN_VOL      = 1

_GRADE_ORDER = {
    "A+": FlowGrade.A_PLUS,
    "A":  FlowGrade.A,
    "B":  FlowGrade.B,
    "C":  FlowGrade.C,
    "IGNORE": FlowGrade.IGNORE,
}


# ── core scan for one ticker ──────────────────────────────────────────────────

def scan_ticker(
    ticker:     str,
    max_dte:    int  = _DEFAULT_MAX_DTE,
    min_oi:     int  = _DEFAULT_MIN_OI,
    min_vol:    int  = _DEFAULT_MIN_VOL,
    colour:     bool = True,
    show_breakdown: bool = False,
    log_file:   Optional[Path] = None,
    log_alerts: bool = True,
    min_grade:  FlowGrade = FLOW_ALERT_MIN_GRADE,
    emit_alert: bool = True,
) -> Optional[object]:
    """
    Run a full options flow scan for a single ticker.

    Returns a FlowResult, or None if data could not be loaded.
    """
    try:
        chain = load_chain(
            ticker  = ticker,
            max_dte = max_dte,
            min_oi  = min_oi,
            min_vol = min_vol,
        )
    except PermissionError as exc:
        print(f"  [ERROR] {exc}", file=sys.stderr)
        return None
    except Exception as exc:
        logger.error("Chain load failed for %s: %s", ticker, exc)
        return None

    if chain.empty:
        logger.info("No chain data for %s — skipping.", ticker)
        return None

    spot = float(chain.iloc[0].get("spot", 0)) if "spot" in chain.columns else 0.0

    features   = compute_flow_features(chain, stock_price=spot)
    detections = detect_unusual_flow(features)
    result     = score_flow(ticker, features, detections, stock_price=spot)

    # Print report
    report = format_flow_report(result, colour=colour, show_breakdown=show_breakdown)
    print(report)

    # Log to CSV
    grade_rank = {
        FlowGrade.IGNORE: 0, FlowGrade.C: 1, FlowGrade.B: 2,
        FlowGrade.A: 3, FlowGrade.A_PLUS: 4,
    }
    min_rank = grade_rank.get(min_grade, 2)

    if log_alerts and grade_rank.get(result.grade, 0) >= min_rank:
        log_flow_alert_csv(result, log_file=log_file)

    return result


# ── scan the full universe ────────────────────────────────────────────────────

def scan_universe(
    tickers:    list[str],
    max_dte:    int  = _DEFAULT_MAX_DTE,
    min_oi:     int  = _DEFAULT_MIN_OI,
    min_vol:    int  = _DEFAULT_MIN_VOL,
    colour:     bool = True,
    log_file:   Optional[Path] = None,
    log_alerts: bool = True,
    min_grade:  FlowGrade = FLOW_ALERT_MIN_GRADE,
    throttle_sec: float = 1.0,
) -> list:
    """
    Scan all tickers in the universe and return a list of FlowResults.
    Throttles requests to avoid Polygon rate limits.
    """
    results = []
    total   = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{total}] Scanning {ticker}…")
        try:
            result = scan_ticker(
                ticker      = ticker,
                max_dte     = max_dte,
                min_oi      = min_oi,
                min_vol     = min_vol,
                colour      = colour,
                show_breakdown = False,
                log_file    = log_file,
                log_alerts  = log_alerts,
                min_grade   = min_grade,
                emit_alert  = False,
            )
            if result is not None:
                results.append(result)
        except Exception as exc:
            logger.error("Error scanning %s: %s", ticker, exc)

        if i < total:
            time.sleep(throttle_sec)

    return results


# ── mode: snapshot ────────────────────────────────────────────────────────────

def cmd_snapshot(args: argparse.Namespace) -> None:
    """One-time scan of the universe and print a summary table."""
    tickers = load_universe(
        tickers  = args.tickers if args.tickers else None,
        file     = args.universe_file,
        extended = args.extended,
    )
    print(f"Scanning {len(tickers)} ticker(s): {', '.join(tickers)}\n")

    min_grade = _GRADE_ORDER.get(args.min_grade, FlowGrade.B)
    log_file  = Path(args.log_file) if args.log_file else None

    results = scan_universe(
        tickers     = tickers,
        max_dte     = args.max_dte,
        min_oi      = args.min_oi,
        min_vol     = args.min_vol,
        colour      = not args.no_colour,
        log_file    = log_file,
        log_alerts  = not args.no_log,
        min_grade   = min_grade,
    )

    if results:
        summary = format_flow_summary(
            results,
            min_grade = min_grade,
            colour    = not args.no_colour,
        )
        print(summary)

        logged = [r for r in results if r.alert_state != FlowAlertState.SILENT]
        if logged and not args.no_log:
            lf = log_file or LOG_FILE
            print(f"\n  [{len(logged)} alert(s) logged → {lf}]")


# ── mode: single ticker ───────────────────────────────────────────────────────

def cmd_single(args: argparse.Namespace) -> None:
    """Deep scan of a single ticker with full breakdown."""
    ticker    = args.ticker.upper()
    log_file  = Path(args.log_file) if args.log_file else None
    min_grade = _GRADE_ORDER.get(args.min_grade, FlowGrade.B)

    print(f"Scanning {ticker} (max_dte={args.max_dte}, min_oi={args.min_oi})…\n")

    scan_ticker(
        ticker         = ticker,
        max_dte        = args.max_dte,
        min_oi         = args.min_oi,
        min_vol        = args.min_vol,
        colour         = not args.no_colour,
        show_breakdown = args.breakdown,
        log_file       = log_file,
        log_alerts     = not args.no_log,
        min_grade      = min_grade,
    )

    if not args.no_log:
        lf = log_file or LOG_FILE
        print(f"\n  [alerts logged → {lf}]")


# ── mode: live ────────────────────────────────────────────────────────────────

def cmd_live(args: argparse.Namespace) -> None:
    """Continuous live loop — rescan the universe every N seconds."""
    tickers = load_universe(
        tickers  = args.tickers if args.tickers else None,
        file     = args.universe_file,
        extended = args.extended,
    )
    interval  = args.interval_sec
    min_grade = _GRADE_ORDER.get(args.min_grade, FlowGrade.B)
    log_file  = Path(args.log_file) if args.log_file else None

    print(
        f"Options Flow Scanner — LIVE mode\n"
        f"Universe: {', '.join(tickers)}\n"
        f"Rescan every {interval}s  |  min grade: {args.min_grade}\n"
        f"Press Ctrl+C to stop.\n"
    )

    try:
        from zoneinfo import ZoneInfo
        tz_et = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        tz_et = pytz.timezone("America/New_York")

    while True:
        try:
            now_et = datetime.now(tz_et).replace(tzinfo=None)
            # Only scan during extended market hours (07:00–20:00 ET)
            if not (7 <= now_et.hour < 20):
                print(f"  [{now_et.strftime('%H:%M')} ET] Outside scan window. Sleeping {interval}s…")
                time.sleep(interval)
                continue

            print(f"\n{'='*56}")
            print(f"  Scan at {now_et.strftime('%Y-%m-%d %H:%M:%S')} ET")
            print(f"{'='*56}\n")

            results = scan_universe(
                tickers     = tickers,
                max_dte     = args.max_dte,
                min_oi      = args.min_oi,
                min_vol     = args.min_vol,
                colour      = not args.no_colour,
                log_file    = log_file,
                log_alerts  = not args.no_log,
                min_grade   = min_grade,
            )

            if results:
                summary = format_flow_summary(
                    results,
                    min_grade = min_grade,
                    colour    = not args.no_colour,
                )
                print(summary)

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as exc:
            logger.error("Live scan error: %s", exc, exc_info=True)

        print(f"\n  [Next scan in {interval}s…]")
        time.sleep(interval)


# ── argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_options_flow_scanner",
        description="Options unusual-flow scanner — detect and alert on unusual options activity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    sub = p.add_subparsers(dest="mode", required=False)

    # ── shared options ────────────────────────────────────────────────────
    def add_shared(sp):
        sp.add_argument("--tickers",     nargs="+", default=None,
                        help="Explicit ticker list (overrides universe file and default)")
        sp.add_argument("--universe-file", default=None,
                        help="Path to a text file with one ticker per line")
        sp.add_argument("--extended",    action="store_true",
                        help="Use the extended 23-ticker universe")
        sp.add_argument("--max-dte",     default=_DEFAULT_MAX_DTE, type=int,
                        help="Maximum days to expiry to include")
        sp.add_argument("--min-oi",      default=_DEFAULT_MIN_OI, type=int,
                        help="Minimum open interest per contract")
        sp.add_argument("--min-vol",     default=_DEFAULT_MIN_VOL, type=int,
                        help="Minimum daily volume per contract")
        sp.add_argument("--min-grade",   default="B", choices=["A+", "A", "B", "C", "IGNORE"],
                        help="Minimum grade to show and log")
        sp.add_argument("--no-colour",   action="store_true",
                        help="Disable ANSI colour output")
        sp.add_argument("--no-log",      action="store_true",
                        help="Do not write alerts to CSV log")
        sp.add_argument("--log-file",    default=None,
                        help="Override alert log CSV path")

    # ── snapshot ──────────────────────────────────────────────────────────
    snap = sub.add_parser("snapshot", help="One-time universe scan",
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_shared(snap)

    # ── single ticker ─────────────────────────────────────────────────────
    single = sub.add_parser("single", help="Deep scan of one ticker",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    single.add_argument("--ticker", required=True, help="Ticker to scan")
    single.add_argument("--breakdown", action="store_true",
                        help="Show full score breakdown and conditions")
    add_shared(single)

    # ── live ──────────────────────────────────────────────────────────────
    live = sub.add_parser("live", help="Continuous live scanning loop",
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    live.add_argument("--interval-sec", default=_DEFAULT_INTERVAL_SEC, type=int,
                      help="Seconds between full universe scans")
    add_shared(live)

    return p


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level  = args.log_level,
        format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt= "%Y-%m-%d %H:%M:%S",
    )

    mode = args.mode or "snapshot"

    if mode == "snapshot":
        # Handle case where no subcommand was given (default to snapshot)
        if not hasattr(args, "tickers"):
            args.tickers      = None
            args.universe_file = None
            args.extended     = False
            args.max_dte      = _DEFAULT_MAX_DTE
            args.min_oi       = _DEFAULT_MIN_OI
            args.min_vol      = _DEFAULT_MIN_VOL
            args.min_grade    = "B"
            args.no_colour    = False
            args.no_log       = False
            args.log_file     = None
        cmd_snapshot(args)

    elif mode == "single":
        cmd_single(args)

    elif mode == "live":
        cmd_live(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

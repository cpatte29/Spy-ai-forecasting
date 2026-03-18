"""
run_setup_scanner.py
────────────────────
SPY Setup Scanner CLI — three operating modes:

  snapshot   One-time scan of the latest bar (default).
  live       Continuous loop, rescanning on each new bar.
  replay     Walk over a historical bar dataset, evaluate each bar,
             and produce a performance summary by setup type.

Usage examples
──────────────
  # Single snapshot (live Polygon data):
  python3 scripts/run_setup_scanner.py snapshot --provider polygon

  # Live mode (rescans every 5 minutes during market hours):
  python3 scripts/run_setup_scanner.py live --provider polygon --interval 300

  # Historical replay evaluation:
  python3 scripts/run_setup_scanner.py replay --lookback-days 60 --provider polygon

  # Replay from a local file:
  python3 scripts/run_setup_scanner.py replay --file path/to/bars.parquet

  # Snapshot with no-colour output (for piping to a file):
  python3 scripts/run_setup_scanner.py snapshot --no-colour

  # Show full score breakdown:
  python3 scripts/run_setup_scanner.py snapshot --breakdown --raw

This script is read-only — it does NOT place trades.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
# Allow running from the scripts/ subdirectory or from spy_ai_model/
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT        = _SCRIPTS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── imports ───────────────────────────────────────────────────────────────────
from config import TICKER, HORIZON_DIR
from data.data_loader import load_bars
from features.feature_engineering import build_features
from models.train_direction import load_direction_model, predict_direction_proba
from models.train_range import load_range_model, predict_range
from confluence_engine.zone_context import get_zone_context
from confluence_engine.confluence_scorer import compute_confluence

from setup_scanner.setup_detector import build_snapshot, detect_setups
from setup_scanner.setup_scorer import rank_all_setups
from setup_scanner.setup_report import format_setup_report, format_replay_summary
from setup_scanner.setup_alerts import process_alerts, LOG_FILE
from setup_scanner.setup_definitions import SetupGrade, SetupType

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
_DEFAULT_LOOKBACK_DAYS = 7
_DEFAULT_INTERVAL      = "5m"
_DEFAULT_LIVE_SLEEP    = 300   # 5 minutes
_BAR_MINS: dict[str, int] = {
    "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60,
}

# ── model loading helpers ─────────────────────────────────────────────────────

def _load_models():
    try:
        dir_model   = load_direction_model()
        range_model = load_range_model()
        return dir_model, range_model
    except Exception as exc:
        logger.error("Failed to load models: %s", exc)
        raise


# ── core scan function ────────────────────────────────────────────────────────

def run_scan(
    df_bars:      pd.DataFrame,
    dir_model,
    range_model,
    interval:     str   = _DEFAULT_INTERVAL,
    show_breakdown: bool = False,
    show_raw:     bool  = False,
    colour:       bool  = True,
    enable_analog: bool = False,
    enable_alerts: bool = True,
    log_file:     Path | None = None,
    scored_ts:    pd.Timestamp | None = None,
) -> dict:
    """
    Run a full setup scan on the provided bar DataFrame.

    Parameters
    ──────────
    df_bars       OHLCV DataFrame up to (and including) the bar to score.
    dir_model     Loaded LightGBM direction model.
    range_model   Loaded LightGBM range model.
    interval      Bar interval (e.g. "5m").
    show_breakdown Print score breakdown for top setup.
    show_raw      Print raw signal values.
    colour        Use ANSI colour codes in output.
    enable_analog Try to run analog engine (requires parquet dataset).
    enable_alerts Emit and log alerts for qualifying setups.
    log_file      Override default alert log file path.
    scored_ts     If set, use this bar timestamp instead of last bar.

    Returns
    ───────
    dict with keys:
        ranked_results  list[SetupResult]
        top             SetupResult
        bar_ts          pd.Timestamp
        price           float
        dir_prob        float
        pred_range      float
    """
    if df_bars.empty:
        raise ValueError("df_bars is empty")

    # ── 1. Identify the scored bar ────────────────────────────────────────
    if scored_ts is None:
        scored_ts = df_bars.index[-1]

    df_scored = df_bars.loc[:scored_ts]
    price     = float(df_scored.iloc[-1]["close"])

    # ── 2. Build features ─────────────────────────────────────────────────
    df_feat = build_features(df_bars)

    # ── 3. Verify feature alignment silently ──────────────────────────────
    training_cols = list(dir_model.feature_name_)
    missing_feat  = [c for c in training_cols if c not in df_feat.columns]
    if missing_feat:
        raise ValueError(f"Feature mismatch: {missing_feat}")

    # ── 4. Extract feature row ────────────────────────────────────────────
    try:
        pos     = df_feat.index.get_loc(scored_ts)
        X       = df_feat.iloc[[pos]][training_cols]
    except KeyError:
        raise ValueError(f"Scored timestamp {scored_ts} not in feature DataFrame")

    if X.isnull().any(axis=1).item():
        nan_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"NaN features at {scored_ts}: {nan_cols}")

    # ── 5. Model predictions ──────────────────────────────────────────────
    dir_prob   = float(predict_direction_proba(dir_model,   X)[0])
    pred_range = float(predict_range(range_model, X)[0])

    # ── 6. Zone context ───────────────────────────────────────────────────
    zone_ctx = {}
    try:
        zone_ctx = get_zone_context(df_scored, current_price=price)
    except Exception as exc:
        logger.warning("Zone context failed: %s", exc)

    # ── 7. Analog engine (optional) ───────────────────────────────────────
    analog = {}
    if enable_analog:
        analog = _try_analog(df_scored, zone_ctx, price)

    # ── 8. Confluence ─────────────────────────────────────────────────────
    confluence = {}
    try:
        confluence = compute_confluence(
            dir_prob   = dir_prob,
            pred_range = pred_range,
            zone_ctx   = zone_ctx,
            analog     = analog,
        )
    except Exception as exc:
        logger.warning("Confluence scorer failed: %s", exc)

    # ── 9. Feature dict for structure signals ─────────────────────────────
    feat_dict = df_feat.iloc[pos].to_dict() if pos < len(df_feat) else {}

    # ── 10. Build snapshot ────────────────────────────────────────────────
    snapshot = build_snapshot(
        df_bars       = df_scored,
        dir_prob      = dir_prob,
        pred_range    = pred_range,
        zone_ctx      = zone_ctx,
        analog        = analog,
        confluence    = confluence,
        features      = feat_dict,
    )

    # ── 11. Detect setups ─────────────────────────────────────────────────
    detections = detect_setups(snapshot)

    # ── 12. Score and rank ────────────────────────────────────────────────
    ranked_results = rank_all_setups(snapshot, detections)

    # ── 13. Print report ──────────────────────────────────────────────────
    report = format_setup_report(
        ranked_results,
        bar_ts         = scored_ts,
        price          = price,
        interval       = interval,
        colour         = colour,
        show_breakdown = show_breakdown,
        show_raw       = show_raw,
    )
    print(report)

    # ── 14. Alerts ────────────────────────────────────────────────────────
    if enable_alerts:
        emitted = process_alerts(
            ranked_results,
            bar_ts   = scored_ts,
            price    = price,
            ticker   = TICKER,
            interval = interval,
            colour   = colour,
            log_file = log_file,
        )
        if emitted:
            print(f"  [{len(emitted)} alert(s) logged → {log_file or LOG_FILE}]")

    top = ranked_results[0] if ranked_results else None

    return {
        "ranked_results": ranked_results,
        "top":            top,
        "bar_ts":         scored_ts,
        "price":          price,
        "dir_prob":       dir_prob,
        "pred_range":     pred_range,
        "zone_ctx":       zone_ctx,
        "analog":         analog,
        "confluence":     confluence,
    }


def _try_analog(df: pd.DataFrame, zone_ctx: dict, price: float) -> dict:
    """
    Attempt to run the analog engine. Returns {} on any failure.
    The analog engine requires a pre-built parquet dataset.
    """
    try:
        from analog_engine.analog_report import run_analog_analysis
        from analog_engine.event_extractor import extract_zone_test_events

        # Require at least 30 bars for a meaningful window
        if len(df) < 30:
            return {}

        # Need a supply zone to define the query event
        supply = zone_ctx.get("supply", {})
        if not supply.get("zone_high"):
            return {}

        query_meta = {
            "zone_high":          supply.get("zone_high"),
            "zone_low":           supply.get("zone_low"),
            "displacement_size":  supply.get("displacement_size", 0.005),
            "volume_spike_ratio": supply.get("volume_spike_ratio", 1.0),
            "bars_since_created": supply.get("bars_since_created", 10),
        }

        # Use the last 25 bars as the pattern window
        window_df = df.iloc[-25:].copy()
        result    = run_analog_analysis(
            window_df,
            query_meta,
            top_n  = 20,
            method = "cosine",
        )
        return result or {}
    except Exception as exc:
        logger.debug("Analog engine skipped: %s", exc)
        return {}


# ── closed-bar guard ──────────────────────────────────────────────────────────

def _get_closed_bar_ts(df: pd.DataFrame, interval: str) -> pd.Timestamp:
    """Return the timestamp of the last fully-closed bar."""
    bar_mins = _BAR_MINS.get(interval, 5)
    bar_dur  = pd.Timedelta(minutes=bar_mins)

    try:
        from zoneinfo import ZoneInfo
        tz_et = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        tz_et = pytz.timezone("America/New_York")

    now_et  = pd.Timestamp(datetime.now(tz_et).replace(tzinfo=None))
    last_ts = df.index[-1]

    if now_et - last_ts < bar_dur:
        if len(df) >= 2:
            logger.info("Latest bar still forming — using previous bar.")
            return df.index[-2]
    return last_ts


# ── snapshot mode ─────────────────────────────────────────────────────────────

def cmd_snapshot(args: argparse.Namespace) -> None:
    """Run a single setup scan on the latest bar."""
    print("Loading models…")
    dir_model, range_model = _load_models()

    print(f"Fetching {TICKER} bars ({args.interval}, {args.lookback_days}d) via {args.provider}…")
    df = load_bars(
        symbol        = TICKER,
        interval      = args.interval,
        lookback_days = args.lookback_days,
        provider      = args.provider,
    )
    if df.empty:
        print(f"ERROR: No bars returned from provider '{args.provider}'.", file=sys.stderr)
        sys.exit(1)

    scored_ts = _get_closed_bar_ts(df, args.interval)

    run_scan(
        df_bars        = df,
        dir_model      = dir_model,
        range_model    = range_model,
        interval       = args.interval,
        show_breakdown = args.breakdown,
        show_raw       = args.raw,
        colour         = not args.no_colour,
        enable_analog  = args.analog,
        enable_alerts  = not args.no_alerts,
        log_file       = Path(args.log_file) if args.log_file else None,
        scored_ts      = scored_ts,
    )


# ── live mode ─────────────────────────────────────────────────────────────────

def cmd_live(args: argparse.Namespace) -> None:
    """
    Continuous live loop — rescan on each new closed bar.
    Runs until interrupted with Ctrl+C.
    """
    print(f"Starting SPY Setup Scanner in LIVE mode (interval={args.interval}, sleep={args.interval_sec}s).")
    print("Press Ctrl+C to stop.\n")

    dir_model, range_model = _load_models()
    last_scanned_ts: Optional[pd.Timestamp] = None

    try:
        from zoneinfo import ZoneInfo
        tz_et = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        tz_et = pytz.timezone("America/New_York")

    while True:
        try:
            now_et = datetime.now(tz_et).replace(tzinfo=None)
            # Only run during extended market window (07:00–20:00 ET)
            if not (7 <= now_et.hour < 20):
                print(f"  [{now_et.strftime('%H:%M')} ET] Outside scan window. Sleeping {args.interval_sec}s…")
                time.sleep(args.interval_sec)
                continue

            df = load_bars(
                symbol        = TICKER,
                interval      = args.interval,
                lookback_days = args.lookback_days,
                provider      = args.provider,
            )
            if df.empty:
                logger.warning("Empty bar response — retrying next cycle.")
                time.sleep(args.interval_sec)
                continue

            scored_ts = _get_closed_bar_ts(df, args.interval)

            if last_scanned_ts is not None and scored_ts <= last_scanned_ts:
                logger.debug("No new bar — last=%s, current=%s", last_scanned_ts, scored_ts)
                time.sleep(args.interval_sec)
                continue

            last_scanned_ts = scored_ts
            print(f"\n[{now_et.strftime('%Y-%m-%d %H:%M:%S')} ET] New bar: {scored_ts}")

            run_scan(
                df_bars        = df,
                dir_model      = dir_model,
                range_model    = range_model,
                interval       = args.interval,
                show_breakdown = args.breakdown,
                show_raw       = False,
                colour         = not args.no_colour,
                enable_analog  = args.analog,
                enable_alerts  = not args.no_alerts,
                log_file       = Path(args.log_file) if args.log_file else None,
                scored_ts      = scored_ts,
            )

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as exc:
            logger.error("Live scan error: %s", exc, exc_info=True)

        time.sleep(args.interval_sec)


# ── replay mode ───────────────────────────────────────────────────────────────

def cmd_replay(args: argparse.Namespace) -> None:
    """
    Walk over a historical bar dataset and evaluate setup detection at each bar.

    For each bar:
      1. Build features and predict direction + range up to that bar.
      2. Detect and score setups.
      3. After HORIZON bars, check realized outcome.
      4. Aggregate statistics per setup type.

    Outputs:
      - Per-bar signal log to replay_log.csv
      - Summary table by setup type printed to terminal
    """
    print("Loading models…")
    dir_model, range_model = _load_models()

    # ── Load historical bars ──────────────────────────────────────────────
    if args.file:
        print(f"Loading bars from file: {args.file}")
        fpath = Path(args.file)
        if fpath.suffix in (".parquet", ".pq"):
            df_all = pd.read_parquet(fpath)
        else:
            df_all = pd.read_csv(fpath, index_col=0, parse_dates=True)
        df_all.index = pd.to_datetime(df_all.index)
    else:
        print(f"Fetching {TICKER} bars ({args.interval}, {args.lookback_days}d) via {args.provider}…")
        df_all = load_bars(
            symbol        = TICKER,
            interval      = args.interval,
            lookback_days = args.lookback_days,
            provider      = args.provider,
        )

    if df_all.empty:
        print("ERROR: No bar data available for replay.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df_all)} bars: {df_all.index[0]} → {df_all.index[-1]}")

    # ── Build full feature set ────────────────────────────────────────────
    print("Building features…")
    df_feat_all = build_features(df_all)
    training_cols = list(dir_model.feature_name_)
    missing_cols  = [c for c in training_cols if c not in df_feat_all.columns]
    if missing_cols:
        print(f"ERROR: Feature mismatch: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    # ── Walk forward ──────────────────────────────────────────────────────
    horizon       = HORIZON_DIR
    bar_mins      = _BAR_MINS.get(args.interval, 5)
    min_history   = max(50, args.min_history)
    total_bars    = len(df_all)
    scan_indices  = range(min_history, total_bars - horizon)

    print(f"Replaying {len(scan_indices)} bars (min_history={min_history}, horizon={horizon})…")

    records = []
    errors  = 0

    for i in scan_indices:
        scored_ts = df_all.index[i]
        df_hist   = df_all.iloc[: i + 1]

        try:
            X = df_feat_all.iloc[[i]][training_cols]
            if X.isnull().any(axis=1).item():
                continue

            dir_prob   = float(predict_direction_proba(dir_model,   X)[0])
            pred_range = float(predict_range(range_model, X)[0])
            price      = float(df_all.iloc[i]["close"])

            # Zone context
            zone_ctx = {}
            try:
                zone_ctx = get_zone_context(df_hist, current_price=price)
            except Exception:
                pass

            # Confluence
            confluence = {}
            try:
                confluence = compute_confluence(
                    dir_prob   = dir_prob,
                    pred_range = pred_range,
                    zone_ctx   = zone_ctx,
                    analog     = {},
                )
            except Exception:
                pass

            feat_dict = df_feat_all.iloc[i].to_dict()
            snapshot  = build_snapshot(
                df_bars    = df_hist,
                dir_prob   = dir_prob,
                pred_range = pred_range,
                zone_ctx   = zone_ctx,
                analog     = {},
                confluence = confluence,
                features   = feat_dict,
            )

            detections     = detect_setups(snapshot)
            ranked_results = rank_all_setups(snapshot, detections)
            top            = ranked_results[0]

            # ── Realized outcome (after horizon bars) ─────────────────────
            future_idx   = i + horizon
            future_close = float(df_all.iloc[future_idx]["close"]) if future_idx < total_bars else None
            realized_dir = None
            realized_move = None
            if future_close is not None:
                realized_dir   = 1 if future_close > price else 0
                realized_move  = (future_close - price) / price
                # High-low range in the window
                window   = df_all.iloc[i + 1 : future_idx + 1]
                hl_range = ((window["high"].max() - window["low"].min()) / price) if len(window) > 0 else 0.0
            else:
                hl_range = None

            # Win: realized direction aligned with setup direction
            win = None
            if realized_dir is not None and top.direction.value != "NEUTRAL":
                if top.direction.value == "LONG":
                    win = realized_dir == 1
                elif top.direction.value == "SHORT":
                    win = realized_dir == 0

            records.append({
                "bar_ts":        scored_ts.isoformat(),
                "price":         price,
                "setup_type":    top.setup_type.value,
                "direction":     top.direction.value,
                "score":         top.score,
                "grade":         top.grade.value,
                "dir_prob":      dir_prob,
                "pred_range":    pred_range,
                "zone_bias":     zone_ctx.get("bias", "NEUTRAL"),
                "realized_dir":  realized_dir,
                "realized_move": realized_move,
                "hl_range":      hl_range,
                "win":           win,
            })

        except Exception as exc:
            errors += 1
            logger.debug("Replay error at %s: %s", scored_ts, exc)

        # Progress
        if (i - min_history) % 100 == 0:
            pct = (i - min_history) / max(len(scan_indices), 1) * 100
            print(f"  Progress: {pct:.0f}% ({i - min_history}/{len(scan_indices)})…", end="\r")

    print(f"\nProcessed {len(records)} bars ({errors} errors).")

    if not records:
        print("No records produced.")
        return

    df_log = pd.DataFrame(records)

    # ── Save per-bar log ──────────────────────────────────────────────────
    log_dir  = _ROOT / "setup_scanner" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "replay_log.csv"
    df_log.to_csv(log_path, index=False)
    print(f"Per-bar replay log saved → {log_path}")

    # ── Build summary ─────────────────────────────────────────────────────
    summary = _build_replay_summary(df_log)
    report  = format_replay_summary(summary, colour=not args.no_colour)
    print(report)

    # Save summary
    summary_path = log_dir / "replay_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved → {summary_path}")


def _build_replay_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-bar replay records into per-setup-type statistics.
    """
    records = []
    for stype in df["setup_type"].unique():
        mask = df["setup_type"] == stype
        sub  = df[mask].copy()

        n = len(sub)
        sub_dir = sub.dropna(subset=["win"])

        win_rate    = sub_dir["win"].mean() if len(sub_dir) > 0 else float("nan")
        avg_move    = sub_dir["realized_move"].mean() if "realized_move" in sub_dir and len(sub_dir) > 0 else float("nan")
        avg_score   = sub["score"].mean()

        grade_counts  = sub["grade"].value_counts()
        a_plus_a_pct  = (grade_counts.get("A+", 0) + grade_counts.get("A", 0)) / max(n, 1)
        ignore_pct    = grade_counts.get("IGNORE", 0) / max(n, 1)

        # False positive: signal fired (not NO_SETUP) but realized win == False
        non_neutral = sub_dir[sub_dir["direction"] != "NEUTRAL"]
        fp_rate     = (1 - non_neutral["win"].mean()) if len(non_neutral) > 0 else float("nan")

        records.append({
            "setup_type":         stype,
            "n_signals":          n,
            "win_rate":           win_rate,
            "avg_move_pct":       avg_move,
            "avg_score":          avg_score,
            "grade_a_plus_a_pct": a_plus_a_pct,
            "grade_ignore_pct":   ignore_pct,
            "false_positive_rate": fp_rate,
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("avg_score", ascending=False).reset_index(drop=True)


# ── argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_setup_scanner",
        description="SPY Setup Scanner — detect and score intraday setups",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level",
    )

    sub = p.add_subparsers(dest="mode", required=False)

    # ── shared options ────────────────────────────────────────────────────
    def add_shared(sp):
        sp.add_argument("--interval",      default=_DEFAULT_INTERVAL,
                        help="Bar interval (must match trained model)")
        sp.add_argument("--lookback-days", default=_DEFAULT_LOOKBACK_DAYS, type=int,
                        help="Calendar days of history to fetch")
        sp.add_argument("--provider",      default="auto",
                        choices=["auto", "polygon", "yfinance"],
                        help="Market data provider")
        sp.add_argument("--breakdown",     action="store_true",
                        help="Show score breakdown for top setup")
        sp.add_argument("--raw",           action="store_true",
                        help="Show raw signal values")
        sp.add_argument("--no-colour",     action="store_true",
                        help="Disable ANSI colour output")
        sp.add_argument("--analog",        action="store_true",
                        help="Enable analog engine (requires pre-built dataset)")
        sp.add_argument("--no-alerts",     action="store_true",
                        help="Do not emit or log alerts")
        sp.add_argument("--log-file",      default=None,
                        help="Override alert log CSV path")

    # ── snapshot ──────────────────────────────────────────────────────────
    snap = sub.add_parser("snapshot", help="One-time scan of the latest bar",
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_shared(snap)

    # ── live ──────────────────────────────────────────────────────────────
    live = sub.add_parser("live", help="Continuous live scanning loop",
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_shared(live)
    live.add_argument("--interval-sec", default=_DEFAULT_LIVE_SLEEP, type=int,
                      help="Seconds between scan iterations")

    # ── replay ────────────────────────────────────────────────────────────
    replay = sub.add_parser("replay", help="Historical replay evaluation",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    replay.add_argument("--interval",      default=_DEFAULT_INTERVAL)
    replay.add_argument("--lookback-days", default=60, type=int,
                        help="Days of history to fetch for replay")
    replay.add_argument("--provider",      default="auto",
                        choices=["auto", "polygon", "yfinance"])
    replay.add_argument("--file",          default=None,
                        help="Path to local OHLCV file (CSV or Parquet) for replay")
    replay.add_argument("--min-history",   default=50, type=int,
                        help="Minimum bars of history before first scan")
    replay.add_argument("--no-colour",     action="store_true")

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
        if not hasattr(args, "breakdown"):
            args.breakdown = False
            args.raw = False
            args.no_colour = False
            args.analog = False
            args.no_alerts = False
            args.log_file = None
            args.interval = _DEFAULT_INTERVAL
            args.lookback_days = _DEFAULT_LOOKBACK_DAYS
            args.provider = "auto"
        cmd_snapshot(args)

    elif mode == "live":
        cmd_live(args)

    elif mode == "replay":
        cmd_replay(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

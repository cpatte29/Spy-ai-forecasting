#!/usr/bin/env python3
"""
scripts/run_confluence_report.py
─────────────────────────────────
Orchestration script: load bars → run forecast model → detect zones →
query analog engine → compute confluence score → print report.

This is a research / reporting tool only.  It does NOT issue orders.
The existing forecasting pipeline (main_pipeline.py) and live loop
(run_live_prediction.py) are completely unchanged.

Usage
─────
  python scripts/run_confluence_report.py --provider polygon
  python scripts/run_confluence_report.py --provider file --file-path data/raw/spy_5m_polygon.parquet
  python scripts/run_confluence_report.py --synthetic
  python scripts/run_confluence_report.py --provider polygon --model-weight 0.7 --zone-weight 0.2 --analog-weight 0.1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_confluence_report")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPY Confluence Report — forecast + analog + zone context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_confluence_report.py --provider polygon\n"
            "  python scripts/run_confluence_report.py"
            " --provider file --file-path data/raw/spy_5m_polygon.parquet\n"
            "  python scripts/run_confluence_report.py --synthetic\n"
        ),
    )
    p.add_argument("--provider",   default="auto",
                   choices=["auto", "polygon", "yfinance", "file"],
                   help="Data provider (default: auto → polygon if key set, else yfinance).")
    p.add_argument("--file-path",  default=None, metavar="PATH",
                   help="Local bar file. Required when --provider file.")
    p.add_argument("--interval",   default="5m", metavar="INTERVAL",
                   help="Bar interval (default: 5m).")
    p.add_argument("--lookback",   default=10,   type=int, metavar="DAYS",
                   help="Calendar days of bars to fetch (default: 10).")
    p.add_argument("--symbol",     default="SPY", metavar="TICKER",
                   help="Ticker symbol (default: SPY).")

    # Zone options
    p.add_argument("--zone-lookback", default=5000, type=int, metavar="BARS",
                   help="Max bars to scan for zones (default: 5000 ≈ 64 sessions at 5m).")
    p.add_argument("--pivot-left",    default=5,    type=int)
    p.add_argument("--pivot-right",   default=5,    type=int)
    p.add_argument("--min-disp",      default=0.003, type=float,
                   help="Min displacement fraction for zone detection (default: 0.003).")

    # Analog options
    p.add_argument("--no-analog",      action="store_true", default=False,
                   help="Skip analog engine query (use if dataset not built yet).")
    p.add_argument("--analog-dataset", default=None, metavar="PATH",
                   help="Override path to analog dataset parquet.")
    p.add_argument("--top-n",          default=24, type=int,
                   help="Analog matches to retrieve (default: 24).")

    # Scorer thresholds
    p.add_argument("--strong-score",   default=0.35, type=float,
                   help="Score threshold for STRONG labels (default: 0.35).")
    p.add_argument("--moderate-score", default=0.12, type=float,
                   help="Score threshold for MODERATE labels (default: 0.12).")

    # Weight overrides (normalised to sum=1 automatically)
    p.add_argument("--model-weight",  default=None, type=float, metavar="W",
                   help="Override model component weight (default: 0.50).")
    p.add_argument("--zone-weight",   default=None, type=float, metavar="W",
                   help="Override zone component weight (default: 0.30).")
    p.add_argument("--analog-weight", default=None, type=float, metavar="W",
                   help="Override analog component weight (default: 0.20).")

    # Misc
    p.add_argument("--synthetic", action="store_true", default=False,
                   help="Use synthetic bars for a smoke test (no API key required).")
    p.add_argument("--verbose",   action="store_true", default=False,
                   help="Enable INFO-level logging.")
    return p.parse_args()


# ── step functions ────────────────────────────────────────────────────────────

def _load_bars(args: argparse.Namespace) -> pd.DataFrame:
    if args.synthetic:
        from data.data_loader import load_synthetic
        print("  Using synthetic bars (smoke test mode) …")
        return load_synthetic(n_days=30)

    from data.data_loader import load_bars

    kwargs: dict = dict(interval=args.interval, symbol=args.symbol)

    if args.provider == "file":
        if not args.file_path:
            print("ERROR: --file-path is required when --provider file", file=sys.stderr)
            sys.exit(1)
        kwargs.update(provider="file", file_path=args.file_path,
                      lookback_days=args.lookback)
    else:
        kwargs.update(provider=args.provider, lookback_days=args.lookback)

    return load_bars(**kwargs)


def _run_model(df_raw: pd.DataFrame) -> dict:
    """Load saved models, build features, predict on last bar."""
    result: dict = {
        "dir_prob":      None,
        "pred_range":    None,
        "current_price": None,
        "bar_ts":        None,
        "feature_count": 0,
        "n_bars":        len(df_raw),
        "error":         None,
    }

    from features.feature_engineering import build_features
    from models.train_direction import load_direction_model, predict_direction_proba
    from models.train_range     import load_range_model,     predict_range

    try:
        dir_model = load_direction_model()
        rng_model = load_range_model()
    except FileNotFoundError as e:
        result["error"] = (
            f"Model file not found: {e}\n"
            "      Run `python main_pipeline.py --mode file"
            " --file-path data/raw/spy_5m_polygon.parquet --interval 5m` first."
        )
        return result

    try:
        feats = build_features(df_raw)
    except Exception as e:
        result["error"] = f"Feature engineering failed: {e}"
        return result

    # Filter to the exact columns used during training
    train_cols = dir_model.booster_.feature_name()
    available  = [c for c in train_cols if c in feats.columns]
    missing    = [c for c in train_cols if c not in feats.columns]
    if missing:
        logger.warning("%d training column(s) missing in live features: %s",
                       len(missing), missing[:5])

    X = feats[available].dropna()
    if X.empty:
        result["error"] = "All feature rows are NaN after dropna() — not enough history."
        return result

    last = X.iloc[[-1]]
    result["bar_ts"]        = last.index[0]
    result["current_price"] = float(df_raw.loc[last.index[0], "close"])
    result["feature_count"] = len(available)
    result["dir_prob"]      = float(predict_direction_proba(dir_model, last)[0])
    result["pred_range"]    = float(predict_range(rng_model, last)[0])
    return result


def _run_analog(df_raw: pd.DataFrame, zone_ctx: dict,
                dataset_path: str | None, top_n: int) -> dict | None:
    """Query the analog engine using the current bar window."""
    ds_path = dataset_path or str(
        ROOT / "analog_engine" / "data" / "supply_zone_events.parquet"
    )

    try:
        from analog_engine.analog_report    import run_analog_analysis
        from analog_engine.similarity_search import load_analog_dataset
    except ImportError as e:
        return {"error": f"analog_engine import failed: {e}", "n_matches": 0}

    if not Path(ds_path).exists():
        return {
            "error": (
                "Analog dataset not found. Build it with:\n"
                "      python analog_engine/analog_dataset_builder.py"
                " --source file --file-path data/raw/spy_5m_polygon.parquet"
            ),
            "n_matches": 0,
        }

    try:
        ds = load_analog_dataset(ds_path)
    except Exception as e:
        return {"error": f"Dataset load error: {e}", "n_matches": 0}

    if len(ds) < top_n:
        return {
            "error": (
                f"Dataset has {len(ds)} events (need ≥ {top_n}). "
                "Rebuild with a longer history window."
            ),
            "n_matches": 0,
        }

    # Build a mock window from recent bars with inferred bar_roles
    window = df_raw.tail(25).copy()
    n = len(window)
    window["bar_role"] = (
        ["before"] * 10 + ["zone"] * 5 + ["after"] * max(0, n - 15)
    )[:n]

    # Use nearest supply zone metadata when available
    supply = zone_ctx.get("supply", {})
    meta: dict = {
        "zone_high":           supply.get("zone_high")  or float(window["high"].max()),
        "zone_low":            supply.get("zone_low")   or float(window["low"].max()  * 1.001),
        "displacement_size":   supply.get("displacement_size", 0.005),
        "volume_spike_ratio":  supply.get("volume_spike_ratio", 1.0),
        "bars_since_creation": supply.get("bars_since_created", 0) or 0,
        "test_number":         1,
        "approach_return":     0.0,
        "approach_speed":      0.0,
    }
    zh = meta["zone_high"]
    zl = meta["zone_low"]
    meta["zone_width_pct"] = (zh - zl) / zh if zh else 0.003
    meta.setdefault("displacement_speed", 0.0003)

    try:
        report = run_analog_analysis(
            df_current_window=window,
            query_event_meta=meta,
            dataset=ds,
            top_n=top_n,
            method="cosine",
            print_report=False,
        )
        report["error"] = None
        return report
    except Exception as e:
        return {"error": f"Analog analysis error: {e}", "n_matches": 0}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    print("\nLoading bars …")
    df_raw = _load_bars(args)
    if df_raw.empty:
        print("ERROR: no bars loaded.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(df_raw)} bars  {df_raw.index[0]}  →  {df_raw.index[-1]}")

    print("Running forecast model …")
    model_out = _run_model(df_raw)
    if model_out.get("error"):
        print(f"  ⚠  {model_out['error']}")

    current_price = (
        model_out.get("current_price") or float(df_raw["close"].iloc[-1])
    )

    print("Detecting supply and demand zones …")
    from confluence_engine.zone_context import get_zone_context
    zone_ctx = get_zone_context(
        df_raw,
        current_price      = current_price,
        bar_interval       = args.interval,
        pivot_left         = args.pivot_left,
        pivot_right        = args.pivot_right,
        min_disp_pct       = args.min_disp,
        zone_lookback_bars = args.zone_lookback,
    )
    s_found = zone_ctx["supply"]["n_zones_found"]
    d_found = zone_ctx["demand"]["n_zones_found"]
    print(f"  Supply zones: {s_found}  |  Demand zones: {d_found}  "
          f"|  Bias: {zone_ctx['bias']}")

    if args.no_analog:
        print("Analog engine: skipped (--no-analog).")
        analog_out = None
    else:
        print("Querying analog engine …")
        analog_out = _run_analog(df_raw, zone_ctx, args.analog_dataset, args.top_n)
        if analog_out and analog_out.get("error"):
            print(f"  ⚠  {analog_out['error']}")
        elif analog_out:
            print(f"  {analog_out.get('n_matches', 0)} matches  "
                  f"rej={analog_out.get('rejection_rate', 0):.0%}  "
                  f"brk={analog_out.get('breakout_rate', 0):.0%}")

    from confluence_engine.confluence_scorer import compute_confluence
    score_out = compute_confluence(
        dir_prob       = model_out.get("dir_prob")   or 0.5,
        pred_range     = model_out.get("pred_range") or 0.0,
        zone_ctx       = zone_ctx,
        analog         = analog_out or {},
        strong_score   = args.strong_score,
        moderate_score = args.moderate_score,
        model_weight   = args.model_weight,
        zone_weight    = args.zone_weight,
        analog_weight  = args.analog_weight,
    )

    from confluence_engine.report_formatter import format_confluence_report
    format_confluence_report(
        current_price = current_price,
        bar_ts        = model_out.get("bar_ts"),
        interval      = args.interval,
        model_out     = model_out,
        zone_ctx      = zone_ctx,
        analog_out    = analog_out,
        score_out     = score_out,
        symbol        = args.symbol,
        print_output  = True,
    )


if __name__ == "__main__":
    main()

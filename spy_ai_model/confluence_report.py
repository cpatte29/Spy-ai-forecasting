#!/usr/bin/env python3
"""
confluence_report.py
────────────────────
Research / reporting tool that combines three signals into a single
confluence label for a given SPY setup:

  1. Forecast model     – direction probability + predicted range
  2. Analog engine      – historical supply-zone match statistics
  3. Zone context       – nearest supply zone relative to current price

Confluence labels
─────────────────
  STRONG_LONG      high-confidence bullish (model ≥ strong threshold +
                   analog not bearish, or no analog data available)
  MODERATE_LONG    moderate bullish (model ≥ moderate threshold)
  NEUTRAL          mixed signals or no edge
  MODERATE_SHORT   moderate bearish
  STRONG_SHORT     high-confidence bearish

This script is a research/reporting tool only.  It does NOT issue orders.
Use run_live_prediction.py for live signals.

Usage
─────
  # Use Polygon for live bars (requires POLYGON_API_KEY):
  python confluence_report.py --provider polygon --interval 5m

  # Use local Parquet file (no API key needed):
  python confluence_report.py \\
      --provider file \\
      --file-path data/raw/spy_5m_polygon.parquet \\
      --interval 5m

  # Override thresholds:
  python confluence_report.py --strong-threshold 0.62 --moderate-threshold 0.56

  # Skip analog engine (if dataset not built yet):
  python confluence_report.py --no-analog

  # Specify a custom analog dataset path:
  python confluence_report.py \\
      --analog-dataset analog_engine/data/supply_zone_events.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("confluence_report")

# ── default thresholds ────────────────────────────────────────────────────────
STRONG_THRESHOLD   = 0.62   # P(up) ≥ this → STRONG_LONG;  P(up) ≤ 1−this → STRONG_SHORT
MODERATE_THRESHOLD = 0.56   # P(up) ≥ this → MODERATE_LONG; etc.

# Analog bias thresholds
ANALOG_BEARISH_REJECTION_RATE = 0.60   # rejection_rate ≥ this → analog bearish
ANALOG_BULLISH_BREAKOUT_RATE  = 0.45   # breakout_rate  ≥ this → analog bullish

SEP  = "═" * 66
DASH = "─" * 66


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPY Confluence Report — model + analog + zone context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python confluence_report.py --provider polygon --interval 5m\n"
            "  python confluence_report.py"
            " --provider file --file-path data/raw/spy_5m_polygon.parquet\n"
            "  python confluence_report.py --no-analog\n"
        ),
    )
    p.add_argument("--provider",    default="auto",
                   choices=["auto", "polygon", "yfinance", "file"],
                   help="Data provider (default: auto)")
    p.add_argument("--file-path",   default=None, metavar="PATH",
                   help="Path to local bar file (required when --provider file).")
    p.add_argument("--interval",    default="5m", metavar="INTERVAL",
                   help="Bar interval (default: 5m).")
    p.add_argument("--lookback",    default=10,   type=int, metavar="DAYS",
                   help="Calendar days to fetch for the live window (default: 10).")
    p.add_argument("--strong-threshold",   default=STRONG_THRESHOLD,   type=float,
                   help=f"Strong confluence threshold (default: {STRONG_THRESHOLD}).")
    p.add_argument("--moderate-threshold", default=MODERATE_THRESHOLD, type=float,
                   help=f"Moderate confluence threshold (default: {MODERATE_THRESHOLD}).")
    p.add_argument("--no-analog",   action="store_true", default=False,
                   help="Skip analog engine (use if dataset not yet built).")
    p.add_argument("--analog-dataset", default=None, metavar="PATH",
                   help="Override path to analog dataset parquet.")
    p.add_argument("--top-n",       default=20, type=int,
                   help="Analog matches to retrieve (default: 20).")
    p.add_argument("--zone-lookback", default=60, type=int, metavar="SESSIONS",
                   help="Sessions of history to scan for supply zones (default: 60).")
    return p.parse_args()


# ── data loading ──────────────────────────────────────────────────────────────

def _load_bars(args: argparse.Namespace) -> pd.DataFrame:
    from data.data_loader import load_bars
    kwargs: dict = dict(interval=args.interval)

    if args.provider == "file":
        if not args.file_path:
            logger.error("--file-path is required when --provider file")
            sys.exit(1)
        kwargs["provider"]  = "file"
        kwargs["file_path"] = args.file_path
        # For a local file, get_latest_stock_bars uses the file's last timestamp
        # as "now" so we can use lookback_days to trim to the most recent window.
        kwargs["lookback_days"] = args.lookback
    else:
        kwargs["provider"]      = args.provider
        kwargs["lookback_days"] = args.lookback

    return load_bars(**kwargs)


# ── model inference ───────────────────────────────────────────────────────────

def _run_model(df_raw: pd.DataFrame, interval: str) -> dict:
    """
    Load saved direction + range models, build features, and predict.

    Returns a dict with:
        dir_prob       float   P(up) for the most recent closed bar
        pred_range     float   predicted high-low range fraction
        current_price  float   close of the scored bar
        bar_ts         Timestamp  timestamp of the scored bar
        feature_count  int
        n_bars         int     total bars in the window
        error          str | None
    """
    from features.feature_engineering import build_features
    from models.train_direction import load_direction_model, predict_direction_proba
    from models.train_range     import load_range_model,     predict_range

    result: dict = {
        "dir_prob":      None,
        "pred_range":    None,
        "current_price": None,
        "bar_ts":        None,
        "feature_count": 0,
        "n_bars":        len(df_raw),
        "error":         None,
    }

    try:
        dir_model = load_direction_model()
        rng_model = load_range_model()
    except FileNotFoundError as e:
        result["error"] = f"Model not found: {e}. Run main_pipeline.py first."
        return result

    try:
        feats = build_features(df_raw)
    except Exception as e:
        result["error"] = f"Feature engineering failed: {e}"
        return result

    # Filter to columns the model was trained on
    train_cols = dir_model.booster_.feature_name()
    available  = [c for c in train_cols if c in feats.columns]
    missing    = [c for c in train_cols if c not in feats.columns]
    if missing:
        logger.warning("Missing %d training columns: %s", len(missing), missing[:5])

    # Drop NaN rows; use last valid bar
    X = feats[available].dropna()
    if X.empty:
        result["error"] = "All feature rows are NaN after dropna()."
        return result

    last_bar   = X.iloc[[-1]]
    result["bar_ts"]        = last_bar.index[0]
    result["current_price"] = float(df_raw.loc[last_bar.index[0], "close"])
    result["feature_count"] = len(available)

    result["dir_prob"]   = float(predict_direction_proba(dir_model, last_bar)[0])
    result["pred_range"] = float(predict_range(rng_model, last_bar)[0])

    return result


# ── zone context ──────────────────────────────────────────────────────────────

def _get_zone_context(df_raw: pd.DataFrame, current_price: float,
                      zone_lookback_sessions: int) -> dict:
    """
    Detect supply zones in the most recent N sessions and find the nearest
    zone above current price.

    Returns a dict with:
        nearest_zone_high  float | None
        nearest_zone_low   float | None
        zone_distance_pct  float | None   (zone_low − price) / price, signed
        n_zones_found      int
        zone_text          str
    """
    result = {
        "nearest_zone_high": None,
        "nearest_zone_low":  None,
        "zone_distance_pct": None,
        "n_zones_found":     0,
        "zone_text":         "(zone detection failed)",
    }

    try:
        from analog_engine.zone_detector import detect_supply_zones
    except ImportError:
        result["zone_text"] = "(analog_engine not importable)"
        return result

    # Trim to the requested session window
    dates = df_raw.index.normalize().unique()
    if len(dates) > zone_lookback_sessions:
        cutoff = dates[-zone_lookback_sessions]
        df_scan = df_raw[df_raw.index >= cutoff]
    else:
        df_scan = df_raw

    try:
        zones = detect_supply_zones(df_scan)
    except Exception as e:
        result["zone_text"] = f"(zone_detector error: {e})"
        return result

    result["n_zones_found"] = len(zones)

    if zones.empty:
        result["zone_text"] = "No supply zones detected in lookback window."
        return result

    # Nearest zone overhead (zone_low > current_price → resistance above)
    overhead = zones[zones["zone_low"] > current_price]
    if overhead.empty:
        result["zone_text"] = (
            f"{len(zones)} zone(s) found, none overhead "
            f"(price is above or inside all zones)"
        )
        return result

    nearest = overhead.sort_values("zone_low").iloc[0]
    zh = float(nearest["zone_high"])
    zl = float(nearest["zone_low"])
    dist_pct = (zl - current_price) / current_price

    result["nearest_zone_high"] = zh
    result["nearest_zone_low"]  = zl
    result["zone_distance_pct"] = dist_pct
    result["zone_text"] = (
        f"{zl:.4f} – {zh:.4f}  "
        f"({dist_pct*100:+.2f}% from current price)"
    )
    return result


# ── analog query ──────────────────────────────────────────────────────────────

def _run_analog(df_raw: pd.DataFrame, zone_ctx: dict,
                dataset_path: str | None, top_n: int) -> dict:
    """
    Build a mock zone window from recent bars and query the analog engine.

    Returns the run_analog_analysis() dict, or a stub on failure.
    """
    stub = {
        "n_matches":         0,
        "rejection_rate":    0.0,
        "breakout_rate":     0.0,
        "inconclusive_rate": 0.0,
        "avg_rejection_move": 0.0,
        "avg_breakout_move":  0.0,
        "avg_similarity":     0.0,
        "error":              None,
    }

    try:
        from analog_engine.analog_report  import run_analog_analysis
        from analog_engine.similarity_search import load_analog_dataset
    except ImportError as e:
        stub["error"] = f"analog_engine import failed: {e}"
        return stub

    # Load dataset
    ds_path = dataset_path or str(
        ROOT / "analog_engine" / "data" / "supply_zone_events.parquet"
    )
    try:
        ds = load_analog_dataset(ds_path)
    except FileNotFoundError:
        stub["error"] = (
            "Analog dataset not found.  Build it first:\n"
            "  python analog_engine/analog_dataset_builder.py"
            " --source file --file-path data/raw/spy_5m_polygon.parquet"
        )
        return stub
    except Exception as e:
        stub["error"] = f"Dataset load error: {e}"
        return stub

    if ds.empty or len(ds) < top_n:
        stub["error"] = (
            f"Analog dataset has only {len(ds)} events "
            f"(need ≥ {top_n}).  Rebuild with more history."
        )
        return stub

    # Build a mock window: last 25 bars with inferred bar_roles
    window = df_raw.tail(25).copy()
    n = len(window)
    roles = (["before"] * 10 + ["zone"] * 5 + ["after"] * max(0, n - 15))[:n]
    window["bar_role"] = roles

    # Inject zone metadata from zone context (or defaults)
    meta: dict = {}
    if zone_ctx.get("nearest_zone_high"):
        meta["zone_high"]       = zone_ctx["nearest_zone_high"]
        meta["zone_low"]        = zone_ctx["nearest_zone_low"]
        zh = meta["zone_high"]
        zl = meta["zone_low"]
        meta["zone_width_pct"]  = (zh - zl) / zh if zh else 0.003
    else:
        recent_high = float(window["high"].max())
        recent_low  = float(window["low"].min())
        meta["zone_high"]      = recent_high
        meta["zone_low"]       = recent_low * 1.001
        meta["zone_width_pct"] = 0.003

    meta.setdefault("displacement_size",  0.005)
    meta.setdefault("displacement_speed", 0.0003)
    meta.setdefault("volume_spike_ratio", 1.0)
    meta.setdefault("bars_since_creation", 0)
    meta.setdefault("test_number",         1)
    meta.setdefault("approach_return",     0.0)
    meta.setdefault("approach_speed",      0.0)

    try:
        report = run_analog_analysis(
            df_current_window=window,
            query_event_meta=meta,
            dataset=ds,
            top_n=top_n,
            method="cosine",
            print_report=False,
        )
    except Exception as e:
        stub["error"] = f"Analog analysis error: {e}"
        return stub

    report["error"] = None
    return report


# ── confluence label logic ────────────────────────────────────────────────────

def _compute_confluence(
    dir_prob:        float,
    analog:          dict,
    strong_thr:      float,
    moderate_thr:    float,
) -> str:
    """
    Combine directional model probability with analog bias into a single label.

    Analog bias categories:
      BEARISH  → rejection_rate ≥ ANALOG_BEARISH_REJECTION_RATE
      BULLISH  → breakout_rate  ≥ ANALOG_BULLISH_BREAKOUT_RATE
      NEUTRAL  → neither (or no analog data)

    Confluence matrix
    ─────────────────
                          dir_prob
                      strong↑  mod↑  neutral  mod↓  strong↓
    analog BULLISH   STRONG   MOD    NEUTRAL  MOD↓   MOD↓
    analog NEUTRAL   STRONG   MOD    NEUTRAL  MOD↓   STRONG↓
    analog BEARISH   MOD      NEUT   NEUTRAL  MOD↓   STRONG↓
    """
    strong_lo  = 1.0 - strong_thr
    moderate_lo = 1.0 - moderate_thr

    has_analog = analog.get("n_matches", 0) >= 5 and analog.get("error") is None

    if has_analog:
        rej_rate = float(analog.get("rejection_rate", 0.0))
        brk_rate = float(analog.get("breakout_rate",  0.0))
        if rej_rate >= ANALOG_BEARISH_REJECTION_RATE:
            analog_bias = "BEARISH"
        elif brk_rate >= ANALOG_BULLISH_BREAKOUT_RATE:
            analog_bias = "BULLISH"
        else:
            analog_bias = "NEUTRAL"
    else:
        analog_bias = "NONE"

    # Direction bucket
    if dir_prob >= strong_thr:
        dir_bucket = "STRONG_UP"
    elif dir_prob >= moderate_thr:
        dir_bucket = "MODERATE_UP"
    elif dir_prob <= strong_lo:
        dir_bucket = "STRONG_DOWN"
    elif dir_prob <= moderate_lo:
        dir_bucket = "MODERATE_DOWN"
    else:
        dir_bucket = "NEUTRAL"

    # Combine
    if dir_bucket == "STRONG_UP":
        if analog_bias == "BEARISH":
            return "MODERATE_LONG"
        return "STRONG_LONG"

    if dir_bucket == "MODERATE_UP":
        if analog_bias == "BEARISH":
            return "NEUTRAL"
        return "MODERATE_LONG"

    if dir_bucket == "STRONG_DOWN":
        if analog_bias == "BULLISH":
            return "MODERATE_SHORT"
        return "STRONG_SHORT"

    if dir_bucket == "MODERATE_DOWN":
        if analog_bias == "BULLISH":
            return "NEUTRAL"
        return "MODERATE_SHORT"

    return "NEUTRAL"


# ── report printer ────────────────────────────────────────────────────────────

_LABEL_COLOR = {
    "STRONG_LONG":    "▲▲  STRONG_LONG",
    "MODERATE_LONG":  " ▲  MODERATE_LONG",
    "NEUTRAL":        " —  NEUTRAL",
    "MODERATE_SHORT": " ▼  MODERATE_SHORT",
    "STRONG_SHORT":   "▼▼  STRONG_SHORT",
}


def _print_report(
    args:        argparse.Namespace,
    model_res:   dict,
    zone_ctx:    dict,
    analog:      dict,
    label:       str,
    df_raw:      pd.DataFrame,
) -> None:
    import datetime as dt

    try:
        from zoneinfo import ZoneInfo
        tz_et = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        tz_et = pytz.timezone("America/New_York")

    now_et = dt.datetime.now(tz_et).replace(tzinfo=None)

    print()
    print(SEP)
    print("  SPY Confluence Report")
    print(SEP)
    print(f"  Generated at     : {now_et.strftime('%Y-%m-%d %H:%M:%S')} ET")
    print(f"  Provider         : {args.provider}")
    print(f"  Interval         : {args.interval}")
    print(f"  Bars in window   : {len(df_raw)}")
    print(DASH)

    # ── Section 1: Forecast model ──────────────────────────────────────────
    print("  1.  Forecast Model")
    print(DASH)

    if model_res.get("error"):
        print(f"  ⚠   {model_res['error']}")
    else:
        price     = model_res["current_price"]
        dir_prob  = model_res["dir_prob"]
        pred_rng  = model_res["pred_range"]
        bar_ts    = model_res["bar_ts"]
        feat_cnt  = model_res["feature_count"]

        strong_thr   = args.strong_threshold
        moderate_thr = args.moderate_threshold

        if dir_prob >= strong_thr:
            dir_label = f"BULLISH (strong ≥ {strong_thr:.0%})"
        elif dir_prob >= moderate_thr:
            dir_label = f"BULLISH (moderate ≥ {moderate_thr:.0%})"
        elif dir_prob <= 1 - strong_thr:
            dir_label = f"BEARISH (strong ≤ {1-strong_thr:.0%})"
        elif dir_prob <= 1 - moderate_thr:
            dir_label = f"BEARISH (moderate ≤ {1-moderate_thr:.0%})"
        else:
            dir_label = "NEUTRAL"

        range_pts = pred_rng * price

        print(f"  Current price    : {price:.4f}")
        print(f"  Scored bar       : {bar_ts}")
        print(f"  P(up)            : {dir_prob:.4f}   [{dir_label}]")
        print(f"  Predicted range  : {pred_rng:.4f}  ({range_pts:.2f} pts)")
        print(f"  Features used    : {feat_cnt}")

    # ── Section 2: Zone context ────────────────────────────────────────────
    print(DASH)
    print("  2.  Nearest Supply Zone")
    print(DASH)
    print(f"  Zones found      : {zone_ctx['n_zones_found']}")
    print(f"  Nearest zone     : {zone_ctx['zone_text']}")

    # ── Section 3: Analog engine ───────────────────────────────────────────
    print(DASH)
    print("  3.  Analog Engine")
    print(DASH)

    if args.no_analog:
        print("  (skipped — --no-analog flag)")
    elif analog.get("error"):
        print(f"  ⚠   {analog['error']}")
    else:
        n = analog.get("n_matches", 0)
        if n == 0:
            print("  No analog matches found.")
        else:
            rej  = analog.get("rejection_rate",    0.0)
            brk  = analog.get("breakout_rate",     0.0)
            inc  = analog.get("inconclusive_rate", 0.0)
            sim  = analog.get("avg_similarity",    0.0)
            rejmv = analog.get("avg_rejection_move", 0.0)
            brkmv = analog.get("avg_breakout_move",  0.0)

            if rej >= ANALOG_BEARISH_REJECTION_RATE:
                analog_verdict = f"BEARISH (rejection ≥ {ANALOG_BEARISH_REJECTION_RATE:.0%})"
            elif brk >= ANALOG_BULLISH_BREAKOUT_RATE:
                analog_verdict = f"BULLISH (breakout ≥ {ANALOG_BULLISH_BREAKOUT_RATE:.0%})"
            else:
                analog_verdict = "NEUTRAL"

            print(f"  Matches          : {n}")
            print(f"  Avg similarity   : {sim:.4f}")
            print(f"  Rejection rate   : {rej*100:.1f} %  (avg move: {rejmv*100:.2f} %)")
            print(f"  Breakout rate    : {brk*100:.1f} %  (avg move: {brkmv*100:.2f} %)")
            print(f"  Inconclusive     : {inc*100:.1f} %")
            print(f"  Analog verdict   : {analog_verdict}")

    # ── Section 4: Confluence label ────────────────────────────────────────
    print(DASH)
    print("  4.  Confluence")
    print(DASH)
    print(f"  Strong threshold  : {args.strong_threshold:.0%}")
    print(f"  Moderate threshold: {args.moderate_threshold:.0%}")
    print()
    label_str = _LABEL_COLOR.get(label, label)
    print(f"  ┌{'─'*40}┐")
    print(f"  │   {label_str:<37}│")
    print(f"  └{'─'*40}┘")
    print()

    # ── Section 5: Quick guidance ──────────────────────────────────────────
    print(DASH)
    guidance = {
        "STRONG_LONG":    "Model + analog both bullish.  Consider long on dip to VWAP or ORB high retest.",
        "MODERATE_LONG":  "Moderate bullish edge.  Wait for structure confirmation before entry.",
        "NEUTRAL":        "No edge.  Mixed or conflicting signals — stay flat or reduce size.",
        "MODERATE_SHORT": "Moderate bearish edge.  Supply zone overhead may cap upside.",
        "STRONG_SHORT":   "Model + zone both bearish.  Look for rejection entry near zone low.",
    }
    print(f"  Guidance: {guidance.get(label, '')}")
    print(SEP)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── 1. Load bars ──────────────────────────────────────────────────────
    print(f"\nLoading {args.interval} bars (provider={args.provider}) …")
    try:
        df_raw = _load_bars(args)
    except Exception as e:
        print(f"ERROR loading bars: {e}", file=sys.stderr)
        sys.exit(1)

    if df_raw.empty:
        print("ERROR: No bars returned.", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(df_raw)} bars  {df_raw.index[0]} → {df_raw.index[-1]}")

    # ── 2. Model inference ────────────────────────────────────────────────
    print("Running forecast model …")
    model_res = _run_model(df_raw, args.interval)

    # ── 3. Zone context ───────────────────────────────────────────────────
    print("Detecting supply zones …")
    current_price = model_res.get("current_price") or float(df_raw["close"].iloc[-1])
    zone_ctx = _get_zone_context(df_raw, current_price, args.zone_lookback)

    # ── 4. Analog engine ──────────────────────────────────────────────────
    analog: dict
    if args.no_analog:
        analog = {"n_matches": 0, "rejection_rate": 0.0, "breakout_rate": 0.0,
                  "inconclusive_rate": 0.0, "avg_rejection_move": 0.0,
                  "avg_breakout_move": 0.0, "avg_similarity": 0.0, "error": None}
    else:
        print("Querying analog engine …")
        analog = _run_analog(df_raw, zone_ctx, args.analog_dataset, args.top_n)

    # ── 5. Confluence label ───────────────────────────────────────────────
    if model_res.get("error") or model_res.get("dir_prob") is None:
        label = "NEUTRAL"
    else:
        label = _compute_confluence(
            dir_prob     = model_res["dir_prob"],
            analog       = analog,
            strong_thr   = args.strong_threshold,
            moderate_thr = args.moderate_threshold,
        )

    # ── 6. Print report ───────────────────────────────────────────────────
    _print_report(args, model_res, zone_ctx, analog, label, df_raw)


if __name__ == "__main__":
    main()

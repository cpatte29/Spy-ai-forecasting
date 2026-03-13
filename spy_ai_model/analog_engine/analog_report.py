"""
analog_report.py
────────────────
Generate a human-readable historical analog report for a live setup.

Given a current bar window (the "setup") this module:
  1. Computes the pattern feature vector for the live window.
  2. Finds the top-N most similar historical zone-test events.
  3. Summarises the outcomes of those analogues.
  4. Prints (and returns) the report.

Public API
──────────
    from analog_engine.analog_report import run_analog_analysis

    report = run_analog_analysis(
        df_current_window,   # DataFrame with bar_role column
        query_event_meta,    # dict with zone_high, zone_low, etc.
    )
    # report is a dict with all statistics and the matched events DataFrame

CLI
───
    python analog_report.py
        --dataset  analog_engine/data/supply_zone_events.parquet
        --top-n    20
        --method   cosine
        (requires a live window — for testing, builds a synthetic one)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analog_engine.similarity_search import (
    load_analog_dataset,
    find_similar_events,
    compute_query_features,
    explain_top_match,
)

logger = logging.getLogger(__name__)

SEP  = "═" * 62
DASH = "─" * 62


# ── main report function ───────────────────────────────────────────────────────

def run_analog_analysis(
    df_current_window: pd.DataFrame,
    query_event_meta:  dict | pd.Series | None = None,
    dataset:           pd.DataFrame | None     = None,
    dataset_path:      str | Path | None       = None,
    top_n:             int                     = 24,
    method:            Literal["cosine","knn"] = "cosine",
    print_report:      bool                    = True,
) -> dict:
    """
    Run the full analog analysis for a live setup window.

    Parameters
    ──────────
    df_current_window   DataFrame with columns open/high/low/close/volume and
                        a "bar_role" column ("before" | "zone" | "after").
                        The "after" bars are optional for the query (they
                        represent the future which is unknown live).
    query_event_meta    Dict or Series describing the zone being tested.
                        Required keys: zone_high, zone_low, zone_width_pct,
                        displacement_size, displacement_speed,
                        volume_spike_ratio, bars_since_creation, test_number,
                        approach_return, approach_speed.
                        If None, a stub with neutral values is used.
    dataset             Pre-loaded analog dataset DataFrame.  If None, loaded
                        from disk (dataset_path or default path).
    dataset_path        Override for the parquet path.
    top_n               Number of historical analogues to retrieve.
    method              Similarity method: "cosine" (default) or "knn".
    print_report        If True, print the formatted report to stdout.

    Returns
    ───────
    dict with keys:
        matches             pd.DataFrame – top_n matched events with scores
        n_matches           int
        rejection_rate      float
        breakout_rate       float
        inconclusive_rate   float
        avg_rejection_move  float  – average max_move_down_pct for rejections
        avg_breakout_move   float  – average max_move_up_pct for breakouts
        avg_bars_to_resolve float
        avg_similarity      float
        query_features      pd.Series – feature vector of the live setup
        report_text         str  – the formatted report string
    """
    # ── load dataset ──────────────────────────────────────────────────────
    if dataset is None:
        dataset = load_analog_dataset(dataset_path)

    # ── default event meta ────────────────────────────────────────────────
    if query_event_meta is None:
        query_event_meta = _infer_meta_from_window(df_current_window)

    # ── query features ────────────────────────────────────────────────────
    query_fvec = compute_query_features(df_current_window, query_event_meta)

    # ── find matches ──────────────────────────────────────────────────────
    matches = find_similar_events(
        query_window=df_current_window,
        query_event_meta=query_event_meta,
        dataset=dataset,
        top_n=top_n,
        method=method,
    )

    # ── compute statistics ────────────────────────────────────────────────
    stats = _compute_statistics(matches)

    # ── pattern breakdown (most common candle pattern after entry) ────────
    candle_pattern = _dominant_candle_pattern(matches)

    # ── format report ─────────────────────────────────────────────────────
    report_text = _format_report(
        query_event_meta=query_event_meta,
        matches=matches,
        stats=stats,
        candle_pattern=candle_pattern,
        method=method,
    )

    if print_report:
        print(report_text)

    return {
        "matches":              matches,
        "n_matches":            int(stats["n"]),
        "rejection_rate":       float(stats["rejection_rate"]),
        "breakout_rate":        float(stats["breakout_rate"]),
        "inconclusive_rate":    float(stats["inconclusive_rate"]),
        "avg_rejection_move":   float(stats["avg_rejection_move"]),
        "avg_breakout_move":    float(stats["avg_breakout_move"]),
        "avg_bars_to_resolve":  float(stats["avg_bars_to_resolve"]),
        "avg_similarity":       float(stats["avg_similarity"]),
        "dominant_pattern":     candle_pattern,
        "query_features":       query_fvec,
        "report_text":          report_text,
    }


# ── statistics ────────────────────────────────────────────────────────────────

def _compute_statistics(matches: pd.DataFrame) -> dict:
    n = len(matches)
    if n == 0:
        return {
            "n": 0, "rejection_rate": 0.0, "breakout_rate": 0.0,
            "inconclusive_rate": 0.0, "avg_rejection_move": 0.0,
            "avg_breakout_move": 0.0, "avg_bars_to_resolve": 0.0,
            "avg_similarity": 0.0,
        }

    rej_mask = matches.get("outcome__rejection", pd.Series([False]*n, dtype=bool))
    brk_mask = matches.get("outcome__breakout",  pd.Series([False]*n, dtype=bool))

    n_rej = int(rej_mask.sum())
    n_brk = int(brk_mask.sum())
    n_inc = n - n_rej - n_brk

    avg_rej_move = (
        float(matches.loc[rej_mask, "outcome__max_move_down_pct"].mean())
        if n_rej > 0 and "outcome__max_move_down_pct" in matches.columns
        else 0.0
    )
    avg_brk_move = (
        float(matches.loc[brk_mask, "outcome__max_move_up_pct"].mean())
        if n_brk > 0 and "outcome__max_move_up_pct" in matches.columns
        else 0.0
    )
    avg_bars = (
        float(matches["outcome__bars_to_resolution"].mean())
        if "outcome__bars_to_resolution" in matches.columns
        else 0.0
    )
    avg_sim = (
        float(matches["similarity_score"].mean())
        if "similarity_score" in matches.columns
        else 0.0
    )

    return {
        "n":                  n,
        "n_rej":              n_rej,
        "n_brk":              n_brk,
        "n_inc":              n_inc,
        "rejection_rate":     n_rej / n,
        "breakout_rate":      n_brk / n,
        "inconclusive_rate":  n_inc / n,
        "avg_rejection_move": avg_rej_move,
        "avg_breakout_move":  avg_brk_move,
        "avg_bars_to_resolve": avg_bars,
        "avg_similarity":     avg_sim,
    }


# ── candle pattern classification ─────────────────────────────────────────────

def _dominant_candle_pattern(matches: pd.DataFrame) -> str:
    """
    Classify the most common post-zone candlestick pattern from the
    matched events.  Uses zone__avg_upper_wick and zone__bearish_fraction
    as proxies (both computed from zone-contact bars).

    Returns a descriptive string.
    """
    if matches.empty:
        return "Insufficient data"

    uw_col  = "zone__avg_upper_wick"
    bf_col  = "zone__bearish_fraction"
    lw_col  = "zone__avg_lower_wick"

    has_uw = uw_col in matches.columns
    has_bf = bf_col in matches.columns
    has_lw = lw_col in matches.columns

    if not (has_uw or has_bf):
        return "Pattern data unavailable"

    avg_uw = float(matches[uw_col].mean()) if has_uw else 0.0
    avg_lw = float(matches[lw_col].mean()) if has_lw else 0.0
    avg_bf = float(matches[bf_col].mean()) if has_bf else 0.5

    # Simple rule-based classification
    if avg_uw > 0.35 and avg_bf > 0.55:
        return "Upper wick rejection followed by bearish close"
    if avg_uw > 0.30 and avg_bf <= 0.55:
        return "Wick rejection — indecision (hammer-like structure)"
    if avg_bf > 0.65 and avg_uw <= 0.25:
        return "Full-body bearish engulf — strong supply response"
    if avg_lw > 0.30 and avg_bf < 0.45:
        return "Lower wick support — buyers contesting zone"
    if avg_bf < 0.40:
        return "Mostly bullish bars — possible breakout accumulation"
    return "Mixed structure — no dominant pattern"


# ── report formatter ──────────────────────────────────────────────────────────

def _format_report(
    query_event_meta: dict | pd.Series,
    matches:          pd.DataFrame,
    stats:            dict,
    candle_pattern:   str,
    method:           str,
) -> str:
    if isinstance(query_event_meta, pd.Series):
        meta = query_event_meta.to_dict()
    else:
        meta = dict(query_event_meta)

    lines = [SEP]
    lines.append("  SPY Analog Engine  –  Supply Zone Report")
    lines.append(SEP)

    # Zone context
    zh = meta.get("zone_high", float("nan"))
    zl = meta.get("zone_low",  float("nan"))
    lines.append(f"  Zone               : {zl:.4f} – {zh:.4f}")
    lines.append(f"  Zone width         : {meta.get('zone_width_pct', 0)*100:.3f} %")
    lines.append(f"  Displacement       : {meta.get('displacement_size', 0)*100:.2f} %")
    lines.append(f"  Vol spike ratio    : {meta.get('volume_spike_ratio', 1.0):.2f}×")
    lines.append(f"  Similarity method  : {method}")
    lines.append(DASH)

    n = stats["n"]
    lines.append(f"  Analogues found    : {n}")
    if n == 0:
        lines.append("  (No historical analogues — try --top-n or build a larger dataset)")
        lines.append(SEP)
        return "\n".join(lines)

    lines.append(f"  Avg similarity     : {stats['avg_similarity']:.4f}")
    lines.append(DASH)

    # Outcome rates
    lines.append(f"  Rejection rate     : {stats['rejection_rate']*100:.1f} %  "
                 f"({stats['n_rej']}/{n} events)")
    lines.append(f"  Breakout rate      : {stats['breakout_rate']*100:.1f} %  "
                 f"({stats['n_brk']}/{n} events)")
    lines.append(f"  Inconclusive rate  : {stats['inconclusive_rate']*100:.1f} %  "
                 f"({stats['n_inc']}/{n} events)")
    lines.append(DASH)

    # Move statistics
    if stats["n_rej"] > 0:
        lines.append(f"  Avg rejection move : {stats['avg_rejection_move']*100:.2f} %  "
                     f"({stats['avg_rejection_move'] * (zh if not np.isnan(zh) else 500):.2f} pts)")
    if stats["n_brk"] > 0:
        lines.append(f"  Avg breakout move  : {stats['avg_breakout_move']*100:.2f} %  "
                     f"({stats['avg_breakout_move'] * (zh if not np.isnan(zh) else 500):.2f} pts)")
    lines.append(f"  Avg bars to resolve: {stats['avg_bars_to_resolve']:.1f}")
    lines.append(DASH)

    # Candle pattern
    lines.append(f"  Dominant pattern   : {candle_pattern}")
    lines.append(DASH)

    # Top-5 most similar events
    lines.append("  Top 5 analogues:")
    show_cols = ["touch_timestamp", "similarity_score",
                 "outcome__resolution", "outcome__max_move_down_pct",
                 "outcome__max_move_up_pct"]
    avail = [c for c in show_cols if c in matches.columns]
    for i, (_, row) in enumerate(matches.head(5).iterrows(), 1):
        ts  = row.get("touch_timestamp", "?")
        sim = row.get("similarity_score", 0.0)
        res = row.get("outcome__resolution", "?")
        dn  = row.get("outcome__max_move_down_pct", 0.0)
        up  = row.get("outcome__max_move_up_pct",  0.0)
        lines.append(
            f"    {i}. {ts}  sim={sim:.4f}  [{res}]  "
            f"↓{dn*100:.2f}%  ↑{up*100:.2f}%"
        )

    lines.append(SEP)
    return "\n".join(lines)


# ── meta inference helper (for live windows without explicit metadata) ──────────

def _infer_meta_from_window(window: pd.DataFrame) -> dict:
    """
    Infer approximate zone metadata from the window itself when no explicit
    zone metadata is provided.  Uses the maximum high of the "zone" bars as
    zone_high and the minimum open/close as zone_low.
    """
    zone_bars = window[window["bar_role"] == "zone"] if "bar_role" in window.columns else window

    if zone_bars.empty:
        zone_bars = window

    zone_high = float(zone_bars["high"].max())
    zone_low  = float(zone_bars[["open", "close"]].min().min())
    zone_width = max(zone_high - zone_low, 1e-6)

    before_bars = window[window["bar_role"] == "before"] if "bar_role" in window.columns else pd.DataFrame()
    if not before_bars.empty and len(before_bars) >= 2:
        approach_return = float(
            (before_bars["close"].iloc[-1] - before_bars["close"].iloc[0])
            / before_bars["close"].iloc[0]
        )
        approach_speed = approach_return / len(before_bars)
    else:
        approach_return = 0.0
        approach_speed  = 0.0

    return {
        "zone_high":           zone_high,
        "zone_low":            zone_low,
        "zone_width_pct":      zone_width / zone_high,
        "displacement_size":   0.005,   # neutral default
        "displacement_speed":  0.0003,
        "volume_spike_ratio":  1.0,
        "bars_since_creation": 0,
        "test_number":         1,
        "approach_return":     approach_return,
        "approach_speed":      approach_speed,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPY Analog Engine – supply zone report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",   default=None,
                   help="Path to analog dataset parquet file")
    p.add_argument("--top-n",     default=24, type=int,
                   help="Number of historical analogues to retrieve")
    p.add_argument("--method",    default="cosine", choices=["cosine", "knn"],
                   help="Similarity method")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Demo: build a minimal synthetic window to show the report structure
    print("Loading dataset …")
    try:
        ds = load_analog_dataset(args.dataset)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nBuild the dataset first:")
        print("  cd spy_ai_model")
        print("  python analog_engine/analog_dataset_builder.py --period 30d --interval 5m")
        sys.exit(1)

    # Use the first event in the dataset as a demo query
    first = ds.iloc[0]
    from data.data_loader import load_from_yfinance
    try:
        df = load_from_yfinance(interval="5m", period="5d")
    except Exception:
        from data.data_loader import load_synthetic
        df = load_synthetic()

    # Build a mock window from recent bars
    window = df.tail(30).copy()
    window["bar_role"] = (
        ["before"] * 10 + ["zone"] * 5 + ["after"] * max(0, len(window) - 15)
    )[:len(window)]

    run_analog_analysis(
        df_current_window=window,
        dataset=ds,
        top_n=args.top_n,
        method=args.method,
        print_report=True,
    )

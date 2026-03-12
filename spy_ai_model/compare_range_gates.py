"""
compare_range_gates.py
───────────────────────
Regime-aware evaluation: test the direction model under four range filters
derived from the range model's own OOS predictions.

The idea
─────────
The range model predicts the expected high-low range of the next 60 minutes.
Filtering trades to only high-predicted-range bars means the model commits
to moves that are likely to have enough amplitude to justify the trade.

Four subsets evaluated
───────────────────────
  All samples           : no range filter (baseline)
  Range > 50th pct      : median-and-above predicted range
  Range > 60th pct      : top-40% range environment
  Range > 70th pct      : top-30% range environment

For each subset, reports
─────────────────────────
  OOS AUC, Win rate, Sharpe ratio, # trades, Max drawdown

Also
─────
  • Saves percentile calibration to models/saved/range_percentiles.json so the
    live loop can load the same absolute thresholds without recomputing.
  • Prints a recommendation for the best range gate to use in production.

Usage
─────
  python compare_range_gates.py
  python compare_range_gates.py --period 60d --threshold 0.55
  python compare_range_gates.py --data-file spy_5m.parquet --save-calibration
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    DIR_PROB_THRESHOLD, HORIZON_DIR, TRANSACTION_COST_BP,
    TICKER, MODEL_DIR,
)
from data.data_loader import load_from_yfinance, load_from_file
from data.dataset_builder import build_dataset
from evaluation.walk_forward import walk_forward_cv
from backtest.strategy_simulation import run_backtest

logger = logging.getLogger(__name__)

# Percentile gates to evaluate (fractions 0-1)
GATES = [0.0, 0.50, 0.60, 0.70]
GATE_LABELS = {
    0.0:  "All samples  (no gate)",
    0.50: "Range > 50th pct",
    0.60: "Range > 60th pct",
    0.70: "Range > 70th pct",
}

CALIBRATION_FILE = MODEL_DIR / "range_percentiles.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def _suppress(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn(*args, **kwargs)


def _safe_auc(y_true, y_prob) -> str:
    """Return AUC string or 'N/A' when there aren't enough samples."""
    if len(y_true) < 10 or len(np.unique(y_true)) < 2:
        return "N/A"
    return f"{roc_auc_score(y_true, y_prob):.4f}"


def _bt_summary(bt: dict) -> dict:
    s = bt.get("summary", {})
    if not s:
        return {"n_trades": 0, "win_rate": float("nan"),
                "sharpe": float("nan"), "max_dd": float("nan")}
    return {
        "n_trades": s.get("n_trades", 0),
        "win_rate": s.get("win_rate", float("nan")),
        "sharpe":   s.get("sharpe_ratio", float("nan")),
        "max_dd":   s.get("max_drawdown", float("nan")),
    }


def _evaluate_gate(
    gate_pct:      float,
    range_thresh:  float,
    oos_dir_proba: np.ndarray,
    oos_dir_true:  np.ndarray,
    oos_range_pred: np.ndarray,
    oos_index:     pd.DatetimeIndex,
    df_raw:        pd.DataFrame,
    threshold:     float,
) -> dict:
    """Evaluate one range gate level."""
    # OOS AUC on the qualifying subset only
    if gate_pct > 0.0:
        mask       = oos_range_pred >= range_thresh
        sub_prob   = oos_dir_proba[mask]
        sub_true   = oos_dir_true[mask]
        sub_n      = int(mask.sum())
    else:
        sub_prob   = oos_dir_proba
        sub_true   = oos_dir_true
        sub_n      = len(oos_dir_proba)

    auc_str = _safe_auc(sub_true, sub_prob)

    # Backtest on the full OOS series but with the range gate applied at entry
    bt = run_backtest(
        df_raw,
        oos_dir_proba,
        oos_index,
        threshold=threshold,
        cost_bp=TRANSACTION_COST_BP,
        hold_bars=HORIZON_DIR,
        oos_range_pred=oos_range_pred if gate_pct > 0.0 else None,
        range_threshold=range_thresh if gate_pct > 0.0 else 0.0,
    )
    bt_s = _bt_summary(bt)

    return {
        "gate_pct":      gate_pct,
        "range_thresh":  range_thresh,
        "subset_n":      sub_n,
        "auc":           auc_str,
        "n_trades":      bt_s["n_trades"],
        "win_rate":      bt_s["win_rate"],
        "sharpe":        bt_s["sharpe"],
        "max_dd":        bt_s["max_dd"],
    }


# ── recommendation ────────────────────────────────────────────────────────────

def _recommend(results: list[dict]) -> dict:
    """
    Pick the best gate:
      1. Must have >= 3 completed trades.
      2. Among qualifying gates, prefer the one with the highest Sharpe.
      3. Tie-break on win rate.
    Returns the recommended result dict.
    """
    candidates = [r for r in results if r["n_trades"] >= 3]
    if not candidates:
        return results[0]   # fall back to no gate

    def _key(r):
        sharpe = r["sharpe"] if not (isinstance(r["sharpe"], float) and np.isnan(r["sharpe"])) else -999.0
        wr     = r["win_rate"] if not (isinstance(r["win_rate"], float) and np.isnan(r["win_rate"])) else 0.0
        return (sharpe, wr)

    return max(candidates, key=_key)


# ── comparison table ──────────────────────────────────────────────────────────

def _print_table(results: list[dict], threshold: float, best: dict) -> None:
    sep  = "═" * 78
    dash = "─" * 78

    def _flt(v, pct=False):
        if isinstance(v, float) and not np.isnan(v):
            return f"{v*100:.1f}%" if pct else f"{v:.4f}"
        return str(v) if not (isinstance(v, float) and np.isnan(v)) else "—"

    print()
    print(sep)
    print(f"  SPY AI  –  Range-Gate Comparison  |  threshold={threshold}")
    print(sep)
    print(f"  {'Gate':<24}  {'Sub-N':>6}  {'OOS AUC':>8}  {'Trades':>6}  "
          f"{'WinRate':>7}  {'Sharpe':>7}  {'MaxDD':>7}")
    print(dash)

    for r in results:
        lbl    = GATE_LABELS.get(r["gate_pct"], f">{r['gate_pct']*100:.0f}th pct")
        marker = " ◀ BEST" if r is best else ""
        print(
            f"  {lbl:<24}  {r['subset_n']:>6}  {r['auc']:>8}  "
            f"{r['n_trades']:>6}  "
            f"{_flt(r['win_rate'], pct=True):>7}  "
            f"{_flt(r['sharpe']):>7}  "
            f"{_flt(r['max_dd']):>7}"
            f"{marker}"
        )

    print(sep)

    # Percentile thresholds
    non_zero = [r for r in results if r["gate_pct"] > 0.0]
    if non_zero:
        print("\n  Absolute range thresholds (from OOS predictions):")
        for r in non_zero:
            label = GATE_LABELS.get(r["gate_pct"], "")
            print(f"    {label:<24}  threshold = {r['range_thresh']:.6f}  "
                  f"({r['range_thresh'] * 570:.2f} pts on $570 SPY)")

    best_lbl = GATE_LABELS.get(best["gate_pct"], str(best["gate_pct"]))
    print(f"\n  ── Recommendation ──")
    print(f"  Best gate : {best_lbl}")
    if best["gate_pct"] == 0.0:
        print("  No range filter improves results; run with no gate (or widen threshold).")
    else:
        print(f"  Set --range-percentile {best['gate_pct']} in live_loop.py / run_live_prediction.py")
        print(f"  Absolute min-range for live use: {best['range_thresh']:.6f}")
    print()


# ── calibration save ──────────────────────────────────────────────────────────

def _save_calibration(results: list[dict], best: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "percentiles": {
            str(int(r["gate_pct"] * 100)): round(r["range_thresh"], 8)
            for r in results if r["gate_pct"] > 0.0
        },
        "recommended_percentile": best["gate_pct"],
        "recommended_threshold":  round(best["range_thresh"], 8) if best["gate_pct"] > 0.0 else 0.0,
    }
    CALIBRATION_FILE.write_text(json.dumps(payload, indent=2))
    print(f"  Calibration saved → {CALIBRATION_FILE}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    period:           str   = "60d",
    interval:         str   = "5m",
    threshold:        float = DIR_PROB_THRESHOLD,
    data_file:        str | None = None,
    save_calibration: bool  = True,
) -> None:

    print("═" * 78)
    print("  SPY AI  –  Regime-Aware Range Gate Analysis")
    print(f"  interval={interval}  period={period}  threshold={threshold}")
    print("═" * 78)

    # 1. Load data ─────────────────────────────────────────────────────────────
    if data_file:
        print(f"\n  Loading bars from {data_file} …")
        df_raw = load_from_file(data_file)
    else:
        print(f"\n  Downloading {TICKER} {interval} bars ({period}) …")
        df_raw = load_from_yfinance(ticker=TICKER, interval=interval, period=period)

    print(f"  Loaded {len(df_raw)} bars  "
          f"{df_raw.index[0].date()} → {df_raw.index[-1].date()}")

    # 2. Build full dataset ────────────────────────────────────────────────────
    print("\n  Building features + labels …")
    df_model = build_dataset(df_raw)
    feature_cols = [c for c in df_model.columns if c not in ("y_dir", "y_range")]
    print(f"  {len(df_model)} rows  |  {len(feature_cols)} features")

    # 3. Run walk-forward CV once to get OOS dir + range predictions ───────────
    print("\n  Running walk-forward CV (single pass for all gates) …")
    wf = _suppress(walk_forward_cv, df_model)

    if not wf.get("fold_results"):
        print("  ⚠  No folds produced – insufficient data.")
        return

    oos_dir_proba  = np.array(wf["oos_dir_proba"])
    oos_dir_true   = np.array(wf["oos_dir_true"])
    oos_range_pred = np.array(wf["oos_range_pred"])
    oos_index      = wf["oos_index"]

    n_folds = len(wf["fold_results"])
    n_oos   = len(oos_dir_proba)
    print(f"  OOS predictions: {n_oos} rows across {n_folds} fold(s)")

    # 4. Compute percentile thresholds from OOS range predictions ─────────────
    pct_thresholds: dict[float, float] = {}
    for gate in GATES:
        if gate > 0.0:
            pct_thresholds[gate] = float(np.percentile(oos_range_pred, gate * 100))
    pct_thresholds[0.0] = 0.0

    print("\n  Range percentile thresholds (from OOS predictions):")
    for gate in GATES:
        if gate > 0.0:
            v = pct_thresholds[gate]
            print(f"    {int(gate*100):2d}th pct  →  {v:.6f}  ({v*570:.2f} pts on $570)")

    # 5. Evaluate each gate ────────────────────────────────────────────────────
    print("\n  Evaluating gates …")
    results = []
    for gate in GATES:
        r = _evaluate_gate(
            gate_pct=gate,
            range_thresh=pct_thresholds[gate],
            oos_dir_proba=oos_dir_proba,
            oos_dir_true=oos_dir_true,
            oos_range_pred=oos_range_pred,
            oos_index=oos_index,
            df_raw=df_raw,
            threshold=threshold,
        )
        results.append(r)
        lbl = GATE_LABELS.get(gate, f">{gate*100:.0f}th")
        print(f"    {lbl:<24}  trades={r['n_trades']}  "
              f"AUC={r['auc']}  sharpe={r['sharpe']:.4f if isinstance(r['sharpe'], float) and not np.isnan(r['sharpe']) else 'N/A'}")

    # 6. Recommend best gate ───────────────────────────────────────────────────
    best = _recommend(results)

    # 7. Print table ───────────────────────────────────────────────────────────
    _print_table(results, threshold, best)

    # 8. Save calibration file ─────────────────────────────────────────────────
    if save_calibration:
        _save_calibration(results, best)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate SPY direction model under range-percentile gates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--period",    default="60d")
    p.add_argument("--interval",  default="5m")
    p.add_argument("--threshold", default=DIR_PROB_THRESHOLD, type=float)
    p.add_argument("--data-file", default=None)
    p.add_argument("--no-save-calibration", action="store_true",
                   help="Skip saving range_percentiles.json")
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
    main(
        period=args.period,
        interval=args.interval,
        threshold=args.threshold,
        data_file=args.data_file,
        save_calibration=not args.no_save_calibration,
    )

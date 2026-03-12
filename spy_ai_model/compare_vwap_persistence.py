"""
compare_vwap_persistence.py
────────────────────────────
Side-by-side walk-forward evaluation of two upgrades on top of the
39-feature session-enhanced set:

  Upgrade 1 – VWAP regime features (4 new columns)
  ─────────────────────────────────────────────────
    vwap_reclaim_flag        : 1 when price crosses back above VWAP
    vwap_loss_flag           : 1 when price drops below VWAP
    vwap_trend_strength      : rolling mean of sign(close−vwap) ∈ [−1, +1]
    vwap_distance_percentile : rolling %-rank of |dist_vwap| vs. past 20 bars

  Upgrade 2 – Signal persistence filter
  ──────────────────────────────────────
    Only generate a LONG_BIAS trade if the model fires the signal on two
    consecutive bars (persist_n=2).  This cuts whipsaw entries at the cost
    of fewer trades.

Comparison setups
─────────────────
  Baseline : 39 features (current session-enhanced set), no persistence filter
  Enhanced : 43 features (39 + 4 VWAP regime),          2-bar persistence filter

Reported metrics (OOS)
───────────────────────
  AUC, Log-loss, Brier, overfit gap, trades, win rate, Sharpe, max drawdown

Usage
─────
  python compare_vwap_persistence.py
  python compare_vwap_persistence.py --threshold 0.55 --period 60d
  python compare_vwap_persistence.py --data-file path/spy_5m.parquet --save-best
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    DIR_PROB_THRESHOLD, HORIZON_DIR, TRANSACTION_COST_BP,
    TICKER, MODEL_DIR, REPORT_DIR,
)
from data.data_loader import load_from_yfinance, load_from_file
from data.dataset_builder import build_dataset, split_features_labels
from evaluation.walk_forward import walk_forward_cv
from backtest.strategy_simulation import run_backtest
from models.train_direction import train_direction_model, predict_direction_proba, save_direction_model
from models.train_range import train_range_model, predict_range, save_range_model
from features.feature_engineering import NEW_SESSION_FEATURES, NEW_VWAP_FEATURES

logger = logging.getLogger(__name__)

# All 10 session features that were added in the previous round
_ALL_PREVIOUS_NEW = NEW_SESSION_FEATURES  # 10 features


# ── helpers ───────────────────────────────────────────────────────────────────

def _suppress(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with stdout/stderr suppressed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn(*args, **kwargs)


def _oos_metrics(wf: dict) -> dict:
    y_true = wf["oos_dir_true"]
    y_prob = wf["oos_dir_proba"]
    return {
        "auc":     round(roc_auc_score(y_true, y_prob), 4),
        "logloss": round(log_loss(y_true, y_prob), 4),
        "brier":   round(brier_score_loss(y_true, y_prob), 4),
        "n_folds": len(wf["fold_results"]),
        "n_oos":   len(y_true),
    }


def _overfit_gap(df_model: pd.DataFrame, feature_cols: list[str]) -> tuple[float, object, pd.DataFrame]:
    val_size = max(50, int(len(df_model) * 0.10))
    df_tr = df_model.iloc[:-val_size]
    df_vl = df_model.iloc[-val_size:]

    X_tr, y_tr, _ = split_features_labels(df_tr[feature_cols + ["y_dir", "y_range"]])
    X_vl, y_vl, _ = split_features_labels(df_vl[feature_cols + ["y_dir", "y_range"]])

    model, fi = _suppress(train_direction_model, X_tr, y_tr, X_vl, y_vl)
    val_proba  = predict_direction_proba(model, X_vl)
    val_auc    = round(roc_auc_score(y_vl, val_proba), 4)
    return val_auc, model, fi


def _run_setup(
    label:        str,
    df_model:     pd.DataFrame,
    feature_cols: list[str],
    df_raw:       pd.DataFrame,
    threshold:    float,
    persist_n:    int  = 1,
    save_models:  bool = False,
) -> dict:
    """
    Full evaluation pipeline for a given feature set and persistence setting:
      walk-forward CV → backtest (with persist_n) → final model (overfit gap).
    """
    persist_str = f"persist={persist_n}" if persist_n > 1 else "no persist"
    print(f"\n  ── {label}  ({len(feature_cols)} features, {persist_str}) ──", flush=True)

    df_subset = df_model[feature_cols + ["y_dir", "y_range"]]

    print("    Running walk-forward CV …", flush=True)
    wf = _suppress(walk_forward_cv, df_subset)

    if not wf["fold_results"]:
        print("    ⚠  No folds produced – insufficient data.", flush=True)
        return {"label": label, "n_features": len(feature_cols), "error": "no folds"}

    metrics = _oos_metrics(wf)

    print(f"    Running backtest (persist_n={persist_n}) …", flush=True)
    bt = run_backtest(
        df_raw,
        wf["oos_dir_proba"],
        wf["oos_index"],
        threshold=threshold,
        cost_bp=TRANSACTION_COST_BP,
        hold_bars=HORIZON_DIR,
        persist_n=persist_n,
    )
    bt_summary = bt.get("summary", {})

    print("    Training final model for overfit gap …", flush=True)
    final_val_auc, final_model, fi = _overfit_gap(df_model, feature_cols)
    overfit_gap = round(final_val_auc - metrics["auc"], 4)

    if save_models:
        val_size    = max(50, int(len(df_subset) * 0.10))
        df_tr       = df_subset.iloc[:-val_size]
        df_vl       = df_subset.iloc[-val_size:]
        Xr_tr, _, yr_tr = split_features_labels(df_tr)
        Xr_vl, _, yr_vl = split_features_labels(df_vl)
        range_model, r_fi = _suppress(train_range_model, Xr_tr, yr_tr, Xr_vl, yr_vl)
        save_direction_model(final_model)
        save_range_model(range_model)
        fi.to_csv(MODEL_DIR / "final_direction_feature_importance.csv", index=False)
        r_fi.to_csv(MODEL_DIR / "final_range_feature_importance.csv", index=False)
        print(f"    Models saved to {MODEL_DIR}", flush=True)

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            for name, df_fi in [("direction", fi), ("range", r_fi)]:
                top = df_fi.head(20).sort_values("importance")
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.barh(top["feature"], top["importance"])
                ax.set_title(f"{name.title()} model – top-20 feature importance")
                ax.set_xlabel("Importance")
                fig.tight_layout()
                fig.savefig(REPORT_DIR / f"{name}_feature_importance_vwap.png", dpi=120)
                plt.close(fig)
        except Exception as exc:
            logger.warning("Could not save feature importance plots: %s", exc)

    print(f"    OOS AUC={metrics['auc']:.4f}  gap={overfit_gap:+.4f}  "
          f"trades={bt_summary.get('n_trades', 0)}", flush=True)

    return {
        "label":         label,
        "n_features":    len(feature_cols),
        "persist_n":     persist_n,
        "oos_auc":       metrics["auc"],
        "oos_logloss":   metrics["logloss"],
        "oos_brier":     metrics["brier"],
        "n_folds":       metrics["n_folds"],
        "n_oos":         metrics["n_oos"],
        "final_val_auc": final_val_auc,
        "overfit_gap":   overfit_gap,
        "n_trades":      bt_summary.get("n_trades", 0),
        "win_rate":      bt_summary.get("win_rate", float("nan")),
        "sharpe":        bt_summary.get("sharpe_ratio", float("nan")),
        "max_dd":        bt_summary.get("max_drawdown", float("nan")),
        "total_return":  bt_summary.get("total_return", float("nan")),
        "fi":            fi,
    }


# ── comparison table ──────────────────────────────────────────────────────────

def _print_comparison(base: dict, enhanced: dict, threshold: float) -> None:
    sep  = "═" * 72
    dash = "─" * 72

    def _delta(b, e, pct=False, lower_better=False):
        if isinstance(b, (int, float)) and isinstance(e, (int, float)) and not (
            np.isnan(float(b)) or np.isnan(float(e))
        ):
            d    = float(e) - float(b)
            sign = "+" if d >= 0 else ""
            txt  = f"{sign}{d:.4f}" if not pct else f"{sign}{d*100:.1f}pp"
            better = (d > 0) != lower_better
            marker = " ▲" if better and abs(d) > 1e-4 else (" ▼" if not better and abs(d) > 1e-4 else "")
            return txt + marker
        return "—"

    def _fmt(v, pct=False):
        if isinstance(v, float) and not np.isnan(v):
            return f"{v*100:.1f}%" if pct else f"{v:.4f}"
        return str(v) if v != float("nan") else "—"

    print()
    print(sep)
    print(f"  VWAP Regime + Persistence Comparison  |  threshold={threshold}")
    print(sep)
    print(f"  {'Metric':<32}  {'Baseline':>10}  {'Enhanced':>10}  {'Delta':>12}")
    print(dash)

    base_persist_str     = f"persist={base['persist_n']}" if base["persist_n"] > 1 else "none"
    enhanced_persist_str = f"persist={enhanced['persist_n']}" if enhanced["persist_n"] > 1 else "none"
    print(f"  {'Features':<32}  {base['n_features']:>10}  {enhanced['n_features']:>10}  {'—':>12}")
    print(f"  {'Signal persistence':<32}  {base_persist_str:>10}  {enhanced_persist_str:>10}  {'—':>12}")
    print(f"  {'Walk-forward folds':<32}  {base['n_folds']:>10}  {enhanced['n_folds']:>10}  {'—':>12}")
    print(f"  {'OOS rows':<32}  {base['n_oos']:>10}  {enhanced['n_oos']:>10}  {'—':>12}")
    print(dash)

    rows = [
        ("OOS AUC",       base["oos_auc"],       enhanced["oos_auc"],       False, False),
        ("OOS Log-loss",  base["oos_logloss"],   enhanced["oos_logloss"],   False, True),
        ("OOS Brier",     base["oos_brier"],     enhanced["oos_brier"],     False, True),
        ("Final val AUC", base["final_val_auc"], enhanced["final_val_auc"], False, False),
        ("Overfit gap",   base["overfit_gap"],   enhanced["overfit_gap"],   False, True),
        (None, None, None, None, None),
        ("Backtest trades",  base["n_trades"],     enhanced["n_trades"],     False, False),
        ("Win rate",         base["win_rate"],     enhanced["win_rate"],     True,  False),
        ("Sharpe ratio",     base["sharpe"],       enhanced["sharpe"],       False, False),
        ("Max drawdown",     base["max_dd"],       enhanced["max_dd"],       False, True),
        ("Total return",     base["total_return"], enhanced["total_return"], True,  False),
    ]

    for row in rows:
        if row[0] is None:
            print(dash)
            continue
        lbl, bv, ev, pct, lb = row
        bstr = _fmt(bv, pct) if not isinstance(bv, int) else str(bv)
        estr = _fmt(ev, pct) if not isinstance(ev, int) else str(ev)
        dstr = _delta(bv, ev, pct, lb) if bv != ev else "—"
        print(f"  {lbl:<32}  {bstr:>10}  {estr:>10}  {dstr:>12}")

    print(sep)

    # ── Verdict ───────────────────────────────────────────────────────────────
    auc_delta    = float(enhanced["oos_auc"]) - float(base["oos_auc"])
    sharpe_b     = float(base["sharpe"])     if not np.isnan(float(base["sharpe"]))     else 0.0
    sharpe_e     = float(enhanced["sharpe"]) if not np.isnan(float(enhanced["sharpe"])) else 0.0
    sharpe_delta = sharpe_e - sharpe_b
    gap_delta    = float(enhanced["overfit_gap"]) - float(base["overfit_gap"])

    if auc_delta > 0.005 or sharpe_delta > 0.5:
        verdict = "IMPROVED  – VWAP regime features + persistence filter add measurable signal."
    elif auc_delta < -0.005 or sharpe_delta < -0.5:
        verdict = "DEGRADED  – upgrades hurt OOS performance."
    else:
        verdict = "NEUTRAL   – no material change in OOS performance."

    gap_note = ""
    if gap_delta < -0.02:
        gap_note = " Overfit gap reduced (better generalisation)."
    elif gap_delta > 0.02:
        gap_note = " Overfit gap increased (watch for over-fitting)."

    win_delta = float(enhanced["win_rate"]) - float(base["win_rate"])
    trade_delta = int(enhanced["n_trades"]) - int(base["n_trades"])
    filter_note = ""
    if enhanced["persist_n"] > 1:
        filter_note = (
            f"\n  Persistence filter: {trade_delta:+d} trades, "
            f"win rate {win_delta*100:+.1f}pp."
        )

    print(f"\n  Verdict: {verdict}{gap_note}{filter_note}")
    print(sep)


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    period:    str   = "60d",
    interval:  str   = "5m",
    threshold: float = DIR_PROB_THRESHOLD,
    data_file: str | None = None,
    save_best: bool  = False,
    persist_n: int   = 2,
) -> None:

    print("═" * 72)
    print("  SPY AI  –  VWAP Regime + Signal Persistence Comparison")
    print(f"  interval={interval}  period={period}  threshold={threshold}  persist_n={persist_n}")
    print("═" * 72)

    # 1. Load data ─────────────────────────────────────────────────────────────
    if data_file:
        print(f"\n  Loading bars from {data_file} …")
        df_raw = load_from_file(data_file)
    else:
        print(f"\n  Downloading {TICKER} {interval} bars ({period}) …")
        df_raw = load_from_yfinance(ticker=TICKER, interval=interval, period=period)

    print(f"  Loaded {len(df_raw)} bars  "
          f"{df_raw.index[0].date()} → {df_raw.index[-1].date()}")

    # 2. Build full dataset (all 43 features) ──────────────────────────────────
    print("\n  Building features + labels …")
    df_model = build_dataset(df_raw)
    all_feature_cols = [c for c in df_model.columns if c not in ("y_dir", "y_range")]

    # Baseline: the 39 session-enhanced features from the previous comparison
    # (all features EXCEPT the new VWAP regime ones)
    baseline_cols = [c for c in all_feature_cols if c not in NEW_VWAP_FEATURES]
    enhanced_cols = all_feature_cols   # 43 features

    print(f"  Total rows           : {len(df_model)}")
    print(f"  Baseline features    : {len(baseline_cols)}  (39-feature session set, no persistence)")
    print(f"  Enhanced features    : {len(enhanced_cols)}  (+ {len(NEW_VWAP_FEATURES)} VWAP regime, persist_n={persist_n})")
    print(f"  New VWAP regime cols : {NEW_VWAP_FEATURES}")

    # 3. Run both setups ────────────────────────────────────────────────────────
    base = _run_setup(
        "Baseline (39 feat, no persist)",
        df_model, baseline_cols, df_raw,
        threshold=threshold, persist_n=1,
    )
    enhanced = _run_setup(
        f"Enhanced (43 feat, persist={persist_n})",
        df_model, enhanced_cols, df_raw,
        threshold=threshold, persist_n=persist_n,
        save_models=save_best,
    )

    if "error" in base or "error" in enhanced:
        print("\n  One or both setups failed – check logs.")
        return

    # 4. Comparison table ───────────────────────────────────────────────────────
    _print_comparison(base, enhanced, threshold)

    # 5. Feature importance for enhanced model ─────────────────────────────────
    print("\n  Top-15 features by importance (enhanced model):")
    fi_top = enhanced["fi"].head(15)
    for _, r in fi_top.iterrows():
        if r["feature"] in NEW_VWAP_FEATURES:
            marker = " ◀ VWAP NEW"
        elif r["feature"] in NEW_SESSION_FEATURES:
            marker = " ◀ SESSION"
        else:
            marker = ""
        print(f"    {r['feature']:<38}  {r['importance']:>8.0f}{marker}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare baseline vs. VWAP-regime + persistence-filtered SPY model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--period",    default="60d",               help="yfinance period string")
    p.add_argument("--interval",  default="5m",                help="Bar interval")
    p.add_argument("--threshold", default=DIR_PROB_THRESHOLD,  type=float,
                   help="Direction probability threshold for backtest")
    p.add_argument("--persist-n", default=2,                   type=int,
                   help="Consecutive bars required to trigger an entry (>=1)")
    p.add_argument("--data-file", default=None,
                   help="Use a local CSV/Parquet file instead of yfinance")
    p.add_argument("--save-best", action="store_true",
                   help="Save the enhanced model artifacts when comparison completes")
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
        save_best=args.save_best,
        persist_n=args.persist_n,
    )

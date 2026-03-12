"""
compare_session_features.py
───────────────────────────
Side-by-side walk-forward evaluation of the baseline feature set (29 features)
against the enhanced set that adds 10 session-aware features (39 total).

New session features
────────────────────
  minutes_to_close       – countdown to 4 PM close
  opening_range_high     – ORB high normalised by current close
  opening_range_low      – ORB low  normalised by current close
  dist_orb_high          – (orb_h – close) / close  (signed distance)
  dist_orb_low           – (close – orb_l) / close  (signed distance)
  orb_break_flag         – +1 above ORB high / –1 below ORB low / 0 inside
  orb_reject_flag        – 1 if wick tested ORB level but close stayed inside
  power_hour_flag        – 1 during 3:00–4:00 PM ET
  lunch_hour_flag        – 1 during 11:30 AM–1:00 PM ET
  day_of_week            – 0=Mon … 4=Fri

Reported metrics (both setups, OOS)
────────────────────────────────────
  AUC, Log-loss, Brier, OOS rows, folds, overfit gap
  Trades, win rate, Sharpe, max drawdown (from backtest simulation)

Usage
─────
  python compare_session_features.py
  python compare_session_features.py --threshold 0.528 --period 60d
  python compare_session_features.py --data-file path/spy_5m.parquet
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
from features.feature_engineering import NEW_SESSION_FEATURES

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _suppress(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with stdout/stderr suppressed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn(*args, **kwargs)


def _oos_metrics(wf: dict) -> dict:
    """Compute OOS direction model metrics from walk-forward results."""
    y_true = wf["oos_dir_true"]
    y_prob = wf["oos_dir_proba"]
    return {
        "auc":      round(roc_auc_score(y_true, y_prob), 4),
        "logloss":  round(log_loss(y_true, y_prob), 4),
        "brier":    round(brier_score_loss(y_true, y_prob), 4),
        "n_folds":  len(wf["fold_results"]),
        "n_oos":    len(y_true),
    }


def _overfit_gap(df_model: pd.DataFrame, feature_cols: list[str]) -> tuple[float, list]:
    """
    Train a final model on the last 90 % of df_model, validate on the last 10 %.
    Returns (val_auc, feature_importance_df).

    This mirrors the logic in main_pipeline.step_save_final_models().
    """
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
    save_models:  bool = False,
) -> dict:
    """
    Full evaluation pipeline for a given feature set:
      walk-forward CV → backtest → final model (overfit gap).

    Returns a flat summary dict.
    """
    print(f"\n  ── {label}  ({len(feature_cols)} features) ──", flush=True)

    df_subset = df_model[feature_cols + ["y_dir", "y_range"]]

    # Walk-forward CV (suppressed training logs)
    print("    Running walk-forward CV …", flush=True)
    wf = _suppress(walk_forward_cv, df_subset)

    if not wf["fold_results"]:
        print("    ⚠  No folds produced – insufficient data.", flush=True)
        return {"label": label, "n_features": len(feature_cols), "error": "no folds"}

    metrics = _oos_metrics(wf)

    # Backtest
    print("    Running backtest …", flush=True)
    bt = run_backtest(
        df_raw,
        wf["oos_dir_proba"],
        wf["oos_index"],
        threshold=threshold,
        cost_bp=TRANSACTION_COST_BP,
        hold_bars=HORIZON_DIR,
    )
    bt_summary = bt.get("summary", {})

    # Final model (overfit gap)
    print("    Training final model for overfit gap …", flush=True)
    final_val_auc, final_model, fi = _overfit_gap(df_model, feature_cols)
    overfit_gap = round(final_val_auc - metrics["auc"], 4)

    # Optionally save the enhanced model
    if save_models:
        # Also train range model
        val_size   = max(50, int(len(df_subset) * 0.10))
        df_tr      = df_subset.iloc[:-val_size]
        df_vl      = df_subset.iloc[-val_size:]
        Xr_tr, _, yr_tr = split_features_labels(df_tr)
        Xr_vl, _, yr_vl = split_features_labels(df_vl)
        range_model, r_fi = _suppress(train_range_model, Xr_tr, yr_tr, Xr_vl, yr_vl)
        save_direction_model(final_model)
        save_range_model(range_model)
        fi.to_csv(MODEL_DIR / "final_direction_feature_importance.csv", index=False)
        r_fi.to_csv(MODEL_DIR / "final_range_feature_importance.csv", index=False)
        print(f"    Models saved to {MODEL_DIR}", flush=True)

        # Save feature importance to reports dir
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
                fig.savefig(REPORT_DIR / f"{name}_feature_importance.png", dpi=120)
                plt.close(fig)
        except Exception as exc:
            logger.warning("Could not save feature importance plots: %s", exc)

    print(f"    OOS AUC={metrics['auc']:.4f}  gap={overfit_gap:+.4f}  "
          f"trades={bt_summary.get('n_trades', 0)}", flush=True)

    return {
        "label":          label,
        "n_features":     len(feature_cols),
        "oos_auc":        metrics["auc"],
        "oos_logloss":    metrics["logloss"],
        "oos_brier":      metrics["brier"],
        "n_folds":        metrics["n_folds"],
        "n_oos":          metrics["n_oos"],
        "final_val_auc":  final_val_auc,
        "overfit_gap":    overfit_gap,
        "n_trades":       bt_summary.get("n_trades", 0),
        "win_rate":       bt_summary.get("win_rate", float("nan")),
        "sharpe":         bt_summary.get("sharpe_ratio", float("nan")),
        "max_dd":         bt_summary.get("max_drawdown", float("nan")),
        "total_return":   bt_summary.get("total_return", float("nan")),
        "fi":             fi,
    }


# ── comparison table ──────────────────────────────────────────────────────────

def _print_comparison(base: dict, enhanced: dict, threshold: float) -> None:
    sep  = "═" * 72
    dash = "─" * 72

    def _delta(b, e, pct=False, lower_better=False):
        """Format the delta between base and enhanced values."""
        if isinstance(b, float) and isinstance(e, float) and not (
            np.isnan(b) or np.isnan(e)
        ):
            d = e - b
            sign = "+" if d >= 0 else ""
            txt = f"{sign}{d:.4f}" if not pct else f"{sign}{d*100:.1f}pp"
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
    print(f"  Session-Feature Comparison  |  threshold={threshold}")
    print(sep)
    print(f"  {'Metric':<30}  {'Baseline':>10}  {'Enhanced':>10}  {'Delta':>14}")
    print(dash)

    rows = [
        ("Features",          base["n_features"],     enhanced["n_features"],   False, False),
        ("Walk-forward folds", base["n_folds"],        enhanced["n_folds"],      False, False),
        ("OOS rows",           base["n_oos"],          enhanced["n_oos"],        False, False),
        (None, None, None, None, None),
        ("OOS AUC",            base["oos_auc"],        enhanced["oos_auc"],      False, False),
        ("OOS Log-loss",       base["oos_logloss"],    enhanced["oos_logloss"],  False, True),
        ("OOS Brier",          base["oos_brier"],      enhanced["oos_brier"],    False, True),
        ("Final val AUC",      base["final_val_auc"],  enhanced["final_val_auc"],False, False),
        ("Overfit gap",        base["overfit_gap"],    enhanced["overfit_gap"],  False, True),
        (None, None, None, None, None),
        ("Backtest trades",    base["n_trades"],       enhanced["n_trades"],     False, False),
        ("Win rate",           base["win_rate"],       enhanced["win_rate"],     True,  False),
        ("Sharpe ratio",       base["sharpe"],         enhanced["sharpe"],       False, False),
        ("Max drawdown",       base["max_dd"],         enhanced["max_dd"],       False, True),
        ("Total return",       base["total_return"],   enhanced["total_return"], True,  False),
    ]

    for row in rows:
        if row[0] is None:
            print(dash)
            continue
        label, bv, ev, pct, lb = row
        bstr = _fmt(bv, pct) if not isinstance(bv, int) else str(bv)
        estr = _fmt(ev, pct) if not isinstance(ev, int) else str(ev)
        dstr = _delta(bv, ev, pct, lb) if isinstance(bv, (int, float)) and bv != ev else "—"
        print(f"  {label:<30}  {bstr:>10}  {estr:>10}  {dstr:>14}")

    print(sep)

    # Verdict
    auc_delta = enhanced["oos_auc"] - base["oos_auc"]
    sharpe_delta = enhanced["sharpe"] - base["sharpe"] if not (
        np.isnan(enhanced["sharpe"]) or np.isnan(base["sharpe"])
    ) else 0.0
    gap_delta = enhanced["overfit_gap"] - base["overfit_gap"]

    if auc_delta > 0.005 or sharpe_delta > 0.5:
        verdict = "IMPROVED  – session features add measurable signal."
    elif auc_delta < -0.005 or sharpe_delta < -0.5:
        verdict = "DEGRADED  – session features hurt OOS performance."
    else:
        verdict = "NEUTRAL   – no material change in OOS performance."

    gap_note = ""
    if gap_delta < -0.02:
        gap_note = " Overfit gap reduced (better generalisation)."
    elif gap_delta > 0.02:
        gap_note = " Overfit gap increased (watch for over-fitting)."

    print(f"\n  Verdict: {verdict}{gap_note}")
    print(sep)


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    period:     str   = "60d",
    interval:   str   = "5m",
    threshold:  float = DIR_PROB_THRESHOLD,
    data_file:  str | None = None,
    save_best:  bool  = False,
) -> None:

    print("═" * 72)
    print("  SPY AI  –  Session Feature Comparison")
    print(f"  interval={interval}  period={period}  threshold={threshold}")
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

    # 2. Build full dataset (with session features) ────────────────────────────
    print("\n  Building features + labels …")
    df_model = build_dataset(df_raw)
    all_feature_cols = [c for c in df_model.columns if c not in ("y_dir", "y_range")]
    baseline_cols    = [c for c in all_feature_cols if c not in NEW_SESSION_FEATURES]
    enhanced_cols    = all_feature_cols

    print(f"  Total rows: {len(df_model)}")
    print(f"  Baseline features : {len(baseline_cols)}")
    print(f"  Enhanced features : {len(enhanced_cols)}")
    print(f"  New session cols  : {NEW_SESSION_FEATURES}")

    # 3. Run both setups ────────────────────────────────────────────────────────
    base     = _run_setup("Baseline", df_model, baseline_cols, df_raw, threshold)
    enhanced = _run_setup(
        "Enhanced", df_model, enhanced_cols, df_raw, threshold,
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
        marker = " ◀ NEW" if r["feature"] in NEW_SESSION_FEATURES else ""
        print(f"    {r['feature']:<35}  {r['importance']:>8.0f}{marker}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare baseline vs. session-enhanced SPY feature set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--period",    default="60d", help="yfinance period string")
    p.add_argument("--interval",  default="5m",  help="Bar interval")
    p.add_argument("--threshold", default=DIR_PROB_THRESHOLD, type=float,
                   help="Direction probability threshold for backtest")
    p.add_argument("--data-file", default=None,
                   help="Use a local CSV/Parquet file instead of yfinance")
    p.add_argument("--save-best", action="store_true",
                   help="Save the enhanced model artifacts if comparison completes")
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
    )

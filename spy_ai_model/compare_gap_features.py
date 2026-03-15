"""
compare_gap_features.py
───────────────────────
Side-by-side walk-forward comparison of the BASELINE model (43 features)
against the ENHANCED model (+9 overnight-gap / prior-day context features).

What is compared
────────────────
  CLASSIFICATION (direction model, OOS):
    • AUC (ROC)
    • Log-loss
    • Brier score
    • Overfit gap  =  mean(in-sample AUC per fold) − mean(OOS AUC per fold)
      A smaller gap means less overfitting; identical overfit gaps mean the
      new features do not introduce additional overfit even if they improve IS.

  BACKTEST (long-only, fixed threshold):
    • Trade count
    • Win rate
    • Annualised return
    • Sharpe ratio (trade-level, annualised)
    • Max drawdown

  NEW FEATURE IMPORTANCE:
    Ranks the 9 gap features within the full enhanced feature set and
    categorises them as HIGH / MEDIUM / LOW based on percentile rank.
    Explains which new features matter most based on average split gain
    across all walk-forward folds.

Usage
─────
  # Fully offline – synthetic data (fast, good for CI):
  python compare_gap_features.py --synthetic

  # Live data download (requires internet):
  python compare_gap_features.py

  # Use a cached CSV (skips network):
  python compare_gap_features.py --data-file data/raw/spy_5m.csv

  # Longer horizon:
  python compare_gap_features.py --synthetic --horizon-dir 60 --horizon-range 12
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
    level=logging.WARNING,   # suppress verbose LightGBM fold output
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("compare_gap_features")

# ── project imports ───────────────────────────────────────────────────────────
from config import REPORT_DIR, TRAIN_DAYS, VAL_DAYS, STEP_DAYS
from data.data_loader import load_synthetic, load_from_file, load_bars
from data.dataset_builder import build_dataset, split_features_labels
from features.feature_engineering import NEW_GAP_FEATURES
from models.train_direction import train_direction_model, predict_direction_proba
from models.train_range import train_range_model, predict_range
from backtest.strategy_simulation import run_backtest
from evaluation.metrics_report import generate_report

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline vs Enhanced gap-feature comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (offline / CI mode)",
    )
    p.add_argument(
        "--synth-days", default=252, type=int, metavar="N",
        help="Trading days for synthetic generation",
    )
    p.add_argument(
        "--data-file", default=None, metavar="PATH",
        help="Path to a pre-downloaded 5m CSV (skips network)",
    )
    p.add_argument(
        "--period", default="60d",
        help="yfinance / Polygon period when downloading live data",
    )
    p.add_argument(
        "--interval", default="5m",
        help="Bar interval for live download",
    )
    p.add_argument(
        "--horizon-dir", default=60, type=int, metavar="N",
        help="Bars ahead for the direction label",
    )
    p.add_argument(
        "--horizon-range", default=12, type=int, metavar="N",
        help="Bars ahead for the range label",
    )
    p.add_argument(
        "--threshold", default=0.530, type=float,
        help="Direction probability threshold for the backtest",
    )
    p.add_argument(
        "--provider", default="auto",
        choices=["auto", "polygon", "yfinance"],
        help="Market data provider for live download",
    )
    return p.parse_args()


# ── Walk-forward with in-sample AUC ──────────────────────────────────────────

def _make_date_folds(
    trading_dates: np.ndarray,
    train_days: int = TRAIN_DAYS,
    val_days:   int = VAL_DAYS,
    step_days:  int = STEP_DAYS,
) -> list[tuple]:
    """Identical fold logic to evaluation/walk_forward.py."""
    n     = len(trading_dates)
    folds = []
    start = 0
    while start + train_days + val_days <= n:
        te  = start + train_days
        ve  = te    + val_days
        folds.append((trading_dates[start:te], trading_dates[te:ve]))
        start += step_days
    return folds


def _wf_with_overfit(
    df_model:         pd.DataFrame,
    label:            str,
    direction_params: dict | None = None,
) -> dict:
    """
    Walk-forward CV that returns both OOS and IS (in-sample) metrics per fold.

    The overfit gap = mean(IS AUC) − mean(OOS AUC).  When this is the same
    for baseline and enhanced, the new features do not add extra overfit.

    Returns
    ───────
    dict with keys:
        oos_auc, oos_logloss, oos_brier,
        mean_is_auc, mean_oos_auc, overfit_gap,
        fold_is_auc, fold_oos_auc,
        oos_dir_proba, oos_dir_true, oos_range_pred, oos_range_true, oos_index,
        fi_dir   (pd.DataFrame – mean feature importance across folds),
        fi_range (pd.DataFrame – mean feature importance across folds),
        n_folds, n_oos_rows,
    """
    from sklearn.metrics import (
        roc_auc_score, log_loss, brier_score_loss,
    )

    trading_dates = np.sort(np.unique(df_model.index.normalize()))
    folds         = _make_date_folds(trading_dates)

    if not folds:
        n     = len(trading_dates)
        train = max(5, int(n * 0.60))
        val   = max(2, int(n * 0.25))
        step  = max(1, int(n * 0.15))
        folds = _make_date_folds(trading_dates, train, val, step)
        if not folds:
            raise ValueError(
                f"Not enough data for even one fold ({n} trading days available)."
            )

    X_all, y_dir_all, y_range_all = split_features_labels(df_model)

    fold_is_auc    = []
    fold_oos_auc   = []
    oos_proba_list = []
    oos_true_list  = []
    oos_rng_pred   = []
    oos_rng_true   = []
    oos_idx_list   = []
    fi_dir_list    = []
    fi_rng_list    = []

    print(f"  {label}: {len(folds)} fold(s) …", end="", flush=True)

    for train_dates, val_dates in folds:
        tr_mask = df_model.index.normalize().isin(train_dates)
        vl_mask = df_model.index.normalize().isin(val_dates)

        X_tr = X_all[tr_mask];  y_d_tr = y_dir_all[tr_mask];  y_r_tr = y_range_all[tr_mask]
        X_vl = X_all[vl_mask];  y_d_vl = y_dir_all[vl_mask];  y_r_vl = y_range_all[vl_mask]

        if len(X_tr) < 100 or len(X_vl) < 10:
            continue

        dir_model, fi_d = train_direction_model(
            X_tr, y_d_tr, X_vl, y_d_vl, params=direction_params
        )
        rng_model, fi_r = train_range_model(X_tr, y_r_tr, X_vl, y_r_vl)

        # In-sample AUC (train set)
        is_prob = predict_direction_proba(dir_model, X_tr)
        fold_is_auc.append(roc_auc_score(y_d_tr, is_prob))

        # OOS AUC (val set)
        oos_prob = predict_direction_proba(dir_model, X_vl)
        fold_oos_auc.append(roc_auc_score(y_d_vl, oos_prob))

        oos_proba_list.extend(oos_prob)
        oos_true_list.extend(y_d_vl.values)
        oos_rng_pred.extend(predict_range(rng_model, X_vl))
        oos_rng_true.extend(y_r_vl.values)
        oos_idx_list.extend(X_vl.index.tolist())
        fi_dir_list.append(fi_d)
        fi_rng_list.append(fi_r)
        print(".", end="", flush=True)

    print()   # newline after fold dots

    oos_proba_arr = np.array(oos_proba_list)
    oos_true_arr  = np.array(oos_true_list)

    oos_auc    = roc_auc_score(oos_true_arr, oos_proba_arr)
    oos_ll     = log_loss(oos_true_arr, oos_proba_arr)
    oos_brier  = brier_score_loss(oos_true_arr, oos_proba_arr)

    mean_is    = float(np.mean(fold_is_auc))  if fold_is_auc  else float("nan")
    mean_oos   = float(np.mean(fold_oos_auc)) if fold_oos_auc else float("nan")
    overfit_gap = mean_is - mean_oos

    def _mean_fi(fi_list: list[pd.DataFrame]) -> pd.DataFrame:
        if not fi_list:
            return pd.DataFrame(columns=["feature", "importance"])
        return (
            pd.concat(fi_list)
              .groupby("feature")["importance"]
              .mean()
              .reset_index()
              .sort_values("importance", ascending=False)
              .reset_index(drop=True)
        )

    return {
        "label":          label,
        "oos_auc":        oos_auc,
        "oos_logloss":    oos_ll,
        "oos_brier":      oos_brier,
        "mean_is_auc":    mean_is,
        "mean_oos_auc":   mean_oos,
        "overfit_gap":    overfit_gap,
        "fold_is_auc":    fold_is_auc,
        "fold_oos_auc":   fold_oos_auc,
        "oos_dir_proba":  oos_proba_arr,
        "oos_dir_true":   oos_true_arr,
        "oos_range_pred": np.array(oos_rng_pred),
        "oos_range_true": np.array(oos_rng_true),
        "oos_index":      pd.DatetimeIndex(oos_idx_list),
        "fi_dir":         _mean_fi(fi_dir_list),
        "fi_range":       _mean_fi(fi_rng_list),
        "n_folds":        len(fold_is_auc),
        "n_oos_rows":     len(oos_true_arr),
    }


# ── Backtest wrapper ──────────────────────────────────────────────────────────

def _run_backtest(df_raw: pd.DataFrame, wf: dict, threshold: float) -> dict:
    """Run the standard long-only backtest and return the summary dict."""
    bt = run_backtest(
        df_raw          = df_raw,
        oos_dir_proba   = wf["oos_dir_proba"],
        oos_index       = wf["oos_index"],
        threshold       = threshold,
        hold_bars       = 60,
        oos_range_pred  = wf["oos_range_pred"],
        range_threshold = 0.0,   # no range gate for a fair comparison
    )
    return bt.get("summary", {})


# ── Console output ────────────────────────────────────────────────────────────

def _fmt(val, fmt: str = ".4f") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "   —   "
    return format(val, fmt)


def _delta_str(base: float, enh: float, higher_is_better: bool = True) -> str:
    """Format the Δ column with a directional indicator."""
    if np.isnan(base) or np.isnan(enh):
        return "   —   "
    delta = enh - base
    sym   = "▲" if (delta > 0) == higher_is_better else "▼"
    if delta == 0:
        sym = "="
    return f"{delta:+.4f} {sym}"


def _print_comparison(
    base:      dict,
    enh:       dict,
    bt_base:   dict,
    bt_enh:    dict,
    threshold: float,
    n_bars:    int,
    n_sessions: int,
    gap_fi:    pd.DataFrame,     # feature importance for gap features only
    full_fi:   pd.DataFrame,     # full enhanced feature importance
) -> None:
    W   = 72
    SEP = "═" * W
    DIV = "─" * W
    print()
    print(SEP)
    print("  BASELINE vs ENHANCED: Overnight Gap + Prior-Day Context Features")
    print(f"  Data: {n_bars:,} bars  |  {n_sessions} sessions  |  {base['n_folds']} folds")
    print(SEP)

    # ── Classification metrics ────────────────────────────────────────────────
    print()
    print("  DIRECTION MODEL — OOS CLASSIFICATION METRICS")
    print(DIV)
    hdr = f"  {'Metric':<28}  {'Baseline':>10}  {'Enhanced':>10}  {'Δ':>12}"
    print(hdr)
    print(DIV)

    rows_cls = [
        ("OOS AUC (ROC)",    base["oos_auc"],     enh["oos_auc"],     True,  ".4f"),
        ("OOS Log-loss",     base["oos_logloss"],  enh["oos_logloss"],  False, ".4f"),
        ("OOS Brier score",  base["oos_brier"],    enh["oos_brier"],    False, ".4f"),
        ("Mean IS  AUC",     base["mean_is_auc"],  enh["mean_is_auc"],  True,  ".4f"),
        ("Mean OOS AUC",     base["mean_oos_auc"], enh["mean_oos_auc"], True,  ".4f"),
        ("Overfit gap (IS−OOS)", base["overfit_gap"], enh["overfit_gap"], False, ".4f"),
    ]
    for name, bv, ev, hib, fmt in rows_cls:
        ds = _delta_str(bv, ev, hib)
        print(f"  {name:<28}  {_fmt(bv, fmt):>10}  {_fmt(ev, fmt):>10}  {ds:>12}")
    print(DIV)

    # ── Backtest metrics ──────────────────────────────────────────────────────
    print()
    print(f"  BACKTEST METRICS (long-only, threshold={threshold:.3f})")
    print(DIV)
    print(hdr)
    print(DIV)

    def _bt(d: dict, key: str) -> float:
        return float(d.get(key, float("nan")))

    rows_bt = [
        ("Trade count",        _bt(bt_base, "n_trades"),          _bt(bt_enh, "n_trades"),          True,  ".0f"),
        ("Win rate",           _bt(bt_base, "win_rate"),           _bt(bt_enh, "win_rate"),           True,  ".4f"),
        ("Avg net PnL",        _bt(bt_base, "avg_net_pnl"),        _bt(bt_enh, "avg_net_pnl"),        True,  ".6f"),
        ("Annualised return",  _bt(bt_base, "annualised_return"),  _bt(bt_enh, "annualised_return"),  True,  ".4f"),
        ("Sharpe ratio",       _bt(bt_base, "sharpe_ratio"),       _bt(bt_enh, "sharpe_ratio"),       True,  ".4f"),
        ("Max drawdown",       _bt(bt_base, "max_drawdown"),       _bt(bt_enh, "max_drawdown"),       False, ".4f"),
    ]
    for name, bv, ev, hib, fmt in rows_bt:
        ds = _delta_str(bv, ev, hib)
        print(f"  {name:<28}  {_fmt(bv, fmt):>10}  {_fmt(ev, fmt):>10}  {ds:>12}")
    print(DIV)


def _print_gap_feature_analysis(
    gap_fi:    pd.DataFrame,
    full_fi:   pd.DataFrame,
    n_total:   int,
) -> None:
    """
    Print a ranked importance table for the 9 new gap features and explain
    which ones matter most with tier labels (HIGH / MEDIUM / LOW).
    """
    W   = 72
    SEP = "═" * W
    DIV = "─" * W

    # Importance percentile thresholds across ALL features
    all_imp = full_fi["importance"].values
    p75     = np.percentile(all_imp, 75)
    p25     = np.percentile(all_imp, 25)

    def _tier(imp: float) -> str:
        if imp >= p75:
            return "HIGH  ▓▓▓▓"
        if imp >= p25:
            return "MEDIUM ▓▓▓ "
        return "LOW    ▓   "

    # Rank each gap feature within the full sorted list
    rank_map = {row["feature"]: i + 1 for i, row in full_fi.iterrows()}

    print()
    print(SEP)
    print("  NEW FEATURE ANALYSIS — Overnight Gap / Prior-Day Context (9 features)")
    print(f"  Ranked within the full {n_total}-feature enhanced set")
    print(DIV)
    print(
        f"  {'Rank':>4}  {'Feature':<28}  {'Importance':>11}  {'Tier'}"
    )
    print(DIV)

    # Sort gap features by importance descending
    gap_sorted = gap_fi.sort_values("importance", ascending=False)
    for _, row in gap_sorted.iterrows():
        feat  = row["feature"]
        imp   = float(row["importance"])
        rank  = rank_map.get(feat, "?")
        tier  = _tier(imp)
        print(f"  {rank:>4}  {feat:<28}  {imp:>11.1f}  {tier}")

    print(DIV)

    # Narrative summary
    high_feats   = gap_sorted[gap_sorted["importance"] >= p75]["feature"].tolist()
    medium_feats = gap_sorted[
        (gap_sorted["importance"] >= p25) & (gap_sorted["importance"] < p75)
    ]["feature"].tolist()
    low_feats    = gap_sorted[gap_sorted["importance"] < p25]["feature"].tolist()

    print()
    print("  INTERPRETATION")
    print(DIV)
    if high_feats:
        print(f"  HIGH impact  ({', '.join(high_feats)})")
        print("    These features rank in the top 25% of all features by split gain.")
        print("    They provide strong signal that LightGBM exploits frequently.")
    if medium_feats:
        print(f"  MEDIUM impact  ({', '.join(medium_feats)})")
        print("    Mid-tier features — useful but not dominant.")
    if low_feats:
        print(f"  LOW impact  ({', '.join(low_feats)})")
        print("    Below-median features — minor role; consider ablating if")
        print("    a smaller feature set is preferred for latency or stability.")
    print(DIV)


# ── Save reports ──────────────────────────────────────────────────────────────

def _save_comparison_csv(
    base: dict,
    enh:  dict,
    bt_base: dict,
    bt_enh:  dict,
    gap_fi:  pd.DataFrame,
    report_dir: Path,
    threshold:  float,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    def _bt(d: dict, k: str) -> float:
        return float(d.get(k, float("nan")))

    rows = []
    for tag, wf, bt in [("baseline", base, bt_base), ("enhanced", enh, bt_enh)]:
        rows.append({
            "run":              tag,
            "n_folds":          wf["n_folds"],
            "n_oos_rows":       wf["n_oos_rows"],
            "oos_auc":          round(wf["oos_auc"],       4),
            "oos_logloss":      round(wf["oos_logloss"],   4),
            "oos_brier":        round(wf["oos_brier"],     4),
            "mean_is_auc":      round(wf["mean_is_auc"],   4),
            "mean_oos_auc":     round(wf["mean_oos_auc"],  4),
            "overfit_gap":      round(wf["overfit_gap"],   4),
            "threshold":        threshold,
            "n_trades":         _bt(bt, "n_trades"),
            "win_rate":         _bt(bt, "win_rate"),
            "avg_net_pnl":      _bt(bt, "avg_net_pnl"),
            "annualised_return":_bt(bt, "annualised_return"),
            "sharpe_ratio":     _bt(bt, "sharpe_ratio"),
            "max_drawdown":     _bt(bt, "max_drawdown"),
        })
    comparison_df = pd.DataFrame(rows)
    comp_path     = report_dir / "gap_feature_comparison.csv"
    comparison_df.to_csv(comp_path, index=False)
    print(f"\n  Comparison table saved → {comp_path}")

    # Enhanced model – full feature importance (updated)
    if not enh["fi_dir"].empty:
        fi_path = report_dir / "enhanced_direction_feature_importance.csv"
        enh["fi_dir"].to_csv(fi_path, index=False)
        print(f"  Enhanced feature importance → {fi_path}")

    # Gap-only feature importance
    if not gap_fi.empty:
        gap_fi_path = report_dir / "gap_feature_importance.csv"
        gap_fi.to_csv(gap_fi_path, index=False)
        print(f"  Gap feature importance      → {gap_fi_path}")

    # Save a feature importance plot for the enhanced model
    _save_fi_plot(enh["fi_dir"], report_dir)


def _save_fi_plot(fi_df: pd.DataFrame, report_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top_n = min(40, len(fi_df))
        df    = fi_df.head(top_n).sort_values("importance")

        # Colour gap features distinctly
        colors = [
            "#e05c5c" if f in NEW_GAP_FEATURES else "#4c8cbe"
            for f in df["feature"]
        ]

        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.32)))
        ax.barh(df["feature"], df["importance"], color=colors)
        ax.set_title(
            f"Enhanced model – top {top_n} features\n"
            "(red bars = new overnight-gap / prior-day features)"
        )
        ax.set_xlabel("Importance (avg split count across folds)")
        fig.tight_layout()
        out = report_dir / "enhanced_direction_feature_importance.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  Feature importance plot     → {out}")
    except Exception as exc:
        logger.warning("Could not save feature importance plot: %s", exc)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print()
    print("Loading data …")
    if args.synthetic:
        df_raw = load_synthetic(n_days=args.synth_days)
        data_desc = f"synthetic ({args.synth_days} trading days)"
    elif args.data_file:
        df_raw    = load_from_file(args.data_file)
        data_desc = f"file ({args.data_file})"
    else:
        df_raw = load_bars(
            interval=args.interval, period=args.period, provider=args.provider
        )
        data_desc = f"live ({args.provider}, {args.period})"

    n_bars    = len(df_raw)
    n_sessions = df_raw.index.normalize().nunique()
    print(f"  Loaded {n_bars:,} bars  |  {n_sessions} sessions  [{data_desc}]")
    print(
        f"  Range: {df_raw.index[0].date()} → {df_raw.index[-1].date()}"
    )

    # ── 2. Build datasets ─────────────────────────────────────────────────────
    print()
    print("Building datasets …")
    df_base = build_dataset(
        df_raw,
        horizon_dir=args.horizon_dir,
        horizon_range=args.horizon_range,
        include_gap_features=False,
    )
    df_enh = build_dataset(
        df_raw,
        horizon_dir=args.horizon_dir,
        horizon_range=args.horizon_range,
        include_gap_features=True,
    )
    n_base_feat = len([c for c in df_base.columns if c not in ("y_dir", "y_range")])
    n_enh_feat  = len([c for c in df_enh.columns  if c not in ("y_dir", "y_range")])
    print(f"  Baseline : {len(df_base):,} rows  |  {n_base_feat} features")
    print(f"  Enhanced : {len(df_enh):,} rows  |  {n_enh_feat} features  "
          f"(+{n_enh_feat - n_base_feat} gap features)")

    # ── 3. Walk-forward CV (both runs) ────────────────────────────────────────
    print()
    print("Running walk-forward CV …")
    wf_base = _wf_with_overfit(df_base, "BASELINE")
    wf_enh  = _wf_with_overfit(df_enh,  "ENHANCED")

    # ── 4. Backtest ───────────────────────────────────────────────────────────
    bt_base = _run_backtest(df_raw, wf_base, args.threshold)
    bt_enh  = _run_backtest(df_raw, wf_enh,  args.threshold)

    # ── 5. Extract gap-feature importances ────────────────────────────────────
    fi_full = wf_enh["fi_dir"]
    gap_fi  = (
        fi_full[fi_full["feature"].isin(NEW_GAP_FEATURES)]
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # ── 6. Print comparison ───────────────────────────────────────────────────
    _print_comparison(
        base       = wf_base,
        enh        = wf_enh,
        bt_base    = bt_base,
        bt_enh     = bt_enh,
        threshold  = args.threshold,
        n_bars     = n_bars,
        n_sessions = n_sessions,
        gap_fi     = gap_fi,
        full_fi    = fi_full,
    )

    _print_gap_feature_analysis(
        gap_fi  = gap_fi,
        full_fi = fi_full,
        n_total = n_enh_feat,
    )

    # ── 7. Generate full evaluation report for the enhanced model ─────────────
    print()
    print("Generating full evaluation report for the enhanced model …")
    enh_report_dir = REPORT_DIR / "enhanced"
    generate_report(
        wf_results = {
            "fold_results":   [],       # generate_report uses oos arrays; fold_results used only for per-fold CSV
            "oos_dir_proba":  wf_enh["oos_dir_proba"],
            "oos_dir_true":   wf_enh["oos_dir_true"],
            "oos_range_pred": wf_enh["oos_range_pred"],
            "oos_range_true": wf_enh["oos_range_true"],
        },
        report_dir = enh_report_dir,
    )

    # ── 8. Save CSV artefacts ─────────────────────────────────────────────────
    _save_comparison_csv(
        base       = wf_base,
        enh        = wf_enh,
        bt_base    = bt_base,
        bt_enh     = bt_enh,
        gap_fi     = gap_fi,
        report_dir = REPORT_DIR,
        threshold  = args.threshold,
    )

    # ── 9. Per-fold details ───────────────────────────────────────────────────
    W   = 72
    SEP = "═" * W
    DIV = "─" * W
    print()
    print(SEP)
    print("  PER-FOLD AUC  (IS = in-sample, OOS = out-of-sample)")
    print(DIV)
    print(f"  {'Fold':>4}  {'Base IS':>9}  {'Base OOS':>9}  "
          f"{'Enh IS':>9}  {'Enh OOS':>9}  {'Δ OOS':>8}")
    print(DIV)

    n_folds = max(len(wf_base["fold_is_auc"]), len(wf_enh["fold_is_auc"]))
    for i in range(n_folds):
        b_is  = wf_base["fold_is_auc"][i]  if i < len(wf_base["fold_is_auc"])  else float("nan")
        b_oos = wf_base["fold_oos_auc"][i] if i < len(wf_base["fold_oos_auc"]) else float("nan")
        e_is  = wf_enh["fold_is_auc"][i]   if i < len(wf_enh["fold_is_auc"])   else float("nan")
        e_oos = wf_enh["fold_oos_auc"][i]  if i < len(wf_enh["fold_oos_auc"])  else float("nan")
        d_oos = e_oos - b_oos
        sym   = "▲" if d_oos > 0 else ("▼" if d_oos < 0 else "=")
        print(
            f"  {i+1:>4}  {b_is:>9.4f}  {b_oos:>9.4f}  "
            f"{e_is:>9.4f}  {e_oos:>9.4f}  {d_oos:>+7.4f}{sym}"
        )

    print(DIV)
    base_gap = wf_base["overfit_gap"]
    enh_gap  = wf_enh["overfit_gap"]
    print(
        f"  {'Mean':>4}  {wf_base['mean_is_auc']:>9.4f}  {wf_base['mean_oos_auc']:>9.4f}  "
        f"{wf_enh['mean_is_auc']:>9.4f}  {wf_enh['mean_oos_auc']:>9.4f}"
    )
    print(DIV)
    print(
        f"  Overfit gap  Baseline={base_gap:+.4f}   Enhanced={enh_gap:+.4f}   "
        f"Δ={enh_gap - base_gap:+.4f}"
    )
    if abs(enh_gap - base_gap) < 0.005:
        print("  ✓  Overfit gap is essentially unchanged — new features do not add overfit.")
    elif enh_gap > base_gap + 0.005:
        print("  ✗  Overfit gap increased — new features may be overfit to training data.")
    else:
        print("  ✓  Overfit gap decreased — new features actually reduce overfit tendency.")
    print(SEP)
    print()


if __name__ == "__main__":
    main()

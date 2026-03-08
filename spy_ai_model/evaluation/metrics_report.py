"""
metrics_report.py
─────────────────
Compute and persist a comprehensive evaluation report for both models.

Metrics reported
────────────────
Direction model
  • AUC (ROC)
  • Log-loss
  • Brier score
  • Calibration curve (plot)
  • Decile lift table (probability deciles vs. actual hit-rate)
  • Per-fold AUC / logloss

Range model
  • MAE, RMSE, R²
  • Decile analysis (predicted range bucket vs. mean actual range)
  • Per-fold MAE / RMSE

Feature importance tables saved as CSV for both models.

Public API
──────────
    generate_report(wf_results, report_dir)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import REPORT_DIR

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_import_sklearn():
    from sklearn.metrics import (
        roc_auc_score, log_loss, brier_score_loss,
        mean_absolute_error, mean_squared_error, r2_score,
        calibration_curve,
    )
    return (
        roc_auc_score, log_loss, brier_score_loss,
        mean_absolute_error, mean_squared_error, r2_score,
        calibration_curve,
    )


def _decile_table_direction(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Bin predictions into deciles and compute actual hit-rate per decile."""
    df = pd.DataFrame({"prob": y_prob, "actual": y_true})
    df["decile"] = pd.qcut(df["prob"], 10, labels=False, duplicates="drop") + 1
    tbl = (
        df.groupby("decile")
          .agg(count=("actual", "count"),
               mean_prob=("prob", "mean"),
               actual_rate=("actual", "mean"))
          .reset_index()
    )
    tbl["lift"] = tbl["actual_rate"] / y_true.mean()
    return tbl


def _decile_table_range(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Bin predictions into deciles and show mean actual range per bucket."""
    df = pd.DataFrame({"pred": y_pred, "actual": y_true})
    df["decile"] = pd.qcut(df["pred"], 10, labels=False, duplicates="drop") + 1
    tbl = (
        df.groupby("decile")
          .agg(count=("actual", "count"),
               mean_pred=("pred", "mean"),
               mean_actual=("actual", "mean"))
          .reset_index()
    )
    return tbl


def _save_calibration_plot(y_true, y_prob, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve as cal_curve

        prob_true, prob_pred = cal_curve(y_true, y_prob, n_bins=10)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(prob_pred, prob_true, "o-", label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration curve – Direction model")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info("Calibration plot saved → %s", out_path)
    except Exception as exc:
        logger.warning("Could not save calibration plot: %s", exc)


def _save_feature_importance_plot(fi_df: pd.DataFrame, title: str, out_path: Path, top_n: int = 30):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = fi_df.head(top_n).sort_values("importance")
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
        ax.barh(df["feature"], df["importance"])
        ax.set_title(title)
        ax.set_xlabel("Importance (split count)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info("Feature importance plot saved → %s", out_path)
    except Exception as exc:
        logger.warning("Could not save feature importance plot: %s", exc)


# ── main report function ──────────────────────────────────────────────────────

def generate_report(wf_results: dict, report_dir: Path = REPORT_DIR) -> dict:
    """
    Build and persist the full evaluation report.

    Parameters
    ----------
    wf_results : dict returned by walk_forward.walk_forward_cv()
    report_dir : directory to write CSV / PNG artefacts

    Returns
    -------
    dict of summary scalars (also printed to log)
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    (
        roc_auc_score, log_loss, brier_score_loss,
        mean_absolute_error, mean_squared_error, r2_score,
        calibration_curve,
    ) = _safe_import_sklearn()

    y_dir_true  = wf_results["oos_dir_true"]
    y_dir_prob  = wf_results["oos_dir_proba"]
    y_rng_true  = wf_results["oos_range_true"]
    y_rng_pred  = wf_results["oos_range_pred"]
    fold_results = wf_results["fold_results"]

    # ── Direction metrics ──────────────────────────────────────────────────────
    dir_auc     = roc_auc_score(y_dir_true, y_dir_prob)
    dir_logloss = log_loss(y_dir_true, y_dir_prob)
    dir_brier   = brier_score_loss(y_dir_true, y_dir_prob)
    base_rate   = y_dir_true.mean()

    logger.info("=" * 60)
    logger.info("DIRECTION MODEL – overall OOS metrics")
    logger.info("  AUC      : %.4f", dir_auc)
    logger.info("  Log-loss : %.4f", dir_logloss)
    logger.info("  Brier    : %.4f", dir_brier)
    logger.info("  Base rate: %.4f", base_rate)
    logger.info("=" * 60)

    # Decile lift
    dir_decile = _decile_table_direction(y_dir_true, y_dir_prob)
    dir_decile.to_csv(report_dir / "direction_decile_lift.csv", index=False)
    logger.info("Direction decile lift:\n%s", dir_decile.to_string(index=False))

    # Calibration plot
    _save_calibration_plot(y_dir_true, y_dir_prob,
                           report_dir / "direction_calibration.png")

    # ── Range metrics ──────────────────────────────────────────────────────────
    rng_mae  = mean_absolute_error(y_rng_true, y_rng_pred)
    rng_rmse = mean_squared_error(y_rng_true, y_rng_pred, squared=False)
    rng_r2   = r2_score(y_rng_true, y_rng_pred)

    logger.info("RANGE MODEL – overall OOS metrics")
    logger.info("  MAE  : %.6f", rng_mae)
    logger.info("  RMSE : %.6f", rng_rmse)
    logger.info("  R²   : %.4f", rng_r2)
    logger.info("=" * 60)

    rng_decile = _decile_table_range(y_rng_true, y_rng_pred)
    rng_decile.to_csv(report_dir / "range_decile.csv", index=False)
    logger.info("Range decile analysis:\n%s", rng_decile.to_string(index=False))

    # ── Per-fold summary ───────────────────────────────────────────────────────
    fold_rows = []
    for fr in fold_results:
        fold_rows.append({
            "fold":       fr.fold,
            "val_start":  fr.val_start,
            "val_end":    fr.val_end,
            "dir_auc":    round(fr.dir_auc,     4),
            "dir_logloss":round(fr.dir_logloss,  4),
            "range_mae":  round(fr.range_mae,    6),
            "range_rmse": round(fr.range_rmse,   6),
        })
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(report_dir / "per_fold_metrics.csv", index=False)
    logger.info("Per-fold metrics:\n%s", fold_df.to_string(index=False))

    # ── Feature importance (average across folds) ──────────────────────────────
    if fold_results:
        fi_dir_list   = [fr.fi_dir   for fr in fold_results if not fr.fi_dir.empty]
        fi_range_list = [fr.fi_range for fr in fold_results if not fr.fi_range.empty]

        for fi_list, label in [(fi_dir_list, "direction"), (fi_range_list, "range")]:
            if fi_list:
                combined = (
                    pd.concat(fi_list)
                      .groupby("feature")["importance"]
                      .mean()
                      .reset_index()
                      .sort_values("importance", ascending=False)
                )
                combined.to_csv(report_dir / f"{label}_feature_importance.csv", index=False)
                _save_feature_importance_plot(
                    combined,
                    f"{label.capitalize()} model – feature importance (avg across folds)",
                    report_dir / f"{label}_feature_importance.png",
                )

    summary = {
        "dir_auc":     dir_auc,
        "dir_logloss": dir_logloss,
        "dir_brier":   dir_brier,
        "rng_mae":     rng_mae,
        "rng_rmse":    rng_rmse,
        "rng_r2":      rng_r2,
        "n_folds":     len(fold_results),
        "n_oos_rows":  len(y_dir_true),
    }

    # Save summary
    pd.DataFrame([summary]).to_csv(report_dir / "summary_metrics.csv", index=False)
    return summary

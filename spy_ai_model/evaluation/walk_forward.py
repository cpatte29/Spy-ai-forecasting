"""
walk_forward.py
───────────────
Walk-forward cross-validation for both the direction and range models.

Strategy
─────────
  • Sort all rows by timestamp.
  • Split unique trading dates into overlapping windows:
      - Train : TRAIN_DAYS trading days
      - Val   : VAL_DAYS  trading days
      - Step  : STEP_DAYS trading days (how far we advance each fold)
  • At each fold:
      1. Train both models on train slice.
      2. Evaluate on val slice.
      3. Collect OOS predictions and ground-truth labels.
  • Return full OOS prediction arrays for final evaluation.

Public API
──────────
    results = walk_forward_cv(df_model)
    # results keys: fold_metrics, oos_dir, oos_range
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TRAIN_DAYS, VAL_DAYS, STEP_DAYS
from data.dataset_builder      import split_features_labels
from models.train_direction    import train_direction_model, predict_direction_proba
from models.train_range        import train_range_model,    predict_range

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold:       int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    val_start:   pd.Timestamp
    val_end:     pd.Timestamp
    # direction
    dir_auc:     float = np.nan
    dir_logloss: float = np.nan
    # range
    range_mae:   float = np.nan
    range_rmse:  float = np.nan
    # OOS preds
    dir_proba:   np.ndarray = field(default_factory=lambda: np.array([]))
    dir_true:    np.ndarray = field(default_factory=lambda: np.array([]))
    range_pred:  np.ndarray = field(default_factory=lambda: np.array([]))
    range_true:  np.ndarray = field(default_factory=lambda: np.array([]))
    # feature importance
    fi_dir:   pd.DataFrame = field(default_factory=pd.DataFrame)
    fi_range: pd.DataFrame = field(default_factory=pd.DataFrame)


def _make_date_folds(trading_dates: np.ndarray) -> list[tuple]:
    """Return list of (train_idx_slice, val_idx_slice) as date-index positions."""
    n      = len(trading_dates)
    folds  = []
    start  = 0

    while start + TRAIN_DAYS + VAL_DAYS <= n:
        train_end = start + TRAIN_DAYS
        val_end   = train_end + VAL_DAYS
        folds.append((
            trading_dates[start      : train_end],
            trading_dates[train_end  : val_end],
        ))
        start += STEP_DAYS

    return folds


def walk_forward_cv(df_model: pd.DataFrame) -> dict:
    """
    Run walk-forward cross-validation.

    Parameters
    ----------
    df_model : pd.DataFrame
        Output of dataset_builder.build_dataset() – features + y_dir + y_range.

    Returns
    -------
    dict with keys:
        'fold_results'  : List[FoldResult]
        'oos_dir_proba' : np.ndarray  – out-of-sample P(up) across all folds
        'oos_dir_true'  : np.ndarray
        'oos_range_pred': np.ndarray
        'oos_range_true': np.ndarray
        'oos_index'     : pd.DatetimeIndex
    """
    from sklearn.metrics import roc_auc_score, log_loss, mean_absolute_error
    from sklearn.metrics import mean_squared_error
    import numpy as _np

    trading_dates = np.sort(np.unique(df_model.index.normalize()))
    folds         = _make_date_folds(trading_dates)

    if not folds:
        raise ValueError(
            f"Not enough data for even one fold. Need at least "
            f"{TRAIN_DAYS + VAL_DAYS} trading days, got {len(trading_dates)}."
        )

    logger.info(
        "Walk-forward CV: %d folds  (train=%dd, val=%dd, step=%dd)",
        len(folds), TRAIN_DAYS, VAL_DAYS, STEP_DAYS,
    )

    fold_results:   List[FoldResult] = []
    oos_dir_proba  = []
    oos_dir_true   = []
    oos_range_pred = []
    oos_range_true = []
    oos_index_list = []

    X_all, y_dir_all, y_range_all = split_features_labels(df_model)

    for fold_num, (train_dates, val_dates) in enumerate(folds, 1):
        train_mask = df_model.index.normalize().isin(train_dates)
        val_mask   = df_model.index.normalize().isin(val_dates)

        X_tr, y_dir_tr, y_rng_tr = (
            X_all[train_mask], y_dir_all[train_mask], y_range_all[train_mask]
        )
        X_vl, y_dir_vl, y_rng_vl = (
            X_all[val_mask], y_dir_all[val_mask], y_range_all[val_mask]
        )

        if len(X_tr) < 100 or len(X_vl) < 10:
            logger.warning("Fold %d skipped – insufficient rows.", fold_num)
            continue

        logger.info(
            "Fold %d: train=%s→%s (%d rows)  val=%s→%s (%d rows)",
            fold_num,
            pd.Timestamp(train_dates[0]).strftime("%Y-%m-%d"),
            pd.Timestamp(train_dates[-1]).strftime("%Y-%m-%d"), len(X_tr),
            pd.Timestamp(val_dates[0]).strftime("%Y-%m-%d"),
            pd.Timestamp(val_dates[-1]).strftime("%Y-%m-%d"),   len(X_vl),
        )

        # Train models
        dir_model,  fi_dir   = train_direction_model(X_tr, y_dir_tr, X_vl, y_dir_vl)
        rng_model,  fi_range = train_range_model(X_tr, y_rng_tr, X_vl, y_rng_vl)

        # Predict on val
        dir_prob  = predict_direction_proba(dir_model, X_vl)
        rng_pred  = predict_range(rng_model, X_vl)

        # Metrics
        auc      = roc_auc_score(y_dir_vl, dir_prob)
        logloss  = log_loss(y_dir_vl, dir_prob)
        mae      = mean_absolute_error(y_rng_vl, rng_pred)
        rmse     = _np.sqrt(mean_squared_error(y_rng_vl, rng_pred))

        logger.info(
            "  → dir AUC=%.4f  logloss=%.4f  |  range MAE=%.5f  RMSE=%.5f",
            auc, logloss, mae, rmse,
        )

        fr = FoldResult(
            fold       = fold_num,
            train_start= pd.Timestamp(train_dates[0]),
            train_end  = pd.Timestamp(train_dates[-1]),
            val_start  = pd.Timestamp(val_dates[0]),
            val_end    = pd.Timestamp(val_dates[-1]),
            dir_auc    = auc,
            dir_logloss= logloss,
            range_mae  = mae,
            range_rmse = rmse,
            dir_proba  = dir_prob,
            dir_true   = y_dir_vl.values,
            range_pred = rng_pred,
            range_true = y_rng_vl.values,
            fi_dir     = fi_dir,
            fi_range   = fi_range,
        )
        fold_results.append(fr)

        oos_dir_proba.extend(dir_prob)
        oos_dir_true.extend(y_dir_vl.values)
        oos_range_pred.extend(rng_pred)
        oos_range_true.extend(y_rng_vl.values)
        oos_index_list.extend(X_vl.index.tolist())

    return {
        "fold_results":   fold_results,
        "oos_dir_proba":  np.array(oos_dir_proba),
        "oos_dir_true":   np.array(oos_dir_true),
        "oos_range_pred": np.array(oos_range_pred),
        "oos_range_true": np.array(oos_range_true),
        "oos_index":      pd.DatetimeIndex(oos_index_list),
    }

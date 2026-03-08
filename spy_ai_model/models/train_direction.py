"""
train_direction.py
──────────────────
Train a LightGBM binary classifier that predicts P(close[t+60] > close[t]).

Public API
──────────
    model, feature_importance = train_direction_model(X_train, y_train,
                                                       X_val,   y_val)
    save_direction_model(model, path)
    load_direction_model(path)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DIRECTION_PARAMS, EARLY_STOPPING_ROUNDS, MODEL_DIR

logger = logging.getLogger(__name__)


def train_direction_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    params:  Optional[dict] = None,
) -> tuple:
    """
    Fit a LightGBM classifier.

    Returns
    -------
    (model, feature_importance_df)
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("Install lightgbm: pip install lightgbm")

    p = {**DIRECTION_PARAMS, **(params or {})}

    # Separate out non-constructor keys
    n_estimators      = p.pop("n_estimators", 500)
    early_stop        = EARLY_STOPPING_ROUNDS

    model = lgb.LGBMClassifier(n_estimators=n_estimators, **p)

    callbacks = [lgb.early_stopping(early_stop, verbose=False),
                 lgb.log_evaluation(period=50)]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )

    logger.info(
        "Direction model trained. Best iteration: %d",
        model.best_iteration_,
    )

    fi = pd.DataFrame({
        "feature":   X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return model, fi


def save_direction_model(model, path: Optional[Path] = None) -> Path:
    path = Path(path) if path else MODEL_DIR / "direction_model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Direction model saved → %s", path)
    return path


def load_direction_model(path: Optional[Path] = None):
    path = Path(path) if path else MODEL_DIR / "direction_model.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Direction model loaded ← %s", path)
    return model


def predict_direction_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return P(y=1) for each row in X."""
    return model.predict_proba(X)[:, 1]

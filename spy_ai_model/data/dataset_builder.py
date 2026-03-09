"""
dataset_builder.py
──────────────────
Orchestrates the full feature + label assembly pipeline and returns a
clean, leakage-free modelling DataFrame.

    df_raw  →  features  →  labels  →  dropna  →  df_model
"""

from __future__ import annotations

import logging

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.feature_engineering import build_features
from labels.label_builder          import build_labels

logger = logging.getLogger(__name__)


def build_dataset(
    df_raw: pd.DataFrame,
    horizon_dir: int | None = None,
    horizon_range: int | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df_raw : pd.DataFrame
        OHLCV bars with columns open/high/low/close/volume,
        indexed by a tz-naive DatetimeIndex sorted ascending.
    horizon_dir : int or None
        Bars ahead for the direction label (overrides config.HORIZON_DIR).
    horizon_range : int or None
        Bars ahead for the range label (overrides config.HORIZON_RANGE).
    horizon : int or None
        Legacy shorthand – sets both horizons when provided.

    Returns
    -------
    pd.DataFrame
        One row per bar that has both valid features *and* valid labels.
        Columns: all feature columns + y_dir + y_range.
        Index  : original DatetimeIndex (subset of df_raw).
    """
    logger.info("Building features from %d raw bars …", len(df_raw))
    df_feat = build_features(df_raw)

    logger.info("Building labels …")
    df_lab  = build_labels(df_raw, horizon_dir=horizon_dir, horizon_range=horizon_range, horizon=horizon)

    # Align on index (inner join – labels require future bars)
    df = df_feat.join(df_lab[["y_dir", "y_range"]], how="inner")

    before = len(df)
    df.dropna(inplace=True)
    logger.info(
        "Dataset: %d rows after dropna (dropped %d rows with NaN).",
        len(df), before - len(df),
    )

    feature_cols = [c for c in df.columns if c not in ("y_dir", "y_range")]
    logger.info("Feature count: %d", len(feature_cols))

    return df


def split_features_labels(df: pd.DataFrame):
    """
    Convenience split into (X, y_dir, y_range).

    Returns
    -------
    X       : pd.DataFrame  feature matrix
    y_dir   : pd.Series     binary direction label
    y_range : pd.Series     continuous range label
    """
    feature_cols = [c for c in df.columns if c not in ("y_dir", "y_range")]
    return df[feature_cols], df["y_dir"], df["y_range"]

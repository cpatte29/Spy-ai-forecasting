"""
similarity_search.py
────────────────────
Find the most similar historical zone-test events for a given live setup.

Two similarity methods are supported (selectable at runtime):

  "cosine"  – cosine similarity on L2-normalised feature vectors.
              Best for high-dimensional vectors where direction matters more
              than magnitude.  Default.

  "knn"     – k-nearest neighbours using Euclidean distance on StandardScaler-
              normalised features.  Requires scikit-learn.
              More sensitive to scale differences between features.

Both return events sorted by similarity (most similar first).

Usage
─────
    from analog_engine.similarity_search import (
        load_analog_dataset,
        compute_query_features,
        find_similar_events,
    )

    # Load the persisted dataset
    dataset = load_analog_dataset()

    # From a live bar window (with "bar_role" column, just like historical windows):
    top_matches = find_similar_events(
        query_window=df_current_window,
        query_event_meta=event_meta_dict,
        dataset=dataset,
        top_n=20,
        method="cosine",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from analog_engine.pattern_features import compute_event_features, feature_names

logger = logging.getLogger(__name__)

_DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "supply_zone_events.parquet"

# Features excluded from similarity (metadata about zone identity, not pattern shape)
_EXCLUDE_FROM_SIM = {
    "meta__bars_since_creation",
    "meta__test_number",
}


# ── dataset I/O ───────────────────────────────────────────────────────────────

def load_analog_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the analog dataset from Parquet.

    Parameters
    ──────────
    path  Override the default path (analog_engine/data/supply_zone_events.parquet).

    Raises
    ──────
    FileNotFoundError if the dataset has not been built yet.
    """
    p = Path(path) if path else _DEFAULT_DATASET
    if not p.exists():
        raise FileNotFoundError(
            f"Analog dataset not found at {p}.\n"
            "Run the builder first:\n"
            "  python analog_engine/analog_dataset_builder.py"
        )
    ds = pd.read_parquet(p)
    logger.info("Loaded analog dataset: %d events, %d columns.", len(ds), ds.ncols
                if hasattr(ds, "ncols") else len(ds.columns))
    return ds


# ── query feature extraction ──────────────────────────────────────────────────

def compute_query_features(
    query_window:     pd.DataFrame,
    query_event_meta: dict | pd.Series,
) -> pd.Series:
    """
    Compute the feature vector for a live (query) setup window.

    Parameters
    ──────────
    query_window       A DataFrame with the same format as historical windows:
                       open/high/low/close/volume columns + "bar_role" column
                       ("before" | "zone" | "after").  The "after" bars are not
                       used in feature computation (they represent the future).
    query_event_meta   A dict or Series with the same keys as an event row:
                       zone_high, zone_low, zone_width_pct, displacement_size,
                       displacement_speed, volume_spike_ratio,
                       bars_since_creation, test_number,
                       approach_return, approach_speed.

    Returns
    ───────
    pd.Series – feature vector (same index as historical feature vectors).
    """
    if isinstance(query_event_meta, dict):
        query_event_meta = pd.Series(query_event_meta)
    return compute_event_features(query_window, query_event_meta)


# ── similarity search ─────────────────────────────────────────────────────────

def find_similar_events(
    query_window:     pd.DataFrame,
    query_event_meta: dict | pd.Series,
    dataset:          pd.DataFrame,
    top_n:            int                           = 20,
    method:           Literal["cosine", "knn"]     = "cosine",
    feature_subset:   list[str] | None              = None,
) -> pd.DataFrame:
    """
    Find the `top_n` most similar historical zone-test events.

    Parameters
    ──────────
    query_window       Live bar window (bar_role column required).
    query_event_meta   Dict/Series of zone metadata for the live setup.
    dataset            Loaded analog dataset (from load_analog_dataset()).
    top_n              Number of matches to return.
    method             "cosine" or "knn".
    feature_subset     Optional list of feature column names to restrict the
                       search to.  None = use all pattern features.

    Returns
    ───────
    pd.DataFrame – top_n rows from `dataset` with an added "similarity_score"
                   column (higher = more similar for cosine; lower distance for knn,
                   converted to a 0-1 score).  Sorted by similarity_score desc.
    """
    # ── resolve feature columns ───────────────────────────────────────────
    all_feat_names = feature_names()
    sim_features   = [
        c for c in all_feat_names
        if c not in _EXCLUDE_FROM_SIM
        and c in dataset.columns
    ]
    if feature_subset:
        sim_features = [c for c in feature_subset if c in sim_features]

    if not sim_features:
        raise ValueError(
            "No feature columns found in dataset that match the expected names. "
            "Rebuild the dataset with analog_dataset_builder.py."
        )

    # ── compute query vector ──────────────────────────────────────────────
    query_fvec = compute_query_features(query_window, query_event_meta)

    # Align to sim_features, fill missing with 0
    q = np.array([float(query_fvec.get(f, 0.0)) for f in sim_features], dtype=float)
    q = np.nan_to_num(q, nan=0.0)

    # ── build historical matrix ───────────────────────────────────────────
    hist_matrix = dataset[sim_features].to_numpy(dtype=float)
    hist_matrix = np.nan_to_num(hist_matrix, nan=0.0)

    # ── compute similarity ────────────────────────────────────────────────
    if method == "cosine":
        scores = _cosine_similarity_batch(q, hist_matrix)
    elif method == "knn":
        scores = _knn_scores(q, hist_matrix)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'cosine' or 'knn'.")

    # ── assemble results ──────────────────────────────────────────────────
    result = dataset.copy()
    result["similarity_score"] = scores

    top = (
        result
        .sort_values("similarity_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return top


# ── similarity implementations ────────────────────────────────────────────────

def _cosine_similarity_batch(q: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between query vector q and every row of matrix.
    Returns an array of floats in [-1, 1]; higher = more similar.
    """
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        logger.warning("Query feature vector is zero — all cosine similarities will be 0.")
        return np.zeros(len(matrix))

    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms = np.where(row_norms == 0, 1e-10, row_norms)  # avoid div/0

    dots = matrix @ q
    return dots / (row_norms * q_norm)


def _knn_scores(q: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Convert Euclidean distances to similarity scores in [0, 1].
    score = 1 / (1 + distance)  so smaller distance → score near 1.
    """
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning(
            "scikit-learn not installed; falling back to unscaled Euclidean KNN. "
            "Install with: pip install scikit-learn"
        )
        dists = np.linalg.norm(matrix - q, axis=1)
        return 1.0 / (1.0 + dists)

    scaler = StandardScaler()
    # Fit scaler on historical data and transform both
    mat_scaled = scaler.fit_transform(matrix)
    q_scaled   = scaler.transform(q.reshape(1, -1)).flatten()
    dists      = np.linalg.norm(mat_scaled - q_scaled, axis=1)
    return 1.0 / (1.0 + dists)


# ── convenience: rank features by contribution ───────────────────────────────

def explain_top_match(
    query_fvec:  pd.Series,
    match_row:   pd.Series,
    top_k:       int = 10,
) -> pd.DataFrame:
    """
    Show which features contributed most to the similarity between the query
    and a specific matched event.

    Returns a DataFrame sorted by absolute cosine contribution (descending).
    """
    feat_cols = [c for c in query_fvec.index if c in match_row.index
                 and c not in _EXCLUDE_FROM_SIM]

    q_vals = np.array([float(query_fvec.get(f, 0.0)) for f in feat_cols])
    m_vals = np.array([float(match_row.get(f, 0.0))  for f in feat_cols])

    q_norm = np.linalg.norm(q_vals)
    m_norm = np.linalg.norm(m_vals)

    contrib = (q_vals / max(q_norm, 1e-10)) * (m_vals / max(m_norm, 1e-10))
    abs_contrib = np.abs(contrib)

    df = pd.DataFrame({
        "feature":     feat_cols,
        "query_val":   q_vals,
        "match_val":   m_vals,
        "contribution": contrib,
        "abs_contrib": abs_contrib,
    })
    return df.sort_values("abs_contrib", ascending=False).head(top_k).reset_index(drop=True)

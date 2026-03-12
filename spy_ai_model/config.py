"""
Central configuration for the SPY AI forecasting pipeline.
All parameters are grouped here so changing one value propagates everywhere.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data" / "raw"
MODEL_DIR  = BASE_DIR / "models" / "saved"
REPORT_DIR = BASE_DIR / "evaluation" / "reports"

for _d in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
TICKER          = "SPY"
BAR_INTERVAL    = "1m"          # 1-minute bars
HORIZON         = 60            # default horizon (kept for back-compat)
HORIZON_DIR     = 60            # direction label: bars ahead to predict close
HORIZON_RANGE   = 60            # range label: bars ahead for high-low window
MARKET_OPEN     = "09:30"
MARKET_CLOSE    = "16:00"

# ── Feature engineering ───────────────────────────────────────────────────────
EMA_WINDOWS       = [9, 20, 50, 200]
SLOPE_LOOKBACK    = 5           # bars used to compute EMA slope
RV_WINDOWS        = [5, 15, 30]
RANGE_WINDOWS     = [5, 15]
VOL_REL_WINDOWS   = [5, 15]
ROLLING_HL_WINDOW = 15
OPENING_RANGE_BARS = 6          # first N bars per session define the opening range
                                # 6 × 5 min = 30-minute opening range for 5m bars

# ── Walk-forward validation ───────────────────────────────────────────────────
TRAIN_DAYS = 63     # ~3 calendar months of trading days
VAL_DAYS   = 10     # ~2 calendar weeks of trading days
STEP_DAYS  = 10     # slide window forward by 2 weeks each fold

# ── Model hyper-parameters ────────────────────────────────────────────────────
DIRECTION_PARAMS = {
    "objective":        "binary",
    "metric":           ["binary_logloss", "auc"],
    "n_estimators":     500,
    "learning_rate":    0.03,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 50,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "n_jobs":           -1,
    "random_state":     42,
    "verbose":          -1,
}

# Moderate preset – balanced regularisation to reduce train/OOS overfit gap
# while preserving enough model capacity for the typical 1500–2000 row folds.
# Key changes vs default:
#   num_leaves    63  → 31    (shallower trees, less capacity)
#   max_depth     -1  → 6     (hard cap on tree depth)
#   min_child_samples 50 → 60  (slight increase, still fits fold sizes)
#   reg_alpha    0.1  → 0.5   (moderate L1 weight penalty)
#   reg_lambda   1.0  → 2.0   (moderate L2 weight penalty)
#   subsample    0.8  → 0.75  (mild bagging noise increase)
#   colsample_bytree 0.8 → 0.75
DIRECTION_PARAMS_MODERATE = {
    **DIRECTION_PARAMS,
    "num_leaves":        31,
    "max_depth":         6,
    "min_child_samples": 60,
    "subsample":         0.75,
    "colsample_bytree":  0.75,
    "reg_alpha":         0.5,
    "reg_lambda":        2.0,
}

# Conservative preset – strong regularisation; best used with larger datasets
# (≥3000 rows per fold). With ~1500-row folds min_child_samples=100 can cause
# early stopping at iteration 1 and probability collapse.
# Key changes vs default:
#   num_leaves    63  → 31    (shallower trees, less capacity)
#   max_depth     -1  → 6     (hard cap on tree depth)
#   min_child_samples 50 → 100 (require more evidence per leaf)
#   reg_alpha    0.1  → 1.0   (stronger L1 weight penalty)
#   reg_lambda   1.0  → 5.0   (stronger L2 weight penalty)
#   subsample    0.8  → 0.7   (more bagging noise)
#   colsample_bytree 0.8 → 0.7
DIRECTION_PARAMS_CONSERVATIVE = {
    **DIRECTION_PARAMS,
    "num_leaves":        31,
    "max_depth":         6,
    "min_child_samples": 100,
    "subsample":         0.7,
    "colsample_bytree":  0.7,
    "reg_alpha":         1.0,
    "reg_lambda":        5.0,
}

RANGE_PARAMS = {
    "objective":        "regression",
    "metric":           ["mae", "rmse"],
    "n_estimators":     500,
    "learning_rate":    0.03,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 50,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "n_jobs":           -1,
    "random_state":     42,
    "verbose":          -1,
}

EARLY_STOPPING_ROUNDS = 50

# ── Synthetic data ────────────────────────────────────────────────────────────
SYNTH_TRADING_DAYS = 252        # ~1 year of 1-min bars
SYNTH_SEED         = 0

# ── Backtest ──────────────────────────────────────────────────────────────────
DIR_PROB_THRESHOLD  = 0.55      # enter long when P(up) > threshold
TRANSACTION_COST_BP = 1.0       # one-way cost in basis points

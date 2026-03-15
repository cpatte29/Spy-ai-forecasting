"""
main_pipeline.py
────────────────
End-to-end orchestration script for the SPY AI forecasting pipeline.

Usage
─────
# Synthetic data (no internet required):
    python main_pipeline.py --mode synthetic

# Real SPY data via yfinance:
    python main_pipeline.py --mode real --period 60d

# Real SPY data from a local file:
    python main_pipeline.py --mode file --file-path /path/to/spy_1m.csv

Steps executed
──────────────
1. Load / generate 1-minute OHLCV bars
2. Build feature matrix (causally, no look-ahead)
3. Attach direction and range labels
4. Run walk-forward cross-validation (train + eval per fold)
5. Generate evaluation report (metrics, decile tables, plots)
6. Run simple backtest on OOS predictions
7. Train final models on full dataset and save .pkl files
"""

import argparse
import logging
import sys

import numpy as np
from pathlib import Path

# ── make sure project root is on PYTHONPATH ────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── logging setup (stdout so output is always visible) ────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main_pipeline")

# ── project imports (after sys.path fix) ──────────────────────────────────────
from config import REPORT_DIR, MODEL_DIR, DIRECTION_PARAMS_MODERATE, DIRECTION_PARAMS_CONSERVATIVE

from data.data_loader import load_synthetic, load_from_yfinance, load_from_file, load_bars
from data.dataset_builder import build_dataset, split_features_labels
from evaluation.walk_forward import walk_forward_cv
from evaluation.metrics_report import generate_report
from backtest.strategy_simulation import run_backtest, print_backtest_report
from models.train_direction import (
    train_direction_model,
    save_direction_model,
    predict_direction_proba,
)
from models.train_range import (
    train_range_model,
    save_range_model,
    predict_range,
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "SPY AI Intraday Forecasting Pipeline\n"
            "\n"
            "Trains direction (up/down) and range (volatility) models on\n"
            "1-minute SPY bars using walk-forward cross-validation.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main_pipeline.py --mode synthetic\n"
            "  python main_pipeline.py --mode synthetic --synth-days 500\n"
            "  python main_pipeline.py --mode real --period 30d\n"
            "  python main_pipeline.py --mode real --interval 5m --period 60d\n"
            "  python main_pipeline.py --mode real --interval 5m --period 60d --horizon-dir 60 --horizon-range 12\n"
            "  python main_pipeline.py --mode real --interval 15m --period 60d\n"
            "  python main_pipeline.py --mode file --file-path spy_1m.csv\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "real", "file"],
        default="synthetic",
        help=(
            "Data source to use:\n"
            "  synthetic – generate realistic fake SPY 1-min bars (default)\n"
            "  real      – download live SPY data via yfinance\n"
            "  file      – load bars from a local CSV or Parquet file"
        ),
    )
    parser.add_argument(
        "--synth-days",
        type=int,
        default=252,
        metavar="N",
        help="Number of synthetic trading days to generate (default: 252).",
    )
    parser.add_argument(
        "--interval",
        default="1m",
        metavar="INTERVAL",
        help=(
            "Bar interval – only used with --mode real.\n"
            "Examples: 1m, 5m, 15m.  "
            "yfinance limits: 1m→~30 days, 5m/15m→~60 days."
        ),
    )
    parser.add_argument(
        "--period",
        default="30d",
        metavar="PERIOD",
        help=(
            "yfinance period string – only used with --mode real.\n"
            "Examples: 7d, 30d, 60d.  "
            "For 5m/15m bars you can use up to 60d."
        ),
    )
    parser.add_argument(
        "--start",
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date for --mode real (overrides --period).",
    )
    parser.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help="End date for --mode real (overrides --period).",
    )
    parser.add_argument(
        "--file-path",
        default=None,
        metavar="PATH",
        help="Path to a local CSV or Parquet file – only used with --mode file.",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "polygon", "yfinance"],
        metavar="PROVIDER",
        help=(
            "Market data provider for --mode real.\n"
            "  auto     – Polygon when POLYGON_API_KEY env var is set, else yfinance\n"
            "  polygon  – Polygon.io (requires POLYGON_API_KEY)\n"
            "  yfinance – yfinance fallback (default when no key is set)\n"
            "For --mode file use --file-path; the file provider is selected automatically.\n"
            "Set the Polygon key with: export POLYGON_API_KEY=<your_key>"
        ),
    )
    parser.add_argument(
        "--horizon-dir",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Direction label horizon in bars (overrides config HORIZON_DIR=60).\n"
            "1m bars → 60 (1 hour).  5m bars → 60 (5 hours, captures trend)."
        ),
    )
    parser.add_argument(
        "--horizon-range",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Range label horizon in bars (overrides config HORIZON_RANGE=60).\n"
            "1m bars → 60 (1 hour).  5m bars → 12 (1 hour, predicts volatility)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="P",
        help=(
            "Direction probability threshold to enter a long trade "
            "(overrides config DIR_PROB_THRESHOLD=0.55).\n"
            "Use a lower value (e.g. 0.51) when model probabilities are compressed."
        ),
    )
    parser.add_argument(
        "--reg-preset",
        choices=["default", "moderate", "conservative"],
        default="default",
        help=(
            "Direction model regularisation preset.\n"
            "  default      – num_leaves=63, max_depth=-1, reg_alpha=0.1, reg_lambda=1\n"
            "  moderate     – num_leaves=31, max_depth=6,  reg_alpha=0.5, reg_lambda=2  (recommended for 60d data)\n"
            "  conservative – num_leaves=31, max_depth=6,  reg_alpha=1.0, reg_lambda=5  (needs ≥90d data)\n"
            "Use 'moderate' or 'conservative' to reduce the train/OOS overfit gap."
        ),
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        default=False,
        help="Skip the strategy backtest simulation step.",
    )
    parser.add_argument(
        "--skip-final-model",
        action="store_true",
        default=False,
        help="Skip training and saving final models on the full dataset.",
    )
    return parser.parse_args()


# ── pipeline steps ─────────────────────────────────────────────────────────────

def step_load_data(args):
    """Return raw OHLCV bars according to the chosen mode."""
    if args.mode == "synthetic":
        logger.info(
            "=== STEP 1: Generating %d days of synthetic 1-min bars ===",
            args.synth_days,
        )
        return load_synthetic(n_days=args.synth_days)

    elif args.mode == "real":
        provider = getattr(args, "provider", "auto")
        logger.info(
            "=== STEP 1: Downloading SPY %s bars (provider=%s) ===",
            args.interval, provider,
        )
        return load_bars(
            interval=args.interval,
            start=args.start,
            end=args.end,
            period=args.period,
            provider=provider,
        )

    elif args.mode == "file":
        if not args.file_path:
            logger.error("--file-path is required when --mode=file")
            sys.exit(1)
        logger.info(
            "=== STEP 1: Loading bars from local file (provider=file) ===\n"
            "  path: %s",
            args.file_path,
        )
        # Route through the provider layer so the file gets the same
        # validation, gap detection, and normalisation as live providers.
        return load_bars(
            provider="file",
            file_path=args.file_path,
            interval=getattr(args, "interval", "5m"),
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def step_build_dataset(df_raw, horizon_dir=None, horizon_range=None):
    logger.info("=== STEP 2: Building features and labels ===")
    df_model = build_dataset(df_raw, horizon_dir=horizon_dir, horizon_range=horizon_range)
    logger.info(
        "Dataset ready: %d rows, %d columns  (%d features)",
        len(df_model),
        len(df_model.columns),
        len(df_model.columns) - 2,   # subtract y_dir, y_range
    )
    return df_model


def step_walk_forward(df_model, direction_params=None):
    logger.info("=== STEP 3: Walk-forward cross-validation ===")
    wf_results = walk_forward_cv(df_model, direction_params=direction_params)
    logger.info(
        "Walk-forward complete: %d folds, %d OOS rows",
        len(wf_results["fold_results"]),
        len(wf_results["oos_dir_true"]),
    )
    return wf_results


def step_generate_report(wf_results):
    logger.info("=== STEP 4: Generating evaluation report ===")
    proba = wf_results["oos_dir_proba"]
    logger.info(
        "OOS dir proba  min=%.4f  p10=%.4f  p50=%.4f  p90=%.4f  max=%.4f",
        proba.min(), np.percentile(proba, 10), np.percentile(proba, 50),
        np.percentile(proba, 90), proba.max(),
    )
    summary = generate_report(wf_results)
    logger.info("Reports saved to: %s", REPORT_DIR)
    return summary


def step_backtest(df_raw, wf_results, hold_bars=None, threshold=None):
    logger.info("=== STEP 5: Strategy backtest simulation ===")
    kwargs = dict(
        df_raw=df_raw,
        oos_dir_proba=wf_results["oos_dir_proba"],
        oos_index=wf_results["oos_index"],
    )
    if hold_bars is not None:
        kwargs["hold_bars"] = hold_bars
    if threshold is not None:
        kwargs["threshold"] = threshold
    bt_results = run_backtest(**kwargs)
    print_backtest_report(bt_results)
    return bt_results


def step_save_final_models(df_model, direction_params=None):
    """Train on full dataset (90/10 split for early stopping) and save .pkl files."""
    logger.info("=== STEP 6: Training final models on full dataset ===")
    X, y_dir, y_range = split_features_labels(df_model)

    split_idx = int(len(X) * 0.9)
    X_tr,     X_vl     = X.iloc[:split_idx],       X.iloc[split_idx:]
    y_dir_tr,  y_dir_vl  = y_dir.iloc[:split_idx],   y_dir.iloc[split_idx:]
    y_rng_tr,  y_rng_vl  = y_range.iloc[:split_idx],  y_range.iloc[split_idx:]

    dir_model, fi_dir = train_direction_model(X_tr, y_dir_tr, X_vl, y_dir_vl,
                                              params=direction_params)
    rng_model, fi_rng = train_range_model(X_tr, y_rng_tr, X_vl, y_rng_vl)

    dir_path = save_direction_model(dir_model)
    rng_path = save_range_model(rng_model)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fi_dir.to_csv(MODEL_DIR / "final_direction_feature_importance.csv", index=False)
    fi_rng.to_csv(MODEL_DIR / "final_range_feature_importance.csv",    index=False)

    logger.info(
        "Final models saved:\n  direction → %s\n  range     → %s",
        dir_path,
        rng_path,
    )
    return dir_model, rng_model


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   SPY AI Intraday Forecasting Pipeline        ║")
    logger.info("╠══════════════════════════════════════════════╣")
    logger.info("║  mode        : %-29s ║", args.mode)
    if args.mode == "synthetic":
        logger.info("║  synth-days  : %-29s ║", args.synth_days)
    elif args.mode == "real":
        logger.info("║  interval    : %-29s ║", args.interval)
        logger.info("║  period      : %-29s ║", args.period)
        logger.info("║  provider    : %-29s ║", getattr(args, "provider", "auto"))
    if args.horizon_dir is not None:
        logger.info("║  horizon-dir : %-29s ║", args.horizon_dir)
    if args.horizon_range is not None:
        logger.info("║  horizon-rng : %-29s ║", args.horizon_range)
    if args.threshold is not None:
        logger.info("║  threshold   : %-29s ║", args.threshold)
    logger.info("║  reg-preset  : %-29s ║", args.reg_preset)
    logger.info("╚══════════════════════════════════════════════╝")

    # Warn early if requested period exceeds yfinance intraday lookback limits.
    # (Only relevant when provider resolves to yfinance.)
    import os as _os
    _effective_provider = getattr(args, "provider", "auto")
    if _effective_provider == "auto" and not _os.environ.get("POLYGON_API_KEY"):
        _effective_provider = "yfinance"
    _yf_limits = {"5m": 60, "15m": 60, "30m": 60, "1m": 30, "2m": 30}
    if args.mode == "real" and _effective_provider == "yfinance" and args.interval in _yf_limits:
        _period_days = {
            "7d": 7, "14d": 14, "30d": 30, "60d": 58, "90d": 88,
            "1mo": 30, "3mo": 88, "6mo": 180, "1y": 365, "2y": 730,
        }
        req_days = _period_days.get(args.period, 0)
        limit = _yf_limits[args.interval]
        if req_days > limit:
            logger.warning(
                "yfinance limits %s bars to %d calendar days. "
                "--period %s (%dd) will be silently clamped to %dd by the data loader. "
                "Use --provider polygon with POLYGON_API_KEY set for longer history.",
                args.interval, limit, args.period, req_days, limit,
            )

    # 1. Load / generate data
    df_raw = step_load_data(args)
    logger.info(
        "Loaded %d bars  |  %s → %s",
        len(df_raw),
        df_raw.index[0].strftime("%Y-%m-%d"),
        df_raw.index[-1].strftime("%Y-%m-%d"),
    )

    # 2. Features + labels
    df_model = step_build_dataset(df_raw, horizon_dir=args.horizon_dir, horizon_range=args.horizon_range)

    # 3. Walk-forward CV
    _PRESET_MAP = {
        "moderate":    DIRECTION_PARAMS_MODERATE,
        "conservative": DIRECTION_PARAMS_CONSERVATIVE,
    }
    dir_params = _PRESET_MAP.get(args.reg_preset)
    wf_results = step_walk_forward(df_model, direction_params=dir_params)

    if not wf_results["fold_results"]:
        logger.error("No valid folds produced – increase --synth-days or data range.")
        sys.exit(1)

    # 4. Evaluation report
    summary = step_generate_report(wf_results)

    # 5. Backtest  (hold period matches direction horizon)
    if not args.skip_backtest:
        step_backtest(df_raw, wf_results, hold_bars=args.horizon_dir, threshold=args.threshold)

    # 6. Final models
    if not args.skip_final_model:
        step_save_final_models(df_model, direction_params=dir_params)

    # ── Final summary ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║              FINAL SUMMARY                    ║")
    logger.info("╠══════════════════════════════════════════════╣")
    logger.info("║  Direction model (out-of-sample)              ║")
    logger.info("║    AUC      : %-30s ║", f"{summary['dir_auc']:.4f}")
    logger.info("║    Log-loss : %-30s ║", f"{summary['dir_logloss']:.4f}")
    logger.info("║    Brier    : %-30s ║", f"{summary['dir_brier']:.4f}")
    logger.info("╠══════════════════════════════════════════════╣")
    logger.info("║  Range model (out-of-sample)                  ║")
    logger.info("║    MAE      : %-30s ║", f"{summary['rng_mae']:.6f}")
    logger.info("║    RMSE     : %-30s ║", f"{summary['rng_rmse']:.6f}")
    logger.info("║    R²       : %-30s ║", f"{summary['rng_r2']:.4f}")
    logger.info("╠══════════════════════════════════════════════╣")
    logger.info(
        "║  Folds: %-4d   OOS rows: %-18d ║",
        summary["n_folds"],
        summary["n_oos_rows"],
    )
    logger.info("╚══════════════════════════════════════════════╝")
    logger.info("Artefacts written to: %s", REPORT_DIR)


if __name__ == "__main__":
    main()

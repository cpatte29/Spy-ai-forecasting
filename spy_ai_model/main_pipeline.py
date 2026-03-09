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
from config import REPORT_DIR, MODEL_DIR

from data.data_loader import load_synthetic, load_from_yfinance, load_from_file
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
        logger.info(
            "=== STEP 1: Downloading SPY %s bars from yfinance ===", args.interval
        )
        return load_from_yfinance(
            period=args.period,
            interval=args.interval,
            start=args.start,
            end=args.end,
        )

    elif args.mode == "file":
        if not args.file_path:
            logger.error("--file-path is required when --mode=file")
            sys.exit(1)
        logger.info("=== STEP 1: Loading bars from %s ===", args.file_path)
        return load_from_file(args.file_path)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def step_build_dataset(df_raw):
    logger.info("=== STEP 2: Building features and labels ===")
    df_model = build_dataset(df_raw)
    logger.info(
        "Dataset ready: %d rows, %d columns  (%d features)",
        len(df_model),
        len(df_model.columns),
        len(df_model.columns) - 2,   # subtract y_dir, y_range
    )
    return df_model


def step_walk_forward(df_model):
    logger.info("=== STEP 3: Walk-forward cross-validation ===")
    wf_results = walk_forward_cv(df_model)
    logger.info(
        "Walk-forward complete: %d folds, %d OOS rows",
        len(wf_results["fold_results"]),
        len(wf_results["oos_dir_true"]),
    )
    return wf_results


def step_generate_report(wf_results):
    logger.info("=== STEP 4: Generating evaluation report ===")
    summary = generate_report(wf_results)
    logger.info("Reports saved to: %s", REPORT_DIR)
    return summary


def step_backtest(df_raw, wf_results):
    logger.info("=== STEP 5: Strategy backtest simulation ===")
    bt_results = run_backtest(
        df_raw=df_raw,
        oos_dir_proba=wf_results["oos_dir_proba"],
        oos_index=wf_results["oos_index"],
    )
    print_backtest_report(bt_results)
    return bt_results


def step_save_final_models(df_model):
    """Train on full dataset (90/10 split for early stopping) and save .pkl files."""
    logger.info("=== STEP 6: Training final models on full dataset ===")
    X, y_dir, y_range = split_features_labels(df_model)

    split_idx = int(len(X) * 0.9)
    X_tr,     X_vl     = X.iloc[:split_idx],       X.iloc[split_idx:]
    y_dir_tr,  y_dir_vl  = y_dir.iloc[:split_idx],   y_dir.iloc[split_idx:]
    y_rng_tr,  y_rng_vl  = y_range.iloc[:split_idx],  y_range.iloc[split_idx:]

    dir_model, fi_dir = train_direction_model(X_tr, y_dir_tr, X_vl, y_dir_vl)
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
    logger.info("╚══════════════════════════════════════════════╝")

    # 1. Load / generate data
    df_raw = step_load_data(args)
    logger.info(
        "Loaded %d bars  |  %s → %s",
        len(df_raw),
        df_raw.index[0].strftime("%Y-%m-%d"),
        df_raw.index[-1].strftime("%Y-%m-%d"),
    )

    # 2. Features + labels
    df_model = step_build_dataset(df_raw)

    # 3. Walk-forward CV
    wf_results = step_walk_forward(df_model)

    if not wf_results["fold_results"]:
        logger.error("No valid folds produced – increase --synth-days or data range.")
        sys.exit(1)

    # 4. Evaluation report
    summary = step_generate_report(wf_results)

    # 5. Backtest
    if not args.skip_backtest:
        step_backtest(df_raw, wf_results)

    # 6. Final models
    if not args.skip_final_model:
        step_save_final_models(df_model)

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

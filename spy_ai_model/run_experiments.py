"""
run_experiments.py
──────────────────
Automated experiment runner that produces the robustness comparison table.

Experiments
───────────
A) 5m bars / 60-bar direction horizon / default params
   → threshold sweep: 0.525, 0.528, 0.530, 0.535

B) 5m bars / 60-bar direction horizon / moderate params
   → threshold sweep: 0.525, 0.528, 0.530, 0.535

C) 5m bars / 60-bar direction horizon / conservative params
   → threshold sweep: 0.525, 0.528, 0.530, 0.535

D) 15m bars / 4-bar direction horizon / default params
   → threshold sweep: 0.510, 0.515, 0.520, 0.525

Usage
─────
    # Normal (downloads live data, caches to data/raw/):
    python run_experiments.py

    # Re-use cached CSV files (skips yfinance):
    python run_experiments.py --data-5m data/raw/spy_5m.csv --data-15m data/raw/spy_15m.csv

    # Fully offline – use synthetic data (for testing/CI):
    python run_experiments.py --synthetic
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
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_experiments")

from config import DIRECTION_PARAMS_MODERATE, DIRECTION_PARAMS_CONSERVATIVE
from data.data_loader import load_from_yfinance, load_synthetic
from data.dataset_builder import build_dataset
from evaluation.walk_forward import walk_forward_cv
from backtest.strategy_simulation import run_backtest
from sklearn.metrics import roc_auc_score


CACHE_DIR = ROOT / "data" / "raw"


def parse_args():
    p = argparse.ArgumentParser(description="SPY AI experiment comparison runner")
    p.add_argument("--data-5m",  default=None, metavar="PATH",
                   help="Path to pre-downloaded 5m CSV (skips yfinance)")
    p.add_argument("--data-15m", default=None, metavar="PATH",
                   help="Path to pre-downloaded 15m CSV (skips yfinance)")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic data for both experiments (offline mode)")
    return p.parse_args()


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_or_download(interval: str, period: str, cache_path: Path | None,
                      override_path: str | None, synthetic: bool,
                      synth_days: int = 252) -> pd.DataFrame:
    """Load data: explicit path > cache file > yfinance download > synthetic."""
    # Explicit override
    if override_path:
        logger.info("Loading %s bars from %s", interval, override_path)
        df = pd.read_csv(override_path, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        return df

    # Synthetic fallback
    if synthetic:
        logger.info("Synthetic mode: generating %d trading days …", synth_days)
        return load_synthetic(n_days=synth_days)

    # Cache hit
    if cache_path and cache_path.exists():
        logger.info("Loading cached %s bars from %s", interval, cache_path)
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        return df

    # Live download + cache
    logger.info("Downloading %s bars from yfinance …", interval)
    df = load_from_yfinance(period=period, interval=interval)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
        logger.info("Cached %s data → %s", interval, cache_path)
    return df


def _sweep_thresholds(df_raw, wf_results, thresholds, hold_bars):
    """Return a list of backtest summary dicts, one per threshold."""
    rows = []
    for thr in thresholds:
        bt = run_backtest(
            df_raw=df_raw,
            oos_dir_proba=wf_results["oos_dir_proba"],
            oos_index=wf_results["oos_index"],
            threshold=thr,
            hold_bars=hold_bars,
        )
        s = bt.get("summary", {})
        rows.append({
            "threshold":    thr,
            "n_trades":     s.get("n_trades",          0),
            "win_rate":     s.get("win_rate",           float("nan")),
            "avg_net_pnl":  s.get("avg_net_pnl",        float("nan")),
            "sharpe":       s.get("sharpe_ratio",        float("nan")),
            "max_dd":       s.get("max_drawdown",        float("nan")),
            "ann_return":   s.get("annualised_return",   float("nan")),
        })
    return rows


def _run_pipeline(df_raw, horizon_dir, horizon_range, direction_params=None, label=""):
    """Build dataset + walk-forward CV.  Returns (df_raw, wf_results, oos_auc)."""
    logger.info("─" * 60)
    logger.info("EXPERIMENT: %s", label)
    _preset_names = {
        id(DIRECTION_PARAMS_MODERATE):    "moderate",
        id(DIRECTION_PARAMS_CONSERVATIVE): "conservative",
    }
    preset_label = _preset_names.get(id(direction_params), "default") if direction_params else "default"
    logger.info("  interval=%s  horizon_dir=%d  horizon_range=%d  params=%s",
                "inferred", horizon_dir, horizon_range, preset_label)
    logger.info("─" * 60)

    df_model = build_dataset(df_raw, horizon_dir=horizon_dir, horizon_range=horizon_range)
    wf_results = walk_forward_cv(df_model, direction_params=direction_params)

    oos_auc = roc_auc_score(wf_results["oos_dir_true"], wf_results["oos_dir_proba"])
    proba   = wf_results["oos_dir_proba"]
    logger.info(
        "OOS AUC=%.4f  proba: min=%.4f p50=%.4f p90=%.4f max=%.4f",
        oos_auc,
        proba.min(), np.percentile(proba, 50),
        np.percentile(proba, 90), proba.max(),
    )
    return wf_results, oos_auc


def _fmt(val, fmt=".4f"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "  —  "
    return format(val, fmt)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Step 1: Load data (download + cache, or use override / synthetic) ──────
    logger.info("=" * 60)
    df_5m = _load_or_download(
        interval="5m", period="60d",
        cache_path=CACHE_DIR / "spy_5m.csv",
        override_path=args.data_5m,
        synthetic=args.synthetic,
        synth_days=252,
    )
    logger.info("5m-equivalent bars: %d rows", len(df_5m))

    # For 15m, generate fresh synthetic data so bar counts are comparable
    df_15m = _load_or_download(
        interval="15m", period="60d",
        cache_path=CACHE_DIR / "spy_15m.csv",
        override_path=args.data_15m,
        synthetic=args.synthetic,
        synth_days=252,
    )
    logger.info("15m-equivalent bars: %d rows", len(df_15m))

    results_table = []

    # ── Experiment A: 5m / default params ─────────────────────────────────────
    wf_A, auc_A = _run_pipeline(
        df_5m, horizon_dir=60, horizon_range=12,
        direction_params=None,
        label="A: 5m | 60-bar horizon | default params",
    )
    thresholds_5m = [0.525, 0.528, 0.530, 0.535]
    sweep_A = _sweep_thresholds(df_5m, wf_A, thresholds_5m, hold_bars=60)
    for row in sweep_A:
        results_table.append({"experiment": "A-default-5m", "oos_auc": auc_A, **row})

    # ── Experiment B: 5m / moderate params ─────────────────────────────────────
    wf_B, auc_B = _run_pipeline(
        df_5m, horizon_dir=60, horizon_range=12,
        direction_params=DIRECTION_PARAMS_MODERATE,
        label="B: 5m | 60-bar horizon | moderate params",
    )
    sweep_B = _sweep_thresholds(df_5m, wf_B, thresholds_5m, hold_bars=60)
    for row in sweep_B:
        results_table.append({"experiment": "B-moderate-5m", "oos_auc": auc_B, **row})

    # ── Experiment C: 5m / conservative params ─────────────────────────────────
    wf_C, auc_C = _run_pipeline(
        df_5m, horizon_dir=60, horizon_range=12,
        direction_params=DIRECTION_PARAMS_CONSERVATIVE,
        label="C: 5m | 60-bar horizon | conservative params",
    )
    sweep_C = _sweep_thresholds(df_5m, wf_C, thresholds_5m, hold_bars=60)
    for row in sweep_C:
        results_table.append({"experiment": "C-conservative-5m", "oos_auc": auc_C, **row})

    # ── Experiment D: 15m / default params ─────────────────────────────────────
    wf_D, auc_D = _run_pipeline(
        df_15m, horizon_dir=4, horizon_range=4,
        direction_params=None,
        label="D: 15m | 4-bar horizon (1h) | default params",
    )
    thresholds_15m = [0.510, 0.515, 0.520, 0.525]
    sweep_D = _sweep_thresholds(df_15m, wf_D, thresholds_15m, hold_bars=4)
    for row in sweep_D:
        results_table.append({"experiment": "D-default-15m", "oos_auc": auc_D, **row})

    # ── Print comparison table ─────────────────────────────────────────────────
    df_out = pd.DataFrame(results_table)

    HEADER = (
        f"{'Experiment':<22} {'OOS AUC':>8} {'Thr':>6} "
        f"{'Trades':>7} {'WinRate':>8} {'AvgPnL':>9} "
        f"{'Sharpe':>8} {'MaxDD':>8} {'AnnRet':>8}"
    )
    SEP = "─" * len(HEADER)

    print()
    print("=" * len(HEADER))
    print("ROBUSTNESS COMPARISON TABLE")
    print("=" * len(HEADER))
    print(HEADER)
    print(SEP)

    prev_exp = None
    for _, r in df_out.iterrows():
        if prev_exp and r["experiment"] != prev_exp:
            print(SEP)
        prev_exp = r["experiment"]
        print(
            f"{r['experiment']:<22} "
            f"{_fmt(r['oos_auc']):>8} "
            f"{r['threshold']:>6.3f} "
            f"{int(r['n_trades']):>7} "
            f"{_fmt(r['win_rate']):>8} "
            f"{_fmt(r['avg_net_pnl'], '.5f'):>9} "
            f"{_fmt(r['sharpe']):>8} "
            f"{_fmt(r['max_dd']):>8} "
            f"{_fmt(r['ann_return']):>8}"
        )

    print("=" * len(HEADER))
    print()

    # ── Recommendation ─────────────────────────────────────────────────────────
    # Score each row: prefer high Sharpe, low |MaxDD|, n_trades >= 5
    scored = df_out.copy()
    scored = scored[scored["n_trades"] >= 5].copy()
    if not scored.empty:
        scored["score"] = (
            scored["sharpe"].fillna(0)
            - 2 * scored["max_dd"].fillna(0).abs()      # penalise drawdown
            + 0.5 * scored["win_rate"].fillna(0)
        )
        best = scored.loc[scored["score"].idxmax()]
        print("RECOMMENDATION")
        print(f"  Best setup  : {best['experiment']}  threshold={best['threshold']:.3f}")
        print(f"  OOS AUC     : {best['oos_auc']:.4f}")
        print(f"  Trades      : {int(best['n_trades'])}")
        print(f"  Win rate    : {best['win_rate']:.4f}")
        print(f"  Sharpe      : {best['sharpe']:.4f}")
        print(f"  Max DD      : {best['max_dd']:.4f}")
        print(f"  Ann return  : {best['ann_return']:.4f}")
    else:
        print("No setup produced ≥ 5 trades with these thresholds.")

    # Save to CSV
    out_path = ROOT / "evaluation" / "reports" / "experiment_comparison.csv"
    df_out.to_csv(out_path, index=False)
    logger.info("Experiment table saved → %s", out_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/run_confluence_backtest.py
───────────────────────────────────
Walk-forward backtest comparing forecast-only, zone-filtered, and
confluence-enhanced strategies.

Usage
─────
  python scripts/run_confluence_backtest.py \
      --file-path data/raw/spy_5m_polygon.parquet \
      --interval 5m \
      --horizon 12

  # Weight tuning experiment:
  python scripts/run_confluence_backtest.py \
      --file-path data/raw/spy_5m_polygon.parquet \
      --model-weight 0.7 --zone-weight 0.2 --analog-weight 0.1

  # Smoke test with synthetic data:
  python scripts/run_confluence_backtest.py --synthetic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_confluence_backtest")

SEP  = "=" * 52
DASH = "-" * 52


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPY Confluence Backtest — forecast / zone-filtered / confluence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--provider",   default="file",
                   choices=["file", "yfinance", "polygon"],
                   help="Data provider (default: file).")
    p.add_argument("--file-path",  default=None, metavar="PATH",
                   help="Local bar file (parquet or csv.gz).")
    p.add_argument("--interval",   default="5m", metavar="INTERVAL",
                   help="Bar interval (default: 5m).")
    p.add_argument("--lookback",   default=30, type=int, metavar="DAYS",
                   help="Days to fetch when provider=polygon/yfinance (default: 30).")
    p.add_argument("--symbol",     default="SPY", metavar="TICKER")

    p.add_argument("--start-idx",  default=500, type=int, metavar="N",
                   help="Warm-up bars before scoring starts (default: 500).")
    p.add_argument("--horizon",    default=12, type=int, metavar="BARS",
                   help="Forward bars for trade exit (default: 12).")
    p.add_argument("--zone-lookback", default=1000, type=int, metavar="BARS",
                   help="Bars of history for zone detection (default: 1000).")
    p.add_argument("--min-label",  default="MODERATE",
                   choices=["MODERATE", "STRONG"],
                   help="Minimum confluence label to enter a trade (default: MODERATE).")
    p.add_argument("--zone-step",  default=20, type=int, metavar="N",
                   help="Recompute zone context every N bars (default: 20). "
                        "Lower = more accurate but slower.")

    p.add_argument("--strong-score",   default=0.35, type=float)
    p.add_argument("--moderate-score", default=0.12, type=float)

    p.add_argument("--model-weight",  default=None, type=float, metavar="W")
    p.add_argument("--zone-weight",   default=None, type=float, metavar="W")
    p.add_argument("--analog-weight", default=None, type=float, metavar="W")

    p.add_argument("--synthetic", action="store_true", default=False,
                   help="Use synthetic bars for a smoke test.")
    p.add_argument("--save-csv",  default=None, metavar="PATH",
                   help="Save per-bar results to CSV.")
    p.add_argument("--verbose",   action="store_true", default=False)
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_bars(args: argparse.Namespace) -> pd.DataFrame:
    if args.synthetic:
        from data.data_loader import load_synthetic
        print("  Using synthetic bars …")
        return load_synthetic(n_days=60)

    from data.data_loader import load_bars
    kwargs: dict = dict(interval=args.interval, symbol=args.symbol)
    if args.provider == "file":
        if not args.file_path:
            print("ERROR: --file-path is required when --provider file", file=sys.stderr)
            sys.exit(1)
        kwargs.update(provider="file", file_path=args.file_path,
                      lookback_days=args.lookback)
    else:
        kwargs.update(provider=args.provider, lookback_days=args.lookback)
    return load_bars(**kwargs)


def _load_models_and_features(df_raw: pd.DataFrame):
    from features.feature_engineering import build_features
    from models.train_direction        import load_direction_model
    from models.train_range            import load_range_model

    dir_model = load_direction_model()
    rng_model = load_range_model()
    feats     = build_features(df_raw)
    train_cols = dir_model.booster_.feature_name()
    return dir_model, rng_model, feats, train_cols


def _print_metrics_table(results: dict) -> None:
    strats = [("Forecast-only",  results["forecast_only"]),
              ("Zone-filtered",  results["zone_filtered"]),
              ("Confluence",     results["confluence"])]

    header = f"{'Strategy':<18} {'Trades':>7} {'Win%':>7} {'AvgRet':>9} {'Sharpe':>8} {'MaxDD':>9} {'PF':>7}"
    print(SEP)
    print(header)
    print(DASH)
    for name, m in strats:
        trades = m.get("trades", 0)
        if trades == 0:
            print(f"  {name:<16}  (no trades)")
            continue
        wr  = m.get("win_rate",      float("nan"))
        ar  = m.get("avg_ret",       float("nan"))
        sh  = m.get("sharpe",        float("nan"))
        dd  = m.get("max_dd",        float("nan"))
        pf  = m.get("profit_factor", float("nan"))

        def _f(v, fmt):
            return fmt.format(v) if v == v else "  nan"   # nan-safe

        print(
            f"  {name:<16} "
            f"{trades:>7d} "
            f"{_f(wr*100, '{:6.1f}%'):>7} "
            f"{_f(ar*1e4,  '{:+7.2f}'):>9} "
            f"{_f(sh,      '{:+7.3f}'):>8} "
            f"{_f(dd*1e4,  '{:+7.1f}'):>9} "
            f"{_f(pf,      '{:7.2f}'):>7}"
        )
    print(SEP)
    print("  AvgRet and MaxDD in basis points (1bp = 0.01%)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    print("\nLoading bars …")
    df_raw = _load_bars(args)
    if df_raw.empty:
        print("ERROR: no bars loaded.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(df_raw)} bars  {df_raw.index[0]}  →  {df_raw.index[-1]}")

    print("Loading models and building features …")
    try:
        dir_model, rng_model, feats, train_cols = _load_models_and_features(df_raw)
    except FileNotFoundError as e:
        print(f"  ⚠  Model not found: {e}")
        print("     Run main_pipeline.py first to train and save model artifacts.")
        sys.exit(1)
    except Exception as e:
        print(f"  ⚠  {e}")
        sys.exit(1)

    n_oos = len(df_raw) - args.start_idx - args.horizon
    print(f"  {len(train_cols)} training features  |  "
          f"{n_oos} OOS bars to score  "
          f"(horizon={args.horizon})")

    print("Running backtest …")
    from confluence_engine.confluence_backtest import run_backtest
    results = run_backtest(
        df_bars        = df_raw,
        dir_model      = dir_model,
        rng_model      = rng_model,
        feats          = feats,
        train_cols     = train_cols,
        start_idx      = args.start_idx,
        horizon        = args.horizon,
        interval       = args.interval,
        zone_lookback  = args.zone_lookback,
        strong_score   = args.strong_score,
        moderate_score = args.moderate_score,
        model_weight   = args.model_weight,
        zone_weight    = args.zone_weight,
        analog_weight  = args.analog_weight,
        min_label      = args.min_label,
        zone_step      = args.zone_step,
        progress       = args.verbose,
    )

    print(f"\n  Bars scored: {results['bars_scored']}\n")
    _print_metrics_table(results)

    if args.save_csv:
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results["bar_results"]).to_csv(out_path, index=False)
        print(f"\n  Per-bar results saved to {out_path}")


if __name__ == "__main__":
    main()

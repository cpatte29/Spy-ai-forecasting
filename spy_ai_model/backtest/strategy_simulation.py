"""
strategy_simulation.py
───────────────────────
Simple event-driven backtest driven by the direction model's OOS predictions.

Strategy
─────────
  • At bar t, if P(up) > DIR_PROB_THRESHOLD → enter long (buy close[t]).
  • Hold for exactly HORIZON bars (60 minutes).
  • Exit at close[t + HORIZON].
  • No pyramiding: skip new signals while in a trade.
  • Transaction cost applied one-way on entry and one-way on exit.

Output metrics
──────────────
  total_return, annualised_return, sharpe_ratio, max_drawdown,
  win_rate, avg_trade_pnl, n_trades

Public API
──────────
    results = run_backtest(df_model, oos_dir_proba, oos_index)
    print_backtest_report(results)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    DIR_PROB_THRESHOLD, TRANSACTION_COST_BP, HORIZON, REPORT_DIR
)

logger = logging.getLogger(__name__)


# ── core simulation ───────────────────────────────────────────────────────────

def run_backtest(
    df_raw:        pd.DataFrame,
    oos_dir_proba: np.ndarray,
    oos_index:     pd.DatetimeIndex,
    threshold:     float = DIR_PROB_THRESHOLD,
    cost_bp:       float = TRANSACTION_COST_BP,
) -> dict:
    """
    Parameters
    ----------
    df_raw        : full 1-minute OHLCV DataFrame (used to look up exit prices)
    oos_dir_proba : OOS predicted probabilities aligned with oos_index
    oos_index     : DatetimeIndex of OOS rows
    threshold     : minimum P(up) to enter a trade
    cost_bp       : one-way transaction cost in basis points

    Returns
    -------
    dict with trade log and summary statistics
    """
    cost_frac = cost_bp / 10_000.0

    # Build lookup: datetime → close price
    close_map = df_raw["close"].to_dict()

    signals = pd.Series(oos_dir_proba, index=oos_index)

    trades      = []
    in_trade    = False
    exit_time   = None

    for ts, prob in signals.items():
        # Check if an open trade should be closed
        if in_trade and ts >= exit_time:
            exit_price = close_map.get(ts) or close_map.get(
                min(close_map.keys(), key=lambda k: abs((k - ts).total_seconds()))
            )
            gross_pnl  = (exit_price - entry_price) / entry_price
            net_pnl    = gross_pnl - 2 * cost_frac   # pay cost on entry + exit
            trades[-1].update({
                "exit_time":  ts,
                "exit_price": exit_price,
                "gross_pnl":  gross_pnl,
                "net_pnl":    net_pnl,
            })
            in_trade = False

        # Open new trade if signal is strong enough and not already in one
        if (not in_trade) and (prob >= threshold):
            entry_price = close_map.get(ts)
            if entry_price is None:
                continue

            # Compute approximate exit time (HORIZON bars later within same session)
            # Find the HORIZON-th bar in oos_index after current
            future_candidates = signals.index[signals.index > ts]
            if len(future_candidates) < HORIZON:
                break   # not enough future bars
            exit_time = future_candidates[HORIZON - 1]

            in_trade = True
            trades.append({
                "entry_time":  ts,
                "entry_price": entry_price,
                "exit_time":   None,
                "exit_price":  None,
                "prob":        prob,
                "gross_pnl":   None,
                "net_pnl":     None,
            })

    # Drop incomplete last trade
    if trades and trades[-1]["net_pnl"] is None:
        trades.pop()

    if not trades:
        logger.warning("No completed trades in backtest.")
        return {"trades": pd.DataFrame(), "summary": {}}

    trade_df = pd.DataFrame(trades)

    # ── Summary statistics ─────────────────────────────────────────────────────
    net_pnls   = trade_df["net_pnl"].values
    n_trades   = len(net_pnls)
    win_rate   = (net_pnls > 0).mean()
    avg_pnl    = net_pnls.mean()
    total_ret  = (1 + net_pnls).prod() - 1.0

    # Equity curve (starting at 1.0)
    equity = np.cumprod(1 + net_pnls)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    max_dd = dd.min()

    # Annualised return – approximate trading days
    days_span = (trade_df["exit_time"].max() - trade_df["entry_time"].min()).days
    if days_span > 0:
        ann_ret = (1 + total_ret) ** (252 / max(days_span, 1)) - 1
    else:
        ann_ret = 0.0

    # Sharpe (trade-level, annualised assuming ~4 trades/day)
    trades_per_year = 4 * 252
    if net_pnls.std() > 0:
        sharpe = (net_pnls.mean() / net_pnls.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    summary = {
        "n_trades":          n_trades,
        "win_rate":          round(win_rate, 4),
        "avg_net_pnl":       round(avg_pnl, 6),
        "total_return":      round(total_ret, 4),
        "annualised_return": round(ann_ret, 4),
        "sharpe_ratio":      round(sharpe, 4),
        "max_drawdown":      round(max_dd, 4),
    }

    return {"trades": trade_df, "summary": summary}


# ── reporting ──────────────────────────────────────────────────────────────────

def print_backtest_report(results: dict, report_dir: Path = REPORT_DIR):
    """Print summary and save trade log + equity curve plot."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary  = results.get("summary", {})
    trade_df = results.get("trades", pd.DataFrame())

    logger.info("=" * 60)
    logger.info("BACKTEST SUMMARY")
    for k, v in summary.items():
        logger.info("  %-25s %s", k, v)
    logger.info("=" * 60)

    if not trade_df.empty:
        trade_df.to_csv(report_dir / "backtest_trades.csv", index=False)
        pd.DataFrame([summary]).to_csv(report_dir / "backtest_summary.csv", index=False)

        # Equity curve plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            equity = np.cumprod(1 + trade_df["net_pnl"].values)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(equity, linewidth=1)
            ax.set_title("Equity Curve (long-only direction strategy)")
            ax.set_xlabel("Trade #")
            ax.set_ylabel("Cumulative return (1 = starting capital)")
            ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(report_dir / "equity_curve.png", dpi=120)
            plt.close(fig)
            logger.info("Equity curve saved → %s", report_dir / "equity_curve.png")
        except Exception as exc:
            logger.warning("Could not save equity curve: %s", exc)

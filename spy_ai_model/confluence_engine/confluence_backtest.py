"""
confluence_backtest.py
──────────────────────
Walk-forward backtest comparing three strategies on held-out bar data:

  1. Forecast-only  – trade whenever the direction model says bullish/bearish
  2. Zone-filtered  – same as (1) but skip if zone context opposes the signal
  3. Confluence     – require a MODERATE or better confluence label

Metrics reported per strategy
──────────────────────────────
  trades        Number of trades entered
  win_rate      Fraction of profitable trades
  avg_ret       Mean return per trade (log-return of the bar range proxy)
  sharpe        Annualised Sharpe ratio of per-trade returns
  max_dd        Maximum drawdown of cumulative equity curve
  profit_factor Gross profit / gross loss

Usage (programmatic)
─────────────────────
  from confluence_engine.confluence_backtest import run_backtest

  results = run_backtest(
      df_bars     = df,          # OHLCV DataFrame
      dir_model   = dir_model,
      rng_model   = rng_model,
      feats       = feats,
      train_cols  = train_cols,
      start_idx   = 500,         # skip first N bars (warm-up)
      interval    = "5m",
  )
  print(results["summary"])

Usage (CLI)
──────────
  python scripts/run_confluence_backtest.py --file-path data/raw/spy_5m_polygon.parquet

Notes
─────
  • Each bar is scored independently using the models' feature row for that bar
    and the zone context of the preceding N bars.
  • Analog engine is skipped by default (slow) unless --use-analog is passed.
  • Trade return is approximated as: sign(signal) × log(close_t+horizon / close_t).
  • No transaction costs or slippage are modelled.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_HORIZON   = 12     # bars forward for the trade exit
_DEFAULT_ZONE_LB   = 1000   # bars of history for zone detection
_STRONG_SCORE      = 0.35
_MODERATE_SCORE    = 0.12
_BARS_PER_YEAR_5M  = 252 * 78   # ~19656 5-min bars per year


# ── helpers ───────────────────────────────────────────────────────────────────

def _signal_direction(label: str) -> int:
    """Return +1 for long, -1 for short, 0 for neutral / N/A."""
    if "LONG"  in label:
        return  1
    if "SHORT" in label:
        return -1
    return 0


def _equity_metrics(returns: list[float]) -> dict:
    """Compute win_rate, avg_ret, sharpe, max_dd, profit_factor from a list of trade log-returns."""
    if not returns:
        return {
            "trades": 0, "win_rate": float("nan"), "avg_ret": float("nan"),
            "sharpe": float("nan"), "max_dd": float("nan"),
            "profit_factor": float("nan"),
        }

    arr   = np.array(returns, dtype=float)
    wins  = arr[arr > 0]
    losses= arr[arr < 0]
    n     = len(arr)

    win_rate      = float(np.mean(arr > 0))
    avg_ret       = float(np.mean(arr))
    std_ret       = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sharpe        = (avg_ret / std_ret * math.sqrt(_BARS_PER_YEAR_5M)) if std_ret > 0 else 0.0
    profit_factor = (float(wins.sum()) / float(-losses.sum())) if losses.size > 0 and losses.sum() < 0 else float("inf")

    # max drawdown on cumulative equity
    cum   = np.cumsum(arr)
    peak  = np.maximum.accumulate(cum)
    dd    = (cum - peak)
    max_dd= float(dd.min())

    return {
        "trades":        n,
        "win_rate":      round(win_rate,      4),
        "avg_ret":       round(avg_ret,       6),
        "sharpe":        round(sharpe,        3),
        "max_dd":        round(max_dd,        6),
        "profit_factor": round(profit_factor, 3),
    }


# ── main backtest function ─────────────────────────────────────────────────────

def run_backtest(
    df_bars:      pd.DataFrame,
    dir_model:    Any,
    rng_model:    Any,
    feats:        pd.DataFrame,
    train_cols:   list[str],
    start_idx:    int          = 500,
    horizon:      int          = _DEFAULT_HORIZON,
    interval:     str          = "5m",
    zone_lookback: int         = _DEFAULT_ZONE_LB,
    use_analog:   bool         = False,
    strong_score:  float       = _STRONG_SCORE,
    moderate_score: float      = _MODERATE_SCORE,
    model_weight:  float | None = None,
    zone_weight:   float | None = None,
    analog_weight: float | None = None,
    min_label:     str         = "MODERATE",   # "STRONG" or "MODERATE"
    zone_step:     int         = 20,           # recompute zones every N bars
    progress:      bool        = True,
) -> dict:
    """
    Run a walk-forward bar-by-bar backtest.

    Parameters
    ──────────
    df_bars       Full OHLCV bar DataFrame.
    dir_model     Trained LightGBM direction model.
    rng_model     Trained LightGBM range model (unused for PnL, kept for API).
    feats         Feature DataFrame aligned to df_bars index.
    train_cols    Feature column names used during model training.
    start_idx     First bar index to start scoring (warm-up period).
    horizon       Bars forward for trade exit.
    interval      Bar interval string for zone context.
    zone_lookback Bars of history fed to zone detector.
    use_analog    If True, run analog query (slow — off by default).
    strong_score  Score threshold for STRONG labels.
    moderate_score Score threshold for MODERATE labels.
    model_weight/zone_weight/analog_weight  Optional weight overrides.
    min_label     Minimum signal strength to enter a confluence trade:
                  "MODERATE" accepts MODERATE_LONG/SHORT and above;
                  "STRONG" accepts only STRONG_LONG/SHORT.
    zone_step     Recompute zone context every N bars (default 20) for speed.
                  Zone detection is O(bars) — recomputing every bar is slow.
    progress      Print a progress dot every 500 bars.

    Returns
    ───────
    dict with keys:
        bars_scored   int
        forecast_only dict  – metrics
        zone_filtered dict  – metrics
        confluence    dict  – metrics
        bar_results   list  – per-bar dict (for further analysis)
    """
    from models.train_direction          import predict_direction_proba
    from models.train_range              import predict_range
    from confluence_engine.zone_context  import get_zone_context
    from confluence_engine.confluence_scorer import compute_confluence

    available_cols = [c for c in train_cols if c in feats.columns]
    X = feats[available_cols].copy()

    n        = len(df_bars)
    end_idx  = n - horizon       # last bar we can score (need horizon bars after)

    fo_returns: list[float] = []
    zf_returns: list[float] = []
    cf_returns: list[float] = []
    bar_results: list[dict] = []

    # Zone context cache — recomputed every zone_step bars for efficiency
    _zone_ctx_cache: dict | None = None
    _zone_ctx_at: int = -999

    for i in range(start_idx, end_idx):
        bar_ts = df_bars.index[i]

        # ── model prediction ──────────────────────────────────────────────
        if bar_ts not in X.index:
            continue
        row = X.loc[[bar_ts]]
        if row.isnull().all(axis=1).iloc[0]:
            continue

        dir_prob  = float(predict_direction_proba(dir_model, row)[0])
        pred_rng  = float(predict_range(rng_model, row)[0])
        forecast_label = (
            "STRONG_LONG"   if dir_prob >= 0.5 + strong_score / 2 else
            "MODERATE_LONG" if dir_prob >= 0.5 + moderate_score / 2 else
            "STRONG_SHORT"  if dir_prob <= 0.5 - strong_score / 2 else
            "MODERATE_SHORT"if dir_prob <= 0.5 - moderate_score / 2 else
            "NEUTRAL"
        )
        fo_dir = _signal_direction(forecast_label)

        # ── zone context (cached, refreshed every zone_step bars) ─────────
        current_price = float(df_bars["close"].iloc[i])

        if _zone_ctx_cache is None or (i - _zone_ctx_at) >= zone_step:
            lb_start = max(0, i - zone_lookback)
            df_window = df_bars.iloc[lb_start : i + 1]
            _zone_ctx_cache = get_zone_context(
                df_window,
                current_price      = current_price,
                bar_interval       = interval,
                zone_lookback_bars = zone_lookback,
            )
            _zone_ctx_at = i
        else:
            # Update current_price in the cached context without re-detecting zones
            _zone_ctx_cache = dict(_zone_ctx_cache)
            _zone_ctx_cache["current_price"] = current_price

        zone_ctx = _zone_ctx_cache

        # Zone filter: skip if zone bias strongly opposes model signal
        zone_bias  = zone_ctx.get("bias", "NEUTRAL")
        zone_opposes = (
            (fo_dir >  0 and zone_bias in ("AT_SUPPLY", "SUPPLY_OVERHEAD")) or
            (fo_dir <  0 and zone_bias in ("AT_DEMAND", "DEMAND_BELOW"))
        )
        zf_dir = 0 if zone_opposes else fo_dir

        # ── confluence score ──────────────────────────────────────────────
        score_out = compute_confluence(
            dir_prob       = dir_prob,
            pred_range     = pred_rng,
            zone_ctx       = zone_ctx,
            analog         = {},
            strong_score   = strong_score,
            moderate_score = moderate_score,
            model_weight   = model_weight,
            zone_weight    = zone_weight,
            analog_weight  = analog_weight,
        )
        cf_label = score_out["label"]

        if min_label == "STRONG":
            cf_dir = _signal_direction(cf_label) if "STRONG" in cf_label else 0
        else:
            cf_dir = _signal_direction(cf_label)

        # ── actual forward return ─────────────────────────────────────────
        exit_close  = float(df_bars["close"].iloc[i + horizon])
        entry_close = current_price
        if entry_close <= 0:
            continue
        log_ret = math.log(exit_close / entry_close)

        # ── accumulate per-strategy returns ───────────────────────────────
        if fo_dir != 0:
            fo_returns.append(fo_dir * log_ret)
        if zf_dir != 0:
            zf_returns.append(zf_dir * log_ret)
        if cf_dir != 0:
            cf_returns.append(cf_dir * log_ret)

        bar_results.append({
            "bar_ts":          bar_ts,
            "current_price":   current_price,
            "dir_prob":        round(dir_prob, 4),
            "forecast_label":  forecast_label,
            "zone_bias":       zone_bias,
            "confluence_label":cf_label,
            "confluence_score":score_out["score"],
            "alignment":       score_out["alignment"],
            "log_ret":         round(log_ret, 6),
            "fo_trade":        fo_dir,
            "zf_trade":        zf_dir,
            "cf_trade":        cf_dir,
        })

        if progress and len(bar_results) % 500 == 0:
            logger.info("Backtest progress: %d / %d bars scored", len(bar_results), end_idx - start_idx)

    return {
        "bars_scored":   len(bar_results),
        "forecast_only": _equity_metrics(fo_returns),
        "zone_filtered": _equity_metrics(zf_returns),
        "confluence":    _equity_metrics(cf_returns),
        "bar_results":   bar_results,
    }

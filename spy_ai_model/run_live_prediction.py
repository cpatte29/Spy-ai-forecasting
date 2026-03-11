"""
run_live_prediction.py
──────────────────────
Fetch the latest SPY 5-minute bars, build features using the exact same
pipeline as training, load the saved direction and range models, and emit
a live prediction.

Outputs
-------
  • Console summary (timestamp, close, dir_probability, range, signal)
  • live_predictions/latest_signal.json

Signal labels
-------------
  LONG_BIAS   P(up) >= threshold
  SHORT_BIAS  P(up) <= (1 - threshold)
  NO_TRADE    probability inside the band

Usage
-----
  python run_live_prediction.py
  python run_live_prediction.py --threshold 0.528
  python run_live_prediction.py --interval 5m --lookback-days 7
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DIR_PROB_THRESHOLD, TICKER
from data.data_loader import load_from_yfinance
from features.feature_engineering import build_features
from models.train_direction import load_direction_model, predict_direction_proba
from models.train_range import load_range_model, predict_range

logger = logging.getLogger(__name__)

LIVE_DIR           = ROOT / "live_predictions"
LATEST_SIGNAL_PATH = LIVE_DIR / "latest_signal.json"

# Minimum number of bars that must exist in today's session before a
# prediction is considered reliable.  ret_30 / rv_30 need 30 prior bars;
# this gate prevents the first few session bars from producing NaN-dominated
# feature vectors.  Historical bars from prior days satisfy the long-window
# features (EMA-200, etc.); this threshold guards the session-reset features.
MIN_SESSION_BARS = 30


# ── helpers ───────────────────────────────────────────────────────────────────

def _count_session_bars(df_feat, ref_ts) -> int:
    """Count bars whose calendar date matches ref_ts (ET tz-naive index)."""
    return int((df_feat.index.date == ref_ts.date()).sum())


def _signal_label(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "LONG_BIAS"
    if prob <= (1.0 - threshold):
        return "SHORT_BIAS"
    return "NO_TRADE"


def _print_prediction(r: dict) -> None:
    bar = "=" * 58
    print(bar)
    print(f"  SPY AI Live Prediction  [{r['generated_at_utc'][:19]} UTC]")
    print(bar)
    print(f"  Bar timestamp   : {r['bar_timestamp']}")
    print(f"  Current close   : ${r['current_close']:.4f}")
    print(f"  Dir probability : {r['dir_probability']:.4f}  (threshold {r['threshold']})")
    print(f"  Predicted range : {r['predicted_range']:.4f}  ({r['range_pts']:.2f} pts)")
    print(f"  Signal          : {r['signal']}")
    print(f"  Session bars    : {r['session_bars']}")
    print(bar)


# ── main ──────────────────────────────────────────────────────────────────────

def run_live_prediction(
    interval:      str        = "5m",
    lookback_days: int        = 7,
    threshold:     float | None = None,
) -> dict:
    """
    Fetch bars, build features, predict, print, and persist to JSON.

    Returns the prediction dict (same structure written to latest_signal.json).
    """
    if threshold is None:
        threshold = DIR_PROB_THRESHOLD

    # 1. Load saved models ────────────────────────────────────────────────────
    dir_model   = load_direction_model()
    range_model = load_range_model()

    # feature_name_ is the ordered list preserved by the LightGBM sklearn API;
    # using it here guarantees the live vector matches the training column order.
    feature_cols: list[str] = dir_model.feature_name_

    # 2. Fetch recent bars ────────────────────────────────────────────────────
    df_raw = load_from_yfinance(
        ticker=TICKER,
        interval=interval,
        period=f"{lookback_days}d",
    )
    if df_raw.empty:
        raise RuntimeError("yfinance returned no bars for the requested period.")

    # 3. Build features – identical call to training pipeline ─────────────────
    df_feat = build_features(df_raw)

    # 4. Session-bar gate ─────────────────────────────────────────────────────
    latest_ts    = df_feat.index[-1]
    session_bars = _count_session_bars(df_feat, latest_ts)

    if session_bars < MIN_SESSION_BARS:
        logger.warning(
            "Only %d session bar(s) available (minimum recommended: %d). "
            "Session-reset features (VWAP slope, short-window RV) may be "
            "unreliable this early in the session.",
            session_bars, MIN_SESSION_BARS,
        )

    # 5. Extract the latest bar's feature vector ───────────────────────────────
    latest_features = df_feat.iloc[[-1]][feature_cols]

    if latest_features.isnull().any(axis=1).item():
        nan_cols = latest_features.columns[latest_features.isnull().any()].tolist()
        raise RuntimeError(
            f"Latest bar ({latest_ts}) has NaN in: {nan_cols}. "
            f"Try --lookback-days {lookback_days + 3} or wait for more session bars."
        )

    # 6. Predict ──────────────────────────────────────────────────────────────
    dir_proba     = float(predict_direction_proba(dir_model,   latest_features)[0])
    pred_range    = float(predict_range(range_model, latest_features)[0])
    current_close = float(df_raw["close"].iloc[-1])
    signal        = _signal_label(dir_proba, threshold)
    now_utc       = datetime.now(timezone.utc).isoformat()

    result = {
        "generated_at_utc": now_utc,
        "bar_timestamp":    latest_ts.isoformat(),
        "ticker":           TICKER,
        "interval":         interval,
        "current_close":    round(current_close, 4),
        "dir_probability":  round(dir_proba, 6),
        "predicted_range":  round(pred_range, 6),
        "range_pts":        round(pred_range * current_close, 3),
        "threshold":        threshold,
        "signal":           signal,
        "session_bars":     session_bars,
    }

    # 7. Print ────────────────────────────────────────────────────────────────
    _print_prediction(result)

    # 8. Save ─────────────────────────────────────────────────────────────────
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_SIGNAL_PATH.write_text(json.dumps(result, indent=2))
    logger.info("Saved latest signal → %s", LATEST_SIGNAL_PATH)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPY AI live prediction – single shot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--interval", default="5m",
        help="Bar interval passed to yfinance (must match model training interval)",
    )
    p.add_argument(
        "--lookback-days", default=7, type=int,
        help="Calendar days of history to fetch (7 is enough for 5m features)",
    )
    p.add_argument(
        "--threshold", default=None, type=float,
        help=f"Entry probability threshold (default from config: {DIR_PROB_THRESHOLD})",
    )
    p.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_live_prediction(
        interval=args.interval,
        lookback_days=args.lookback_days,
        threshold=args.threshold,
    )

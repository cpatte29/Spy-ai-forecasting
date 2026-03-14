"""
run_live_prediction.py
──────────────────────
Fetch the latest SPY bars, build features using the exact same pipeline as
training, load the saved direction and range models, and emit a live prediction.

Improvements over v1
────────────────────
  1. Feature-alignment diagnostic  – always printed; compares training vs live
     feature sets and raises if any training column is absent.
  2. Closed-bar guarantee          – never scores a partially-formed bar; falls
     back to the preceding bar when the latest bar is still in progress.
  3. Combined signal gate          – LONG/SHORT only when *both* the probability
     threshold AND a configurable minimum predicted range are met.
  4. Richer console output         – confidence bucket, range bucket, ET clock,
     elapsed time since last bar, and range-gate pass/fail.
  5. Paper-trade log               – every prediction appended to
     live_predictions/paper_trade_log.csv; realized outcomes backfilled
     automatically once the outcome window has elapsed.
  6. Configurable MIN_SESSION_BARS – default 12 (1 hour in); lower with
     --min-session-bars for early-session use (see note at bottom of file).

Signal labels
─────────────
  LONG_BIAS   P(up) >= threshold  AND  pred_range >= range_gate
  SHORT_BIAS  P(up) <= 1-threshold AND  pred_range >= range_gate
  NO_TRADE    anything else

Range gate modes
─────────────────
  --min-range 0.003          fixed absolute threshold (legacy)
  --range-percentile 0.60    load the 60th-pct threshold from
                             models/saved/range_percentiles.json (calibrated
                             from OOS predictions by compare_range_gates.py)

Usage
─────
  python run_live_prediction.py
  python run_live_prediction.py --threshold 0.55 --range-percentile 0.60
  python run_live_prediction.py --min-range 0.003
  python run_live_prediction.py --min-session-bars 6   # early-session mode
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DIR_PROB_THRESHOLD, HORIZON_DIR, TICKER, MODEL_DIR
from data.data_loader import load_from_yfinance, load_bars
from features.feature_engineering import build_features
from models.train_direction import load_direction_model, predict_direction_proba
from models.train_range import load_range_model, predict_range

logger = logging.getLogger(__name__)

LIVE_DIR             = ROOT / "live_predictions"
LATEST_SIGNAL_PATH   = LIVE_DIR / "latest_signal.json"
PAPER_TRADE_LOG      = LIVE_DIR / "paper_trade_log.csv"
RANGE_CALIBRATION    = MODEL_DIR / "range_percentiles.json"

# ── session-bar threshold ─────────────────────────────────────────────────────
# 12 bars × 5 min = 60 min  →  predictions start around 10:30 ET.
# At this point intraday VWAP is meaningful, short-window momentum is anchored
# in the current session, and the model is well within its training regime.
# Lower with --min-session-bars 6 for early-session use (see MIN_SESSION_BARS
# note at the bottom of this file).
DEFAULT_MIN_SESSION_BARS = 12

# ── bar-duration lookup ───────────────────────────────────────────────────────
_BAR_MINS: dict[str, int] = {
    "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60,
}

# ── paper-trade CSV schema ────────────────────────────────────────────────────
_PAPER_FIELDS = [
    "signal_timestamp",
    "generated_at_utc",
    "signal",
    "entry_price",
    "dir_probability",
    "predicted_range",
    "range_pts",
    "threshold",
    "min_range",
    "threshold_pass",   # prob alone crossed the threshold boundary
    "range_pass",       # predicted_range >= min_range
    "realized_dir",     # 1/0  – backfilled once outcome window elapses
    "realized_range_actual",  # (max_high - min_low)/entry  – backfilled
    "realized_close",   # close at t+horizon  – backfilled
    "backfill_ts",      # UTC timestamp of backfill run
]


# ── 1. Feature-alignment diagnostic ──────────────────────────────────────────

def _verify_feature_alignment(dir_model, df_feat: pd.DataFrame) -> list[str]:
    """
    Compare the model's stored training features to the live-computed set.

    Always prints a diagnostic block showing counts, missing cols, extra cols,
    and the full ordered inference list.  Raises ValueError if any training
    column is absent from the live computation (which would silently break
    the model's learned splits).

    Returns the ordered list of training feature names to use for inference.
    """
    training_cols  = list(dir_model.feature_name_)
    live_cols_set  = set(df_feat.columns)
    training_set   = set(training_cols)

    missing = [c for c in training_cols if c not in live_cols_set]
    extra   = [c for c in df_feat.columns if c not in training_set]

    print("── Feature alignment ───────────────────────────────────")
    print(f"  Training features : {len(training_cols)}")
    print(f"  Live features     : {len(df_feat.columns)}")
    print(f"  Missing cols      : {missing if missing else 'none'}")
    print(f"  Extra cols        : {extra   if extra   else 'none'}")
    print(f"  Inference order (training column order):")
    for i, col in enumerate(training_cols, 1):
        print(f"    {i:2d}. {col}")
    print("────────────────────────────────────────────────────────")

    if missing:
        raise ValueError(
            f"Live feature computation is missing training columns: {missing}. "
            "Re-run the training pipeline or check feature_engineering.py."
        )

    return training_cols


# ── 2. Closed-bar guarantee ───────────────────────────────────────────────────

def _get_scored_bar_ts(df_raw: pd.DataFrame, interval: str) -> pd.Timestamp:
    """
    Return the timestamp of the last *fully closed* bar in df_raw.

    A 5-minute bar with open_ts T is closed when now_et >= T + 5 min.
    yfinance occasionally returns a partially-formed bar for the current
    in-progress period.  When that is detected, we fall back to the
    preceding (definitely closed) bar.

    df_raw.index is tz-naive ET (America/New_York stripped of tz info),
    so now_et is computed in the same reference frame.
    """
    bar_mins = _BAR_MINS.get(interval, 5)
    bar_dur  = pd.Timedelta(minutes=bar_mins)

    try:
        from zoneinfo import ZoneInfo
        tz_et = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        tz_et = pytz.timezone("America/New_York")

    # tz-naive ET, matching df_raw.index
    now_et  = pd.Timestamp(datetime.now(tz_et).replace(tzinfo=None))
    last_ts = df_raw.index[-1]
    elapsed = now_et - last_ts

    if elapsed < bar_dur:
        if len(df_raw) < 2:
            raise RuntimeError(
                f"Latest bar ({last_ts}) is still forming "
                f"({elapsed.total_seconds()/60:.1f}/{bar_mins} min elapsed) "
                "and there is no prior bar to fall back to."
            )
        prev_ts = df_raw.index[-2]
        logger.info(
            "Latest bar (%s) is still forming (%.1f of %d min elapsed). "
            "Scoring previous bar (%s) instead.",
            last_ts, elapsed.total_seconds() / 60, bar_mins, prev_ts,
        )
        return prev_ts

    return last_ts


# ── 3. Range-gate calibration loader ─────────────────────────────────────────

def _load_range_gate(range_percentile: float, min_range: float) -> tuple[float, str]:
    """
    Resolve the effective min-range threshold and a human-readable label.

    Priority:
      1. range_percentile > 0  →  load absolute threshold from calibration file.
      2. min_range > 0         →  use the fixed absolute value directly.
      3. Both zero             →  no gate (return 0.0).

    Returns (effective_threshold, gate_label).
    """
    if range_percentile > 0.0:
        pct_key = str(int(range_percentile * 100))
        if RANGE_CALIBRATION.exists():
            try:
                cal  = json.loads(RANGE_CALIBRATION.read_text())
                thr  = float(cal["percentiles"].get(pct_key, 0.0))
                if thr > 0.0:
                    return thr, f"{int(range_percentile*100)}th-pct ({thr:.6f})"
            except Exception as exc:
                logger.warning("Could not read range calibration: %s", exc)
        logger.warning(
            "range_percentile=%.2f requested but no valid calibration found at %s. "
            "Run compare_range_gates.py first. Falling back to no range gate.",
            range_percentile, RANGE_CALIBRATION,
        )
        return 0.0, "none (calibration missing)"

    if min_range > 0.0:
        return min_range, f"fixed ({min_range:.6f})"

    return 0.0, "none"


# ── 4. Signal + bucket helpers ────────────────────────────────────────────────

def _signal_label(
    prob:       float,
    threshold:  float,
    pred_range: float,
    min_range:  float,
) -> str:
    """
    Combined gate: direction probability AND minimum predicted range.

    LONG_BIAS   prob >= threshold        AND pred_range >= min_range
    SHORT_BIAS  prob <= (1 - threshold)  AND pred_range >= min_range
    NO_TRADE    anything else
    """
    range_ok = pred_range >= min_range
    if prob >= threshold and range_ok:
        return "LONG_BIAS"
    if prob <= (1.0 - threshold) and range_ok:
        return "SHORT_BIAS"
    return "NO_TRADE"


def _confidence_bucket(prob: float) -> str:
    """
    How far the probability sits from the 0.50 random baseline.

    STRONG    |prob - 0.5| >= 0.05   (≥10 percentage-point edge)
    MODERATE  |prob - 0.5| >= 0.025  (≥5 pp edge)
    WEAK      |prob - 0.5| <  0.025  (<5 pp edge)
    """
    dev = abs(prob - 0.5)
    if dev >= 0.05:
        return "STRONG"
    if dev >= 0.025:
        return "MODERATE"
    return "WEAK"


def _range_bucket(pred_range: float) -> str:
    """
    Volatility tier for a predicted normalized price range.
    Thresholds calibrated for SPY 5-minute bars.

    VERY_WIDE  >= 1.0%   (major intraday move expected)
    WIDE       >= 0.6%
    MODERATE   >= 0.3%
    TIGHT      <  0.3%   (low-volatility / sideways bar)
    """
    if pred_range >= 0.010:
        return "VERY_WIDE"
    if pred_range >= 0.006:
        return "WIDE"
    if pred_range >= 0.003:
        return "MODERATE"
    return "TIGHT"


def _count_session_bars(df_feat: pd.DataFrame, ref_ts: pd.Timestamp) -> int:
    """Count bars in df_feat whose calendar date matches ref_ts."""
    return int((df_feat.index.date == ref_ts.date()).sum())


# ── 4. Rich console output ────────────────────────────────────────────────────

def _print_prediction(r: dict) -> None:
    sep  = "=" * 60
    dash = "─" * 60

    bar_ts      = pd.Timestamp(r["bar_timestamp"])
    now_et      = pd.Timestamp(r["now_et"])
    elapsed_min = max(0, int((now_et - bar_ts).total_seconds() // 60))

    prob  = r["dir_probability"]
    arrow = "▲" if prob >= 0.5 else "▼"

    print(sep)
    print(f"  SPY AI Live Prediction  [{r['generated_at_utc'][:19]} UTC]")
    print(sep)
    print(f"  Current time (ET)    : {r['now_et']}")
    print(f"  Last closed bar      : {r['bar_timestamp']}  (+{elapsed_min}m ago)")
    print(f"  Current close        : ${r['current_close']:.4f}")
    print(dash)
    print(f"  Dir probability  {arrow}   : {prob:.4f}  (threshold {r['threshold']})")
    print(f"  Confidence bucket    : {r['confidence_bucket']}")
    print(f"  Predicted range      : {r['predicted_range']:.4f}  ({r['range_pts']:.2f} pts)")
    print(f"  Range bucket         : {r['range_bucket']}")
    gate_thr   = r.get("effective_range_gate", r.get("min_range", 0.0))
    gate_label = r.get("range_gate_label", "none")
    if gate_thr > 0.0:
        gate_pass = "PASS" if r["predicted_range"] >= gate_thr else "FAIL"
        print(f"  Range gate           : {gate_label}  → {gate_pass}")
    print(dash)
    print(f"  Signal               : {r['signal']}")
    bar_mins    = _BAR_MINS.get(r.get("interval", "5m"), 5)
    session_min = r["session_bars"] * bar_mins
    print(f"  Session bars         : {r['session_bars']}  ({session_min} min into session)")
    print(sep)


# ── 5. Paper-trade log ────────────────────────────────────────────────────────

def _append_paper_trade(result: dict, min_range: float) -> bool:
    """
    Append a prediction row to paper_trade_log.csv.

    Stores every prediction (LONG, SHORT, and NO_TRADE) so that the full
    probability distribution can be analyzed in post-session review.

    threshold_pass – prob alone crossed the threshold boundary (independent of
                     the range gate), useful for isolating range-gate impact.
    range_pass     – predicted_range >= min_range.

    Returns True if the row was written, False if already present.
    """
    if PAPER_TRADE_LOG.exists():
        with PAPER_TRADE_LOG.open(newline="") as f:
            seen = {row.get("signal_timestamp", "") for row in csv.DictReader(f)}
        if result["bar_timestamp"] in seen:
            logger.debug("Paper trade already logged for %s.", result["bar_timestamp"])
            return False

    prob = result["dir_probability"]
    thr  = result["threshold"]
    row  = {
        "signal_timestamp":      result["bar_timestamp"],
        "generated_at_utc":      result["generated_at_utc"],
        "signal":                result["signal"],
        "entry_price":           result["current_close"],
        "dir_probability":       result["dir_probability"],
        "predicted_range":       result["predicted_range"],
        "range_pts":             result["range_pts"],
        "threshold":             result["threshold"],
        "min_range":             min_range,
        "threshold_pass":        prob >= thr or prob <= (1.0 - thr),
        "range_pass":            result["predicted_range"] >= min_range,
        "realized_dir":          "",
        "realized_range_actual": "",
        "realized_close":        "",
        "backfill_ts":           "",
    }

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not PAPER_TRADE_LOG.exists()
    with PAPER_TRADE_LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PAPER_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return True


def _try_backfill(df_raw: pd.DataFrame, interval: str, horizon: int) -> int:
    """
    Attempt to fill realized_dir / realized_range_actual / realized_close
    for paper-trade rows whose outcome window has fully elapsed.

    Uses the already-fetched df_raw; no extra network call.

    The direction label mirrors the training definition:
      realized_dir = 1  if  close[signal_ts + horizon * bar_dur] > entry_price
      realized_dir = 0  otherwise

    The range label mirrors y_range from label_builder.py:
      realized_range = (max_high - min_low) / entry_price
      over bars strictly in (signal_ts, signal_ts + horizon * bar_dur]

    Returns the number of rows updated.
    """
    if not PAPER_TRADE_LOG.exists():
        return 0

    bar_mins    = _BAR_MINS.get(interval, 5)
    horizon_dur = pd.Timedelta(minutes=bar_mins * horizon)

    with PAPER_TRADE_LOG.open(newline="") as f:
        rows = list(csv.DictReader(f))

    updated = 0
    for row in rows:
        if row.get("realized_dir"):        # already backfilled
            continue
        try:
            signal_ts   = pd.Timestamp(row["signal_timestamp"])
            entry_price = float(row["entry_price"])
        except (KeyError, ValueError):
            continue

        realized_ts_target = signal_ts + horizon_dur
        future_idx = df_raw.index[df_raw.index >= realized_ts_target]
        if len(future_idx) == 0:
            continue                         # outcome window not in df_raw yet

        realized_ts    = future_idx[0]
        realized_close = float(df_raw.loc[realized_ts, "close"])
        realized_dir   = 1 if realized_close > entry_price else 0

        # Range over the outcome window (identical definition to y_range in training)
        window = df_raw.loc[
            (df_raw.index > signal_ts) & (df_raw.index <= realized_ts)
        ]
        realized_range = (
            (window["high"].max() - window["low"].min()) / entry_price
            if len(window) > 0
            else ""
        )

        row["realized_dir"]          = realized_dir
        row["realized_range_actual"] = f"{realized_range:.6f}" if realized_range != "" else ""
        row["realized_close"]        = f"{realized_close:.4f}"
        row["backfill_ts"]           = datetime.now(timezone.utc).isoformat()
        updated += 1

    if updated > 0:
        with PAPER_TRADE_LOG.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_PAPER_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Backfilled %d paper-trade row(s) with realized outcomes.", updated)

    return updated


# ── main ──────────────────────────────────────────────────────────────────────

def run_live_prediction(
    interval:          str         = "5m",
    lookback_days:     int         = 7,
    threshold:         float | None = None,
    min_range:         float       = 0.0,
    range_percentile:  float       = 0.0,
    min_session_bars:  int         = DEFAULT_MIN_SESSION_BARS,
    horizon:           int | None  = None,
    provider:          str         = "auto",
) -> dict:
    """
    Fetch bars → build features → verify alignment → score last closed bar →
    predict → print → save latest_signal.json → append paper trade → backfill.

    Parameters
    ----------
    interval          Bar interval string (must match model training, e.g. "5m").
    lookback_days     Calendar days of history to fetch.
    threshold         Direction probability threshold (default: config value).
    min_range         Fixed absolute minimum predicted range for a signal.
                      0.0 disables.  Ignored when range_percentile > 0.
    range_percentile  Percentile gate (e.g. 0.60 = 60th pct).  Loads the
                      corresponding absolute threshold from
                      models/saved/range_percentiles.json (written by
                      compare_range_gates.py).  Takes priority over min_range.
    min_session_bars  Minimum session bars before prediction is trusted.
    horizon           Bars ahead for backfill (default: config.HORIZON_DIR).
    """
    if threshold is None:
        threshold = DIR_PROB_THRESHOLD
    if horizon is None:
        horizon = HORIZON_DIR

    # Resolve the effective range gate (percentile overrides fixed min_range)
    effective_range_gate, gate_label = _load_range_gate(range_percentile, min_range)

    # 1. Load models ──────────────────────────────────────────────────────────
    dir_model   = load_direction_model()
    range_model = load_range_model()

    # 2. Fetch recent bars ────────────────────────────────────────────────────
    df_raw = load_bars(
        symbol=TICKER,
        interval=interval,
        lookback_days=lookback_days,
        provider=provider,
        run_health_check=True,   # prints provider / freshness diagnostics
    )
    if df_raw.empty:
        raise RuntimeError(
            f"Provider '{provider}' returned no bars for the requested period. "
            "Check connectivity, API key, and market hours."
        )

    # 3. Build features – identical call to training pipeline ─────────────────
    df_feat = build_features(df_raw)

    # 4. Verify feature alignment (always printed) ────────────────────────────
    feature_cols = _verify_feature_alignment(dir_model, df_feat)

    # 5. Identify the last fully-closed bar ───────────────────────────────────
    scored_ts    = _get_scored_bar_ts(df_raw, interval)
    session_bars = _count_session_bars(df_feat, scored_ts)

    if session_bars < min_session_bars:
        logger.warning(
            "Only %d session bar(s) available (min_session_bars=%d, "
            "≈%d min into session).  VWAP slope and short-window momentum "
            "features are thin this early.  Prediction issued; treat with caution.",
            session_bars, min_session_bars,
            session_bars * _BAR_MINS.get(interval, 5),
        )

    # 6. Extract feature vector for the scored bar ────────────────────────────
    scored_pos      = df_feat.index.get_loc(scored_ts)
    latest_features = df_feat.iloc[[scored_pos]][feature_cols]

    if latest_features.isnull().any(axis=1).item():
        nan_cols = latest_features.columns[latest_features.isnull().any()].tolist()
        raise RuntimeError(
            f"Scored bar ({scored_ts}) has NaN in: {nan_cols}. "
            f"Try --lookback-days {lookback_days + 3} or wait for more session bars."
        )

    # 7. Predict ──────────────────────────────────────────────────────────────
    dir_proba     = float(predict_direction_proba(dir_model,   latest_features)[0])
    pred_range    = float(predict_range(range_model, latest_features)[0])
    current_close = float(df_raw.iloc[scored_pos]["close"])
    signal        = _signal_label(dir_proba, threshold, pred_range, effective_range_gate)
    conf_bucket   = _confidence_bucket(dir_proba)
    rng_bucket    = _range_bucket(pred_range)
    now_utc       = datetime.now(timezone.utc).isoformat()

    try:
        from zoneinfo import ZoneInfo
        tz_et = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        tz_et = pytz.timezone("America/New_York")
    now_et_str = pd.Timestamp(datetime.now(tz_et).replace(tzinfo=None)).isoformat(
        timespec="seconds"
    )

    result = {
        "generated_at_utc":    now_utc,
        "now_et":              now_et_str,
        "bar_timestamp":       scored_ts.isoformat(),
        "ticker":              TICKER,
        "interval":            interval,
        "current_close":       round(current_close, 4),
        "dir_probability":     round(dir_proba, 6),
        "predicted_range":     round(pred_range, 6),
        "range_pts":           round(pred_range * current_close, 3),
        "threshold":           threshold,
        "min_range":           min_range,
        "range_percentile":    range_percentile,
        "effective_range_gate": round(effective_range_gate, 8),
        "range_gate_label":    gate_label,
        "confidence_bucket":   conf_bucket,
        "range_bucket":        rng_bucket,
        "signal":              signal,
        "session_bars":        session_bars,
    }

    # 8. Print ────────────────────────────────────────────────────────────────
    _print_prediction(result)

    # 9. Save latest_signal.json ──────────────────────────────────────────────
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_SIGNAL_PATH.write_text(json.dumps(result, indent=2))
    logger.info("Saved latest signal → %s", LATEST_SIGNAL_PATH)

    # 10. Paper-trade log + backfill ──────────────────────────────────────────
    _append_paper_trade(result, effective_range_gate)
    backfilled = _try_backfill(df_raw, interval, horizon)
    if backfilled:
        print(f"  [paper trade] Backfilled {backfilled} row(s) with realized outcomes.")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPY AI live prediction – single shot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--interval", default="5m",
        help="Bar interval (must match model training interval)",
    )
    p.add_argument(
        "--lookback-days", default=7, type=int,
        help="Calendar days of history to fetch",
    )
    p.add_argument(
        "--threshold", default=None, type=float,
        help=f"Direction probability threshold (default from config: {DIR_PROB_THRESHOLD})",
    )
    p.add_argument(
        "--min-range", default=0.0, type=float,
        help="Fixed absolute minimum predicted range (0 = disabled; ignored if --range-percentile set)",
    )
    p.add_argument(
        "--range-percentile", default=0.0, type=float,
        help="Percentile gate from calibration file (e.g. 0.60 = 60th pct). "
             "Requires compare_range_gates.py to have been run first.",
    )
    p.add_argument(
        "--min-session-bars", default=DEFAULT_MIN_SESSION_BARS, type=int,
        help="Minimum current-session bars before prediction is trusted",
    )
    p.add_argument(
        "--horizon", default=None, type=int,
        help=f"Bars ahead for backfill (default: {HORIZON_DIR})",
    )
    p.add_argument(
        "--provider", default="auto",
        choices=["auto", "polygon", "yfinance"],
        help=(
            "Market data provider.  'auto' uses Polygon when POLYGON_API_KEY "
            "is set, else yfinance.  Set POLYGON_API_KEY env var before use."
        ),
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
        min_range=args.min_range,
        range_percentile=args.range_percentile,
        min_session_bars=args.min_session_bars,
        horizon=args.horizon,
        provider=args.provider,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MIN_SESSION_BARS ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The original value was 30 (150 min, ~12:00 ET).  This was too conservative.
#
# Feature sensitivity by session depth
# ─────────────────────────────────────
#   ≥ 1 bar   All candlestick geometry, EMA distances (warm from prior days),
#             rv_5/15/30 (warm from prior days), momentum ret_1/3/5/15/30
#             (prior-day anchored for small lags), dist_vwap (trivially 0 at
#             bar 1, meaningful by bar 3).
#
#   ≥ 5 bars  vwap_slope_5 becomes non-NaN.  The model's VWAP slope features
#             are now fully computed.
#
#   ≥ 12 bars (~60 min, ~10:30 ET)  ← DEFAULT
#             ret_15/30 have meaningful intraday anchoring.  VWAP slope is
#             stable.  rv_15/30 are still cross-session but intraday vol
#             context is accumulating.  This is a good general-purpose setting.
#
#   ≥ 30 bars (~150 min, ~12:00 ET)  ← original
#             Purely intraday rv_30.  Maximally conservative; eliminates the
#             morning session entirely.
#
# Recommendation
# ──────────────
# Keep this model as a LATER-MORNING / AFTERNOON bias tool with the default
# of 12 bars (~10:30 ET start).  Do NOT build a separate early-session model
# from the same artifacts; the training data includes early-session bars, but
# the model's accuracy at bar 1–5 is empirically lower because:
#
#   (a) VWAP slope is trivially small / zero → low signal quality.
#   (b) Intraday momentum (ret_15, ret_30) is anchored in yesterday's close,
#       not today's price action.
#   (c) SPY open-range volatility (first 15–30 min) is structurally higher
#       and harder for a single model to generalize.
#
# If you need early-session signals, the right path is a *separately trained*
# model whose training data is restricted to bars 1–12 of each session, so
# it learns the open-range feature distribution explicitly.  Using this model
# with --min-session-bars 6 is acceptable for situational awareness but should
# not be used for trade signals without that retrain.

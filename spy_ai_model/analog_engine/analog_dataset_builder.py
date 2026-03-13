"""
analog_dataset_builder.py
─────────────────────────
Assemble the full historical supply-zone analog dataset and persist it.

Pipeline
────────
  1. Load SPY bars          (yfinance, file, or synthetic)
  2. Detect supply zones    (zone_detector)
  3. Find zone-test events  (event_extractor)
  4. Compute outcomes       (price action AFTER the zone window)
  5. Compute pattern features (pattern_features)
  6. Join everything        → one row per zone-test event
  7. Save to Parquet        → analog_engine/data/supply_zone_events.parquet

Outcome columns
───────────────
  outcome__rejection          bool  – price moved down from zone touch
  outcome__breakout           bool  – price closed above zone_high by > threshold
  outcome__max_move_up_pct    float – max upward move in outcome window / touch_close
  outcome__max_move_down_pct  float – max downward move in outcome window / touch_close
  outcome__close_change_pct   float – (close_at_resolution - touch_close) / touch_close
  outcome__bars_to_resolution int   – bars until breakout or rejection confirmed
  outcome__resolution         str   – "rejection" | "breakout" | "inconclusive"

Usage
─────
  python analog_dataset_builder.py

  # From another module:
  from analog_engine.analog_dataset_builder import build_analog_dataset
  ds = build_analog_dataset()

  # Use a local CSV file instead:
  from analog_engine.analog_dataset_builder import build_analog_dataset
  ds = build_analog_dataset(source="file", file_path="data/spy_1m.parquet")

CLI flags
─────────
  --source        yfinance | file | synthetic   (default: yfinance)
  --file-path     path to local bar file        (required if source=file)
  --period        yfinance period string        (default: 30d)
  --interval      bar interval                  (default: 5m)
  --output        output parquet path
  --pivot-left    pivot detection left bars     (default: 5)
  --pivot-right   pivot detection right bars    (default: 5)
  --min-disp      min displacement %           (default: 0.003)
  --outcome-bars  bars ahead for outcome       (default: 24)
  --log-level     DEBUG|INFO|WARNING|ERROR      (default: INFO)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analog_engine.zone_detector   import detect_supply_zones
from analog_engine.event_extractor import detect_zone_tests
from analog_engine.pattern_features import build_feature_matrix

logger = logging.getLogger(__name__)

# Default output path (inside the analog_engine package)
_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "supply_zone_events.parquet"


# ── outcome computation ────────────────────────────────────────────────────────

def _compute_outcomes(
    df:           pd.DataFrame,
    events:       pd.DataFrame,
    windows:      dict[int, pd.DataFrame],
    outcome_bars: int   = 24,
    reject_pct:   float = 0.003,   # 0.30 % down → rejection
    breakout_pct: float = 0.002,   # 0.20 % close above zone_high → breakout
) -> pd.DataFrame:
    """
    For each event, look at the `outcome_bars` bars AFTER the zone window
    ends and classify the outcome.

    Returns a DataFrame indexed by event_id with outcome__ columns.
    """
    pos_map = {ts: i for i, ts in enumerate(df.index)}
    records: list[dict] = []

    for _, event_row in events.iterrows():
        eid       = int(event_row["event_id"])
        window    = windows.get(eid)
        zone_high = float(event_row["zone_high"])

        if window is None or window.empty:
            records.append(_empty_outcome(eid))
            continue

        # Start of outcome window: first bar after the window
        last_window_ts = window.index[-1]
        if last_window_ts not in pos_map:
            records.append(_empty_outcome(eid))
            continue

        start_pos  = pos_map[last_window_ts] + 1
        end_pos    = min(start_pos + outcome_bars, len(df))
        outcome_df = df.iloc[start_pos:end_pos]

        if outcome_df.empty:
            records.append(_empty_outcome(eid))
            continue

        touch_close = float(window[window["bar_role"] == "zone"]["close"].iloc[-1]
                            if not window[window["bar_role"] == "zone"].empty
                            else window["close"].iloc[-1])

        highs  = outcome_df["high"].to_numpy(dtype=float)
        lows   = outcome_df["low"].to_numpy(dtype=float)
        closes = outcome_df["close"].to_numpy(dtype=float)

        max_up   = (highs.max()  - touch_close) / max(touch_close, 1e-8)
        max_down = (touch_close - lows.min())    / max(touch_close, 1e-8)
        final_chg = (closes[-1]  - touch_close)  / max(touch_close, 1e-8)

        # Resolution logic: whichever threshold is reached first
        resolution      = "inconclusive"
        bars_to_resolve = int(len(outcome_df))

        for i, (h, l, c) in enumerate(zip(highs, lows, closes)):
            down_move = (touch_close - l) / max(touch_close, 1e-8)
            up_close  = (c - zone_high)  / max(zone_high, 1e-8)

            if down_move >= reject_pct and resolution == "inconclusive":
                resolution      = "rejection"
                bars_to_resolve = i + 1
                break
            if up_close >= breakout_pct:
                resolution      = "breakout"
                bars_to_resolve = i + 1
                break

        records.append({
            "event_id":                    eid,
            "outcome__rejection":          resolution == "rejection",
            "outcome__breakout":           resolution == "breakout",
            "outcome__resolution":         resolution,
            "outcome__max_move_up_pct":    round(max_up,    6),
            "outcome__max_move_down_pct":  round(max_down,  6),
            "outcome__close_change_pct":   round(final_chg, 6),
            "outcome__bars_to_resolution": bars_to_resolve,
        })

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records).set_index("event_id")
    return out


def _empty_outcome(eid: int) -> dict:
    return {
        "event_id":                    eid,
        "outcome__rejection":          False,
        "outcome__breakout":           False,
        "outcome__resolution":         "inconclusive",
        "outcome__max_move_up_pct":    0.0,
        "outcome__max_move_down_pct":  0.0,
        "outcome__close_change_pct":   0.0,
        "outcome__bars_to_resolution": 0,
    }


# ── dataset builder ────────────────────────────────────────────────────────────

def build_analog_dataset(
    source:        str         = "yfinance",
    file_path:     str | None  = None,
    period:        str         = "30d",
    interval:      str         = "5m",
    pivot_left:    int         = 5,
    pivot_right:   int         = 5,
    min_disp_pct:  float       = 0.003,
    outcome_bars:  int         = 24,
    bars_before:   int         = 10,
    bars_after:    int         = 10,
    output_path:   Path | None = None,
) -> pd.DataFrame:
    """
    Run the full pipeline and return the analog dataset as a DataFrame.

    Each row = one zone-test event.
    Columns  = event metadata + pattern features + outcome metrics.

    Parameters
    ──────────
    source        "yfinance" | "file" | "synthetic"
    file_path     Required when source="file".
    period        yfinance period string.
    interval      Bar interval string; must be consistent with the data.
    pivot_left/right  Bars either side for pivot detection.
    min_disp_pct  Minimum displacement to qualify as a supply zone.
    outcome_bars  Bars after zone window to evaluate outcome.
    bars_before   Bars to capture before zone touch.
    bars_after    Bars to capture after zone exit.
    output_path   Where to save the parquet; defaults to
                  analog_engine/data/supply_zone_events.parquet.

    Returns
    ───────
    pd.DataFrame – the full analog dataset.
    """
    output_path = output_path or _DEFAULT_OUTPUT

    # ── 1. Load bars ──────────────────────────────────────────────────────────
    logger.info("Loading SPY bars (source=%s, period=%s, interval=%s) …",
                source, period, interval)

    if source == "yfinance":
        from data.data_loader import load_from_yfinance
        df = load_from_yfinance(interval=interval, period=period)
    elif source == "file":
        if not file_path:
            raise ValueError("source='file' requires file_path.")
        from data.data_loader import load_from_file
        df = load_from_file(file_path)
    elif source == "synthetic":
        from data.data_loader import load_synthetic
        df = load_synthetic()
    else:
        raise ValueError(f"Unknown source: {source!r}. Use 'yfinance', 'file', or 'synthetic'.")

    logger.info("Loaded %d bars  (%s → %s).", len(df), df.index[0], df.index[-1])

    # ── 2. Detect supply zones ────────────────────────────────────────────────
    logger.info("Detecting supply zones …")
    zones = detect_supply_zones(
        df,
        pivot_left=pivot_left,
        pivot_right=pivot_right,
        min_displacement_pct=min_disp_pct,
    )
    logger.info("Found %d supply zones.", len(zones))

    if zones.empty:
        logger.warning("No supply zones detected. Try reducing --min-disp or --pivot-left/right.")
        return pd.DataFrame()

    # ── 3. Find zone-test events ──────────────────────────────────────────────
    logger.info("Finding zone-test events …")
    events, windows = detect_zone_tests(
        df, zones,
        bars_before=bars_before,
        bars_after=bars_after,
    )
    logger.info("Found %d zone-test events.", len(events))

    if events.empty:
        logger.warning("No zone-test events found.")
        return pd.DataFrame()

    # ── 4. Compute outcomes ───────────────────────────────────────────────────
    logger.info("Computing outcomes …")
    outcomes = _compute_outcomes(df, events, windows, outcome_bars=outcome_bars)

    # ── 5. Compute pattern features ───────────────────────────────────────────
    logger.info("Computing pattern feature vectors …")
    features = build_feature_matrix(events, windows)

    # ── 6. Join everything ────────────────────────────────────────────────────
    logger.info("Joining metadata, features, and outcomes …")

    # events: indexed 0-N with event_id column
    events_indexed = events.set_index("event_id")

    dataset = events_indexed.join(features, how="left")
    dataset = dataset.join(outcomes,        how="left")

    # Cast outcome bool columns
    for col in ["outcome__rejection", "outcome__breakout"]:
        if col in dataset.columns:
            dataset[col] = dataset[col].astype(bool)

    dataset = dataset.sort_values("touch_timestamp").reset_index()
    logger.info("Dataset shape: %s", dataset.shape)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    logger.info("Saved analog dataset → %s", output_path)

    _print_dataset_summary(dataset)
    return dataset


# ── summary helper ────────────────────────────────────────────────────────────

def _print_dataset_summary(dataset: pd.DataFrame) -> None:
    sep  = "=" * 60
    dash = "─" * 60
    print(sep)
    print("  Analog Dataset Summary")
    print(sep)
    print(f"  Total events         : {len(dataset)}")
    if "touch_timestamp" in dataset.columns:
        print(f"  Date range           : {dataset['touch_timestamp'].iloc[0]}  →  "
              f"{dataset['touch_timestamp'].iloc[-1]}")
    if "outcome__resolution" in dataset.columns:
        vc = dataset["outcome__resolution"].value_counts()
        total = len(dataset)
        for label, cnt in vc.items():
            print(f"  {label:<24}: {cnt:4d}  ({cnt/total*100:.1f} %)")
    if "outcome__max_move_down_pct" in dataset.columns:
        rej = dataset[dataset.get("outcome__rejection", False)]
        if len(rej) > 0:
            print(dash)
            print(f"  Avg rejection move   : {rej['outcome__max_move_down_pct'].mean()*100:.2f} %")
        brk = dataset[dataset.get("outcome__breakout", False)]
        if len(brk) > 0:
            print(f"  Avg breakout move    : {brk['outcome__max_move_up_pct'].mean()*100:.2f} %")
    feature_cols = [c for c in dataset.columns if c.startswith(("approach__", "zone__", "meta__"))]
    print(f"  Feature columns      : {len(feature_cols)}")
    print(sep)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the SPY supply-zone analog dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source",        default="yfinance",
                   choices=["yfinance", "file", "synthetic"])
    p.add_argument("--file-path",     default=None,
                   help="Path to local bar file (required if --source=file)")
    p.add_argument("--period",        default="30d",
                   help="yfinance period string (e.g. 7d, 30d, 60d)")
    p.add_argument("--interval",      default="5m",
                   help="Bar interval (e.g. 1m, 5m, 15m)")
    p.add_argument("--pivot-left",    default=5, type=int)
    p.add_argument("--pivot-right",   default=5, type=int)
    p.add_argument("--min-disp",      default=0.003, type=float,
                   help="Min displacement fraction (e.g. 0.003 = 0.3 %%)")
    p.add_argument("--outcome-bars",  default=24, type=int,
                   help="Bars after zone window to measure outcome")
    p.add_argument("--output",        default=None,
                   help="Output parquet path (default: analog_engine/data/supply_zone_events.parquet)")
    p.add_argument("--log-level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    build_analog_dataset(
        source=args.source,
        file_path=args.file_path,
        period=args.period,
        interval=args.interval,
        pivot_left=args.pivot_left,
        pivot_right=args.pivot_right,
        min_disp_pct=args.min_disp,
        outcome_bars=args.outcome_bars,
        output_path=Path(args.output) if args.output else None,
    )

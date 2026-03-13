"""
zone_detector.py
────────────────
Detect supply zones mechanically from a bar DataFrame.

Definition
──────────
A supply zone is formed when:
  1. A pivot high exists: the bar's high is the highest in a symmetric
     window of (pivot_left + 1 + pivot_right) bars.
  2. Within the next `max_displacement_bars` bars, price falls at least
     `min_displacement_pct` from the pivot high (displacement down).
  3. Optionally, the pivot bar's volume exceeds `volume_spike_factor`
     × the rolling 20-bar average volume.

Zone boundaries
───────────────
  zone_high = pivot bar high
  zone_low  = min(pivot bar open, pivot bar close)
              i.e., the body of the pivot candle — the "base" of the zone
              where supply was clustered before the impulse.

Output columns
──────────────
  creation_timestamp      DatetimeIndex value of the pivot bar
  zone_high               float  – top of the supply zone
  zone_low                float  – bottom of the supply zone (candle body base)
  displacement_size       float  – (pivot_high - min_low) / pivot_high
  displacement_speed      float  – displacement_size / bars_to_min_low
  pivot_bar_volume        int
  avg_volume_20           float  – 20-bar rolling avg volume at pivot bar
  volume_spike_ratio      float  – pivot_volume / avg_volume_20
  bars_to_displacement_low int   – bars from pivot to the displacement trough

Input
─────
  df : pd.DataFrame
      tz-naive DatetimeIndex, columns open/high/low/close/volume
      (standard project bar format from data_loader.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── pivot-high detection ───────────────────────────────────────────────────────

def find_pivot_highs(
    df: pd.DataFrame,
    left:  int = 5,
    right: int = 5,
) -> pd.Series:
    """
    Return a boolean Series marking pivot-high bars.

    A bar at position i is a pivot high when:
        df["high"].iloc[i] == max(df["high"].iloc[i-left : i+right+1])
    (strictly dominant; ties go to the later bar).

    Bars within `left` of the start or `right` of the end cannot be pivots.
    """
    if len(df) < left + right + 1:
        return pd.Series(False, index=df.index)

    highs = df["high"].to_numpy(dtype=float)
    n     = len(highs)
    is_pivot = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        window_max = highs[i - left : i + right + 1].max()
        if highs[i] == window_max:
            # break ties: only the rightmost equal value is the pivot
            if highs[i] > highs[i + 1 : i + right + 1].max() if right > 0 else True:
                is_pivot[i] = True

    return pd.Series(is_pivot, index=df.index)


# ── supply zone detection ──────────────────────────────────────────────────────

def detect_supply_zones(
    df: pd.DataFrame,
    pivot_left:            int   = 5,
    pivot_right:           int   = 5,
    min_displacement_pct:  float = 0.003,   # 0.30 % minimum drop from pivot high
    max_displacement_bars: int   = 20,      # scan at most N bars for the trough
    volume_spike_factor:   float = 1.5,
    require_volume_spike:  bool  = False,
) -> pd.DataFrame:
    """
    Detect supply zones from historical bars.

    Parameters
    ──────────
    df                     Bar DataFrame (project-standard format).
    pivot_left / right     Bars either side required for a pivot high.
    min_displacement_pct   Minimum drop from pivot high to qualify.
    max_displacement_bars  Window (in bars) after the pivot to find the trough.
    volume_spike_factor    Volume multiplier threshold for spike check.
    require_volume_spike   If True, only keep pivots with a volume spike.

    Returns
    ───────
    pd.DataFrame  – one row per detected supply zone, sorted by
                    creation_timestamp ascending.  Empty DataFrame if none found.
    """
    if df.empty:
        return _empty_zones()

    # 20-bar rolling average volume (used for volume spike ratio)
    vol_ma = df["volume"].rolling(20, min_periods=1).mean()

    # map timestamp → positional index for fast slicing
    pos_map: dict = {ts: i for i, ts in enumerate(df.index)}

    pivot_mask = find_pivot_highs(df, left=pivot_left, right=pivot_right)
    pivot_timestamps = df.index[pivot_mask].tolist()

    records: list[dict] = []

    for ts in pivot_timestamps:
        pos       = pos_map[ts]
        row       = df.iloc[pos]
        piv_high  = float(row["high"])
        piv_open  = float(row["open"])
        piv_close = float(row["close"])
        piv_vol   = float(row["volume"])
        avg_vol   = float(vol_ma.iloc[pos])

        # ── optional volume spike check ────────────────────────────────────
        if require_volume_spike and avg_vol > 0:
            if piv_vol < volume_spike_factor * avg_vol:
                continue

        # ── displacement window ────────────────────────────────────────────
        disp_slice = df.iloc[pos + 1 : pos + 1 + max_displacement_bars]
        if disp_slice.empty:
            continue

        min_low_idx = int(disp_slice["low"].argmin())
        min_low_val = float(disp_slice["low"].iloc[min_low_idx])
        min_low_ts  = disp_slice.index[min_low_idx]

        displacement_size = (piv_high - min_low_val) / piv_high
        if displacement_size < min_displacement_pct:
            continue

        bars_to_low        = pos_map[min_low_ts] - pos
        displacement_speed = displacement_size / max(bars_to_low, 1)

        # zone boundaries: body of pivot candle (where resting sell orders sit)
        zone_high = piv_high
        zone_low  = min(piv_open, piv_close)   # candle body base

        records.append({
            "creation_timestamp":       ts,
            "zone_high":                round(zone_high, 4),
            "zone_low":                 round(zone_low,  4),
            "displacement_size":        round(displacement_size, 6),
            "displacement_speed":       round(displacement_speed, 9),
            "pivot_bar_volume":         int(piv_vol),
            "avg_volume_20":            round(avg_vol, 1),
            "volume_spike_ratio":       round(piv_vol / avg_vol if avg_vol > 0 else 1.0, 4),
            "bars_to_displacement_low": int(bars_to_low),
        })

    if not records:
        return _empty_zones()

    zones = (
        pd.DataFrame(records)
        .sort_values("creation_timestamp")
        .reset_index(drop=True)
    )
    return zones


# ── helpers ────────────────────────────────────────────────────────────────────

def _empty_zones() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "creation_timestamp",
        "zone_high",
        "zone_low",
        "displacement_size",
        "displacement_speed",
        "pivot_bar_volume",
        "avg_volume_20",
        "volume_spike_ratio",
        "bars_to_displacement_low",
    ])


def zone_summary(zones: pd.DataFrame) -> None:
    """Print a quick human-readable summary of detected zones."""
    if zones.empty:
        print("No supply zones detected.")
        return

    print(f"Supply zones detected : {len(zones)}")
    print(f"  Date range          : {zones['creation_timestamp'].iloc[0]}  →  "
          f"{zones['creation_timestamp'].iloc[-1]}")
    print(f"  Avg displacement    : {zones['displacement_size'].mean():.4f}  "
          f"({zones['displacement_size'].mean()*100:.2f} %)")
    print(f"  Avg bars to trough  : {zones['bars_to_displacement_low'].mean():.1f}")
    print(f"  Avg vol spike ratio : {zones['volume_spike_ratio'].mean():.2f}×")

"""
event_extractor.py
──────────────────
Detect when price returns to test a known supply zone, then capture the
bar window around each touch event.

Zone-test definition
────────────────────
A zone test occurs at bar i when:
  bar.high  >= zone_low    (price enters the zone from below)
  bar.low   <= zone_high   (bar overlaps the zone)

  AND bar i is at least `min_bars_after_creation` bars after the zone's
  creation bar (prevents scoring the displacement bar itself).

Zone lifecycle
──────────────
  • A zone can be tested at most `max_tests_per_zone` times.
  • A zone is considered "broken" (no longer a valid supply zone) if a bar
    CLOSES above zone_high by more than `broken_threshold_pct` — at that
    point the zone is retired and no further tests are recorded.
  • After each test, the scanner skips forward past the inside-zone bars
    so overlapping touches are counted as one test.

Window structure per event
──────────────────────────
  bars_before  bars running up to (not including) the first touch
  zone_bars    consecutive bars that overlap the zone
  bars_after   bars after the zone exit (or end of data)

  All three are concatenated into a single DataFrame with a "bar_role"
  column: "before" | "zone" | "after".

Output
──────
  events  : pd.DataFrame  – one row per zone-test event, metadata only
  windows : dict[int, pd.DataFrame]  – event_id → raw bar window slice

Events DataFrame columns
────────────────────────
  event_id                int
  zone_idx                int   – index in the zones DataFrame
  creation_timestamp      Timestamp
  touch_timestamp         Timestamp  – first bar that enters the zone
  zone_high               float
  zone_low                float
  zone_width_pct          float  – (zone_high - zone_low) / zone_high
  bars_since_creation     int    – bars from creation to first touch
  approach_return         float  – price return over the `bars_before` approach
  approach_speed          float  – approach_return / bars_before (per-bar)
  bars_inside_zone        int    – how many bars the price spent in the zone
  test_number             int    – 1 = first test, 2 = second test, …
  displacement_size       float  – from zone metadata
  displacement_speed      float  – from zone metadata
  volume_spike_ratio      float  – from zone metadata
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ── main extractor ─────────────────────────────────────────────────────────────

def detect_zone_tests(
    df: pd.DataFrame,
    zones: pd.DataFrame,
    bars_before:             int   = 10,
    bars_after:              int   = 10,
    min_bars_after_creation: int   = 1,
    max_tests_per_zone:      int   = 3,
    broken_threshold_pct:    float = 0.001,   # 0.10 % close above zone_high → broken
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """
    Find all zone-test events for every zone in `zones`.

    Parameters
    ──────────
    df                       Bar DataFrame (project-standard format).
    zones                    Output of zone_detector.detect_supply_zones().
    bars_before              Context bars to capture before the touch.
    bars_after               Context bars to capture after the zone exit.
    min_bars_after_creation  Ignore touches within this many bars of the pivot.
    max_tests_per_zone       Retire the zone after this many tests.
    broken_threshold_pct     Fraction above zone_high that signals a breakout /
                             zone failure.  Zone is retired on that close.

    Returns
    ───────
    (events_df, windows)
      events_df : pd.DataFrame with one row per event (metadata).
      windows   : dict mapping event_id → window DataFrame.
    """
    if zones.empty or df.empty:
        return _empty_events(), {}

    # Positional lookup for fast index arithmetic
    pos_map: dict = {ts: i for i, ts in enumerate(df.index)}

    events:  list[dict]              = []
    windows: dict[int, pd.DataFrame] = {}
    event_id = 0

    for zone_idx, zone in zones.iterrows():
        creation_ts = zone["creation_timestamp"]
        zone_high   = float(zone["zone_high"])
        zone_low    = float(zone["zone_low"])

        if creation_ts not in pos_map:
            logger.debug("Zone creation ts %s not found in df index — skipping.", creation_ts)
            continue

        creation_pos = pos_map[creation_ts]
        scan_start   = creation_pos + min_bars_after_creation
        tests_found  = 0
        zone_broken  = False
        pos          = scan_start

        while pos < len(df) and not zone_broken and tests_found < max_tests_per_zone:
            bar       = df.iloc[pos]
            bar_high  = float(bar["high"])
            bar_low   = float(bar["low"])
            bar_close = float(bar["close"])
            bar_ts    = df.index[pos]

            # ── zone-broken check (close above zone_high) ─────────────────
            if bar_close > zone_high * (1.0 + broken_threshold_pct):
                zone_broken = True
                logger.debug("Zone %d broken at %s (close=%.4f > zone_high=%.4f).",
                             zone_idx, bar_ts, bar_close, zone_high)
                break

            # ── touch check ───────────────────────────────────────────────
            touches = bar_high >= zone_low and bar_low <= zone_high
            if not touches:
                pos += 1
                continue

            # ── first touch bar found ─────────────────────────────────────
            touch_pos = pos

            # Find consecutive inside-zone bars (price remains overlapping zone)
            inside_end_pos = touch_pos
            for j in range(touch_pos, min(touch_pos + 50, len(df))):
                jbar_h = float(df.iloc[j]["high"])
                jbar_l = float(df.iloc[j]["low"])
                if jbar_h >= zone_low and jbar_l <= zone_high:
                    inside_end_pos = j
                else:
                    break

            bars_inside = inside_end_pos - touch_pos + 1

            # ── window boundaries ─────────────────────────────────────────
            window_start = max(0, touch_pos - bars_before)
            window_end   = min(len(df), inside_end_pos + bars_after + 1)

            raw_window = df.iloc[window_start:window_end].copy()

            # Label each bar's role
            roles = []
            for i, wts in enumerate(raw_window.index):
                abs_pos = pos_map[wts]
                if abs_pos < touch_pos:
                    roles.append("before")
                elif abs_pos <= inside_end_pos:
                    roles.append("zone")
                else:
                    roles.append("after")
            raw_window["bar_role"] = roles

            # ── approach return ───────────────────────────────────────────
            if window_start < touch_pos:
                approach_open_close = float(df.iloc[window_start]["close"])
                touch_close         = float(bar["close"])
                approach_return     = (touch_close - approach_open_close) / approach_open_close
                approach_bars       = touch_pos - window_start
                approach_speed      = approach_return / max(approach_bars, 1)
            else:
                approach_return = 0.0
                approach_speed  = 0.0

            # ── record ────────────────────────────────────────────────────
            events.append({
                "event_id":              event_id,
                "zone_idx":              int(zone_idx),
                "creation_timestamp":    creation_ts,
                "touch_timestamp":       bar_ts,
                "zone_high":             zone_high,
                "zone_low":              zone_low,
                "zone_width_pct":        round((zone_high - zone_low) / zone_high, 6),
                "bars_since_creation":   touch_pos - creation_pos,
                "approach_return":       round(approach_return, 6),
                "approach_speed":        round(approach_speed, 8),
                "bars_inside_zone":      bars_inside,
                "test_number":           tests_found + 1,
                "displacement_size":     float(zone["displacement_size"]),
                "displacement_speed":    float(zone["displacement_speed"]),
                "volume_spike_ratio":    float(zone.get("volume_spike_ratio", 1.0)),
            })

            windows[event_id] = raw_window
            event_id     += 1
            tests_found  += 1

            # Skip past the inside-zone bars to avoid double-counting
            pos = inside_end_pos + 1

    if not events:
        return _empty_events(), {}

    events_df = (
        pd.DataFrame(events)
        .sort_values("touch_timestamp")
        .reset_index(drop=True)
    )
    return events_df, windows


# ── helpers ────────────────────────────────────────────────────────────────────

def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "event_id", "zone_idx", "creation_timestamp", "touch_timestamp",
        "zone_high", "zone_low", "zone_width_pct", "bars_since_creation",
        "approach_return", "approach_speed", "bars_inside_zone",
        "test_number", "displacement_size", "displacement_speed",
        "volume_spike_ratio",
    ])


def event_summary(events: pd.DataFrame) -> None:
    """Print a quick human-readable summary of detected zone-test events."""
    if events.empty:
        print("No zone-test events detected.")
        return

    print(f"Zone-test events detected : {len(events)}")
    print(f"  Date range             : {events['touch_timestamp'].iloc[0]}  →  "
          f"{events['touch_timestamp'].iloc[-1]}")
    print(f"  Avg bars since creation: {events['bars_since_creation'].mean():.1f}")
    print(f"  Avg approach return    : {events['approach_return'].mean()*100:.3f} %")
    print(f"  Avg bars inside zone   : {events['bars_inside_zone'].mean():.1f}")
    test_dist = events["test_number"].value_counts().sort_index()
    for t, cnt in test_dist.items():
        print(f"    test #{t}: {cnt} events")

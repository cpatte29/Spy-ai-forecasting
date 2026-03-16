"""
report_formatter.py
───────────────────
Format the assembled confluence data into a human-readable console report.

The formatter takes pre-computed dicts from the three signal layers
(model output, zone context, analog analysis) plus the scorer output
and returns both a printed string and a structured summary dict.

Usage
─────
  from confluence_engine.report_formatter import format_confluence_report

  text, summary = format_confluence_report(
      current_price = 580.42,
      bar_ts        = pd.Timestamp("2024-03-15 14:00:00"),
      interval      = "5m",
      model_out     = {"dir_prob": 0.61, "pred_range": 0.013, ...},
      zone_ctx      = get_zone_context(...),
      analog_out    = run_analog_analysis(...),
      score_out     = compute_confluence(...),
      print_output  = True,
  )
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd

SEP  = "=" * 44
DASH = "-" * 44

# Map label → display string with directional arrows
_LABEL_DISPLAY = {
    "STRONG_LONG":    "▲▲  STRONG_LONG",
    "MODERATE_LONG":  " ▲  MODERATE_LONG",
    "NEUTRAL":        " —  NEUTRAL",
    "MODERATE_SHORT": " ▼  MODERATE_SHORT",
    "STRONG_SHORT":   "▼▼  STRONG_SHORT",
}


def format_confluence_report(
    current_price: float,
    bar_ts:        pd.Timestamp | None,
    interval:      str,
    model_out:     dict,
    zone_ctx:      dict,
    analog_out:    dict | None,
    score_out:     dict,
    symbol:        str = "SPY",
    print_output:  bool = True,
) -> tuple[str, dict]:
    """
    Format the full confluence report.

    Parameters
    ──────────
    current_price  Float close price of the scored bar.
    bar_ts         Timestamp of the scored bar.
    interval       Bar size string for display (e.g. "5m").
    model_out      Dict from the forecast model inference step.
                   Expected keys: dir_prob, pred_range, feature_count, n_bars.
    zone_ctx       Dict from get_zone_context().
    analog_out     Dict from run_analog_analysis() or None.
    score_out      Dict from compute_confluence().
    symbol         Ticker symbol for display.
    print_output   If True, print the report to stdout.

    Returns
    ───────
    (report_text : str, summary : dict)
    """
    lines: list[str] = []

    def _add(s: str = "") -> None:
        lines.append(s)

    # ── header ────────────────────────────────────────────────────────────
    _add(SEP)
    _add(f"  {symbol} Confluence Report")
    _add(SEP)

    ts_str = bar_ts.strftime("%Y-%m-%d %H:%M") if bar_ts else "–"
    _add(f"  Bar          : {ts_str} ET  [{interval}]")
    _add(f"  Current Price: {current_price:.4f}")
    _add(DASH)

    # ── section 1: forecast model ─────────────────────────────────────────
    _add("  Forecast Model:")
    dir_prob   = model_out.get("dir_prob")
    pred_range = model_out.get("pred_range")
    err        = model_out.get("error")

    if err:
        _add(f"    ⚠  {err}")
    elif dir_prob is not None:
        range_pts  = (pred_range or 0.0) * current_price
        _add(f"    Direction Prob : {dir_prob:.4f}  "
             f"({'bullish' if dir_prob >= 0.5 else 'bearish'})")
        _add(f"    Predicted Range: {(pred_range or 0):.4f}  ({range_pts:.2f} pts)")
    else:
        _add("    (model output unavailable)")
    _add(DASH)

    # ── section 2: zone context ───────────────────────────────────────────
    _add("  Zone Context:")
    supply = zone_ctx.get("supply", {})
    demand = zone_ctx.get("demand", {})
    bias   = zone_ctx.get("bias", "NEUTRAL")

    _add(f"    Bias         : {bias}")

    # Supply zone
    if supply.get("zone_high") is not None:
        zh    = supply["zone_high"]
        zl    = supply["zone_low"]
        dist  = supply.get("distance_to_zone_pct")
        flag  = supply.get("inside_zone_flag", False)
        since = supply.get("bars_since_created")
        n_s   = supply.get("n_zones_found", 0)

        inside_str = "  ← INSIDE ZONE" if flag else ""
        dist_str   = f"  (distance: {dist*100:.3f}%)" if dist is not None and not flag else ""
        _add(f"    Nearest Supply : {zl:.4f} – {zh:.4f}{inside_str}{dist_str}")
        if since is not None:
            _add(f"    Bars since created: {since}")
        _add(f"    Supply zones found: {n_s}")
    else:
        _add(f"    Nearest Supply : none detected  ({supply.get('n_zones_found', 0)} zones found)")

    # Demand zone
    if demand.get("zone_high") is not None:
        zh    = demand["zone_high"]
        zl    = demand["zone_low"]
        dist  = demand.get("distance_to_zone_pct")
        flag  = demand.get("inside_zone_flag", False)
        n_d   = demand.get("n_zones_found", 0)

        inside_str = "  ← INSIDE ZONE" if flag else ""
        dist_str   = f"  (distance: {dist*100:.3f}%)" if dist is not None and not flag else ""
        _add(f"    Nearest Demand : {zl:.4f} – {zh:.4f}{inside_str}{dist_str}")
        _add(f"    Demand zones found: {n_d}")
    else:
        _add(f"    Nearest Demand : none detected  ({demand.get('n_zones_found', 0)} zones found)")

    _add(DASH)

    # ── section 3: analog engine ──────────────────────────────────────────
    _add("  Analog Engine:")

    if analog_out is None:
        _add("    (skipped)")
    elif analog_out.get("error"):
        _add(f"    ⚠  {analog_out['error']}")
    else:
        n_matches = analog_out.get("n_matches", 0)
        if n_matches == 0:
            _add("    No analog matches found.")
        else:
            rej  = analog_out.get("rejection_rate",    0.0)
            brk  = analog_out.get("breakout_rate",     0.0)
            inc  = analog_out.get("inconclusive_rate", 0.0)
            sim  = analog_out.get("avg_similarity",    0.0)
            dom  = analog_out.get("dominant_pattern",  "–")
            rejmv = analog_out.get("avg_rejection_move", 0.0)
            brkmv = analog_out.get("avg_breakout_move",  0.0)

            _add(f"    Analog matches   : {n_matches}")
            _add(f"    Avg similarity   : {sim:.4f}")
            _add(f"    Rejection rate   : {rej*100:.1f} %"
                 f"  (avg move: {rejmv*100:.2f} %)")
            _add(f"    Breakout rate    : {brk*100:.1f} %"
                 f"  (avg move: {brkmv*100:.2f} %)")
            _add(f"    Inconclusive     : {inc*100:.1f} %")
            _add(f"    Dominant pattern : {dom}")

    _add(DASH)

    # ── section 4: scorer breakdown ───────────────────────────────────────
    _add("  Score Breakdown:")
    comps   = score_out.get("components", {})
    weights = score_out.get("weights",    {})
    m_score = comps.get("model_score",  0.0)
    z_score = comps.get("zone_score",   0.0)
    a_score = comps.get("analog_score", 0.0)
    total   = score_out.get("score", 0.0)

    _add(f"    Model  ({weights.get('model', 0.5)*100:.0f}%): "
         f"{m_score:+.4f}  →  weighted {m_score*weights.get('model',0.5):+.4f}")
    _add(f"    Zone   ({weights.get('zone', 0.3)*100:.0f}%): "
         f"{z_score:+.4f}  →  weighted {z_score*weights.get('zone',0.3):+.4f}")
    _add(f"    Analog ({weights.get('analog', 0.2)*100:.0f}%): "
         f"{a_score:+.4f}  →  weighted {a_score*weights.get('analog',0.2):+.4f}")
    _add(f"    Total score : {total:+.4f}")

    _add(DASH)

    # ── section 5: confluence label ───────────────────────────────────────
    label      = score_out.get("label", "NEUTRAL")
    label_disp = _LABEL_DISPLAY.get(label, label)

    _add("  Confluence Score:")
    _add()
    _add(f"      {label_disp}")
    _add()
    _add(SEP)

    report_text = "\n".join(lines)

    if print_output:
        print(report_text)

    # ── structured summary (for programmatic use) ─────────────────────────
    summary: dict[str, Any] = {
        "symbol":          symbol,
        "bar_ts":          bar_ts,
        "current_price":   current_price,
        "dir_prob":        model_out.get("dir_prob"),
        "pred_range":      model_out.get("pred_range"),
        "supply_zone_high": supply.get("zone_high"),
        "supply_zone_low":  supply.get("zone_low"),
        "supply_distance":  supply.get("distance_to_zone_pct"),
        "supply_inside":    supply.get("inside_zone_flag", False),
        "demand_zone_high": demand.get("zone_high"),
        "demand_zone_low":  demand.get("zone_low"),
        "demand_distance":  demand.get("distance_to_zone_pct"),
        "demand_inside":    demand.get("inside_zone_flag", False),
        "zone_bias":        zone_ctx.get("bias"),
        "n_supply_zones":   supply.get("n_zones_found", 0),
        "n_demand_zones":   demand.get("n_zones_found", 0),
        "n_analog_matches": analog_out.get("n_matches", 0) if analog_out else 0,
        "rejection_rate":   analog_out.get("rejection_rate") if analog_out else None,
        "breakout_rate":    analog_out.get("breakout_rate")  if analog_out else None,
        "confluence_score": score_out.get("score"),
        "confluence_label": label,
        "model_score":      comps.get("model_score"),
        "zone_score":       comps.get("zone_score"),
        "analog_score":     comps.get("analog_score"),
    }

    return report_text, summary

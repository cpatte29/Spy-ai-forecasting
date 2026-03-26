"""
setup_report.py
───────────────
Format a ranked list of SetupResults into a human-readable terminal report.

Usage
─────
  from setup_scanner.setup_report import format_setup_report

  text = format_setup_report(
      ranked_results,        # list[SetupResult], highest score first
      bar_ts    = ts,
      price     = 667.19,
      interval  = "5m",
  )
  print(text)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from setup_scanner.setup_definitions import (
    AlertState,
    ALERT_GRADE_MAP,
    ALERT_MIN_GRADE,
    SetupGrade,
    SetupResult,
    SetupType,
)
from setup_scanner.volume_integration import volume_regime_label

# ── grade display colours (ANSI) ──────────────────────────────────────────────
# Only used when colour=True is passed to format_setup_report

_ANSI_RESET  = "\033[0m"
_ANSI_BOLD   = "\033[1m"
_ANSI_GREEN  = "\033[92m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RED    = "\033[91m"
_ANSI_CYAN   = "\033[96m"
_ANSI_DIM    = "\033[2m"

_GRADE_COLOUR = {
    SetupGrade.A_PLUS: _ANSI_GREEN  + _ANSI_BOLD,
    SetupGrade.A:      _ANSI_GREEN,
    SetupGrade.B:      _ANSI_YELLOW,
    SetupGrade.C:      _ANSI_DIM,
    SetupGrade.IGNORE: _ANSI_DIM,
}

_DIRECTION_SYMBOL = {
    "LONG":    "▲",
    "SHORT":   "▼",
    "NEUTRAL": "─",
}

_ALERT_PREFIX = {
    AlertState.HIGH_CONVICTION: "🔴 HIGH_CONVICTION",
    AlertState.ACTIONABLE:      "🟡 ACTIONABLE",
    AlertState.WATCHLIST:       "🟢 WATCHLIST",
    AlertState.SILENT:          "   SILENT",
}


def _coloured(text: str, code: str, colour: bool) -> str:
    if not colour:
        return text
    return f"{code}{text}{_ANSI_RESET}"


def _grade_line(result: SetupResult, colour: bool) -> str:
    g    = result.grade.value
    code = _GRADE_COLOUR.get(result.grade, "")
    sym  = _DIRECTION_SYMBOL.get(result.direction.value, "─")
    return _coloured(f"{g:3s}  {sym} {result.setup_type.value}  (score {result.score}/100)", code, colour)


def _fmt_price(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f}"


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v*100:.2f}%"


# ── main report builder ───────────────────────────────────────────────────────

def _model_confidence_label(dir_prob: float | None, colour: bool) -> str:
    """Short model-confidence tag based on distance of dir_prob from 0.5."""
    if dir_prob is None:
        return "n/a"
    edge = abs(dir_prob - 0.50)
    if edge >= 0.10:
        label, code = f"STRONG  ({dir_prob:.2f})", _ANSI_GREEN
    elif edge >= 0.05:
        label, code = f"MODERATE({dir_prob:.2f})", _ANSI_YELLOW
    else:
        label, code = f"WEAK    ({dir_prob:.2f})", _ANSI_RED
    return _coloured(label, code, colour)


def format_setup_report(
    ranked_results: list[SetupResult],
    bar_ts:         pd.Timestamp | str | None = None,
    price:          float | None              = None,
    interval:       str                       = "5m",
    colour:         bool                      = True,
    show_breakdown: bool                      = False,
    show_raw:       bool                      = False,
    max_alts:       int                       = 3,
    ticker:         str                       = "SPY",
) -> str:
    """
    Build the setup scanner terminal report.

    Default (show_breakdown=False): compact 6-line summary.
    With show_breakdown=True: full verbose report including conditions,
    missing confirmations, score breakdown, guidance, and raw signals.

    Parameters
    ──────────
    ranked_results   list[SetupResult] sorted by score descending.
    bar_ts           Timestamp of the scored bar.
    price            Current close price.
    interval         Bar interval string (e.g. "5m").
    colour           If True, add ANSI colour codes.
    show_breakdown   If True, print full verbose report.
    show_raw         If True, print raw signal values dict (breakdown only).
    max_alts         Number of alternative setups to show.
    ticker           Symbol being scanned (displayed in header).

    Returns
    ───────
    Formatted multi-line string.
    """
    dash = "─" * 56

    if bar_ts is not None:
        ts_str = pd.Timestamp(bar_ts).strftime("%Y-%m-%d %H:%M") + " ET"
    else:
        ts_str = "n/a"

    price_str = f"${price:.2f}" if price is not None else "n/a"

    if not ranked_results:
        return f"{dash}\n  {ticker}  {ts_str}  {price_str}  ({interval})  — no results\n{dash}"

    top         = ranked_results[0]
    alert_state = ALERT_GRADE_MAP.get(top.grade, AlertState.SILENT)
    alert_label = _ALERT_PREFIX.get(alert_state, "")
    grade_c     = _GRADE_COLOUR.get(top.grade, "")
    sym         = _DIRECTION_SYMBOL.get(top.direction.value, "─")

    # ── volume context ────────────────────────────────────────────────────
    vol_raw = top.raw or {}
    vol_ctx_for_report = {
        "rvol":                vol_raw.get("rvol"),
        "volume_regime":       vol_raw.get("volume_regime", "NORMAL"),
        "vol_imbalance":       vol_raw.get("vol_imbalance"),
        "imbalance_label":     vol_raw.get("imbalance_label", "NEUTRAL"),
        "breakout_confirmed":  vol_raw.get("breakout_confirmed", False),
        "rejection_confirmed": vol_raw.get("rejection_confirmed", False),
        "low_participation":   vol_raw.get("low_participation", False),
    }
    regime      = vol_ctx_for_report.get("volume_regime", "NORMAL")
    low_p       = vol_ctx_for_report.get("low_participation", False)
    regime_col  = _ANSI_GREEN if regime == "HIGH" else (_ANSI_RED if low_p else "")
    vol_line    = volume_regime_label(vol_ctx_for_report)

    # ── model confidence ──────────────────────────────────────────────────
    dir_prob    = vol_raw.get("dir_prob")
    conf_label  = _model_confidence_label(dir_prob, colour)

    # ── alternatives line ─────────────────────────────────────────────────
    alts = [r for r in ranked_results[1:] if r.setup_type != SetupType.NO_SETUP]
    no_setup_r = [r for r in ranked_results if r.setup_type == SetupType.NO_SETUP]
    alt_parts  = [
        _coloured(
            f"{r.setup_type.value} {r.grade.value}{r.score}",
            _GRADE_COLOUR.get(r.grade, ""),
            colour,
        )
        for r in alts[:2]
    ]
    if no_setup_r:
        ns = no_setup_r[0]
        alt_parts.append(_coloured(f"NO_SETUP {ns.grade.value}{ns.score}", _ANSI_DIM, colour))
    alts_str = "  │  ".join(alt_parts) if alt_parts else "─"

    # ── invalidation / target ─────────────────────────────────────────────
    inv_str = _fmt_price(top.invalidation_level) if top.invalidation_level else "─"
    if top.target_zone is not None:
        lo, hi  = top.target_zone
        tgt_str = f"{_fmt_price(lo)}–{_fmt_price(hi)}"
    else:
        tgt_str = "─"

    # ══════════════════════════════════════════════════════════════════════
    #  COMPACT report  (default)
    # ══════════════════════════════════════════════════════════════════════
    lines: list[str] = [dash]
    lines.append(
        f"  {_coloured(ticker, _ANSI_BOLD, colour)}"
        f"  {ts_str}  {price_str}  ({interval})"
    )
    lines.append(
        f"  {_coloured(top.setup_type.value, grade_c + _ANSI_BOLD, colour)}"
        f"  {_coloured(top.grade.value, grade_c, colour)}"
        f"  {top.score}/100"
        f"  {sym} {top.direction.value}"
        f"   {alert_label}"
    )
    lines.append(f"  Vol   : {_coloured(vol_line, regime_col, colour)}")
    lines.append(f"  Model : {conf_label}")
    lines.append(f"  Inv   : {inv_str}   Target: {tgt_str}")
    lines.append(f"  Alts  : {alts_str}")

    if not show_breakdown:
        lines.append(dash)
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════════
    #  VERBOSE addition  (--breakdown)
    # ══════════════════════════════════════════════════════════════════════
    sep = "=" * 56
    lines.append(sep)

    # Conditions
    if top.conditions_met:
        lines.append("  Why:")
        for cond in top.conditions_met:
            lines.append(f"    • {cond}")

    if top.missing_confirms:
        lines.append("")
        lines.append("  Missing confirmations:")
        for miss in top.missing_confirms:
            lines.append(f"    ○ {miss}")

    # Guidance
    if top.guidance:
        lines.append("")
        lines.append(f"  Guidance: {top.guidance}")

    # Score breakdown
    if top.score_breakdown:
        lines.append("")
        lines.append("  Score breakdown:")
        bd = top.score_breakdown
        component_order = [
            "forecast_prob", "range_forecast", "zone_proximity", "zone_strength",
            "analog_alignment", "structure", "volume_score", "chop_penalty", "confirmation",
        ]
        for key in component_order:
            val   = bd.get(key, 0)
            bar_w = max(0, min(20, int(abs(val) / 2)))
            bar_s = "█" * bar_w
            sign  = " " if val >= 0 else "−"
            label = key.replace("_", " ").capitalize()
            lines.append(f"    {label:20s} {sign}{abs(val):3d} {_coloured(bar_s, _ANSI_CYAN, colour)}")
        lines.append(f"    {'TOTAL':20s}   {bd.get('total', 0):3d}")

    # Raw signals
    if show_raw and top.raw:
        lines.append("")
        lines.append("  Raw signals:")
        for k, v in top.raw.items():
            if v is None:
                continue
            if isinstance(v, float):
                lines.append(f"    {k:30s}: {v:.4f}")
            else:
                lines.append(f"    {k:30s}: {v}")

    lines.append(sep)
    return "\n".join(lines)


def format_replay_summary(
    summary_df,   # pd.DataFrame with per-setup-type stats
    colour: bool = True,
) -> str:
    """
    Format the historical replay evaluation summary table.
    """
    import pandas as pd

    lines: list[str] = []
    sep  = "=" * 80
    dash = "─" * 80

    lines.append(sep)
    lines.append(_coloured("  SPY Setup Scanner — Historical Replay Summary", _ANSI_BOLD, colour))
    lines.append(sep)

    if summary_df is None or summary_df.empty:
        lines.append("  No replay data available.")
        lines.append(sep)
        return "\n".join(lines)

    # Header
    col_w = 30
    lines.append(
        f"  {'Setup Type':{col_w}s}  {'N':>5}  "
        f"{'Win%':>6}  {'AvgMove':>8}  {'AvgScore':>9}  "
        f"{'A+/A':>6}  {'IGNORE':>7}  {'FalsePos%':>10}"
    )
    lines.append(dash)

    for _, row in summary_df.iterrows():
        stype = str(row.get("setup_type", "?"))
        n     = int(row.get("n_signals", 0))
        win   = float(row.get("win_rate",      0.0)) * 100
        move  = float(row.get("avg_move_pct",  0.0)) * 100
        ascore = float(row.get("avg_score",    0.0))
        a_pct = float(row.get("grade_a_plus_a_pct", 0.0)) * 100
        ig_pct= float(row.get("grade_ignore_pct",   0.0)) * 100
        fp    = float(row.get("false_positive_rate", 0.0)) * 100

        # Colour win rate
        if win >= 55:
            w_col = _ANSI_GREEN
        elif win >= 45:
            w_col = _ANSI_YELLOW
        else:
            w_col = _ANSI_RED

        lines.append(
            f"  {stype:{col_w}s}  {n:5d}  "
            f"{_coloured(f'{win:5.1f}%', w_col, colour):>6}  "
            f"{move:+7.2f}%  "
            f"{ascore:8.1f}  "
            f"{a_pct:5.1f}%  "
            f"{ig_pct:6.1f}%  "
            f"{fp:9.1f}%"
        )

    lines.append(sep)
    return "\n".join(lines)

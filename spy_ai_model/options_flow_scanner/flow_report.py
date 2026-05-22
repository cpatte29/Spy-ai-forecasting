"""
flow_report.py
──────────────
Terminal report formatting and CSV alert logging for the options flow scanner.

Report format (default compact)
────────────────────────────────
  ──── NVDA  $882.50  ────────────────────────────────
    BULLISH   A  82/100   ACTIONABLE
    Top: NVDA 2026-04-17 950C   vol 12,840  vol/OI 1.42
    Premium: $2.1M calls vs $0.4M puts  (CP ratio 2.8x)
    Flags: call sweep · concentrated strike/expiry
    Guidance: Research catalyst before entry.
  ────────────────────────────────────────────────────

With --breakdown: adds full condition list, caveats, score breakdown.

CSV log
───────
  options_flow_scanner/logs/flow_alerts.csv
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from options_flow_scanner.flow_scorer import (
    FlowAlertState,
    FlowBias,
    FlowGrade,
    FlowResult,
    FLOW_ALERT_MAP,
    FLOW_ALERT_MIN_GRADE,
)

logger = logging.getLogger(__name__)

# ── log file ───────────────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
LOG_DIR     = _MODULE_DIR / "logs"
LOG_FILE    = LOG_DIR / "flow_alerts.csv"

_CSV_FIELDS = [
    "scan_ts",
    "ticker",
    "stock_price",
    "bias",
    "score",
    "grade",
    "alert_state",
    "call_vol",
    "put_vol",
    "cp_ratio",
    "call_vol_oi",
    "put_vol_oi",
    "total_premium",
    "call_premium",
    "put_premium",
    "top_contract_ticker",
    "top_contract_type",
    "top_contract_strike",
    "top_contract_expiry",
    "top_contract_vol",
    "top_contract_oi",
    "top_contract_vol_oi",
    "call_avg_dte",
    "put_avg_dte",
    "far_otm_dominated",
    "conditions_count",
    "caveats_count",
    "guidance",
]

# ── ANSI colours ───────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_DIM    = "\033[2m"

_GRADE_COLOUR = {
    FlowGrade.A_PLUS: _GREEN  + _BOLD,
    FlowGrade.A:      _GREEN,
    FlowGrade.B:      _YELLOW,
    FlowGrade.C:      _DIM,
    FlowGrade.IGNORE: _DIM,
}

_BIAS_COLOUR = {
    FlowBias.BULLISH: _GREEN,
    FlowBias.BEARISH: _RED,
    FlowBias.MIXED:   _YELLOW,
    FlowBias.NEUTRAL: _DIM,
}

_ALERT_PREFIX = {
    FlowAlertState.HIGH_CONVICTION: "🔴 HIGH_CONVICTION",
    FlowAlertState.ACTIONABLE:      "🟡 ACTIONABLE",
    FlowAlertState.WATCHLIST:       "🟢 WATCHLIST",
    FlowAlertState.SILENT:          "   SILENT",
}

_BIAS_ARROW = {
    FlowBias.BULLISH: "▲",
    FlowBias.BEARISH: "▼",
    FlowBias.MIXED:   "◆",
    FlowBias.NEUTRAL: "─",
}


def _c(text: str, code: str, colour: bool) -> str:
    return f"{code}{text}{_RESET}" if colour else text


def _fmt_contract(top: dict) -> str:
    """Format the top contract as a short label, e.g. 'NVDA 2026-04-17 950C'."""
    if not top:
        return "─"
    ticker  = top.get("opt_ticker", "")
    strike  = top.get("strike", "")
    expiry  = top.get("expiry", "")
    ctype   = (top.get("contract_type") or "").upper()[:1]  # C or P
    vol     = int(top.get("volume", 0))
    voi     = top.get("vol_oi_ratio", 0)

    label = ""
    if ticker:
        label = ticker
    elif strike and expiry:
        label = f"{expiry} {strike}{ctype}"

    if vol:
        label += f"   vol {vol:,}"
    if voi:
        label += f"  vol/OI {voi:.2f}x"
    return label


# ── main report ───────────────────────────────────────────────────────────────

def format_flow_report(
    result:         FlowResult,
    colour:         bool = True,
    show_breakdown: bool = False,
) -> str:
    """
    Format a single-ticker flow result as a terminal report.

    Parameters
    ──────────
    result          FlowResult from score_flow().
    colour          ANSI colour output.
    show_breakdown  If True, include full condition list and score breakdown.

    Returns
    ───────
    Formatted string ready for print().
    """
    dash = "─" * 56
    lines: list[str] = [dash]

    price_str  = f"${result.stock_price:.2f}" if result.stock_price else "n/a"
    grade_col  = _GRADE_COLOUR.get(result.grade, "")
    bias_col   = _BIAS_COLOUR.get(result.bias, "")
    bias_arrow = _BIAS_ARROW.get(result.bias, "─")
    alert_lbl  = _ALERT_PREFIX.get(result.alert_state, "")

    # Header
    lines.append(
        f"  {_c(result.ticker, _BOLD, colour)}"
        f"  {price_str}"
    )

    # Bias + grade + alert
    lines.append(
        f"  {_c(bias_arrow + ' ' + result.bias.value, bias_col + _BOLD, colour)}"
        f"   {_c(result.grade.value, grade_col, colour)}"
        f"  {result.score}/100"
        f"   {alert_lbl}"
    )

    # Top contract
    top = result.top_contract
    lines.append(f"  Top : {_fmt_contract(top)}")

    # Premium summary
    feat = result.features
    cp   = feat.get("cp_ratio", 1.0)
    call_prem = feat.get("call_premium", 0.0)
    put_prem  = feat.get("put_premium", 0.0)
    lines.append(
        f"  Prem: ${call_prem/1e6:.1f}M calls  "
        f"${put_prem/1e6:.1f}M puts  "
        f"(CP {cp:.1f}x)"
    )

    # Detection flags (compact)
    flag_names = [
        d.get("signal_type", "").replace("_", " ").lower()
        for d in result.detections
        if d.get("bias") not in ("NEUTRAL", "AMBIGUOUS")
    ]
    if flag_names:
        lines.append(f"  Flags: {' · '.join(flag_names[:4])}")

    if result.caveats:
        lines.append(
            f"  {_c('⚠ ' + result.caveats[0][:80], _YELLOW, colour)}"
        )

    # Guidance
    lines.append(f"  Guide: {result.guidance[:100]}")

    if not show_breakdown:
        lines.append(dash)
        return "\n".join(lines)

    # ── verbose breakdown ─────────────────────────────────────────────────────
    sep = "=" * 56
    lines.append(sep)

    if result.conditions:
        lines.append("  Reasons:")
        for cond in result.conditions[:6]:
            lines.append(f"    • {cond[:100]}")

    if result.caveats:
        lines.append("")
        lines.append("  Caveats:")
        for cav in result.caveats:
            lines.append(f"    ⚠ {cav[:100]}")

    bd = result.score_breakdown
    if bd:
        lines.append("")
        lines.append("  Score breakdown:")
        component_order = [
            "flow_intensity", "premium_size", "directional_skew",
            "contract_quality", "concentration", "stock_confirm",
            "far_otm_penalty", "mixed_penalty", "low_prem_penalty",
        ]
        for key in component_order:
            val = bd.get(key)
            if val is None:
                continue
            bar_w = max(0, min(20, int(abs(val) / 2)))
            bar_s = "█" * bar_w
            sign  = " " if val >= 0 else "−"
            label = key.replace("_", " ").capitalize()
            lines.append(
                f"    {label:22s} {sign}{abs(val):3d} "
                f"{_c(bar_s, _CYAN, colour)}"
            )
        lines.append(f"    {'TOTAL':22s}   {bd.get('total', 0):3d}")

    lines.append(sep)
    return "\n".join(lines)


def format_flow_summary(
    results:   list[FlowResult],
    min_grade: FlowGrade = FlowGrade.IGNORE,
    colour:    bool      = True,
) -> str:
    """
    Format a multi-ticker summary table sorted by score descending.

    Parameters
    ──────────
    results    list[FlowResult] from a full universe scan.
    min_grade  Minimum grade to include in the table.
    colour     ANSI colour output.

    Returns
    ───────
    Formatted string.
    """
    sep  = "=" * 72
    dash = "─" * 72
    lines: list[str] = [sep]
    lines.append(_c("  Options Flow Scanner — Universe Summary", _BOLD, colour))
    lines.append(sep)

    grade_rank = {
        FlowGrade.A_PLUS: 5, FlowGrade.A: 4, FlowGrade.B: 3,
        FlowGrade.C: 2, FlowGrade.IGNORE: 1,
    }
    min_rank = grade_rank.get(min_grade, 1)

    filtered = [
        r for r in results
        if grade_rank.get(r.grade, 0) >= min_rank
        and r.bias != FlowBias.NEUTRAL
    ]
    filtered.sort(key=lambda r: r.score, reverse=True)

    if not filtered:
        lines.append("  No tickers met the minimum threshold.")
        lines.append(sep)
        return "\n".join(lines)

    # Header row
    lines.append(
        f"  {'Ticker':6s}  {'Price':>8s}  {'Bias':8s}  "
        f"{'Gr':3s}  {'Scr':4s}  {'Alert':16s}  "
        f"{'CP Ratio':8s}  {'Premium':>10s}"
    )
    lines.append(dash)

    for r in filtered:
        price_s  = f"${r.stock_price:.2f}" if r.stock_price else "n/a"
        bias_col = _BIAS_COLOUR.get(r.bias, "")
        g_col    = _GRADE_COLOUR.get(r.grade, "")
        prem_m   = r.features.get("total_premium", 0.0) / 1e6
        cp       = r.features.get("cp_ratio", 1.0)
        alert    = _ALERT_PREFIX.get(r.alert_state, "")

        lines.append(
            f"  {_c(r.ticker, _BOLD, colour):6s}  {price_s:>8s}  "
            f"{_c(r.bias.value, bias_col, colour):8s}  "
            f"{_c(r.grade.value, g_col, colour):3s}  {r.score:4d}  "
            f"{alert:16s}  "
            f"{cp:8.2f}x  ${prem_m:8.2f}M"
        )

    lines.append(sep)
    return "\n".join(lines)


# ── CSV logging ───────────────────────────────────────────────────────────────

def log_flow_alert_csv(
    result:   FlowResult,
    log_file: Path | None = None,
) -> bool:
    """
    Append a flow alert record to the CSV log file.

    Parameters
    ──────────
    result    FlowResult to log.
    log_file  Override log path (defaults to LOG_FILE).

    Returns
    ───────
    True on success, False on error.
    """
    log_path = Path(log_file) if log_file else LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        try:
            from zoneinfo import ZoneInfo
            tz_et = ZoneInfo("America/New_York")
        except ImportError:
            import pytz
            tz_et = pytz.timezone("America/New_York")

        scan_ts = datetime.now(tz_et).strftime("%Y-%m-%d %H:%M:%S")

        top = result.top_contract or {}
        feat = result.features    or {}

        row = {
            "scan_ts":              scan_ts,
            "ticker":               result.ticker,
            "stock_price":          result.stock_price,
            "bias":                 result.bias.value,
            "score":                result.score,
            "grade":                result.grade.value,
            "alert_state":          result.alert_state.value,
            "call_vol":             feat.get("call_vol", ""),
            "put_vol":              feat.get("put_vol", ""),
            "cp_ratio":             feat.get("cp_ratio", ""),
            "call_vol_oi":          feat.get("call_vol_oi", ""),
            "put_vol_oi":           feat.get("put_vol_oi", ""),
            "total_premium":        feat.get("total_premium", ""),
            "call_premium":         feat.get("call_premium", ""),
            "put_premium":          feat.get("put_premium", ""),
            "top_contract_ticker":  top.get("opt_ticker", ""),
            "top_contract_type":    top.get("contract_type", ""),
            "top_contract_strike":  top.get("strike", ""),
            "top_contract_expiry":  top.get("expiry", ""),
            "top_contract_vol":     top.get("volume", ""),
            "top_contract_oi":      top.get("open_interest", ""),
            "top_contract_vol_oi":  top.get("vol_oi_ratio", ""),
            "call_avg_dte":         feat.get("call_avg_dte", ""),
            "put_avg_dte":          feat.get("put_avg_dte", ""),
            "far_otm_dominated":    feat.get("far_otm_dominated", ""),
            "conditions_count":     len(result.conditions),
            "caveats_count":        len(result.caveats),
            "guidance":             result.guidance[:200],
        }

        write_header = (
            not log_path.exists() or log_path.stat().st_size == 0
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=_CSV_FIELDS, extrasaction="ignore"
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        return True

    except Exception as exc:
        logger.error("Failed to log flow alert for %s: %s", result.ticker, exc)
        return False


def emit_flow_alert(result: FlowResult, colour: bool = True) -> None:
    """Print a compact alert line when a ticker crosses the alert threshold."""
    if result.alert_state == FlowAlertState.SILENT:
        return

    grade_col  = _GRADE_COLOUR.get(result.grade, "")
    bias_col   = _BIAS_COLOUR.get(result.bias, "")
    alert_lbl  = _ALERT_PREFIX.get(result.alert_state, "")
    arrow      = _BIAS_ARROW.get(result.bias, "─")

    price_s = f"${result.stock_price:.2f}" if result.stock_price else ""

    print(
        f"  {alert_lbl}  "
        f"{_c(result.ticker, _BOLD, colour):6s} {price_s:>8s}  "
        f"{_c(arrow + ' ' + result.bias.value, bias_col, colour):10s}  "
        f"{_c(result.grade.value, grade_col, colour)} {result.score}/100"
    )

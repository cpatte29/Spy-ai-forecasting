"""
data/quality.py
───────────────
Intraday bar quality validator for SPY OHLCV data.

Works on any DataFrame in the project schema:
    tz-naive ET DatetimeIndex  |  columns: open, high, low, close, volume

Checks
──────
  Per-bar (row-level):
    • OHLCV sanity          – high >= open/close >= low, volume > 0
    • Price return spike    – |log(close[t] / close[t-1])| within session
    • Bar range spike       – (high − low) / close intrabar
    • Inter-bar gap         – |open[t] − close[t-1]| / close[t-1] intraday
    • Volume spike          – rolling z-score and multiple-of-median
    • Timestamp alignment   – minute value on expected interval grid

  Per-session (date-level):
    • Missing bars          – bar count vs expected count for interval
    • Duplicate timestamps  – any dups within the session
    • Session price range   – max(high)/min(low) − 1 (circuit-breaker flag)

  Global:
    • Stale live data       – latest bar vs current ET wall-clock (live mode)

Thresholds (default)
─────────────────────
  All defaults are calibrated for SPY 5-minute bars.  Pass a custom
  QualityThresholds instance to the BarValidator constructor to override.

  max_bar_return        0.5%   – |log-ret| per bar (99.9th pct ≈ 0.35%)
  max_bar_range         1.0%   – (H−L)/close per bar
  max_interbar_gap      0.5%   – intraday open vs prior close
  max_volume_zscore     6.0    – rolling 20-bar log-volume z-score
  max_volume_mult       10.0   – volume / rolling 20-bar median
  min_session_coverage  70%    – fraction of expected bars (allows half-days)
  max_stale_minutes     15.0   – latest bar age in market hours (live mode)

Usage
─────
  # Standalone (returns ValidationResult, prints + saves report):
  from data.quality import validate_bars
  result = validate_bars(df, interval="5m", report_dir=REPORT_DIR)

  # Class-based (full control):
  from data.quality import BarValidator
  v = BarValidator(interval="5m", symbol="SPY")
  r = v.validate(df, live=True)
  v.print_report(r)
  v.save_report(r, report_dir=REPORT_DIR)

  # Compact live-inference check (no report file):
  result = validate_bars(df, interval="5m", live=True,
                         print_output=True, compact=True)
  if result.critical > 0:
      ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]

# ── Session constants ─────────────────────────────────────────────────────────

# Regular session = 09:30 → 15:55 (last bar open) for 5m, 15:59 for 1m, etc.
# Formula: ceil(390 / bar_minutes) where 390 = 6.5h × 60 min
_EXPECTED_BARS: dict[str, int] = {
    "1m":  390,   # 6.5 h × 60
    "2m":  195,   # 6.5 h × 30
    "5m":  78,    # 6.5 h × 12
    "10m": 39,    # 6.5 h × 6
    "15m": 26,    # 6.5 h × 4
    "30m": 13,    # 6.5 h × 2
    "60m": 7,     # ceil(6.5)
    "1h":  7,
    "1d":  1,
}

# Interval → interval length in minutes
_BAR_MINUTES: dict[str, int] = {
    "1m":  1, "2m":  2, "5m":  5,  "10m": 10,
    "15m": 15, "30m": 30, "60m": 60, "1h":  60, "1d":  390,
}

# Valid minute-of-hour values for timestamp alignment check
def _valid_minutes_for_interval(interval: str) -> set[int]:
    bar_min = _BAR_MINUTES.get(interval, 1)
    if bar_min >= 60:
        return {30}          # hourly bars can land anywhere; skip minute check
    return {m for m in range(60) if m % bar_min == 0}


# ── Severity helper ───────────────────────────────────────────────────────────

_SEV_ORDER = {"ok": 0, "info": 1, "warning": 2, "critical": 3}

def _max_sev(a: str, b: str) -> str:
    return a if _SEV_ORDER.get(a, 0) >= _SEV_ORDER.get(b, 0) else b


# ── Thresholds ────────────────────────────────────────────────────────────────

@dataclass
class QualityThresholds:
    """
    Validation thresholds for intraday bar checks.

    All price thresholds are expressed as fractions, not percentages
    (0.005 = 0.5%, 0.010 = 1.0%).

    Defaults are calibrated for SPY 5-minute bars.  Loosen slightly for
    1-minute bars (higher noise) or tighten for longer intervals.
    """

    # ── Per-bar: price ────────────────────────────────────────────────────
    # |log(close[t] / close[t-1])| within a session.
    # SPY 5m 99.9th pct ≈ 0.35%; 0.50% captures clear anomalies.
    # At 2× (1.0%) we escalate to critical.
    max_bar_return:       float = 0.005   # 0.5%

    # (high - low) / close for a single bar.
    # SPY 5m normal: 0.05–0.2%; 1.0% ≈ $5.60 on $560 SPY → clearly bad.
    max_bar_range:        float = 0.010   # 1.0%

    # |open[t] - close[t-1]| / close[t-1] for consecutive intraday bars.
    # Intraday gaps > 0.5% suggest a missing bar or bad tick.
    max_interbar_gap:     float = 0.005   # 0.5%

    # ── Per-bar: volume ───────────────────────────────────────────────────
    # Rolling 20-bar log-volume z-score.
    # 6σ → ~1 in 500 million under normality; catches true outliers.
    max_volume_zscore:    float = 6.0

    # Volume / rolling 20-bar median.  Flags extreme spikes while allowing
    # legitimate open/close surges which are often 2–4× normal.
    max_volume_mult:      float = 10.0

    # ── Per-session ───────────────────────────────────────────────────────
    # Fraction of expected bars in a session.  Below → session flagged.
    # 70% lets genuine half-sessions (early close ≈ 3.5h / 6.5h = 54%) pass
    # at warning level rather than critical.
    min_session_coverage: float = 0.70   # 70%

    # Max (max_high − min_low) / min_low for one session.
    # SPY rarely moves > 5% intraday; above this suggests bad data.
    max_session_range:    float = 0.05   # 5%

    # ── Live / staleness ──────────────────────────────────────────────────
    # Minutes since last bar before flagging as stale in live mode.
    max_stale_minutes:    float = 15.0


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Container for all quality check outputs.

    Attributes
    ──────────
    passed          True when critical == 0.
    warnings        Count of warning-level issues.
    critical        Count of critical-level issues.
    total_bars      Total bar count in the validated DataFrame.
    total_sessions  Unique trading-day count.
    flagged_bars    Bars with at least one quality flag.
    flagged_sessions Sessions with at least one quality flag.
    session_report  pd.DataFrame – one row per session, quality columns.
    bar_flags       pd.DataFrame – one row per flagged bar (may be empty).
    summary         dict – serialisable high-level summary.
    is_stale        True when data is stale (live mode only).
    stale_minutes   Minutes since last bar (live mode only).
    """
    passed:           bool
    warnings:         int
    critical:         int
    total_bars:       int
    total_sessions:   int
    flagged_bars:     int
    flagged_sessions: int
    session_report:   pd.DataFrame
    bar_flags:        pd.DataFrame
    summary:          dict
    is_stale:         bool  = False
    stale_minutes:    float = 0.0


# ── Validator ─────────────────────────────────────────────────────────────────

class BarValidator:
    """
    Comprehensive data quality validator for intraday OHLCV bars.

    Parameters
    ──────────
    interval    Bar interval string: "1m", "5m", "15m", etc.
                Drives expected-bar-count and alignment grid checks.
    symbol      Ticker symbol (informational – appears in report headers).
    thresholds  Custom QualityThresholds instance, or None for SPY-5m defaults.
    """

    def __init__(
        self,
        interval:   str                       = "5m",
        symbol:     str                       = "SPY",
        thresholds: Optional[QualityThresholds] = None,
    ) -> None:
        self.interval    = interval
        self.symbol      = symbol
        self.thr         = thresholds or QualityThresholds()
        self._expected   = _EXPECTED_BARS.get(interval, 78)
        self._bar_min    = _BAR_MINUTES.get(interval, 5)
        self._valid_mins = _valid_minutes_for_interval(interval)

    # ── Public: validate ──────────────────────────────────────────────────────

    def validate(
        self,
        df:   pd.DataFrame,
        live: bool = False,
    ) -> ValidationResult:
        """
        Run all quality checks and return a ValidationResult.

        Parameters
        ──────────
        df    OHLCV bars in project schema (tz-naive ET DatetimeIndex).
              The DataFrame is not modified.
        live  If True, also checks for stale data (latest bar vs. ET now).
        """
        # ── 0. Schema guard ────────────────────────────────────────────────
        if df.empty:
            return self._empty_result()

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"BarValidator: DataFrame is missing columns {missing}. "
                f"Available: {list(df.columns)}"
            )

        # ── 1. Pre-compute helpers ─────────────────────────────────────────
        # same_session[i] = True when bar i shares its date with bar i-1
        # (i.e., it is NOT the first bar of a new session)
        dates       = pd.Series(df.index.date, index=df.index)
        same_session = (dates == dates.shift(1)).values   # bool array

        # ── 2. Collect bar-level flags ─────────────────────────────────────
        bar_flags_list: list[dict] = []
        bar_flags_list += self._check_ohlcv_sanity(df)
        bar_flags_list += self._check_price_returns(df, same_session)
        bar_flags_list += self._check_bar_ranges(df)
        bar_flags_list += self._check_interbar_gaps(df, same_session)
        bar_flags_list += self._check_volume(df)
        bar_flags_list += self._check_timestamp_alignment(df)

        # ── 3. Session-level aggregation ───────────────────────────────────
        session_list = self._check_sessions(df, bar_flags_list)

        # ── 4. Staleness (live mode) ───────────────────────────────────────
        is_stale, stale_minutes = (False, 0.0)
        if live:
            is_stale, stale_minutes = self._check_staleness(df)

        # ── 5. Count severity ──────────────────────────────────────────────
        warnings = 0
        critical = 0

        for bf in bar_flags_list:
            if bf["severity"] == "critical":
                critical += 1
            else:
                warnings += 1

        for sr in session_list:
            if sr["severity"] == "critical":
                critical += 1
            elif sr["severity"] == "warning":
                warnings += 1

        if is_stale:
            critical += 1

        flagged_sessions = sum(1 for sr in session_list if sr["is_suspicious"])
        flagged_bars     = len(bar_flags_list)

        # ── 6. Build DataFrames ────────────────────────────────────────────
        session_df = self._build_session_df(session_list)
        bar_df     = self._build_bar_df(bar_flags_list)

        # ── 7. Summary dict ────────────────────────────────────────────────
        summary = {
            "symbol":           self.symbol,
            "interval":         self.interval,
            "total_bars":       len(df),
            "total_sessions":   len(session_list),
            "date_range_start": str(df.index.min().date()),
            "date_range_end":   str(df.index.max().date()),
            "expected_bars_per_session": self._expected,
            "flagged_bars":     flagged_bars,
            "flagged_sessions": flagged_sessions,
            "warnings":         warnings,
            "critical":         critical,
            "passed":           critical == 0,
            "is_stale":         is_stale,
            "stale_minutes":    round(stale_minutes, 1),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        return ValidationResult(
            passed           = (critical == 0),
            warnings         = warnings,
            critical         = critical,
            total_bars       = len(df),
            total_sessions   = len(session_list),
            flagged_bars     = flagged_bars,
            flagged_sessions = flagged_sessions,
            session_report   = session_df,
            bar_flags        = bar_df,
            summary          = summary,
            is_stale         = is_stale,
            stale_minutes    = stale_minutes,
        )

    # ── Public: print_report ──────────────────────────────────────────────────

    def print_report(
        self,
        result:      ValidationResult,
        report_paths: Optional[tuple[Path, Path]] = None,
        compact:     bool = False,
    ) -> None:
        """
        Print a formatted quality report to stdout.

        Parameters
        ──────────
        result       ValidationResult from validate().
        report_paths Tuple of (sessions_path, bars_path) from save_report(),
                     printed at the bottom when provided.
        compact      If True, only print summary + critical issues (for live use).
        """
        s   = result.summary
        sep = "═" * 64
        div = "─" * 64
        OK  = "  ✓"
        BAD = "  ✗"

        status_str = "PASS" if result.passed else "FAIL"
        if result.critical == 0 and result.warnings > 0:
            status_str = "WARN"

        print(sep)
        print(
            f"  {self.symbol}  ·  Data Quality Report"
            f"  [{self.interval} | {s['total_sessions']} sessions"
            f" | {s['total_bars']:,} bars]"
        )
        print(sep)
        print(
            f"  Status   :  {status_str}"
            f"   {result.critical} critical · {result.warnings} warnings"
        )
        print(
            f"  Range    :  {s['date_range_start']}  →  {s['date_range_end']}"
        )
        print(
            f"  Expected :  {s['expected_bars_per_session']} bars/session"
        )
        print(div)

        # ── Global checks ──────────────────────────────────────────────────
        if not compact:
            print("  GLOBAL CHECKS")

            # Staleness
            if result.is_stale:
                print(
                    f"{BAD}  Stale data          "
                    f"{result.stale_minutes:.0f} min since last bar"
                    f"  ← CRITICAL"
                )
            elif result.summary.get("is_stale") is False and result.stale_minutes > 0:
                print(
                    f"{OK}  Data freshness      "
                    f"{result.stale_minutes:.0f} min since last bar"
                )

            # Total duplicate count (across all sessions)
            total_dups = (
                result.session_report["duplicate_count"].sum()
                if not result.session_report.empty
                else 0
            )
            if total_dups:
                print(f"{BAD}  Duplicate timestamps  {int(total_dups)} total")
            else:
                print(f"{OK}  Duplicate timestamps  none")

            # Alignment
            total_misaligned = (
                result.session_report["misaligned_count"].sum()
                if not result.session_report.empty
                else 0
            )
            if total_misaligned:
                print(
                    f"{BAD}  Timestamp alignment  "
                    f"{int(total_misaligned)} bars not on {self.interval} grid"
                )
            else:
                print(f"{OK}  Timestamp alignment  all bars on {self.interval} grid")

            # Coverage summary
            if not result.session_report.empty:
                low_cov = (
                    result.session_report["coverage_pct"] < self.thr.min_session_coverage
                ).sum()
                if low_cov:
                    print(
                        f"{BAD}  Session coverage     "
                        f"{int(low_cov)} session(s) below "
                        f"{self.thr.min_session_coverage*100:.0f}% threshold"
                    )
                else:
                    print(f"{OK}  Session coverage     all sessions ≥ "
                          f"{self.thr.min_session_coverage*100:.0f}%")

            print(div)

        # ── Suspicious sessions ────────────────────────────────────────────
        sus_sessions = (
            result.session_report[result.session_report["is_suspicious"]]
            if not result.session_report.empty
            else pd.DataFrame()
        )
        if not sus_sessions.empty:
            print(
                f"  SESSIONS  ({len(sus_sessions)} of "
                f"{result.total_sessions} suspicious)"
            )
            for _, row in sus_sessions.iterrows():
                cov_str = f"{row['coverage_pct']*100:.1f}%"
                sev_tag = "← CRITICAL" if row["severity"] == "critical" else ""
                print(
                    f"{BAD}  {row['session_date']}  "
                    f"{int(row['bar_count'])}/{int(row['expected_bars'])} bars"
                    f"  ({cov_str})   {row['issues']}  {sev_tag}"
                )
            print(div)
        elif not compact:
            print(f"  SESSIONS  all {result.total_sessions} sessions OK")
            print(div)

        # ── Flagged bars ───────────────────────────────────────────────────
        if not result.bar_flags.empty:
            # Show all flags in compact mode; cap at 20 in full mode
            show_df = result.bar_flags
            cap     = 20
            if not compact and len(show_df) > cap:
                tail = len(show_df) - cap
                show_df = show_df.head(cap)
            else:
                tail = 0

            print(f"  BARS  ({result.flagged_bars} suspicious)")
            for _, row in show_df.iterrows():
                ts_str  = str(row["timestamp"])[:16]
                val_str = self._format_flag_value(row["check"], float(row["value"]))
                thr_str = self._format_flag_value(row["check"], float(row["threshold"]))
                sev_tag = "← CRITICAL" if row["severity"] == "critical" else ""
                print(
                    f"{BAD}  {ts_str}   {row['check']:<20}"
                    f"  val={val_str}  thr={thr_str}  {sev_tag}"
                )
            if tail > 0:
                print(f"       … {tail} more bar flags (see saved report)")
            print(div)
        elif not compact:
            print(f"  BARS  all {result.total_bars:,} bars within thresholds")
            print(div)

        # ── Report paths ───────────────────────────────────────────────────
        if report_paths:
            print(f"  Report → {report_paths[0]}")
            if not report_paths[1].stat().st_size < 100:   # non-trivial bars file
                print(f"           {report_paths[1]}")

        print(sep)

    # ── Public: save_report ───────────────────────────────────────────────────

    def save_report(
        self,
        result:     ValidationResult,
        report_dir: Path,
    ) -> tuple[Path, Path]:
        """
        Save quality report CSVs to report_dir.

        Files written
        ─────────────
        data_quality_<YYYYMMDD_HHMMSS>_sessions.csv  – one row per session
        data_quality_<YYYYMMDD_HHMMSS>_bars.csv      – one row per flagged bar

        Returns
        ───────
        (sessions_path, bars_path)
        """
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        sess_path = report_dir / f"data_quality_{ts_tag}_sessions.csv"
        bars_path = report_dir / f"data_quality_{ts_tag}_bars.csv"

        result.session_report.to_csv(sess_path, index=False)
        result.bar_flags.to_csv(bars_path, index=False)

        logger.info(
            "Quality report saved → %s  (%d suspicious sessions, %d flagged bars)",
            sess_path.parent,
            result.flagged_sessions,
            result.flagged_bars,
        )
        return sess_path, bars_path

    # ── Bar-level checks ──────────────────────────────────────────────────────

    def _check_ohlcv_sanity(self, df: pd.DataFrame) -> list[dict]:
        """
        Verify fundamental OHLCV relationships.

        Critical:  high < low (impossible bar), high < open, high < close,
                   low > open, low > close.
        Warning:   volume <= 0 (likely a bad tick).
        """
        flags: list[dict] = []
        o, h, l, c, v = (df[x] for x in REQUIRED_COLUMNS)

        def _flag_mask(mask: pd.Series, check: str, value_col: pd.Series,
                       thr: float, sev: str, desc_tmpl: str) -> None:
            for ts in df.index[mask]:
                val = float(value_col.loc[ts])
                flags.append({
                    "timestamp":    ts,
                    "session_date": str(ts.date()),
                    "check":        check,
                    "value":        round(val, 6),
                    "threshold":    thr,
                    "severity":     sev,
                    "description":  desc_tmpl.format(val=val),
                })

        # high < low
        _flag_mask(h < l,  "ohlcv_high_lt_low",
                   (h - l).abs(), 0.0, "critical",
                   "high < low  (diff={val:.4f})")
        # high below open or close
        _flag_mask(h < o - 1e-8, "ohlcv_high_lt_open",
                   (o - h), 0.0, "critical",
                   "high < open  (spread={val:.4f})")
        _flag_mask(h < c - 1e-8, "ohlcv_high_lt_close",
                   (c - h), 0.0, "critical",
                   "high < close  (spread={val:.4f})")
        # low above open or close
        _flag_mask(l > o + 1e-8, "ohlcv_low_gt_open",
                   (l - o), 0.0, "critical",
                   "low > open  (spread={val:.4f})")
        _flag_mask(l > c + 1e-8, "ohlcv_low_gt_close",
                   (l - c), 0.0, "critical",
                   "low > close  (spread={val:.4f})")
        # non-positive volume
        _flag_mask(v <= 0, "ohlcv_volume_zero",
                   v, 0.0, "warning",
                   "volume={val:.0f}  (must be > 0)")

        return flags

    def _check_price_returns(
        self,
        df:           pd.DataFrame,
        same_session: np.ndarray,
    ) -> list[dict]:
        """
        Flag bars where |log(close[t] / close[t-1])| exceeds the threshold.

        Only checks bars that are continuations within the same session
        (skips the first bar of each day to avoid flagging overnight gaps).

        Severity:
          warning   threshold   ≤ |ret| < 2× threshold
          critical  |ret| ≥ 2× threshold
        """
        flags:   list[dict] = []
        thr      = self.thr.max_bar_return
        log_ret  = np.log(df["close"] / df["close"].shift(1)).abs().values
        idx      = df.index

        for i in range(1, len(df)):
            if not same_session[i]:
                continue
            ret = log_ret[i]
            if ret >= thr:
                sev = "critical" if ret >= thr * 2 else "warning"
                flags.append({
                    "timestamp":    idx[i],
                    "session_date": str(idx[i].date()),
                    "check":        "price_return",
                    "value":        round(float(ret), 6),
                    "threshold":    thr,
                    "severity":     sev,
                    "description":  f"|log-ret|={ret*100:.3f}%  (thr={thr*100:.2f}%)",
                })
        return flags

    def _check_bar_ranges(self, df: pd.DataFrame) -> list[dict]:
        """
        Flag bars where (high − low) / close exceeds the threshold.

        Applied to every bar (session-open bars are included because an
        abnormal high-low range within any single bar is always suspicious).
        """
        flags:  list[dict] = []
        thr     = self.thr.max_bar_range
        ranges  = ((df["high"] - df["low"]) / df["close"]).values
        idx     = df.index

        for i, rng in enumerate(ranges):
            if rng >= thr:
                sev = "critical" if rng >= thr * 2 else "warning"
                flags.append({
                    "timestamp":    idx[i],
                    "session_date": str(idx[i].date()),
                    "check":        "bar_range",
                    "value":        round(float(rng), 6),
                    "threshold":    thr,
                    "severity":     sev,
                    "description":  f"(H-L)/close={rng*100:.3f}%  (thr={thr*100:.2f}%)",
                })
        return flags

    def _check_interbar_gaps(
        self,
        df:           pd.DataFrame,
        same_session: np.ndarray,
    ) -> list[dict]:
        """
        Flag bars where |open[t] − close[t-1]| / close[t-1] exceeds the threshold.

        Only applied within sessions (skips the first bar of each day).
        Intraday gaps suggest a missing bar or bad tick; severity is always
        'warning' because trading halts can cause legitimate intraday gaps.
        """
        flags: list[dict] = []
        thr    = self.thr.max_interbar_gap
        opens  = df["open"].values
        closes = df["close"].values
        idx    = df.index

        for i in range(1, len(df)):
            if not same_session[i]:
                continue
            prev_close = closes[i - 1]
            if prev_close == 0:
                continue
            gap = abs(opens[i] - prev_close) / prev_close
            if gap >= thr:
                sev = "critical" if gap >= thr * 3 else "warning"
                flags.append({
                    "timestamp":    idx[i],
                    "session_date": str(idx[i].date()),
                    "check":        "interbar_gap",
                    "value":        round(float(gap), 6),
                    "threshold":    thr,
                    "severity":     sev,
                    "description":  (
                        f"|open-prev_close|/prev_close={gap*100:.3f}%"
                        f"  (thr={thr*100:.2f}%)"
                    ),
                })
        return flags

    def _check_volume(self, df: pd.DataFrame) -> list[dict]:
        """
        Flag bars with abnormally high volume via two independent methods:

        z-score:  rolling 20-bar log-volume z-score > max_volume_zscore
        multiple: volume / rolling 20-bar median    > max_volume_mult

        The more severe of the two findings is used when both fire on the
        same bar.  SPY has legitimately high open/close volume so the
        rolling window is intentionally short (20 bars) and thresholds are
        generous.
        """
        flags:   list[dict] = []
        idx      = df.index
        vol      = df["volume"].replace(0, np.nan)
        log_vol  = np.log(vol.clip(lower=1))

        WIN = 20
        roll_mean   = log_vol.rolling(WIN, min_periods=5).mean()
        roll_std    = log_vol.rolling(WIN, min_periods=5).std().replace(0, np.nan)
        z_scores    = ((log_vol - roll_mean) / roll_std).fillna(0).values

        roll_median = vol.rolling(WIN, min_periods=5).median().replace(0, np.nan)
        vol_mult    = (vol / roll_median).fillna(0).values

        thr_z   = self.thr.max_volume_zscore
        thr_m   = self.thr.max_volume_mult
        fired   : set[int] = set()   # track already-flagged bar indices

        # z-score flags
        for i, z in enumerate(z_scores):
            if z >= thr_z:
                sev = "critical" if z >= thr_z * 1.5 else "warning"
                flags.append({
                    "timestamp":    idx[i],
                    "session_date": str(idx[i].date()),
                    "check":        "volume_zscore",
                    "value":        round(float(z), 2),
                    "threshold":    thr_z,
                    "severity":     sev,
                    "description":  f"log-vol z-score={z:.1f}  (thr={thr_z:.1f}σ)",
                })
                fired.add(i)

        # multiple-of-median flags (skip bars already caught by z-score)
        for i, mult in enumerate(vol_mult):
            if i in fired:
                continue
            if mult >= thr_m:
                sev = "critical" if mult >= thr_m * 2 else "warning"
                flags.append({
                    "timestamp":    idx[i],
                    "session_date": str(idx[i].date()),
                    "check":        "volume_mult",
                    "value":        round(float(mult), 2),
                    "threshold":    thr_m,
                    "severity":     sev,
                    "description":  f"volume={mult:.1f}× median  (thr={thr_m:.1f}×)",
                })
        return flags

    def _check_timestamp_alignment(self, df: pd.DataFrame) -> list[dict]:
        """
        Flag bars whose minute-of-hour does not land on the expected interval grid.

        e.g., for 5m bars the valid minutes are {0, 5, 10, …, 55}.
        A bar timestamped at :03 or :47 suggests a provider formatting error
        or data merged from a different interval.

        Skipped entirely for hourly and daily intervals where any minute is valid.
        """
        flags: list[dict] = []
        if self._bar_min >= 60:
            return flags            # hourly+ bars: alignment check not meaningful

        valid   = self._valid_mins
        minutes = df.index.minute
        idx     = df.index

        for i, minute in enumerate(minutes):
            if minute not in valid:
                flags.append({
                    "timestamp":    idx[i],
                    "session_date": str(idx[i].date()),
                    "check":        "timestamp_align",
                    "value":        float(minute),
                    "threshold":    float(min(valid)),
                    "severity":     "warning",
                    "description":  (
                        f"minute={minute}  not on {self.interval} grid"
                        f"  (expected one of {sorted(valid)[:5]}…)"
                    ),
                })
        return flags

    # ── Session-level aggregation ─────────────────────────────────────────────

    def _check_sessions(
        self,
        df:              pd.DataFrame,
        bar_flags_list:  list[dict],
    ) -> list[dict]:
        """
        Aggregate bar-level flags by session date and add session-level checks.

        Session checks performed:
          • Missing bars      – bar_count < min_session_coverage × expected
          • Duplicate timestamps
          • Misaligned timestamps (from timestamp_align bar flags)
          • Session price range (max_high / min_low - 1 > max_session_range)
        """
        sessions: list[dict] = []

        # Index bar flags by date for O(1) lookup
        flags_by_date: dict[str, list[dict]] = {}
        for bf in bar_flags_list:
            d = bf["session_date"]
            flags_by_date.setdefault(d, []).append(bf)

        dates_series = pd.Series(df.index.date, index=df.index)

        for date, group in df.groupby(dates_series):
            date_str    = str(date)
            bar_count   = len(group)
            expected    = self._expected
            coverage    = bar_count / expected

            # Duplicates within this session
            dup_count   = int(group.index.duplicated().sum())

            # Misaligned timestamps from the bar-level check
            session_bar_flags  = flags_by_date.get(date_str, [])
            misaligned_count   = sum(
                1 for bf in session_bar_flags if bf["check"] == "timestamp_align"
            )

            # Session price range
            s_high = float(group["high"].max())
            s_low  = float(group["low"].min())
            session_range = (s_high - s_low) / s_low if s_low > 0 else 0.0

            issues:  list[str] = []
            severity: str = "ok"

            # Coverage check
            if coverage < self.thr.min_session_coverage:
                pct = coverage * 100
                issues.append(
                    f"coverage {pct:.1f}% ({bar_count}/{expected} bars)"
                )
                severity = _max_sev(
                    severity,
                    "critical" if coverage < 0.50 else "warning",
                )

            # Duplicate check
            if dup_count > 0:
                issues.append(f"{dup_count} duplicate timestamp(s)")
                severity = _max_sev(severity, "warning")

            # Misalignment check
            if misaligned_count > 0:
                issues.append(f"{misaligned_count} misaligned timestamp(s)")
                severity = _max_sev(severity, "warning")

            # Session range check (circuit-breaker / bad-data indicator)
            if session_range > self.thr.max_session_range:
                issues.append(
                    f"session range {session_range*100:.1f}%"
                    f"  (thr={self.thr.max_session_range*100:.0f}%)"
                )
                severity = _max_sev(severity, "warning")

            sessions.append({
                "session_date":      date_str,
                "bar_count":         bar_count,
                "expected_bars":     expected,
                "coverage_pct":      round(coverage, 4),
                "duplicate_count":   dup_count,
                "misaligned_count":  misaligned_count,
                "session_range_pct": round(session_range, 6),
                "bar_flags_count":   len(session_bar_flags),
                "is_suspicious":     len(issues) > 0,
                "severity":          severity,
                "issues":            "; ".join(issues),
            })

        return sessions

    # ── Staleness check ───────────────────────────────────────────────────────

    @staticmethod
    def _check_staleness(df: pd.DataFrame) -> tuple[bool, float]:
        """
        Return (is_stale, minutes_since_last_bar) using current ET wall-clock.

        Stale threshold: 15 minutes (mirrors BaseProvider.health_check).
        """
        try:
            from zoneinfo import ZoneInfo
            tz_et = ZoneInfo("America/New_York")
        except ImportError:
            import pytz
            tz_et = pytz.timezone("America/New_York")

        import datetime as _dt
        now_et = pd.Timestamp(_dt.datetime.now(tz_et).replace(tzinfo=None))
        last_ts = df.index[-1]
        stale_min = (now_et - last_ts).total_seconds() / 60.0
        is_stale  = stale_min > 15.0
        return is_stale, max(0.0, stale_min)

    # ── Empty result ──────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result() -> ValidationResult:
        return ValidationResult(
            passed          = False,
            warnings        = 1,
            critical        = 0,
            total_bars      = 0,
            total_sessions  = 0,
            flagged_bars    = 0,
            flagged_sessions= 0,
            session_report  = pd.DataFrame(),
            bar_flags       = pd.DataFrame(),
            summary         = {"total_bars": 0, "warnings": 1, "critical": 0,
                               "passed": False},
            is_stale        = True,
            stale_minutes   = float("inf"),
        )

    # ── DataFrame builders ────────────────────────────────────────────────────

    @staticmethod
    def _build_session_df(sessions: list[dict]) -> pd.DataFrame:
        if not sessions:
            return pd.DataFrame(columns=[
                "session_date", "bar_count", "expected_bars", "coverage_pct",
                "duplicate_count", "misaligned_count", "session_range_pct",
                "bar_flags_count", "is_suspicious", "severity", "issues",
            ])
        return pd.DataFrame(sessions)

    @staticmethod
    def _build_bar_df(flags: list[dict]) -> pd.DataFrame:
        if not flags:
            return pd.DataFrame(columns=[
                "timestamp", "session_date", "check", "value",
                "threshold", "severity", "description",
            ])
        df = pd.DataFrame(flags)
        df.sort_values(["timestamp", "check"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ── Value formatter ───────────────────────────────────────────────────────

    @staticmethod
    def _format_flag_value(check: str, value: float) -> str:
        """Format a flag value for human-readable console output."""
        pct_checks = {"price_return", "bar_range", "interbar_gap"}
        if check in pct_checks:
            return f"{value*100:.3f}%"
        if check == "volume_zscore":
            return f"{value:.1f}σ"
        if check == "volume_mult":
            return f"{value:.1f}×"
        if check == "timestamp_align":
            return f":{int(value):02d}"
        return f"{value:.4f}"


# ── Convenience function ──────────────────────────────────────────────────────

def validate_bars(
    df:           pd.DataFrame,
    interval:     str            = "5m",
    symbol:       str            = "SPY",
    live:         bool           = False,
    report_dir:   Optional[Path] = None,
    print_output: bool           = True,
    compact:      bool           = False,
    thresholds:   Optional[QualityThresholds] = None,
) -> ValidationResult:
    """
    Convenience wrapper: validate a bar DataFrame and optionally print/save the report.

    Parameters
    ──────────
    df           OHLCV bars in project schema.
    interval     Bar interval string ("1m", "5m", "15m", etc.).
    symbol       Ticker symbol (used in report headers).
    live         If True, also checks for stale data.
    report_dir   If provided, saves sessions + bars CSVs to this directory.
    print_output If True (default), prints the report to stdout.
    compact      If True, only prints summary + critical issues (for live use).
    thresholds   Custom QualityThresholds, or None for SPY-5m defaults.

    Returns
    ───────
    ValidationResult – use result.passed, result.critical, result.bar_flags, etc.

    Examples
    ────────
    # Historical download – full report saved to evaluation/reports/
    result = validate_bars(df, interval="5m", report_dir=REPORT_DIR)

    # Live inference – compact check, no file saved
    result = validate_bars(df, interval="5m", live=True, compact=True)
    if result.critical > 0:
        logger.warning("Data quality issues: %d critical", result.critical)
    """
    validator = BarValidator(interval=interval, symbol=symbol, thresholds=thresholds)
    result    = validator.validate(df, live=live)

    paths: Optional[tuple[Path, Path]] = None
    if report_dir is not None:
        paths = validator.save_report(result, report_dir)

    if print_output:
        validator.print_report(result, report_paths=paths, compact=compact)

    return result

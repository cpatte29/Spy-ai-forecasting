"""
base_provider.py
────────────────
Abstract base class for all market data providers.

Every provider must normalise its output to the project-standard DataFrame:
  • tz-naive DatetimeIndex in America/New_York (ET), sorted ascending
  • columns: open, high, low, close, volume  (float64 / float64 / … / float64)
  • only regular-session bars (09:30 – 16:00 ET)
  • no duplicate timestamps

Downstream code (feature engineering, labels, live inference) depends only
on this contract — swapping providers requires no changes outside this layer.

Options scaffolding
───────────────────
get_option_chain() and get_option_quotes() are placeholder methods.
They raise NotImplementedError until a provider implements them.
This allows the rest of the codebase to import and reference these methods
without breaking; the options layer can be filled in later.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Required output columns – in order
REQUIRED_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]


class BaseProvider(ABC):
    """
    Abstract market data provider.

    Subclasses must implement:
        get_stock_bars()
        get_latest_stock_bars()
        provider_name  (property)
    """

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier, e.g. 'polygon' or 'yfinance'."""

    # ── stock bars ────────────────────────────────────────────────────────────

    @abstractmethod
    def get_stock_bars(
        self,
        symbol:   str,
        interval: str,
        start:    Optional[str] = None,
        end:      Optional[str] = None,
        period:   Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars.

        Parameters
        ──────────
        symbol    Ticker symbol, e.g. "SPY".
        interval  Bar size string: "1m", "5m", "15m", "1h".
        start     ISO date string "YYYY-MM-DD" (optional).
        end       ISO date string "YYYY-MM-DD" (optional).
        period    Period shorthand: "7d", "30d", "60d", "1y", etc.
                  Used when start/end are not provided.

        Returns
        ───────
        pd.DataFrame – project-standard format (see module docstring).
        Empty DataFrame if no data is available.
        """

    @abstractmethod
    def get_latest_stock_bars(
        self,
        symbol:        str,
        interval:      str,
        lookback_days: int,
    ) -> pd.DataFrame:
        """
        Fetch recent bars suitable for live inference.

        Equivalent to get_stock_bars with start = today - lookback_days
        and end = tomorrow (so today's intraday bars are included).

        Parameters
        ──────────
        symbol        Ticker symbol.
        interval      Bar size string.
        lookback_days Calendar days to look back from today.

        Returns
        ───────
        pd.DataFrame – project-standard format.
        """

    # ── health diagnostics ────────────────────────────────────────────────────

    def health_check(
        self,
        symbol:   str,
        interval: str,
        df:       Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Evaluate data freshness and return a health status dict.

        If df is provided (already fetched), reuses it.
        Otherwise fetches 1 day of bars for the check.

        Returns dict with keys:
            provider_name         str
            latest_bar_ts         pd.Timestamp or None
            is_today              bool
            stale                 bool  (> 15 min behind during mkt hours)
            stale_minutes         float
            rows_fetched          int
        """
        import datetime

        try:
            from zoneinfo import ZoneInfo
            tz_et = ZoneInfo("America/New_York")
        except ImportError:
            import pytz
            tz_et = pytz.timezone("America/New_York")

        if df is None or df.empty:
            try:
                df = self.get_latest_stock_bars(symbol, interval, lookback_days=3)
            except Exception as exc:
                logger.warning("health_check: bar fetch failed: %s", exc)
                df = pd.DataFrame()

        now_et = pd.Timestamp(datetime.datetime.now(tz_et).replace(tzinfo=None))

        if df.empty:
            return {
                "provider_name": self.provider_name,
                "latest_bar_ts": None,
                "is_today":      False,
                "stale":         True,
                "stale_minutes": float("inf"),
                "rows_fetched":  0,
            }

        latest_ts     = df.index[-1]
        is_today      = latest_ts.date() == now_et.date()
        stale_minutes = (now_et - latest_ts).total_seconds() / 60.0
        stale         = stale_minutes > 15.0

        return {
            "provider_name": self.provider_name,
            "latest_bar_ts": latest_ts,
            "is_today":      is_today,
            "stale":         stale,
            "stale_minutes": round(stale_minutes, 1),
            "rows_fetched":  len(df),
        }

    def print_health(
        self,
        symbol:   str,
        interval: str,
        df:       Optional[pd.DataFrame] = None,
    ) -> dict:
        """Run health_check and print the results to stdout."""
        h = self.health_check(symbol, interval, df)
        dash = "─" * 54
        print(dash)
        print(f"  Provider health  [{h['provider_name']}]")
        print(dash)
        print(f"  Provider         : {h['provider_name']}")
        if h["latest_bar_ts"] is not None:
            print(f"  Latest bar       : {h['latest_bar_ts']}")
            print(f"  Is today's data  : {'YES' if h['is_today'] else 'NO  ← stale date'}")
            stale_str = (
                f"YES  ← {h['stale_minutes']:.0f} min behind"
                if h["stale"] else
                f"NO   ({h['stale_minutes']:.0f} min since last bar)"
            )
            print(f"  Stale            : {stale_str}")
        else:
            print("  Latest bar       : (no data)")
        print(f"  Rows fetched     : {h['rows_fetched']}")
        print(dash)
        return h

    # ── validation helpers (shared) ───────────────────────────────────────────

    @staticmethod
    def validate(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
        """
        Apply standard validation rules to a normalised DataFrame.

        Checks (in order):
          1. Not empty
          2. Required columns present (open/high/low/close/volume)
          3. Timezone consistency – index must be tz-naive ET; if tz-aware,
             converts to ET and strips tz info with a warning.
          4. Duplicate timestamps – deduplicates (keep first), logs count.
          5. Sort order – re-sorts ascending if out of order.
          6. NaN in OHLCV columns – logs count of affected rows.
          7. Bar-gap detection – infers the bar interval from the data and
             reports any intra-session gaps that suggest missing bars.

        Logs warnings for recoverable issues; raises ValueError only for
        unrecoverable schema errors (missing columns).

        Returns the validated (and potentially de-duplicated / sorted)
        DataFrame.
        """
        tag = f"[{context}] " if context else ""

        # ── 1. Empty check ────────────────────────────────────────────────
        if df.empty:
            logger.warning("%sDataFrame is empty – no bars returned.", tag)
            return df

        # ── 2. Required columns ───────────────────────────────────────────
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{tag}Missing required columns: {missing}")

        # ── 3. Timezone consistency ───────────────────────────────────────
        if df.index.tz is not None:
            logger.warning(
                "%sIndex is tz-aware (%s) – converting to tz-naive ET. "
                "Providers should strip timezone before returning.",
                tag, df.index.tz,
            )
            df = df.copy()
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)

        # ── 4. Duplicate timestamps ───────────────────────────────────────
        n_dupes = df.index.duplicated().sum()
        if n_dupes:
            logger.warning(
                "%s%d duplicate timestamp(s) – keeping first occurrence.",
                tag, n_dupes,
            )
            df = df[~df.index.duplicated(keep="first")]

        # ── 5. Sort order ─────────────────────────────────────────────────
        if not df.index.is_monotonic_increasing:
            logger.warning("%sIndex is not sorted ascending – re-sorting.", tag)
            df = df.sort_index()

        # ── 6. NaN in OHLCV ──────────────────────────────────────────────
        nan_rows = df[REQUIRED_COLUMNS].isnull().any(axis=1).sum()
        if nan_rows:
            logger.warning(
                "%s%d row(s) have NaN in OHLCV columns.", tag, nan_rows,
            )

        # ── 7. Bar-gap detection ──────────────────────────────────────────
        BaseProvider._check_bar_gaps(df, tag)

        return df

    @staticmethod
    def _check_bar_gaps(df: pd.DataFrame, log_tag: str = "") -> None:
        """
        Infer the bar interval and report intra-session gaps.

        Algorithm
        ─────────
        1. Compute all consecutive time-deltas.
        2. Filter out overnight gaps (> 1 hour) – these are expected.
        3. Infer the "base interval" as the most common intra-session delta.
        4. Flag any intra-session delta that is > 1.5× the base interval as a
           gap, i.e. at least one bar is missing between those two timestamps.
        5. Sum the excess time across all gaps to estimate the missing-bar count.

        This is provider-agnostic: it works for any interval (1m, 5m, 15m, …)
        without needing the interval to be passed in.

        Logs a WARNING when gaps are found, DEBUG otherwise.
        """
        if len(df) < 3:
            return

        deltas = pd.Series(df.index).diff().dropna()

        # Overnight / weekend gaps are expected – ignore anything > 90 min
        # (longest normal session gap is the open itself: ~17.5 h)
        intra = deltas[deltas <= pd.Timedelta(minutes=90)]

        if intra.empty:
            return

        # Base interval = most frequent intra-session delta
        base_interval = intra.mode().iloc[0]

        # Gaps: consecutive deltas more than 1.5× the base interval
        # (0.5 slack covers minor rounding differences in some feeds)
        threshold = base_interval * 1.5
        gaps      = intra[intra > threshold]

        if gaps.empty:
            logger.debug(
                "%sBar-gap check OK  base_interval=%s  bars=%d",
                log_tag, base_interval, len(df),
            )
            return

        # Estimate missing bars: each gap contributes (gap / base - 1) missing bars
        missing_bar_count = int(
            round((gaps / base_interval - 1).clip(lower=0).sum())
        )

        # Find timestamps bracketing the largest single gap.
        # gaps is a pd.Series whose index values are positional integers
        # aligned with df.index[1:], so position N → df.index[N].
        largest_gap_pos  = int(gaps.idxmax())
        gap_start_ts     = df.index[largest_gap_pos - 1]
        gap_end_ts       = df.index[largest_gap_pos]

        logger.warning(
            "%s%d intra-session gap(s) detected → ~%d missing bar(s)  "
            "(base_interval=%s)  largest gap: %s → %s (%s)",
            log_tag,
            len(gaps),
            missing_bar_count,
            base_interval,
            gap_start_ts,
            gap_end_ts,
            gaps.max(),
        )

    # ── options scaffolding (placeholder) ─────────────────────────────────────

    def get_option_chain(
        self,
        symbol:      str,
        expiry_date: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Retrieve the option chain for a given symbol and expiry.

        TODO: Implement when options support is added.
              For Polygon: GET /v3/snapshot/options/{underlyingAsset}
              Fields to include: strike, expiry, call_put, bid, ask,
              mid, volume, open_interest, iv, delta, gamma, theta, vega.

        Parameters
        ──────────
        symbol       Underlying ticker, e.g. "SPY".
        expiry_date  ISO date "YYYY-MM-DD", or None for all expiries.

        Returns
        ───────
        pd.DataFrame – one row per option contract.
        """
        raise NotImplementedError(
            f"{self.provider_name}.get_option_chain() is not yet implemented. "
            "This is a placeholder for future options support."
        )

    def get_option_quotes(
        self,
        contract_ticker: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Retrieve quote history for a single option contract.

        TODO: Implement when options support is added.
              For Polygon: GET /v2/ticks/options/{ticker}/quotes
              or GET /v2/aggs/ticker/{optionsTicker}/range/{m}/{t}/{from}/{to}

        Parameters
        ──────────
        contract_ticker  OCC-formatted contract ticker,
                         e.g. "O:SPY241220C00500000".

        Returns
        ───────
        pd.DataFrame – bid/ask/mid/volume time series.
        """
        raise NotImplementedError(
            f"{self.provider_name}.get_option_quotes() is not yet implemented. "
            "This is a placeholder for future options support."
        )

    def filter_liquid_options(
        self,
        chain_df: pd.DataFrame,
        min_volume:        int   = 100,
        min_open_interest: int   = 500,
        max_spread_pct:    float = 0.05,
    ) -> pd.DataFrame:
        """
        TODO: Filter an option chain DataFrame to liquid contracts only.
              Criteria: volume >= min_volume, OI >= min_open_interest,
              (ask - bid) / mid <= max_spread_pct.
        """
        raise NotImplementedError(
            "filter_liquid_options() is not yet implemented."
        )

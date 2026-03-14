"""
yfinance_provider.py
────────────────────
yfinance market data provider — wraps the existing data_loader logic so
it plugs cleanly into the provider abstraction without code duplication.

This provider is intentionally kept as a FALLBACK / RESEARCH option:
  • No API key required.
  • Free, but rate-limited and occasionally unreliable for intraday data.
  • 1m bars limited to the last ~30 calendar days.
  • 5m / 15m bars limited to the last ~60 calendar days.

For production / live use, prefer PolygonProvider.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from data.providers.base_provider import BaseProvider, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

# yfinance absolute lookback limits (calendar days)
_YF_MAX_LOOKBACK: dict[str, int] = {
    "1m": 30, "2m": 30,
    "5m": 60, "15m": 60, "30m": 60, "60m": 60, "1h": 60,
}

# Per-request chunk sizes (yfinance rejects large ranges for fine intervals)
_YF_CHUNK_DAYS: dict[str, int] = {
    "1m": 7, "2m": 7,
    "5m": 59, "15m": 59, "30m": 59, "60m": 59, "1h": 59,
}

# Period shorthand → calendar days
_PERIOD_DAYS: dict[str, int] = {
    "1d":  1,   "3d":  3,   "7d":  7,   "14d": 14,
    "30d": 30,  "60d": 58,  "90d": 88,  "180d": 180,
    "1mo": 30,  "3mo": 88,  "6mo": 180, "1y":  365, "2y":  730,
}


class YFinanceProvider(BaseProvider):
    """
    yfinance-backed provider.

    Wraps the existing data_loader.load_from_yfinance() logic in a
    provider-compatible interface.  No new logic is introduced; this is
    a thin adapter so the rest of the codebase can remain provider-agnostic.
    """

    @property
    def provider_name(self) -> str:
        return "yfinance"

    # ── public interface ──────────────────────────────────────────────────────

    def get_stock_bars(
        self,
        symbol:   str,
        interval: str,
        start:    Optional[str] = None,
        end:      Optional[str] = None,
        period:   Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Download bars via yfinance using chunked requests to work around
        the per-request time-range limits.

        Delegates directly to data_loader.load_from_yfinance() so all
        existing logic (chunking, tz handling, column normalisation, market-
        hours filter) is reused without duplication.
        """
        # If neither start/end nor period are given, default to 30d
        effective_period = period or "30d"

        try:
            import sys
            from pathlib import Path
            # Ensure project root is on sys.path so data_loader can be found
            _root = Path(__file__).resolve().parents[2]
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))

            from data.data_loader import load_from_yfinance
        except ImportError as exc:
            raise ImportError(
                "data.data_loader.load_from_yfinance could not be imported. "
                "Ensure you are running from the spy_ai_model directory."
            ) from exc

        logger.info(
            "yfinance: fetching %s %s bars  period=%s  start=%s  end=%s",
            symbol, interval, effective_period, start, end,
        )

        df = load_from_yfinance(
            ticker=symbol,
            period=effective_period,
            interval=interval,
            start=start,
            end=end,
        )

        df = self.validate(df, context=f"yfinance/{symbol}/{interval}")
        logger.info("yfinance: returned %d bars.", len(df))
        return df

    def get_latest_stock_bars(
        self,
        symbol:        str,
        interval:      str,
        lookback_days: int,
    ) -> pd.DataFrame:
        """
        Fetch the most recent `lookback_days` calendar days of bars.

        Clamps lookback_days to the yfinance limit for the given interval
        and logs a warning if truncation occurs.
        """
        max_days = _YF_MAX_LOOKBACK.get(interval)
        if max_days and lookback_days > max_days:
            logger.warning(
                "yfinance limits %s bars to %d calendar days. "
                "Clamping lookback from %d → %d days.",
                interval, max_days, lookback_days, max_days,
            )
            lookback_days = max_days

        # Build start / end dates
        end_dt   = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        start_dt = end_dt - pd.Timedelta(days=lookback_days)

        return self.get_stock_bars(
            symbol=symbol,
            interval=interval,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
        )

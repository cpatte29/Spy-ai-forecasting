"""
polygon_provider.py
───────────────────
Polygon.io market data provider.

Authentication
──────────────
Reads the API key from the POLYGON_API_KEY environment variable.
Never hardcode keys in source files or pass them on the command line.

API used
────────
  Aggregate bars (v2):
    GET /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
    Parameters: adjusted=true, sort=asc, limit=50000

  Pagination:
    Polygon returns a "next_url" field when results exceed the per-page limit.
    This provider follows next_url automatically until all bars are fetched.

Timestamp handling
──────────────────
  Polygon timestamps are Unix milliseconds in UTC.
  Conversion: ms → UTC datetime → tz_convert("America/New_York") → tz_localize(None)
  This produces the tz-naive ET DatetimeIndex the rest of the project expects.

Supported intervals
───────────────────
  "1m"  → multiplier=1,  timespan="minute"
  "5m"  → multiplier=5,  timespan="minute"
  "15m" → multiplier=15, timespan="minute"
  "1h"  → multiplier=1,  timespan="hour"
  "1d"  → multiplier=1,  timespan="day"

Optional dependency
───────────────────
  The official `polygon-api-client` is used when installed.
  Falls back to `requests` if it is not installed.
  Install the official client for richer future feature support:
    pip install polygon-api-client

⚠️  API key security
─────────────────────
  DO NOT commit POLYGON_API_KEY to git.
  Add it to your shell profile (.bashrc / .zshrc) or a local .env file.
  Use `export POLYGON_API_KEY=<your_key>` before running any script.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

from data.providers.base_provider import BaseProvider, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

_POLYGON_BASE = "https://api.polygon.io"

# Interval → (multiplier, timespan) for Polygon aggregate API
_INTERVAL_MAP: dict[str, tuple[int, str]] = {
    "1m":  (1,  "minute"),
    "2m":  (2,  "minute"),
    "5m":  (5,  "minute"),
    "15m": (15, "minute"),
    "30m": (30, "minute"),
    "1h":  (1,  "hour"),
    "60m": (1,  "hour"),
    "1d":  (1,  "day"),
}

# period shorthand → calendar days
_PERIOD_DAYS: dict[str, int] = {
    "1d":  1,   "3d":  3,   "7d":  7,   "14d": 14,
    "30d": 30,  "60d": 60,  "90d": 90,  "180d": 180,
    "1mo": 30,  "3mo": 90,  "6mo": 180, "1y":  365, "2y":  730,
}


class PolygonProvider(BaseProvider):
    """
    Polygon.io aggregate bar provider.

    Uses the POLYGON_API_KEY environment variable.  Raises ValueError on
    first API call if the key is missing so that import itself never fails.
    """

    @property
    def provider_name(self) -> str:
        return "polygon"

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
        Fetch historical aggregate bars from Polygon.io.

        Parameters
        ──────────
        symbol    Ticker (e.g. "SPY").
        interval  Bar size: "1m", "5m", "15m", "1h".
        start     "YYYY-MM-DD" (inclusive).
        end       "YYYY-MM-DD" (exclusive; use tomorrow for today's bars).
        period    Period string: "30d", "60d", "1y" etc.
                  Used only when start/end are not both provided.

        Returns
        ───────
        pd.DataFrame – normalised OHLCV bars, tz-naive ET index.
        """
        from_date, to_date = self._resolve_dates(start, end, period)
        multiplier, timespan = self._parse_interval(interval)

        logger.info(
            "Polygon: fetching %s %s bars  %s → %s",
            symbol, interval, from_date, to_date,
        )

        raw_results = self._fetch_aggs(symbol, multiplier, timespan, from_date, to_date)

        if not raw_results:
            logger.warning(
                "Polygon returned 0 results for %s [%s %s → %s].",
                symbol, interval, from_date, to_date,
            )
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        df = self._normalise(raw_results)
        df = self._filter_market_hours(df)
        df = self.validate(df, context=f"polygon/{symbol}/{interval}")
        logger.info("Polygon: returned %d bars after market-hours filter.", len(df))
        return df

    def get_latest_stock_bars(
        self,
        symbol:        str,
        interval:      str,
        lookback_days: int,
    ) -> pd.DataFrame:
        """
        Fetch the most recent `lookback_days` calendar days of bars.

        end is set to tomorrow so today's intraday session is included.
        """
        today    = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
        from_dt  = today - pd.Timedelta(days=lookback_days)
        to_dt    = today + pd.Timedelta(days=1)          # include today
        from_str = from_dt.strftime("%Y-%m-%d")
        to_str   = to_dt.strftime("%Y-%m-%d")
        return self.get_stock_bars(symbol, interval, start=from_str, end=to_str)

    # ── internal: API interaction ──────────────────────────────────────────────

    def _get_api_key(self) -> str:
        key = os.environ.get("POLYGON_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "POLYGON_API_KEY environment variable is not set.\n"
                "Export it before running:\n"
                "  export POLYGON_API_KEY=<your_key>\n\n"
                "⚠️  Never hardcode or commit your API key to git."
            )
        return key

    def _fetch_aggs(
        self,
        ticker:     str,
        multiplier: int,
        timespan:   str,
        from_date:  str,
        to_date:    str,
    ) -> list[dict]:
        """
        Fetch aggregate bars, following pagination (next_url) automatically.

        Tries the official polygon-api-client first; falls back to raw
        requests if it is not installed.
        """
        try:
            return self._fetch_aggs_official(ticker, multiplier, timespan, from_date, to_date)
        except ImportError:
            logger.debug(
                "polygon-api-client not installed – using raw requests fallback. "
                "Install with: pip install polygon-api-client"
            )
            return self._fetch_aggs_requests(ticker, multiplier, timespan, from_date, to_date)

    def _fetch_aggs_official(
        self,
        ticker:     str,
        multiplier: int,
        timespan:   str,
        from_date:  str,
        to_date:    str,
    ) -> list[dict]:
        """Use the official polygon-api-client library."""
        from polygon import RESTClient  # type: ignore[import]

        key    = self._get_api_key()
        client = RESTClient(api_key=key)

        try:
            # list_aggs() is a generator; wrap in list() to fully consume it.
            # Older clients expose get_aggs() which returns a list directly.
            if hasattr(client, "list_aggs"):
                aggs = list(client.list_aggs(
                    ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=from_date,
                    to=to_date,
                    adjusted=True,
                    sort="asc",
                    limit=50000,
                ))
            else:
                aggs = client.get_aggs(
                    ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=from_date,
                    to=to_date,
                    adjusted=True,
                    sort="asc",
                    limit=50000,
                )
        except Exception as exc:
            logger.error("Polygon official client error: %s", exc)
            raise

        # Convert Agg objects to plain dicts for uniform downstream handling
        results: list[dict] = []
        for agg in aggs:
            results.append({
                "o": getattr(agg, "open",   None),
                "h": getattr(agg, "high",   None),
                "l": getattr(agg, "low",    None),
                "c": getattr(agg, "close",  None),
                "v": getattr(agg, "volume", None),
                "t": getattr(agg, "timestamp", None),
            })
        return results

    def _fetch_aggs_requests(
        self,
        ticker:     str,
        multiplier: int,
        timespan:   str,
        from_date:  str,
        to_date:    str,
    ) -> list[dict]:
        """Raw HTTP fallback – no third-party polygon client required."""
        try:
            import requests as req
        except ImportError:
            raise ImportError(
                "Neither polygon-api-client nor requests is installed.\n"
                "Install one of:\n"
                "  pip install polygon-api-client\n"
                "  pip install requests"
            )

        key = self._get_api_key()
        url = (
            f"{_POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/"
            f"{multiplier}/{timespan}/{from_date}/{to_date}"
        )
        params: dict = {
            "adjusted": "true",
            "sort":     "asc",
            "limit":    50000,
            "apiKey":   key,
        }

        all_results: list[dict] = []
        page = 0

        while True:
            page += 1
            logger.debug("Polygon HTTP request page %d: %s", page, url)
            try:
                resp = req.get(url, params=params, timeout=30)
            except req.exceptions.RequestException as exc:
                raise RuntimeError(f"Polygon HTTP request failed: {exc}") from exc

            if resp.status_code == 403:
                raise PermissionError(
                    "Polygon API returned 403 Forbidden. "
                    "Check that POLYGON_API_KEY is valid and has the right plan."
                )
            if resp.status_code == 429:
                raise RuntimeError(
                    "Polygon API rate limit hit (429 Too Many Requests). "
                    "Add delays between requests or upgrade your plan."
                )

            resp.raise_for_status()
            data = resp.json()

            status = data.get("status", "")
            if status not in ("OK", "DELAYED", ""):
                err = data.get("error") or data.get("message") or status
                raise RuntimeError(f"Polygon API error: {err}")

            batch = data.get("results") or []
            all_results.extend(batch)
            logger.debug("Polygon page %d: got %d bars (total so far: %d).",
                         page, len(batch), len(all_results))

            next_url = data.get("next_url")
            if not next_url:
                break

            # next_url already contains all query params except apiKey
            url    = next_url
            params = {"apiKey": key}

        return all_results

    # ── internal: normalisation ────────────────────────────────────────────────

    @staticmethod
    def _normalise(results: list[dict]) -> pd.DataFrame:
        """
        Convert raw Polygon aggregate result dicts to the project-standard DataFrame.

        Polygon field mapping:
          t  → timestamp (ms since Unix epoch, UTC)
          o  → open
          h  → high
          l  → low
          c  → close
          v  → volume
        """
        timestamps = pd.to_datetime(
            [r["t"] for r in results],
            unit="ms",
            utc=True,
        ).tz_convert("America/New_York").tz_localize(None)  # tz-naive ET

        df = pd.DataFrame(
            {
                "open":   [float(r["o"]) for r in results],
                "high":   [float(r["h"]) for r in results],
                "low":    [float(r["l"]) for r in results],
                "close":  [float(r["c"]) for r in results],
                "volume": [float(r["v"]) for r in results],
            },
            index=timestamps,
        )
        df.index.name = None
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
        """Keep only bars inside [09:30, 16:00] ET."""
        if df.empty:
            return df
        t = df.index.time
        open_t  = pd.Timestamp("1970-01-01 09:30").time()
        close_t = pd.Timestamp("1970-01-01 16:00").time()
        return df.loc[(t >= open_t) & (t <= close_t)].copy()

    # ── internal: date / interval helpers ─────────────────────────────────────

    @staticmethod
    def _parse_interval(interval: str) -> tuple[int, str]:
        """Translate project interval string to (multiplier, timespan)."""
        if interval not in _INTERVAL_MAP:
            raise ValueError(
                f"Unsupported interval {interval!r}. "
                f"Supported: {sorted(_INTERVAL_MAP)}"
            )
        return _INTERVAL_MAP[interval]

    @staticmethod
    def _resolve_dates(
        start:  Optional[str],
        end:    Optional[str],
        period: Optional[str],
    ) -> tuple[str, str]:
        """
        Resolve (from_date, to_date) strings for the Polygon API.

        Priority:
          1. Both start and end provided  → use directly.
          2. Only start provided          → end = tomorrow.
          3. period provided              → start = today - N days, end = tomorrow.
          4. Fallback                     → last 30 days.
        """
        tomorrow = (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        today    = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

        if start and end:
            return start, end
        if start:
            return start, tomorrow
        if period:
            days  = _PERIOD_DAYS.get(period)
            if days is None:
                # Try parsing numeric suffix e.g. "45d"
                try:
                    days = int(period.rstrip("d").rstrip("D"))
                except ValueError:
                    raise ValueError(
                        f"Unrecognised period {period!r}. "
                        f"Use one of {sorted(_PERIOD_DAYS)} or a plain number like '45d'."
                    )
            from_dt = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
            return from_dt.strftime("%Y-%m-%d"), tomorrow

        # fallback: last 30 days
        fallback_start = (pd.Timestamp.utcnow() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        logger.warning(
            "No start/end/period specified for Polygon request – defaulting to last 30 days."
        )
        return fallback_start, tomorrow

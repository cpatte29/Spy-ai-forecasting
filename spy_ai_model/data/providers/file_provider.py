"""
file_provider.py
────────────────
LocalFileProvider – loads OHLCV bars from a local CSV or Parquet file.

Use this provider when you want to:
  • Run the full pipeline offline with a pre-recorded dataset.
  • Backtest against historical data exported from any source.
  • Write integration tests without making network calls.

File requirements
─────────────────
  • CSV (.csv / .txt) or Parquet (.parquet / .pq) format.
  • A datetime column used as the index (first column, or a column named
    "datetime", "date", "timestamp", "time", or "Datetime").
  • Columns: open, high, low, close, volume  (case-insensitive).
  • Timestamps may be tz-aware (any tz) or tz-naive; tz-aware values are
    converted to tz-naive America/New_York (ET) automatically.

Provider name
─────────────
  Returns "file:<filename>" so log lines clearly identify the source.

CLI / factory usage
───────────────────
  from data.providers import get_provider

  p = get_provider("file", file_path="/path/to/spy_5m.csv")
  df = p.get_stock_bars("SPY", "5m")

  # Or with a date filter:
  df = p.get_stock_bars("SPY", "5m", start="2025-01-01", end="2025-03-01")

  # Latest N calendar days of data in the file:
  df = p.get_latest_stock_bars("SPY", "5m", lookback_days=30)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure project root is on sys.path so sibling imports work when this
# module is executed directly or from any working directory.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.providers.base_provider import BaseProvider, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

# Period shorthand → calendar days (mirrors polygon_provider + yfinance_provider)
_PERIOD_DAYS: dict[str, int] = {
    "1d":  1,   "3d":  3,   "7d":  7,   "14d": 14,
    "30d": 30,  "60d": 60,  "90d": 90,  "180d": 180,
    "1mo": 30,  "3mo": 90,  "6mo": 180, "1y":  365, "2y":  730,
}

# Candidate column names that could serve as the datetime index
_DATETIME_CANDIDATES = {"datetime", "date", "timestamp", "time", "Datetime", "Date"}


class LocalFileProvider(BaseProvider):
    """
    Market data provider backed by a local CSV or Parquet file.

    Parameters
    ──────────
    file_path   Absolute or relative path to the data file.
                Accepted extensions: .csv  .txt  .parquet  .pq

    Notes
    ─────
    • get_latest_stock_bars() filters to the most recent lookback_days
      of data present in the file, anchored at the file's last timestamp —
      not "today".  This lets you test live-inference code against historical
      snapshots without changing the call sites.
    • The symbol and interval arguments to get_stock_bars() are informational
      only; the file is assumed to contain a single instrument.
    """

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)
        if not self._file_path.exists():
            raise FileNotFoundError(
                f"LocalFileProvider: file not found: {self._file_path}\n"
                f"Check the path and make sure the file has been exported."
            )
        logger.info(
            "LocalFileProvider initialised  ←  %s  (%s bytes)",
            self._file_path,
            f"{self._file_path.stat().st_size:,}",
        )

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        """Returns 'file:<filename>' so log lines clearly identify the source."""
        return f"file:{self._file_path.name}"

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
        Load bars from the local file, with optional date-range filtering.

        Parameters
        ──────────
        symbol    Informational only — the file is assumed to contain a
                  single instrument.
        interval  Informational only — used in log messages and context tags.
        start     ISO date "YYYY-MM-DD" (inclusive).  None = no lower bound.
        end       ISO date "YYYY-MM-DD" (exclusive).  None = no upper bound.
        period    Period shorthand "30d", "60d", etc.  Computes start from
                  the last timestamp in the file when start is not given.
                  Ignored when start is provided.

        Returns
        ───────
        pd.DataFrame – project-standard OHLCV, tz-naive ET, market hours only.
        """
        logger.info(
            "LocalFileProvider: loading %s  interval=%s  start=%s  end=%s  period=%s",
            self._file_path.name, interval, start, end, period,
        )

        df = self._load_and_normalise()

        if df.empty:
            logger.warning("LocalFileProvider: file loaded but produced 0 usable bars.")
            return df

        start_ts, end_ts = self._resolve_dates(df, start, end, period)

        if start_ts is not None:
            before = len(df)
            df = df.loc[df.index >= start_ts]
            logger.debug(
                "LocalFileProvider: start filter %s  %d → %d rows.",
                start_ts.date(), before, len(df),
            )
        if end_ts is not None:
            before = len(df)
            df = df.loc[df.index < end_ts]
            logger.debug(
                "LocalFileProvider: end filter %s  %d → %d rows.",
                end_ts.date(), before, len(df),
            )

        df = self.validate(df, context=f"file/{self._file_path.name}/{interval}")
        logger.info(
            "LocalFileProvider: returned %d bars  [%s → %s]",
            len(df),
            df.index[0].date() if not df.empty else "—",
            df.index[-1].date() if not df.empty else "—",
        )
        return df

    def get_latest_stock_bars(
        self,
        symbol:        str,
        interval:      str,
        lookback_days: int,
    ) -> pd.DataFrame:
        """
        Return the most recent `lookback_days` of data present in the file.

        The lookback is anchored at the file's last timestamp, not today's
        date.  This allows integration testing against historical snapshots
        without changing downstream code.

        Example: if the file ends on 2025-06-30 and lookback_days=7, this
        returns bars from 2025-06-24 onward.
        """
        # Load without date filter to find the file's last timestamp.
        df_full = self._load_and_normalise()
        if df_full.empty:
            return df_full

        anchor   = df_full.index.max()
        start_dt = anchor - pd.Timedelta(days=lookback_days)
        end_dt   = anchor + pd.Timedelta(days=1)   # inclusive of anchor date

        logger.info(
            "LocalFileProvider: get_latest_stock_bars  lookback=%dd  "
            "anchor=%s  start=%s",
            lookback_days, anchor.date(), start_dt.date(),
        )

        return self.get_stock_bars(
            symbol=symbol,
            interval=interval,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
        )

    # ── internal: loading + normalisation ─────────────────────────────────────

    def _load_and_normalise(self) -> pd.DataFrame:
        """
        Read the raw file, normalise column names, strip timezone, and filter
        to regular market-session bars (09:30 – 16:00 ET).

        Raises
        ──────
        ValueError   – unsupported file format, missing required columns.
        """
        path = self._file_path
        # Use full filename to detect compound extensions (.csv.gz, .csv.bz2)
        name_lower = path.name.lower()
        suffix     = path.suffix.lower()

        # ── Read raw file ──────────────────────────────────────────────────
        if suffix in (".parquet", ".pq"):
            df = pd.read_parquet(path)
        elif suffix in (".csv", ".txt"):
            df = self._read_csv(path)
        elif name_lower.endswith((".csv.gz", ".csv.bz2", ".csv.xz", ".txt.gz")):
            # pandas read_csv infers compression from the extension automatically
            df = self._read_csv(path)
        else:
            raise ValueError(
                f"LocalFileProvider: unsupported format '{path.suffix}'. "
                "Accepted: .csv  .txt  .csv.gz  .parquet  .pq"
            )

        # ── Ensure DatetimeIndex ───────────────────────────────────────────
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try promoting a named column to the index
            matched = [c for c in df.columns if c in _DATETIME_CANDIDATES]
            if matched:
                df.set_index(matched[0], inplace=True)
                logger.debug(
                    "LocalFileProvider: promoted column '%s' to index.", matched[0]
                )
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as exc:
                    raise ValueError(
                        f"LocalFileProvider: could not parse a datetime index from "
                        f"'{path.name}'. "
                        f"Ensure the first column contains datetime values. ({exc})"
                    ) from exc
        else:
            df.index = pd.to_datetime(df.index)

        # ── Strip timezone → tz-naive ET ───────────────────────────────────
        if df.index.tz is not None:
            logger.debug(
                "LocalFileProvider: converting tz-aware index (%s) to tz-naive ET.",
                df.index.tz,
            )
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)

        # ── Normalise column names ─────────────────────────────────────────
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance-style MultiIndex: (metric, ticker) → keep metric only
            df.columns = [str(c[0]).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"LocalFileProvider: '{path.name}' is missing required columns "
                f"{missing}.  Available columns: {list(df.columns)}"
            )
        df = df[REQUIRED_COLUMNS].copy()

        # ── Market-hours filter ────────────────────────────────────────────
        try:
            from config import MARKET_OPEN, MARKET_CLOSE
        except ImportError:
            MARKET_OPEN  = "09:30"
            MARKET_CLOSE = "16:00"

        open_t  = pd.Timestamp(f"1970-01-01 {MARKET_OPEN}").time()
        close_t = pd.Timestamp(f"1970-01-01 {MARKET_CLOSE}").time()
        t = df.index.time
        before = len(df)
        df = df.loc[(t >= open_t) & (t <= close_t)].copy()
        if len(df) < before:
            logger.debug(
                "LocalFileProvider: dropped %d pre/post-market bars.",
                before - len(df),
            )

        df.sort_index(inplace=True)
        return df

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        """
        Read a CSV file, trying common date/index patterns.

        yfinance exports sometimes use 'Datetime' as the first column;
        standard exports use the first unnamed column as the index.
        """
        # Try standard read with first column as index + date parsing
        try:
            df = pd.read_csv(path, parse_dates=True, index_col=0)
            # If the index parsed as strings rather than datetimes, retry
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, infer_datetime_format=True)
            return df
        except Exception:
            pass

        # Fallback: read flat, let _load_and_normalise() promote a datetime col
        return pd.read_csv(path)

    @staticmethod
    def _resolve_dates(
        df:     pd.DataFrame,
        start:  Optional[str],
        end:    Optional[str],
        period: Optional[str],
    ) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Resolve (start_ts, end_ts) for filtering the loaded DataFrame.

        Priority:
          1. start provided → use start (and end if given, else no upper bound).
          2. period provided → anchor end at file's last timestamp.
          3. Neither → (None, None) – return all bars.
        """
        if start:
            start_ts = pd.Timestamp(start)
            end_ts   = pd.Timestamp(end) if end else None
            return start_ts, end_ts

        if period:
            days = _PERIOD_DAYS.get(period)
            if days is None:
                try:
                    days = int(period.rstrip("dD"))
                except ValueError:
                    logger.warning(
                        "LocalFileProvider: unrecognised period %r – returning all bars.",
                        period,
                    )
                    return None, None

            if df.empty:
                return None, None

            end_ts   = df.index.max() + pd.Timedelta(days=1)
            start_ts = end_ts - pd.Timedelta(days=days)
            logger.debug(
                "LocalFileProvider: period=%s → start=%s  end=%s",
                period, start_ts.date(), end_ts.date(),
            )
            return start_ts, end_ts

        return None, None

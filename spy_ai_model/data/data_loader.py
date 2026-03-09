"""
data_loader.py
──────────────
Load SPY 1-minute OHLCV bars from:
  • yfinance  (real data, requires internet)
  • a local CSV/Parquet file
  • the synthetic generator

Output is always a clean pd.DataFrame with columns
  open, high, low, close, volume
indexed by a tz-naive DatetimeIndex limited to regular market hours.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TICKER, MARKET_OPEN, MARKET_CLOSE, DATA_DIR

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars inside [MARKET_OPEN, MARKET_CLOSE]."""
    times = df.index.time
    open_t  = pd.Timestamp(f"1970-01-01 {MARKET_OPEN}").time()
    close_t = pd.Timestamp(f"1970-01-01 {MARKET_CLOSE}").time()
    mask = (times >= open_t) & (times <= close_t)
    return df.loc[mask].copy()


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, drop extras, verify required cols exist."""
    # yfinance ≥0.2 returns a MultiIndex (metric, ticker); flatten to metric only
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")
    return df[required]


def _strip_timezone(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df


# ── public loaders ────────────────────────────────────────────────────────────

def load_from_yfinance(
    ticker:     str = TICKER,
    period:     str = "1y",
    interval:   str = "1m",
    start: Optional[str] = None,
    end:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Download bars via yfinance.

    yfinance limits 1-minute history to ~30 days per call, so for longer
    periods this function fetches in 7-day chunks and concatenates.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance")

    import datetime as dt

    if start and end:
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)
    else:
        end_dt   = pd.Timestamp.today().normalize()
        # map period string to days
        period_days = {
            "7d": 7, "14d": 14, "30d": 30, "60d": 58, "90d": 88,
            "1mo": 30, "3mo": 88, "6mo": 180, "1y": 365, "2y": 730,
        }
        days = period_days.get(period, 30)
        start_dt = end_dt - pd.Timedelta(days=days)

    # yfinance per-request limits: 1m→7 days, 5m/15m/30m→60 days, 1h→730 days
    _chunk_days = {"1m": 7, "2m": 7, "5m": 59, "15m": 59, "30m": 59, "60m": 59, "1h": 59}
    chunk_size = dt.timedelta(days=_chunk_days.get(interval, 7))

    # yfinance absolute lookback limits: data older than this many calendar days
    # cannot be fetched regardless of chunk size.  Clamp start_dt and warn early.
    _max_lookback = {"1m": 30, "2m": 30, "5m": 60, "15m": 60, "30m": 60}
    if interval in _max_lookback:
        earliest_allowed = end_dt - pd.Timedelta(days=_max_lookback[interval] - 1)
        if start_dt < earliest_allowed:
            logger.warning(
                "%s bars: yfinance limits history to %d calendar days. "
                "Clamping start from %s → %s (requested period exceeds API limit).",
                interval, _max_lookback[interval],
                start_dt.date(), earliest_allowed.date(),
            )
            start_dt = earliest_allowed
    frames = []
    chunk_start = start_dt

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + chunk_size, end_dt)
        logger.info("Fetching %s from %s to %s …", ticker, chunk_start.date(), chunk_end.date())
        raw = yf.download(
            ticker,
            start=chunk_start.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if not raw.empty:
            frames.append(raw)
        chunk_start = chunk_end

    if not frames:
        raise ValueError("yfinance returned no data.")

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")]
    df = _strip_timezone(df)
    df = _standardise_columns(df)
    df = _filter_market_hours(df)
    df.sort_index(inplace=True)
    return df


def load_from_file(path: str | Path) -> pd.DataFrame:
    """
    Load bars from a local CSV or Parquet file.

    The file must have a datetime column (or index) plus open/high/low/close/volume.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".txt"):
        df = pd.read_csv(path, parse_dates=True, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    df.index = pd.DatetimeIndex(df.index)
    df = _strip_timezone(df)
    df = _standardise_columns(df)
    df = _filter_market_hours(df)
    df.sort_index(inplace=True)
    return df


def load_synthetic(n_days: int = 252, seed: int = 0) -> pd.DataFrame:
    """Return a freshly generated synthetic bar DataFrame."""
    from data.generate_synthetic import build_synthetic_bars
    logger.info("Generating %d days of synthetic 1-minute bars …", n_days)
    return build_synthetic_bars(n_days=n_days, seed=seed)


def save_bars(df: pd.DataFrame, name: str = "spy_1m") -> Path:
    """Persist bars to DATA_DIR as Parquet."""
    out = DATA_DIR / f"{name}.parquet"
    df.to_parquet(out)
    logger.info("Saved %d bars to %s", len(df), out)
    return out

"""
generate_synthetic.py
─────────────────────
Produces realistic-looking SPY 1-minute OHLCV bars using a geometric
Brownian motion with intraday volume/volatility patterns.

Usage
-----
    from data.generate_synthetic import build_synthetic_bars
    df = build_synthetic_bars(n_days=252)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    MARKET_OPEN, MARKET_CLOSE, SYNTH_TRADING_DAYS, SYNTH_SEED
)


def _trading_minutes(date: datetime) -> pd.DatetimeIndex:
    """Return 1-min timestamps for a single trading day."""
    open_dt  = datetime.combine(date, datetime.strptime(MARKET_OPEN,  "%H:%M").time())
    close_dt = datetime.combine(date, datetime.strptime(MARKET_CLOSE, "%H:%M").time())
    return pd.date_range(open_dt, close_dt, freq="1min")


def _intraday_vol_pattern(n: int) -> np.ndarray:
    """U-shaped volatility: high at open/close, low mid-day."""
    t = np.linspace(0, 1, n)
    pattern = 1.5 - np.sin(np.pi * t) + 0.3 * np.exp(-20 * (t - 0.0) ** 2)
    return pattern / pattern.mean()


def _intraday_vol_pattern_volume(n: int) -> np.ndarray:
    """Volume: high at open, dip mid-day, spike at close."""
    t = np.linspace(0, 1, n)
    pattern = (
        3.0 * np.exp(-15 * t)
        + 0.4
        + 2.0 * np.exp(-15 * (t - 1.0) ** 2)
    )
    return pattern / pattern.mean()


def build_synthetic_bars(
    n_days: int = SYNTH_TRADING_DAYS,
    seed:   int = SYNTH_SEED,
    start_price: float = 450.0,
    annual_vol:  float = 0.18,
    annual_drift: float = 0.08,
) -> pd.DataFrame:
    """
    Generate *n_days* of synthetic 1-minute SPY OHLCV bars.

    Returns
    -------
    pd.DataFrame  columns: open, high, low, close, volume
                  index  : DatetimeIndex (tz-naive, market hours only)
    """
    rng = np.random.default_rng(seed)

    # Compute per-minute drift / vol from annual figures
    minutes_per_year = 252 * 390
    mu_bar  = annual_drift / minutes_per_year
    sig_bar = annual_vol   / np.sqrt(minutes_per_year)

    # Build list of trading dates (Mon-Fri, skip simple holidays by skipping
    # the first ~10 US federal holidays per year heuristically)
    start_date = datetime(2023, 1, 3)
    dates: list[datetime] = []
    d = start_date
    while len(dates) < n_days:
        if d.weekday() < 5:          # Mon-Fri
            dates.append(d)
        d += timedelta(days=1)

    records = []
    price = start_price

    for day in dates:
        minutes = _trading_minutes(day)
        n       = len(minutes)

        vol_pat = _intraday_vol_pattern(n)
        vol_vol = _intraday_vol_pattern_volume(n)

        # Simulate minute returns
        eps    = rng.standard_normal(n)
        ret    = mu_bar + sig_bar * vol_pat * eps
        closes = price * np.cumprod(1 + ret)

        # Construct OHLC from close sequence
        opens  = np.empty(n)
        opens[0] = price
        opens[1:] = closes[:-1]

        # Intra-bar noise for high/low
        bar_range = np.abs(closes - opens) + sig_bar * vol_pat * np.abs(rng.standard_normal(n)) * price * 0.5
        highs  = np.maximum(opens, closes) + bar_range * rng.uniform(0.0, 0.5, n)
        lows   = np.minimum(opens, closes) - bar_range * rng.uniform(0.0, 0.5, n)
        lows   = np.maximum(lows, 1.0)     # no negative prices

        # Volume: base 5M shares/day, shaped intraday
        base_vol = 5_000_000 / 390
        volumes  = (base_vol * vol_vol * rng.lognormal(0, 0.3, n)).astype(int)
        volumes  = np.maximum(volumes, 1)

        for i, ts in enumerate(minutes):
            records.append({
                "datetime": ts,
                "open":     round(float(opens[i]),  2),
                "high":     round(float(highs[i]),  2),
                "low":      round(float(lows[i]),   2),
                "close":    round(float(closes[i]), 2),
                "volume":   int(volumes[i]),
            })

        price = float(closes[-1])   # carry over to next day

    df = pd.DataFrame(records).set_index("datetime")
    df.index = pd.DatetimeIndex(df.index)
    return df


if __name__ == "__main__":
    df = build_synthetic_bars()
    print(f"Generated {len(df):,} bars over {df.index.normalize().nunique()} days")
    print(df.head())
    print(df.tail())

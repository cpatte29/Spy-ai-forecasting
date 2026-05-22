"""
ticker_universe.py
──────────────────
Manage the ticker universe for the options flow scanner.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_UNIVERSE: list[str] = [
    "SPY", "QQQ", "AAPL", "NVDA", "TSLA", "META", "AMD", "MSFT", "AMZN",
]

# Broader liquid mid/large-cap universe for later expansion
EXTENDED_UNIVERSE: list[str] = DEFAULT_UNIVERSE + [
    "GOOGL", "NFLX", "BABA", "TSM", "COIN", "PLTR", "SOFI", "HOOD",
    "GLD", "TLT", "IWM", "XLF", "XLE", "ARKK",
]


def load_universe(
    tickers: list[str] | None = None,
    file: str | Path | None = None,
    extended: bool = False,
) -> list[str]:
    """
    Return the list of tickers to scan.

    Priority: explicit tickers > file > extended > default.
    """
    if tickers:
        return [t.upper().strip() for t in tickers if t.strip()]

    if file is not None:
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(f"Universe file not found: {path}")
        raw = path.read_text().splitlines()
        loaded = [t.strip().upper() for t in raw if t.strip() and not t.startswith("#")]
        if not loaded:
            raise ValueError(f"Universe file is empty: {path}")
        return loaded

    return EXTENDED_UNIVERSE if extended else DEFAULT_UNIVERSE

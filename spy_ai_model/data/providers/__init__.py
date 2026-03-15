"""
data.providers
──────────────
Market data provider abstraction layer.

Usage
─────
    from data.providers import get_provider

    # Auto-select: Polygon when POLYGON_API_KEY exists, else yfinance
    provider = get_provider("auto")
    df = provider.get_stock_bars("SPY", "5m", period="30d")

    # Explicit provider
    provider = get_provider("polygon")
    df = provider.get_latest_stock_bars("SPY", "5m", lookback_days=7)

    # Local file (CSV or Parquet)
    provider = get_provider("file", file_path="/path/to/spy_5m.csv")
    df = provider.get_stock_bars("SPY", "5m", start="2025-01-01")

Providers
─────────
  auto      – Polygon when POLYGON_API_KEY env var is set, else yfinance
  polygon   – Polygon.io REST API  (requires POLYGON_API_KEY env var)
  yfinance  – yfinance fallback     (research / development use)
  file      – local CSV / Parquet   (requires file_path= keyword argument)

Options scaffolding
───────────────────
  All providers expose get_option_chain() and get_option_quotes() but these
  raise NotImplementedError until a provider implements them.  The interface
  is defined in base_provider.BaseProvider so import sites never need to change.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.providers.base_provider import BaseProvider


def get_provider(name: str = "auto", **kwargs) -> "BaseProvider":
    """
    Factory function that returns a configured provider instance.

    Parameters
    ──────────
    name  "auto"     – use Polygon when POLYGON_API_KEY is set, else yfinance
          "polygon"  – require Polygon.io (raises at first call if key missing)
          "yfinance" – always use yfinance (research / fallback)
          "file"     – load from a local CSV or Parquet file
                       Requires: file_path=<str | Path>

    **kwargs
          file_path  Required when name="file".  Path to the data file.

    Returns
    ───────
    BaseProvider subclass instance ready to call.

    Examples
    ────────
    >>> get_provider("auto")
    >>> get_provider("polygon")
    >>> get_provider("yfinance")
    >>> get_provider("file", file_path="/data/spy_5m.csv")
    """
    if name == "auto":
        name = "polygon" if os.environ.get("POLYGON_API_KEY") else "yfinance"

    if name == "polygon":
        from data.providers.polygon_provider import PolygonProvider
        return PolygonProvider()

    if name == "yfinance":
        from data.providers.yfinance_provider import YFinanceProvider
        return YFinanceProvider()

    if name == "file":
        file_path = kwargs.get("file_path") or kwargs.get("path")
        if not file_path:
            raise ValueError(
                "get_provider('file') requires a file_path= keyword argument.\n"
                "Example: get_provider('file', file_path='/data/spy_5m.csv')"
            )
        from data.providers.file_provider import LocalFileProvider
        return LocalFileProvider(file_path)

    raise ValueError(
        f"Unknown provider {name!r}. "
        f"Valid options: 'auto', 'polygon', 'yfinance', 'file'."
    )


__all__ = ["get_provider"]

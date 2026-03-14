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

    # Explicit
    provider = get_provider("polygon")
    df = provider.get_latest_stock_bars("SPY", "5m", lookback_days=7)

Providers
─────────
  polygon   – Polygon.io REST API  (requires POLYGON_API_KEY env var)
  yfinance  – yfinance fallback     (research / development use)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.providers.base_provider import BaseProvider


def get_provider(name: str = "auto") -> "BaseProvider":
    """
    Factory function that returns a configured provider instance.

    Parameters
    ──────────
    name  "auto"     – use Polygon when POLYGON_API_KEY is set, else yfinance
          "polygon"  – require Polygon.io (raises if key missing)
          "yfinance" – always use yfinance (development / fallback)

    Returns
    ───────
    BaseProvider subclass instance ready to call.
    """
    if name == "auto":
        name = "polygon" if os.environ.get("POLYGON_API_KEY") else "yfinance"

    if name == "polygon":
        from data.providers.polygon_provider import PolygonProvider
        return PolygonProvider()

    if name == "yfinance":
        from data.providers.yfinance_provider import YFinanceProvider
        return YFinanceProvider()

    raise ValueError(
        f"Unknown provider {name!r}. Valid options: 'polygon', 'yfinance', 'auto'."
    )


__all__ = ["get_provider"]

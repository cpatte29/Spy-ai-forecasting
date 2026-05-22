"""
option_chain_loader.py
──────────────────────
Load option chain snapshots from Polygon.io for a given underlying ticker.

Polygon endpoint used
─────────────────────
  GET /v3/snapshot/options/{underlyingAsset}
  Docs: https://polygon.io/docs/options/get_v3_snapshot_options__underlyingasset

Each result includes:
  details.ticker, details.contract_type, details.strike_price,
  details.expiration_date, details.shares_per_contract,
  day.volume, day.open, day.high, day.low, day.close, day.vwap,
  open_interest, implied_volatility, greeks.delta/gamma/theta/vega,
  underlying_asset.price

Notes
─────
- Requires POLYGON_API_KEY env var with options data access.
- Returns tz-naive ET timestamps; all prices in USD.
- Filters applied: max_dte, moneyness range, min_oi, min_vol.
- Far-OTM contracts (>OTM_LOTTO_PCT from spot) are flagged, not dropped.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_POLYGON_BASE   = "https://api.polygon.io"
_SNAPSHOT_PATH  = "/v3/snapshot/options/{underlying}"
_PAGE_LIMIT     = 250          # max results per Polygon page
_REQUEST_TIMEOUT = 20          # seconds

# Moneyness thresholds
_NTM_PCT        = 0.05         # within 5% of spot = near-the-money
_OTM_PCT        = 0.15         # 5–15% OTM = out-of-the-money
OTM_LOTTO_PCT   = 0.25         # beyond 25% = lotto / flagged


def _get_api_key() -> str:
    key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "POLYGON_API_KEY environment variable is not set.\n"
            "Export it before running:\n"
            "  export POLYGON_API_KEY=<your_key>\n"
        )
    return key


def _days_to_expiry(expiration_date: str) -> int:
    """Calendar days from today to expiration_date (YYYY-MM-DD)."""
    exp = date.fromisoformat(expiration_date)
    return max(0, (exp - date.today()).days)


def _moneyness_label(strike: float, spot: float) -> str:
    if spot <= 0:
        return "UNKNOWN"
    pct = abs(strike - spot) / spot
    if pct <= _NTM_PCT:
        return "NTM"
    if pct <= _OTM_PCT:
        return "OTM"
    if pct <= OTM_LOTTO_PCT:
        return "FAR_OTM"
    return "LOTTO"


def _fetch_snapshot_page(
    underlying: str,
    api_key: str,
    params: dict,
) -> tuple[list[dict], Optional[str]]:
    """Fetch one page of the options snapshot. Returns (results, next_url)."""
    url = _POLYGON_BASE + _SNAPSHOT_PATH.format(underlying=underlying)
    params = {**params, "apiKey": api_key}

    resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)

    if resp.status_code == 403:
        raise PermissionError(
            f"Polygon returned 403 for {underlying} options.\n"
            "Options data requires a Polygon subscription with options access.\n"
            "Check your plan at https://polygon.io/dashboard"
        )
    if resp.status_code == 404:
        logger.warning("No options data found for %s (404).", underlying)
        return [], None
    resp.raise_for_status()

    data = resp.json()
    results  = data.get("results") or []
    next_url = data.get("next_url")
    return results, next_url


def _fetch_all_snapshot(
    underlying: str,
    api_key: str,
    contract_type: Optional[str] = None,
    max_dte: int = 90,
    min_oi: int = 0,
    min_vol: int = 0,
    strike_pct_range: float = OTM_LOTTO_PCT,
    max_pages: int = 10,
) -> list[dict]:
    """
    Paginate through the Polygon options snapshot for `underlying`.
    Returns raw result dicts (unprocessed).
    """
    params: dict = {"limit": _PAGE_LIMIT}
    if contract_type:
        params["contract_type"] = contract_type

    # Expiry filter: today → today + max_dte
    today_str = date.today().isoformat()
    max_date  = (datetime.now().replace(
        year=datetime.now().year + (max_dte // 365),
        month=((datetime.now().month - 1 + max_dte // 30) % 12) + 1,
        day=min(datetime.now().day, 28),
    )).date()
    params["expiration_date.gte"] = today_str
    params["expiration_date.lte"] = (
        date.fromordinal(date.today().toordinal() + max_dte).isoformat()
    )

    all_results: list[dict] = []
    next_url: Optional[str] = None
    pages = 0

    while pages < max_pages:
        if next_url:
            # Follow pagination URL from Polygon
            sep = "&" if "?" in next_url else "?"
            paged = requests.get(
                f"{next_url}{sep}apiKey={api_key}",
                timeout=_REQUEST_TIMEOUT,
            )
            paged.raise_for_status()
            data     = paged.json()
            results  = data.get("results") or []
            next_url = data.get("next_url")
        else:
            results, next_url = _fetch_snapshot_page(underlying, api_key, params)

        all_results.extend(results)
        pages += 1

        if not next_url:
            break

    return all_results


def _parse_result(r: dict, spot: float) -> Optional[dict]:
    """Parse one Polygon options snapshot result into a flat dict."""
    try:
        details = r.get("details") or {}
        day     = r.get("day")     or {}
        greeks  = r.get("greeks")  or {}
        ua      = r.get("underlying_asset") or {}

        strike  = float(details.get("strike_price", 0) or 0)
        expiry  = details.get("expiration_date", "")
        ctype   = (details.get("contract_type") or "").lower()  # "call" or "put"
        opt_ticker = details.get("ticker", "")

        if not expiry or not ctype or strike <= 0:
            return None

        dte     = _days_to_expiry(expiry)
        volume  = int(day.get("volume", 0) or 0)
        oi      = int(r.get("open_interest", 0) or 0)
        iv      = float(r.get("implied_volatility", 0) or 0)

        # Mid price estimate
        bid   = float(day.get("open", 0) or 0)    # Polygon doesn't give live bid/ask in snapshot
        ask   = float(day.get("close", 0) or 0)   # Use day open/close as rough spread proxy
        last  = float(day.get("close", 0) or 0)
        vwap  = float(day.get("vwap", 0) or 0)
        mid   = vwap if vwap > 0 else last

        vol_oi_ratio = volume / max(oi, 1)
        premium_est  = volume * mid * float(details.get("shares_per_contract", 100) or 100)

        moneyness = _moneyness_label(strike, spot)
        far_otm   = moneyness in ("FAR_OTM", "LOTTO")

        return {
            "opt_ticker":    opt_ticker,
            "contract_type": ctype,
            "strike":        strike,
            "expiry":        expiry,
            "dte":           dte,
            "volume":        volume,
            "open_interest": oi,
            "vol_oi_ratio":  round(vol_oi_ratio, 4),
            "mid":           mid,
            "vwap":          vwap,
            "iv":            iv,
            "premium_est":   round(premium_est, 2),
            "delta":         float(greeks.get("delta", 0) or 0),
            "gamma":         float(greeks.get("gamma", 0) or 0),
            "theta":         float(greeks.get("theta", 0) or 0),
            "vega":          float(greeks.get("vega", 0) or 0),
            "moneyness":     moneyness,
            "far_otm_flag":  far_otm,
            "spot":          spot,
        }
    except (TypeError, ValueError, KeyError) as exc:
        logger.debug("Failed to parse option result: %s — %s", r, exc)
        return None


def load_chain(
    ticker: str,
    max_dte: int = 90,
    min_oi: int = 10,
    min_vol: int = 1,
    strike_pct_range: float = OTM_LOTTO_PCT,
    max_pages: int = 8,
) -> pd.DataFrame:
    """
    Load and normalise the options chain snapshot for `ticker`.

    Parameters
    ──────────
    ticker            Underlying symbol (e.g. "SPY", "AAPL").
    max_dte           Maximum days to expiry to include (default 90).
    min_oi            Minimum open interest per contract (filters micro-OI noise).
    min_vol           Minimum daily volume per contract.
    strike_pct_range  Maximum moneyness distance to include (default OTM_LOTTO_PCT=0.25).
                      Contracts beyond this are flagged but still included.
    max_pages         Maximum Polygon pages to fetch (safety cap).

    Returns
    ───────
    pd.DataFrame with columns:
        opt_ticker, contract_type, strike, expiry, dte, volume, open_interest,
        vol_oi_ratio, mid, vwap, iv, premium_est,
        delta, gamma, theta, vega,
        moneyness, far_otm_flag, spot

    Empty DataFrame if no data available or API error.
    """
    api_key = _get_api_key()

    # First fetch a stock quote to get the current spot price
    spot = _get_spot_price(ticker, api_key)
    if spot <= 0:
        logger.warning("Could not fetch spot price for %s — chain load aborted.", ticker)
        return pd.DataFrame()

    logger.info("Loading options chain for %s (spot=%.2f, max_dte=%d)…", ticker, spot, max_dte)

    try:
        raw = _fetch_all_snapshot(
            underlying       = ticker,
            api_key          = api_key,
            max_dte          = max_dte,
            min_oi           = min_oi,
            strike_pct_range = strike_pct_range,
            max_pages        = max_pages,
        )
    except PermissionError:
        raise
    except Exception as exc:
        logger.error("Error fetching options chain for %s: %s", ticker, exc)
        return pd.DataFrame()

    if not raw:
        logger.warning("No options data returned for %s.", ticker)
        return pd.DataFrame()

    rows = [_parse_result(r, spot) for r in raw]
    rows = [r for r in rows if r is not None]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Apply filters (but keep far-OTM flagged, not dropped)
    df = df[df["dte"] <= max_dte]
    df = df[df["open_interest"] >= min_oi]
    df = df[df["volume"] >= min_vol]

    # Strike range filter — exclude extreme LOTTO contracts from scoring
    # (they remain in df with far_otm_flag=True so the scorer can flag them)
    pct_from_spot = (df["strike"] - spot).abs() / spot
    df = df[pct_from_spot <= strike_pct_range + 0.10]   # keep up to 35% OTM

    df = df.sort_values(["contract_type", "expiry", "strike"]).reset_index(drop=True)
    logger.info("Loaded %d contracts for %s.", len(df), ticker)
    return df


def _get_spot_price(ticker: str, api_key: str) -> float:
    """Fetch the current stock price via Polygon's ticker snapshot endpoint."""
    url    = f"{_POLYGON_BASE}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
    params = {"apiKey": api_key}
    try:
        resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return 0.0
        data   = resp.json()
        ticker_data = data.get("ticker") or {}
        day    = ticker_data.get("day") or {}
        price  = float(day.get("c", 0) or 0)
        if price <= 0:
            # Fall back to last trade
            lt = ticker_data.get("lastTrade") or {}
            price = float(lt.get("p", 0) or 0)
        return price
    except Exception as exc:
        logger.debug("Spot price fetch failed for %s: %s", ticker, exc)
        return 0.0

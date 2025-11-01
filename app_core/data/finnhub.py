"""Helpers for interacting with the Finnhub market data API."""

from __future__ import annotations

import datetime as dt
import os
from typing import Any, Dict, Optional, Tuple

import requests

try:  # pragma: no cover - Python < 3.9 fallback
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore


US_EASTERN = ZoneInfo("America/New_York")

API_KEY = os.getenv(
    "FINNHUB_API_KEY",
    "d3ifbshr01qn6oiodof0d3ifbshr01qn6oiodofg",
)

PROFILE_URL = "https://finnhub.io/api/v1/stock/profile2"
QUOTE_URL = "https://finnhub.io/api/v1/quote"

Timeout = Optional[float]


def fetch_company_profile(
    symbol: str,
    *,
    session: requests.Session | None = None,
    timeout: Timeout = 10,
) -> Optional[Dict[str, Any]]:
    """Return company metadata for ``symbol`` or ``None`` on failure."""

    client = session or requests
    try:
        resp = client.get(
            PROFILE_URL,
            params={"symbol": symbol.upper(), "token": API_KEY},
            timeout=timeout,
        )
        if not resp.ok:
            return None
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    sector = str(data.get("finnhubIndustry") or data.get("sector") or "").strip()
    industry = str(data.get("industry") or data.get("gicsSector") or "").strip()
    name = str(data.get("name") or data.get("ticker") or "").strip()
    exchange = str(data.get("exchange") or data.get("exchangeSymbol") or "").strip()

    return {
        "sector": sector,
        "industry": industry,
        "name": name,
        "exchange": exchange,
        "source": "finnhub",
        "last_updated": dt.datetime.now(US_EASTERN).isoformat(),
    }


def fetch_quote(
    symbol: str,
    *,
    session: requests.Session | None = None,
    timeout: Timeout = 5,
) -> Optional[Dict[str, Any]]:
    """Return the latest quote snapshot for ``symbol``."""

    client = session or requests
    try:
        resp = client.get(
            QUOTE_URL,
            params={"symbol": symbol.upper(), "token": API_KEY},
            timeout=timeout,
        )
        if not resp.ok:
            return None
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return None

    if not isinstance(payload, dict):
        return None

    return {
        "price": payload.get("c"),
        "prev_close": payload.get("pc"),
        "timestamp": payload.get("t"),
    }


def check_status(
    symbol: str = "AAPL",
    *,
    session: requests.Session | None = None,
    timeout: Timeout = 5,
) -> Tuple[bool, str]:
    """Health-check helper returning ``(ok, detail)`` for Finnhub."""

    snapshot = fetch_quote(symbol, session=session, timeout=timeout)
    if snapshot is None:
        return False, "返回内容异常"

    price = snapshot.get("price")
    prev_close = snapshot.get("prev_close")
    timestamp = snapshot.get("timestamp")

    if price is not None and prev_close is not None and timestamp:
        detail = f"现价 {price}｜昨收 {prev_close}"
        return True, detail

    return False, "缺少现价/昨收数据"


__all__ = [
    "API_KEY",
    "fetch_company_profile",
    "fetch_quote",
    "check_status",
]

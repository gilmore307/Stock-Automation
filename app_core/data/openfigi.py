"""Helpers for interacting with the OpenFIGI mapping API."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import requests

API_URL = "https://api.openfigi.com/v3/mapping"

API_KEY = os.getenv(
    "OPENFIGI_API_KEY",
    "9e242491-ee71-47c0-9e04-49d2e952c15c",
)


def map_ticker(
    symbol: str,
    *,
    session: requests.Session | None = None,
    timeout: float | None = 5,
) -> List[Dict[str, Any]]:
    """Return mapping rows for ``symbol`` or an empty list on failure."""

    client = session or requests
    headers = {
        "Content-Type": "application/json",
    }
    if API_KEY:
        headers["X-OPENFIGI-APIKEY"] = API_KEY

    try:
        resp = client.post(
            API_URL,
            headers=headers,
            json=[{"idType": "TICKER", "idValue": symbol.upper()}],
            timeout=timeout,
        )
        if not resp.ok:
            return []
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return []

    if isinstance(payload, list) and payload:
        entry = payload[0]
        data_rows = entry.get("data") if isinstance(entry, dict) else None
        if isinstance(data_rows, list):
            return data_rows
    return []


def check_status(
    symbol: str = "AAPL",
    *,
    session: requests.Session | None = None,
    timeout: float | None = 5,
) -> Tuple[bool, str]:
    """Return ``(ok, detail)`` for an OpenFIGI health check."""

    rows = map_ticker(symbol, session=session, timeout=timeout)
    if rows:
        mapping = rows[0]
        figi = (mapping or {}).get("figi")
        name = (mapping or {}).get("name") or (mapping or {}).get("securityName")
        if figi:
            name_part = f"｜{name}" if name else ""
            return True, f"{symbol.upper()} → {figi}{name_part}"
        return False, "缺少 FIGI 字段"
    return False, "缺少映射条目"


__all__ = [
    "API_KEY",
    "map_ticker",
    "check_status",
]

"""Helpers for retrieving macro data from the FRED API."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import requests

API_URL = "https://api.stlouisfed.org/fred/series/observations"

API_KEY = os.getenv(
    "FRED_API_KEY",
    "5c9129e297742bb633b85e498edf83fa",
)


def fetch_latest_observation(
    series_id: str,
    *,
    session: requests.Session | None = None,
    timeout: float | None = 5,
) -> Dict[str, Any] | None:
    """Return the latest observation for ``series_id`` or ``None``."""

    client = session or requests
    params = {
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
    }
    try:
        resp = client.get(API_URL, params=params, timeout=timeout)
        if not resp.ok:
            return None
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return None

    if not isinstance(payload, dict):
        return None

    observations = payload.get("observations")
    if isinstance(observations, list) and observations:
        latest = observations[0]
        if isinstance(latest, dict):
            return latest
    return None


def check_status(
    series_id: str = "DGS3MO",
    *,
    session: requests.Session | None = None,
    timeout: float | None = 5,
) -> Tuple[bool, str]:
    """Return ``(ok, detail)`` for a FRED health check."""

    latest = fetch_latest_observation(series_id, session=session, timeout=timeout)
    if not isinstance(latest, dict):
        return False, "缺少观测数据"

    value = latest.get("value")
    date_text = latest.get("date")
    if value and value != ".":
        return True, f"{date_text} → {value}"

    return False, "观测值缺失"


__all__ = [
    "API_KEY",
    "fetch_latest_observation",
    "check_status",
]

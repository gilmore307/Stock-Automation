"""Helpers for downloading data from the Nasdaq earnings calendar API."""

from __future__ import annotations

import datetime as dt
import json
import time
from typing import Any, Dict, Iterable, List, Tuple

import requests

EARNINGS_API_URL = "https://api.nasdaq.com/api/calendar/earnings"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/market-activity/earnings",
}


def _extract_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    data_block = payload.get("data") if isinstance(payload.get("data"), dict) else None
    rows_candidate: Iterable[Dict[str, Any]] | None = None

    if isinstance(data_block, dict):
        calendar_block = data_block.get("calendar")
        if isinstance(calendar_block, dict):
            calendar_rows = calendar_block.get("rows")
            if isinstance(calendar_rows, list):
                rows_candidate = calendar_rows
        if rows_candidate is None:
            direct_rows = data_block.get("rows")
            if isinstance(direct_rows, list):
                rows_candidate = direct_rows

    if rows_candidate is None:
        fallback_rows = payload.get("rows")
        if isinstance(fallback_rows, list):
            rows_candidate = fallback_rows

    return list(rows_candidate or [])


def fetch_earnings(
    date: dt.date,
    *,
    session: requests.Session | None = None,
    timeout: float | None = 20,
    retries: int = 1,
) -> List[Dict[str, Any]]:
    """Fetch Nasdaq earnings rows for ``date`` with a simple retry."""

    client = session or requests
    params = {"date": date.strftime("%Y-%m-%d")}

    for attempt in range(retries + 1):
        try:
            resp = client.get(
                EARNINGS_API_URL,
                params=params,
                headers=HEADERS,
                timeout=timeout,
            )
            resp.raise_for_status()
        except requests.RequestException:
            if attempt >= retries:
                raise
            time.sleep(1)
            continue

        try:
            payload = resp.json()
        except json.JSONDecodeError:
            if attempt >= retries:
                raise
            time.sleep(1)
            continue

        if not isinstance(payload, dict):
            return []

        rows = _extract_rows(payload)
        if not rows:
            return []

        results: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = (
                row.get("symbol")
                or row.get("Symbol")
                or row.get("companyTickerSymbol")
                or ""
            )
            time_field = (
                row.get("time")
                or row.get("Time")
                or row.get("EPSTime")
                or row.get("timeStatus")
                or row.get("when")
                or ""
            )
            name = (
                row.get("companyName")
                or row.get("name")
                or row.get("Company")
                or ""
            )
            symbol = str(symbol).strip()
            if not symbol:
                continue
            results.append(
                {
                    "symbol": symbol.upper(),
                    "company": str(name).strip(),
                    "time": str(time_field).strip(),
                    "raw": row,
                }
            )
        return results

    return []


def check_status(
    date: dt.date | str,
    *,
    session: requests.Session | None = None,
    timeout: float | None = 5,
) -> Tuple[bool, str]:
    """Return ``(ok, detail)`` for a connection test against Nasdaq."""

    client = session or requests
    if isinstance(date, dt.date):
        date_str = date.isoformat()
    else:
        date_str = str(date)

    try:
        resp = client.get(
            EARNINGS_API_URL,
            params={"date": date_str},
            headers=HEADERS,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return False, f"请求异常：{exc}"[:160]

    if not resp.ok:
        return False, f"HTTP {resp.status_code}"

    try:
        payload = resp.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        rows = _extract_rows(payload)
        if rows:
            sample = rows[0]
            symbol = (
                sample.get("symbol")
                or sample.get("Symbol")
                or sample.get("companyTickerSymbol")
                or ""
            )
            time_field = (
                sample.get("time")
                or sample.get("Time")
                or sample.get("EPSTime")
                or sample.get("timeStatus")
                or sample.get("when")
                or ""
            )
            symbol = str(symbol).strip().upper()
            time_field = str(time_field).strip() or "未提供时间"
            if symbol:
                detail = f"示例：{symbol}｜{time_field}"
                return True, detail
            return False, "返回行缺少代码字段"
        return False, "缺少财报行数据"

    return False, "返回内容异常"


__all__ = [
    "EARNINGS_API_URL",
    "HEADERS",
    "fetch_earnings",
    "check_status",
]

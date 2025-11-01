"""Thin wrapper around :mod:`tvDatafeed` with shared credentials handling."""

from __future__ import annotations

import datetime as dt
import os
import threading
from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:  # pragma: no cover - optional dependency
    from tvDatafeed import Interval, TvDatafeed
except Exception:  # pragma: no cover - library may be missing
    Interval = None  # type: ignore[assignment]
    TvDatafeed = None  # type: ignore[assignment]


__all__ = [
    "TVDataError",
    "PricePoint",
    "ensure_client",
    "fetch_hist",
    "fetch_price_point",
    "is_available",
]


DEFAULT_INTERVAL = getattr(Interval, "in_1_minute", "1m")


class TVDataError(RuntimeError):
    """Raised when tvDatafeed is unavailable or returns invalid data."""


@dataclass(slots=True)
class PricePoint:
    """Container for a single price observation."""

    price: float
    timestamp: dt.datetime


_CLIENT_LOCK = threading.Lock()
_CLIENT: Optional[TvDatafeed] = None


def _get_credentials() -> tuple[Optional[str], Optional[str]]:
    """Return the tvDatafeed credentials from the environment."""

    username = os.getenv("TVDATAFEED_USERNAME")
    password = os.getenv("TVDATAFEED_PASSWORD")
    return username or None, password or None


def is_available() -> bool:
    """Return ``True`` if tvDatafeed can be instantiated."""

    if TvDatafeed is None:
        return False
    username, password = _get_credentials()
    return bool(username and password)


def ensure_client() -> Optional[TvDatafeed]:
    """Return a cached :class:`TvDatafeed` instance if credentials are set."""

    if TvDatafeed is None:
        return None

    username, password = _get_credentials()
    if not username or not password:
        return None

    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        try:
            _CLIENT = TvDatafeed(username=username, password=password)
        except Exception as exc:  # pragma: no cover - network/login errors
            raise TVDataError(f"tvDatafeed 登录失败：{exc}") from exc
    return _CLIENT


def _to_naive_utc(value: dt.datetime | None) -> Optional[dt.datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(dt.timezone.utc).replace(tzinfo=None)


def fetch_hist(
    symbol: str,
    exchange: str,
    *,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    interval: object = DEFAULT_INTERVAL,
    n_bars: int | None = None,
    extended_session: bool = False,
) -> pd.DataFrame:
    """Fetch historical data for ``symbol`` within ``[start, end]``."""

    client = ensure_client()
    if client is None:
        raise TVDataError("tvDatafeed 未启用（缺少库或凭据）。")

    params = {
        "symbol": symbol,
        "exchange": exchange,
        "interval": interval,
        "extended_session": extended_session,
    }
    start_naive = _to_naive_utc(start)
    end_naive = _to_naive_utc(end)
    if start_naive is not None:
        params["start_date"] = start_naive
    if end_naive is not None:
        params["end_date"] = end_naive
    if n_bars is not None:
        params["n_bars"] = n_bars

    try:
        data = client.get_hist(**params)
    except Exception as exc:  # pragma: no cover - network issues
        raise TVDataError(f"tvDatafeed 请求失败：{exc}") from exc

    if data is None or isinstance(data, list):  # pragma: no cover - defensive guard
        raise TVDataError("tvDatafeed 返回空数据。")

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise TVDataError("tvDatafeed 返回结果为空。")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TVDataError("tvDatafeed 返回的索引不是时间序列。")

    return data


def fetch_price_point(
    data: pd.DataFrame,
    target: dt.datetime,
    *,
    tolerance: dt.timedelta = dt.timedelta(minutes=5),
    field: str = "close",
) -> Optional[PricePoint]:
    """Return the price nearest to ``target`` within ``tolerance`` minutes."""

    if data.empty:
        return None

    index = data.index
    if index.tz is None:
        index_utc = index.tz_localize(dt.timezone.utc, nonexistent="shift_forward")
    else:
        index_utc = index.tz_convert(dt.timezone.utc)

    target_utc = target.astimezone(dt.timezone.utc)
    diffs = (index_utc - target_utc).abs()
    try:
        position = diffs.argmin()
    except ValueError:
        return None

    nearest_delta = diffs[position]
    if isinstance(nearest_delta, dt.timedelta):
        delta = nearest_delta
    else:  # pragma: no cover - pandas Timedelta
        delta = dt.timedelta(seconds=float(nearest_delta.total_seconds()))

    if delta > tolerance:
        return None

    row = data.iloc[position]
    if field not in row:
        # allow alternate casing (e.g. "Close")
        alt_field = field.capitalize()
        if alt_field in row:
            field = alt_field
        else:
            return None

    try:
        price = float(row[field])
    except (TypeError, ValueError):
        return None

    timestamp = index_utc[position].to_pydatetime()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)

    return PricePoint(price=price, timestamp=timestamp)

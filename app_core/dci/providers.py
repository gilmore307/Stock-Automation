"""Utility helpers to source DCI input payloads from multiple backends."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import requests

from .config import provider_defaults


logger = logging.getLogger(__name__)

__all__ = [
    "DCIDataProvider",
    "DCIDataProviderError",
    "FileDCIDataProvider",
    "DirectoryDCIDataProvider",
    "HTTPDCIDataProvider",
    "register_dci_provider",
    "get_default_provider",
    "load_dci_payloads",
]


class DCIDataProviderError(RuntimeError):
    """Raised when a provider cannot deliver a usable payload."""


class DCIDataProvider:
    """Base class for DCI payload providers."""

    def load(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError


_CUSTOM_PROVIDER: DCIDataProvider | None = None
_PROVIDER_DEFAULTS = provider_defaults()


def register_dci_provider(provider: DCIDataProvider | None) -> None:
    """Register an explicit provider instance used for subsequent loads."""

    global _CUSTOM_PROVIDER
    _CUSTOM_PROVIDER = provider
    if hasattr(load_dci_payloads, "_cache"):
        delattr(load_dci_payloads, "_cache")


@dataclass
class FileDCIDataProvider(DCIDataProvider):
    """Load all inputs from a single JSON file."""

    path: Path

    def load(self) -> Dict[str, Dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError as exc:
            raise DCIDataProviderError(str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise DCIDataProviderError(str(exc)) from exc
        return _normalise_payload(payload)


@dataclass
class DirectoryDCIDataProvider(DCIDataProvider):
    """Load inputs from a directory of JSON files."""

    directory: Path
    glob: str = "*.json"

    def load(self) -> Dict[str, Dict[str, Any]]:
        if not self.directory.exists():
            raise DCIDataProviderError(f"Directory not found: {self.directory}")

        payload: Dict[str, Dict[str, Any]] = {}
        for path in sorted(self.directory.glob(self.glob)):
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    candidate = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue

            if isinstance(candidate, dict):
                symbol_key = _infer_symbol_key(candidate, path)
                if symbol_key:
                    payload[symbol_key] = candidate
                    continue

            if path.stem:
                payload[path.stem.upper()] = candidate if isinstance(candidate, dict) else {}

        if not payload:
            raise DCIDataProviderError(
                f"No usable DCI payloads found in directory: {self.directory}"
            )
        return payload


@dataclass
class HTTPDCIDataProvider(DCIDataProvider):
    """Load inputs from an HTTP endpoint returning JSON payloads."""

    url: str
    timeout: float = 10.0

    def load(self) -> Dict[str, Dict[str, Any]]:
        try:
            response = requests.get(self.url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DCIDataProviderError(str(exc)) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise DCIDataProviderError("Response is not valid JSON") from exc

        return _normalise_payload(payload)


def _infer_symbol_key(candidate: Dict[str, Any], path: Path) -> str | None:
    symbol = candidate.get("symbol") or candidate.get("ticker")
    if isinstance(symbol, str) and symbol.strip():
        return symbol.strip().upper()
    if path.stem:
        return path.stem.upper()
    return None


def _normalise_payload(raw: Any) -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}

    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(key, str) and isinstance(value, dict):
                payload[key.upper()] = value
        return payload

    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            symbol = _infer_symbol_key(entry, Path(""))
            if symbol:
                payload[symbol] = entry
        return payload

    return payload


def get_default_provider() -> DCIDataProvider:
    """Resolve the preferred provider based on environment configuration."""

    if _CUSTOM_PROVIDER is not None:
        return _CUSTOM_PROVIDER

    url = os.getenv("DCI_DATA_URL") or (_PROVIDER_DEFAULTS.url if _PROVIDER_DEFAULTS.url else None)
    if url:
        return HTTPDCIDataProvider(url)

    directory = os.getenv("DCI_DATA_DIR") or (
        str(_PROVIDER_DEFAULTS.directory) if _PROVIDER_DEFAULTS.directory else None
    )
    if directory:
        return DirectoryDCIDataProvider(Path(directory))

    path = os.getenv("DCI_DATA_PATH") or (
        str(_PROVIDER_DEFAULTS.path) if _PROVIDER_DEFAULTS.path else None
    )
    if path:
        return FileDCIDataProvider(Path(path))

    raise DCIDataProviderError("未配置可用的 DCI 数据提供器")


def load_dci_payloads(force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """Load DCI payloads from the resolved provider with simple caching."""

    if force_reload or not hasattr(load_dci_payloads, "_cache"):
        try:
            provider = get_default_provider()
            payload = provider.load()
        except DCIDataProviderError as exc:
            logger.warning("载入 DCI 数据失败：%s", exc)
            payload = {}
        load_dci_payloads._cache = payload  # type: ignore[attr-defined]
        return dict(payload)

    return dict(load_dci_payloads._cache)  # type: ignore[attr-defined]

"""Configuration utilities for DCI calibration parameters."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence


_CONFIG_ENV_VAR = "DCI_CONFIG_PATH"
_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class Curve:
    """Simple piecewise-linear curve with monotonic constraints."""

    x: Sequence[float]
    y: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError("Curve requires matching x/y lengths.")
        if len(self.x) < 2:
            raise ValueError("Curve requires at least two points.")
        object.__setattr__(self, "x", tuple(self.x))
        object.__setattr__(self, "y", tuple(self.y))

    def __call__(self, value: float) -> float:
        if value <= self.x[0]:
            return float(self.y[0])
        if value >= self.x[-1]:
            return float(self.y[-1])
        # Binary search for segment
        lo = 0
        hi = len(self.x) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if value < self.x[mid]:
                hi = mid
            else:
                lo = mid
        x0, y0 = self.x[lo], self.y[lo]
        x1, y1 = self.x[hi], self.y[hi]
        if math.isclose(x0, x1):
            return float(y0)
        t = (value - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))

    def is_monotonic_increasing(self) -> bool:
        last = self.y[0]
        for value in self.y[1:]:
            if value + 1e-9 < last:
                return False
            last = value
        return True

    def is_monotonic_decreasing(self) -> bool:
        last = self.y[0]
        for value in self.y[1:]:
            if value - 1e-9 > last:
                return False
            last = value
        return True

    def is_concave(self) -> bool:
        slopes = []
        for i in range(len(self.x) - 1):
            dx = self.x[i + 1] - self.x[i]
            if math.isclose(dx, 0.0):
                slopes.append(0.0)
            else:
                slopes.append((self.y[i + 1] - self.y[i]) / dx)
        for left, right in zip(slopes, slopes[1:]):
            if right - left > 1e-9:
                return False
        return True


@dataclass(frozen=True)
class BetaParams:
    beta0: float
    beta1: float

    @classmethod
    def from_mapping(cls, mapping: dict[str, float]) -> "BetaParams":
        beta0 = float(mapping.get("beta0", 0.0))
        beta1 = float(mapping.get("beta1", 1.0))
        if beta1 <= 0.0:
            beta1 = 1.0
        return cls(beta0=beta0, beta1=beta1)


@dataclass(frozen=True)
class QualityGate:
    threshold: float
    cap: float


@dataclass(frozen=True)
class DCIConfig:
    beta: BetaParams
    gamma: float
    kappa_d: float
    eta_macro: float
    ci_scale: Curve
    q_scale: Curve
    penalty_curve: Curve
    quality_gate: QualityGate
    platt_path: Path | None


DEFAULT_CONFIG = {
    "beta": {"beta0": 0.0, "beta1": 1.0},
    "gamma": 0.35,
    "kappa_d": 0.4,
    "eta_macro": 0.8,
    "ci_scale": {
        "x": [0.0, 30.0, 50.0, 70.0, 100.0],
        "y": [0.55, 0.65, 0.75, 0.85, 0.9],
    },
    "q_scale": {
        "x": [0.0, 40.0, 60.0, 80.0, 100.0],
        "y": [0.95, 0.85, 0.7, 0.6, 0.55],
    },
    "penalty_curve": {
        "x": [0.0, 2.0, 5.0, 10.0],
        "y": [-10.0, -6.0, -2.0, 0.0],
    },
    "quality_gate": {"threshold": 60.0, "cap": 55.0},
    "platt": {"path": ""},
}


def _read_json_file(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    return {}


def _merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    out = dict(base)
    for key, value in override.items():
        if key not in base:
            out[key] = value
            continue
        if isinstance(value, dict) and isinstance(base[key], dict):
            out[key] = _merge(base[key], value)
        else:
            out[key] = value
    return out


_CACHED_CONFIG: DCIConfig | None = None
_PLATT_OVERRIDES: dict[str, BetaParams] | None = None


def _build_curve(raw: dict[str, Iterable[float]], *, kind: str) -> Curve:
    x = list(raw.get("x", ()))
    y = list(raw.get("y", ()))
    if len(x) != len(y) or len(x) < 2:
        raise ValueError(f"Invalid curve definition for {kind}.")
    points = sorted(zip(x, y), key=lambda item: item[0])
    x_sorted = [float(px) for px, _ in points]
    y_sorted = [float(py) for _, py in points]
    curve = Curve(x=tuple(x_sorted), y=tuple(y_sorted))
    return curve


def _load_platt_overrides(path: Path) -> dict[str, BetaParams]:
    mapping = _read_json_file(path)
    overrides: dict[str, BetaParams] = {}
    for symbol, params in mapping.items():
        if not isinstance(symbol, str) or not isinstance(params, dict):
            continue
        beta = BetaParams.from_mapping(params)
        overrides[symbol.upper()] = beta
    return overrides


def load_config() -> DCIConfig:
    global _CACHED_CONFIG, _PLATT_OVERRIDES
    if _CACHED_CONFIG is not None:
        return _CACHED_CONFIG

    config_path: Path | None = None
    env_path = os.environ.get(_CONFIG_ENV_VAR)
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            config_path = candidate
    if config_path is None and _DEFAULT_CONFIG_PATH.is_file():
        config_path = _DEFAULT_CONFIG_PATH

    raw_config = dict(DEFAULT_CONFIG)
    if config_path is not None:
        raw_config = _merge(raw_config, _read_json_file(config_path))

    beta = BetaParams.from_mapping(raw_config.get("beta", {}))

    try:
        ci_curve = _build_curve(raw_config.get("ci_scale", {}), kind="ci_scale")
        if not ci_curve.is_monotonic_increasing():
            raise ValueError
    except Exception:
        ci_curve = _build_curve(DEFAULT_CONFIG["ci_scale"], kind="ci_scale")

    try:
        q_curve = _build_curve(raw_config.get("q_scale", {}), kind="q_scale")
        if not q_curve.is_monotonic_decreasing():
            raise ValueError
    except Exception:
        q_curve = _build_curve(DEFAULT_CONFIG["q_scale"], kind="q_scale")

    try:
        penalty_curve = _build_curve(raw_config.get("penalty_curve", {}), kind="penalty_curve")
        if not penalty_curve.is_concave():
            raise ValueError
        if any(value > 0.0 for value in penalty_curve.y):
            raise ValueError
    except Exception:
        penalty_curve = _build_curve(DEFAULT_CONFIG["penalty_curve"], kind="penalty_curve")

    gate_raw = raw_config.get("quality_gate", {})
    threshold = float(gate_raw.get("threshold", DEFAULT_CONFIG["quality_gate"]["threshold"]))
    cap = float(gate_raw.get("cap", DEFAULT_CONFIG["quality_gate"]["cap"]))
    quality_gate = QualityGate(threshold=threshold, cap=cap)

    gamma = float(raw_config.get("gamma", DEFAULT_CONFIG["gamma"]))
    kappa_d = float(raw_config.get("kappa_d", DEFAULT_CONFIG["kappa_d"]))
    eta_macro = _clamp(float(raw_config.get("eta_macro", DEFAULT_CONFIG["eta_macro"])))

    platt_path_value = raw_config.get("platt", {}).get("path", "")
    platt_path = Path(platt_path_value).expanduser() if platt_path_value else None
    if platt_path is not None and not platt_path.is_file():
        platt_path = None

    _CACHED_CONFIG = DCIConfig(
        beta=beta,
        gamma=gamma,
        kappa_d=kappa_d,
        eta_macro=eta_macro,
        ci_scale=ci_curve,
        q_scale=q_curve,
        penalty_curve=penalty_curve,
        quality_gate=quality_gate,
        platt_path=platt_path,
    )

    _PLATT_OVERRIDES = None
    return _CACHED_CONFIG


def get_beta_for_symbol(symbol: str) -> BetaParams:
    config = load_config()
    if config.platt_path is None:
        return config.beta

    global _PLATT_OVERRIDES
    if _PLATT_OVERRIDES is None:
        _PLATT_OVERRIDES = _load_platt_overrides(config.platt_path)

    override = _PLATT_OVERRIDES.get(symbol.upper()) if _PLATT_OVERRIDES else None
    return override or config.beta


def evaluate_ci_scale(ci_value: float, quality: float) -> float:
    config = load_config()
    ci_eff = ci_value
    if quality >= config.quality_gate.threshold:
        ci_eff = min(ci_value, config.quality_gate.cap)
    return _clamp(config.ci_scale(ci_eff))


def evaluate_quality_scale(quality: float) -> float:
    config = load_config()
    return _clamp(config.q_scale(quality))


def evaluate_penalty(em_pct: float) -> float:
    config = load_config()
    return config.penalty_curve(em_pct)


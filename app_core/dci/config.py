"""Configuration utilities for DCI calibration parameters."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence


_CONFIG_ENV_VAR = "DCI_CONFIG_PATH"
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "dci.yaml"


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
        object.__setattr__(self, "x", tuple(float(value) for value in self.x))
        object.__setattr__(self, "y", tuple(float(value) for value in self.y))

    def __call__(self, value: float) -> float:
        if value <= self.x[0]:
            return float(self.y[0])
        if value >= self.x[-1]:
            return float(self.y[-1])
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
    def from_mapping(cls, mapping: Mapping[str, float]) -> "BetaParams":
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
class ProviderDefaults:
    url: str | None
    directory: Path | None
    path: Path | None


@dataclass(frozen=True)
class DefaultsConfig:
    field_defaults: Mapping[str, float]
    missing_factor_z: float
    option_expiry_days: int | None
    has_weekly_option: bool
    option_expiry_type: str | None
    provider: ProviderDefaults

    def get_field(self, name: str, fallback: float = 0.0) -> float:
        return float(self.field_defaults.get(name, fallback))


@dataclass(frozen=True)
class BucketDef:
    code: str
    label: str
    target: float


@dataclass(frozen=True)
class FactorDef:
    bucket: str
    prior: float


@dataclass(frozen=True)
class WeightsConfig:
    l2_limit: float
    intercept_limit: float
    buckets: Mapping[str, BucketDef]
    factors: Mapping[str, FactorDef]


@dataclass(frozen=True)
class WeeklyGateConfig:
    require_weekly: bool
    weekly_max_days: int | None


@dataclass(frozen=True)
class ShockCIGateConfig:
    min_ci: float | None
    max_ci: float | None


@dataclass(frozen=True)
class GatingConfig:
    min_em_pct: float
    weekly: WeeklyGateConfig
    shock_ci: ShockCIGateConfig | None
    skip_bucket: str


@dataclass(frozen=True)
class ToggleDef:
    env: str
    default: bool

    def resolve(self) -> bool:
        value = os.getenv(self.env)
        if value is None:
            return self.default
        return str(value).strip().lower() not in {"0", "false", "no", "off"}


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
    defaults: DefaultsConfig
    weights: WeightsConfig
    gates: GatingConfig
    toggles: Mapping[str, ToggleDef]


DEFAULT_CONFIG: Dict[str, object] = {
    "defaults": {
        "missing_factor_z": 0.0,
        "option_expiry_days": 7,
        "has_weekly_option": True,
        "option_expiry_type": "W",
        "fields": {
            "z_cons": 0.0,
            "z_narr": 0.0,
            "CI": 50.0,
            "Q": 50.0,
            "D": 0.0,
            "EM_pct": 5.0,
            "S_stab": 0.8,
            "shock_flag": 0,
        },
        "provider": {"url": "", "directory": "", "path": ""},
    },
    "weights": {
        "l2_limit": 0.05,
        "intercept_limit": 0.25,
        "buckets": {
            "A": {"label": "预期/基本面动能", "target": 0.45},
            "B": {"label": "经营质量/利润动能", "target": 0.25},
            "C": {"label": "价格/成交动能", "target": 0.20},
            "D": {"label": "相对估值", "target": 0.10},
        },
        "factors": {
            "EPS_Rev_30d": {"bucket": "A", "prior": 0.20},
            "Sales_Rev_30d": {"bucket": "A", "prior": 0.10},
            "Guide_Drift": {"bucket": "A", "prior": 0.10},
            "Backlog_Bookings_Delta": {"bucket": "A", "prior": 0.05},
            "GM_YoY_Delta": {"bucket": "B", "prior": 0.10},
            "OPM_YoY_Delta": {"bucket": "B", "prior": 0.10},
            "FCF_Margin_Slope": {"bucket": "B", "prior": 0.05},
            "Ret20_rel": {"bucket": "C", "prior": 0.10},
            "Ret60_rel": {"bucket": "C", "prior": 0.05},
            "UDVol20": {"bucket": "C", "prior": 0.05},
            "Value_vs_Sector": {"bucket": "D", "prior": 0.07},
            "EarningsYield_vs_Sector": {"bucket": "D", "prior": 0.03},
        },
    },
    "bounds": {
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
    },
    "gates": {
        "min_em_pct": 2.0,
        "require_weekly": True,
        "weekly_max_days": 9,
        "shock_ci": {"min_ci": 25.0, "max_ci": 85.0},
        "skip_bucket": "跳过标的",
    },
    "toggles": {
        "auto_baseline": {"env": "DCI_AUTO_BASELINE", "default": True},
        "structured_log": {"env": "DCI_STRUCTURED_LOG", "default": False},
    },
}


def _read_config_file(path: Path) -> Dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError:
        return {}

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
    if isinstance(data, dict):
        return data  # type: ignore[return-value]
    return {}


def _merge(base: MutableMapping[str, object], override: Mapping[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = dict(base)
    for key, value in override.items():
        if key not in base:
            out[key] = value
            continue
        base_value = base[key]
        if isinstance(base_value, MutableMapping) and isinstance(value, Mapping):
            out[key] = _merge(base_value, value)
        else:
            out[key] = value
    return out


def _build_curve(raw: Mapping[str, Iterable[float]], *, kind: str) -> Curve:
    x = list(raw.get("x", ()))
    y = list(raw.get("y", ()))
    if len(x) != len(y) or len(x) < 2:
        raise ValueError(f"Invalid curve definition for {kind}.")
    points = sorted(zip(x, y), key=lambda item: float(item[0]))
    x_sorted = [float(px) for px, _ in points]
    y_sorted = [float(py) for _, py in points]
    return Curve(x=tuple(x_sorted), y=tuple(y_sorted))


def _path_or_none(value: object) -> Path | None:
    if isinstance(value, str) and value.strip():
        candidate = Path(value).expanduser()
        if candidate.is_file() or not candidate.exists():
            return candidate
    return None


def _build_defaults(section: Mapping[str, object]) -> DefaultsConfig:
    fields_raw = section.get("fields", {})
    field_defaults = {str(k): float(v) for k, v in dict(fields_raw).items()} if isinstance(fields_raw, Mapping) else {}

    missing_factor = float(section.get("missing_factor_z", 0.0))
    option_days_raw = section.get("option_expiry_days")
    option_days = int(option_days_raw) if isinstance(option_days_raw, (int, float)) else None

    has_weekly = bool(section.get("has_weekly_option", True))
    option_type_raw = section.get("option_expiry_type")
    option_type = str(option_type_raw) if isinstance(option_type_raw, str) and option_type_raw.strip() else None

    provider_raw = section.get("provider", {})
    provider = ProviderDefaults(
        url=str(provider_raw.get("url")) if isinstance(provider_raw, Mapping) and provider_raw.get("url") else None,
        directory=_path_or_none(provider_raw.get("directory")) if isinstance(provider_raw, Mapping) else None,
        path=_path_or_none(provider_raw.get("path")) if isinstance(provider_raw, Mapping) else None,
    )

    return DefaultsConfig(
        field_defaults=field_defaults,
        missing_factor_z=missing_factor,
        option_expiry_days=option_days,
        has_weekly_option=has_weekly,
        option_expiry_type=option_type,
        provider=provider,
    )


def _build_weights(section: Mapping[str, object]) -> WeightsConfig:
    l2_limit = float(section.get("l2_limit", 0.05))
    intercept_limit = float(section.get("intercept_limit", 0.25))

    buckets_raw = section.get("buckets", {})
    bucket_defs: Dict[str, BucketDef] = {}
    if isinstance(buckets_raw, Mapping):
        for code, spec in buckets_raw.items():
            if not isinstance(spec, Mapping):
                continue
            bucket_defs[str(code)] = BucketDef(
                code=str(code),
                label=str(spec.get("label", code)),
                target=float(spec.get("target", 0.0)),
            )

    factors_raw = section.get("factors", {})
    factor_defs: Dict[str, FactorDef] = {}
    if isinstance(factors_raw, Mapping):
        for name, spec in factors_raw.items():
            if not isinstance(spec, Mapping):
                continue
            bucket = str(spec.get("bucket", ""))
            prior = float(spec.get("prior", 0.0))
            factor_defs[str(name)] = FactorDef(bucket=bucket, prior=prior)

    return WeightsConfig(
        l2_limit=l2_limit,
        intercept_limit=intercept_limit,
        buckets=bucket_defs,
        factors=factor_defs,
    )


def _build_gates(section: Mapping[str, object]) -> GatingConfig:
    min_em_pct = float(section.get("min_em_pct", 0.0))
    weekly = WeeklyGateConfig(
        require_weekly=bool(section.get("require_weekly", False)),
        weekly_max_days=(
            int(section.get("weekly_max_days"))
            if isinstance(section.get("weekly_max_days"), (int, float))
            else None
        ),
    )

    shock_section = section.get("shock_ci")
    shock_cfg: ShockCIGateConfig | None
    if isinstance(shock_section, Mapping):
        min_ci = shock_section.get("min_ci")
        max_ci = shock_section.get("max_ci")
        shock_cfg = ShockCIGateConfig(
            min_ci=float(min_ci) if isinstance(min_ci, (int, float)) else None,
            max_ci=float(max_ci) if isinstance(max_ci, (int, float)) else None,
        )
    else:
        shock_cfg = None

    skip_bucket = str(section.get("skip_bucket", "跳过标的"))

    return GatingConfig(
        min_em_pct=min_em_pct,
        weekly=weekly,
        shock_ci=shock_cfg,
        skip_bucket=skip_bucket,
    )


def _build_toggles(section: Mapping[str, object]) -> Dict[str, ToggleDef]:
    toggles: Dict[str, ToggleDef] = {}
    for key, spec in section.items():
        if not isinstance(spec, Mapping):
            continue
        env = str(spec.get("env", "")).strip() or key.upper()
        default = bool(spec.get("default", False))
        toggles[str(key)] = ToggleDef(env=env, default=default)
    return toggles


_CACHED_CONFIG: DCIConfig | None = None
_PLATT_OVERRIDES: Dict[str, BetaParams] | None = None


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

    base_config: Dict[str, object] = dict(DEFAULT_CONFIG)
    if config_path is not None:
        file_config = _read_config_file(config_path)
        base_config = _merge(base_config, file_config)

    bounds = base_config.get("bounds", {})
    beta = BetaParams.from_mapping(bounds.get("beta", {})) if isinstance(bounds, Mapping) else BetaParams(0.0, 1.0)

    try:
        ci_curve = _build_curve(bounds.get("ci_scale", {}), kind="ci_scale")  # type: ignore[arg-type]
        if not ci_curve.is_monotonic_increasing():
            raise ValueError
    except Exception:
        ci_curve = _build_curve(DEFAULT_CONFIG["bounds"]["ci_scale"], kind="ci_scale")  # type: ignore[index]

    try:
        q_curve = _build_curve(bounds.get("q_scale", {}), kind="q_scale")  # type: ignore[arg-type]
        if not q_curve.is_monotonic_decreasing():
            raise ValueError
    except Exception:
        q_curve = _build_curve(DEFAULT_CONFIG["bounds"]["q_scale"], kind="q_scale")  # type: ignore[index]

    try:
        penalty_curve = _build_curve(bounds.get("penalty_curve", {}), kind="penalty_curve")  # type: ignore[arg-type]
        if not penalty_curve.is_concave():
            raise ValueError
        if any(value > 0.0 for value in penalty_curve.y):
            raise ValueError
    except Exception:
        penalty_curve = _build_curve(DEFAULT_CONFIG["bounds"]["penalty_curve"], kind="penalty_curve")  # type: ignore[index]

    quality_raw = bounds.get("quality_gate", {}) if isinstance(bounds, Mapping) else {}
    quality_gate = QualityGate(
        threshold=float(quality_raw.get("threshold", DEFAULT_CONFIG["bounds"]["quality_gate"]["threshold"])),  # type: ignore[index]
        cap=float(quality_raw.get("cap", DEFAULT_CONFIG["bounds"]["quality_gate"]["cap"])),  # type: ignore[index]
    )

    gamma = float(bounds.get("gamma", DEFAULT_CONFIG["bounds"]["gamma"]))  # type: ignore[index]
    kappa_d = float(bounds.get("kappa_d", DEFAULT_CONFIG["bounds"]["kappa_d"]))  # type: ignore[index]
    eta_macro = _clamp(float(bounds.get("eta_macro", DEFAULT_CONFIG["bounds"]["eta_macro"])))  # type: ignore[index]

    platt_section = bounds.get("platt", {}) if isinstance(bounds, Mapping) else {}
    platt_path = _path_or_none(platt_section.get("path")) if isinstance(platt_section, Mapping) else None
    if platt_path is not None and not platt_path.is_file():
        platt_path = None

    defaults_section = base_config.get("defaults", {})
    defaults = _build_defaults(defaults_section if isinstance(defaults_section, Mapping) else {})

    weights_section = base_config.get("weights", {})
    weights = _build_weights(weights_section if isinstance(weights_section, Mapping) else {})

    gates_section = base_config.get("gates", {})
    gates = _build_gates(gates_section if isinstance(gates_section, Mapping) else {})

    toggles_section = base_config.get("toggles", {})
    toggles = _build_toggles(toggles_section if isinstance(toggles_section, Mapping) else {})

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
        defaults=defaults,
        weights=weights,
        gates=gates,
        toggles=toggles,
    )

    _PLATT_OVERRIDES = None
    return _CACHED_CONFIG


def _load_platt_overrides(path: Path) -> Dict[str, BetaParams]:
    mapping = _read_config_file(path)
    overrides: Dict[str, BetaParams] = {}
    for symbol, params in mapping.items():
        if not isinstance(symbol, str) or not isinstance(params, Mapping):
            continue
        beta = BetaParams.from_mapping(params)  # type: ignore[arg-type]
        overrides[symbol.upper()] = beta
    return overrides


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


def resolve_toggle(name: str) -> bool:
    config = load_config()
    toggle = config.toggles.get(name)
    if toggle is None:
        return False
    return toggle.resolve()


def provider_defaults() -> ProviderDefaults:
    return load_config().defaults.provider


def default_missing_factor_z() -> float:
    return load_config().defaults.missing_factor_z


def default_field_value(name: str) -> float:
    return load_config().defaults.get_field(name, 0.0)


def gating_config() -> GatingConfig:
    return load_config().gates


def weights_config() -> WeightsConfig:
    return load_config().weights


def defaults_config() -> DefaultsConfig:
    return load_config().defaults


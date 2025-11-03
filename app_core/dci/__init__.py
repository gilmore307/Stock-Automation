"""Directional Certainty Index (DCI) computation utilities."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from typing import Dict, Iterable, List, Optional, Tuple

from .config import (
    evaluate_ci_scale,
    evaluate_penalty,
    evaluate_quality_scale,
    get_beta_for_symbol,
    load_config,
    resolve_toggle,
)


logger = logging.getLogger(__name__)


ROBUST_EPS = 1e-9
ROBUST_CLIP = 3.0


@dataclass(frozen=True)
class FactorInput:
    """Input for a single factor.

    The value can either be a pre-computed robust z-score (preferred) or a tuple
    containing (value, median, mad). The caller is responsible for ensuring that
    the units across securities are comparable when using the raw value path.
    """

    z: float | None = None
    value: float | None = None
    median: float | None = None
    mad: float | None = None

    def scaled(self) -> float:
        """Return the compressed score in [-1, 1] using robust z + tanh."""

        if self.z is not None:
            z_val = float(self.z)
        elif (
            self.value is not None
            and self.median is not None
            and self.mad is not None
        ):
            denom = 1.4826 * float(self.mad) + ROBUST_EPS
            z_val = 3.0 * (float(self.value) - float(self.median)) / denom
        else:
            raise ValueError("FactorInput requires either z or (value, median, mad).")

        z_val = max(-ROBUST_CLIP, min(ROBUST_CLIP, z_val))
        return math.tanh(z_val / 2.0)


@dataclass(frozen=True)
class DCIInputs:
    symbol: str
    factors: Dict[str, FactorInput]
    z_cons: float
    z_narr: float
    crowding_index: float
    quality_score: float
    disagreement: float
    expected_move_pct: float
    stability: float
    shock_flag: int
    factor_weights: Dict[str, float] | None = None
    has_weekly_option: bool = True
    option_expiry_days: int | None = None
    option_expiry_type: str | None = None


@dataclass(frozen=True)
class DCIResult:
    symbol: str
    direction: int
    p_up: float
    dci_base: float
    dci_penalised: float
    dci_final: float
    position_weight: float
    certainty: float
    position_bucket: str
    shrink_factors: Dict[str, float]
    base_score: float
    scaled_factors: Dict[str, float]
    factor_weights: Dict[str, float]
    gating_passed: bool
    gating_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class DCIGatingDecision:
    passed: bool
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class FactorDescriptor:
    """Descriptor tying a factor to its bucket and prior weight."""

    bucket: str
    prior: float


@dataclass(frozen=True)
class BucketSpec:
    """Specification for target bucket proportions."""

    code: str
    label: str
    target: float


_MODEL_CFG = load_config()
_WEIGHTS_CFG = _MODEL_CFG.weights

FACTOR_BUCKET_SPECS: Dict[str, BucketSpec] = {}
for code, spec in _WEIGHTS_CFG.buckets.items():
    FACTOR_BUCKET_SPECS[code] = BucketSpec(code=spec.code, label=spec.label, target=spec.target)

if not FACTOR_BUCKET_SPECS:
    raise ValueError("No DCI factor buckets configured.")

FACTOR_DESCRIPTORS: Dict[str, FactorDescriptor] = {}
for name, spec in _WEIGHTS_CFG.factors.items():
    if spec.bucket not in FACTOR_BUCKET_SPECS:
        raise ValueError(f"Factor {name} references unknown bucket {spec.bucket}.")
    FACTOR_DESCRIPTORS[name] = FactorDescriptor(bucket=spec.bucket, prior=spec.prior)

BASE_FACTOR_WEIGHTS: Dict[str, float] = {
    name: desc.prior for name, desc in FACTOR_DESCRIPTORS.items()
}

FACTOR_BUCKET_FACTORS: Dict[str, List[str]] = {}
for factor, descriptor in FACTOR_DESCRIPTORS.items():
    FACTOR_BUCKET_FACTORS.setdefault(descriptor.bucket, []).append(factor)

for bucket_code, spec in FACTOR_BUCKET_SPECS.items():
    factors = FACTOR_BUCKET_FACTORS.get(bucket_code)
    if not factors:
        raise ValueError(f"Bucket {bucket_code} has no factors configured.")
    bucket_sum = sum(BASE_FACTOR_WEIGHTS[f] for f in factors)
    if not math.isclose(bucket_sum, spec.target, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"Bucket {bucket_code} prior weights sum {bucket_sum:.4f} "
            f"!= target {spec.target:.4f}"
        )


# Backward compatibility alias for legacy imports in other modules.
FACTOR_WEIGHTS: Dict[str, float] = BASE_FACTOR_WEIGHTS


@dataclass
class FactorWeightState:
    weights: Dict[str, float]
    industry_bias: Dict[str, float]
    style_bias: Dict[str, float]
    last_delta: Dict[str, float]


@dataclass(frozen=True)
class WeightDriftReport:
    """Summary emitted after a weight update."""

    weights: Dict[str, float]
    deltas: Dict[str, float]
    bucket_totals: Dict[str, float]
    l1_norm: float
    l2_norm: float
    max_drift: float
    mean_drift: float
    projected: bool
    warnings: Tuple[str, ...]
    industry_bias: Dict[str, float]
    style_bias: Dict[str, float]


FACTOR_WEIGHT_L2_LIMIT = _WEIGHTS_CFG.l2_limit
MAX_INTERCEPT_MAGNITUDE = _WEIGHTS_CFG.intercept_limit

_DEFAULTS_CFG = _MODEL_CFG.defaults
_GATING_CFG = _MODEL_CFG.gates
_AUTO_BASELINE_ENABLED = resolve_toggle("auto_baseline")
_STRUCTURED_LOG_ENABLED = resolve_toggle("structured_log")


def _initial_state() -> FactorWeightState:
    weights = dict(BASE_FACTOR_WEIGHTS)
    last_delta = {name: 0.0 for name in weights}
    return FactorWeightState(weights=weights, industry_bias={}, style_bias={}, last_delta=last_delta)


_ACTIVE_FACTOR_STATE = _initial_state()
_LAST_WEIGHT_REPORT: Optional[WeightDriftReport] = None


def _project_bucket_weights(candidate: Dict[str, float]) -> Dict[str, float]:
    """Project arbitrary weights so each bucket sum matches its target."""

    projected: Dict[str, float] = {}
    for bucket, factors in FACTOR_BUCKET_FACTORS.items():
        spec = FACTOR_BUCKET_SPECS[bucket]
        non_negative = {f: max(0.0, float(candidate.get(f, 0.0))) for f in factors}
        bucket_sum = sum(non_negative.values())
        if bucket_sum <= 0.0:
            baseline = {f: BASE_FACTOR_WEIGHTS[f] for f in factors}
            bucket_sum = sum(baseline.values())
            non_negative = baseline
        scale = spec.target / bucket_sum
        for factor in factors:
            projected[factor] = non_negative[factor] * scale
    return projected


def _compute_bucket_totals(weights: Dict[str, float]) -> Dict[str, float]:
    return {
        bucket: sum(weights[f] for f in factors)
        for bucket, factors in FACTOR_BUCKET_FACTORS.items()
    }


def _clamp_bias_map(
    current: Dict[str, float],
    override: Optional[Dict[str, float]],
) -> Dict[str, float]:
    updated = dict(current)
    if override is None:
        return updated
    for key, value in override.items():
        updated[key] = max(-MAX_INTERCEPT_MAGNITUDE, min(MAX_INTERCEPT_MAGNITUDE, float(value)))
    keys_to_drop = [key for key, value in updated.items() if abs(value) < 1e-9]
    for key in keys_to_drop:
        updated.pop(key)
    return updated


def _build_weight_report(
    new_weights: Dict[str, float],
    delta: Dict[str, float],
    projected: bool,
    industry_bias: Dict[str, float],
    style_bias: Dict[str, float],
) -> WeightDriftReport:
    l1_norm = sum(abs(v) for v in delta.values())
    l2_norm = math.sqrt(sum(v * v for v in delta.values()))
    max_drift = max((abs(v) for v in delta.values()), default=0.0)
    mean_drift = l1_norm / len(delta) if delta else 0.0
    bucket_totals = _compute_bucket_totals(new_weights)
    warnings = []
    for bucket, spec in FACTOR_BUCKET_SPECS.items():
        total = bucket_totals[bucket]
        if not math.isclose(total, spec.target, rel_tol=1e-6, abs_tol=1e-6):
            warnings.append(
                f"bucket {bucket} sum {total:.6f} deviates from target {spec.target:.6f}"
            )
    if l2_norm > FACTOR_WEIGHT_L2_LIMIT + 1e-9:
        warnings.append(
            f"L2 drift {l2_norm:.6f} exceeded limit {FACTOR_WEIGHT_L2_LIMIT:.6f}"
        )
    return WeightDriftReport(
        weights=dict(new_weights),
        deltas=dict(delta),
        bucket_totals=bucket_totals,
        l1_norm=l1_norm,
        l2_norm=l2_norm,
        max_drift=max_drift,
        mean_drift=mean_drift,
        projected=projected,
        warnings=tuple(warnings),
        industry_bias=dict(industry_bias),
        style_bias=dict(style_bias),
    )


def set_factor_weights(
    weights: Dict[str, float],
    *,
    industry_bias: Optional[Dict[str, float]] = None,
    style_bias: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Set the active factor weights enforcing bucket/regularisation constraints."""

    global _ACTIVE_FACTOR_STATE, _LAST_WEIGHT_REPORT

    current = _ACTIVE_FACTOR_STATE
    candidate = dict(current.weights)
    for name in BASE_FACTOR_WEIGHTS:
        if name in weights:
            candidate[name] = float(weights[name])

    projected_weights = _project_bucket_weights(candidate)
    delta = {
        name: projected_weights[name] - current.weights[name]
        for name in BASE_FACTOR_WEIGHTS
    }
    l2_norm = math.sqrt(sum(value * value for value in delta.values()))
    projected = False
    if l2_norm > FACTOR_WEIGHT_L2_LIMIT:
        scale = FACTOR_WEIGHT_L2_LIMIT / l2_norm
        shrunk = {
            name: current.weights[name] + delta[name] * scale
            for name in BASE_FACTOR_WEIGHTS
        }
        projected_weights = _project_bucket_weights(shrunk)
        delta = {
            name: projected_weights[name] - current.weights[name]
            for name in BASE_FACTOR_WEIGHTS
        }
        projected = True

    industry_bias_state = _clamp_bias_map(current.industry_bias, industry_bias)
    style_bias_state = _clamp_bias_map(current.style_bias, style_bias)

    _ACTIVE_FACTOR_STATE = FactorWeightState(
        weights=dict(projected_weights),
        industry_bias=industry_bias_state,
        style_bias=style_bias_state,
        last_delta=dict(delta),
    )

    _LAST_WEIGHT_REPORT = _build_weight_report(
        projected_weights,
        delta,
        projected,
        industry_bias_state,
        style_bias_state,
    )

    return dict(projected_weights)


def get_factor_weights() -> Dict[str, float]:
    """Return a copy of the currently active factor weights."""

    return dict(_ACTIVE_FACTOR_STATE.weights)


def get_factor_biases() -> Dict[str, Dict[str, float]]:
    """Return the currently configured industry/style intercepts."""

    return {
        "industry": dict(_ACTIVE_FACTOR_STATE.industry_bias),
        "style": dict(_ACTIVE_FACTOR_STATE.style_bias),
    }


def get_last_weight_report() -> Optional[WeightDriftReport]:
    """Return the most recent weight drift report, if available."""

    return _LAST_WEIGHT_REPORT


def adjust_factor_weights(
    deltas: Dict[str, float],
    *,
    learning_rate: float = 1.0,
    industry_bias_delta: Optional[Dict[str, float]] = None,
    style_bias_delta: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Apply incremental updates to factor weights and return the result."""

    if not deltas and not industry_bias_delta and not style_bias_delta:
        return get_factor_weights()

    current_weights = _ACTIVE_FACTOR_STATE.weights
    candidate = {
        name: current_weights[name] + learning_rate * float(deltas.get(name, 0.0))
        for name in BASE_FACTOR_WEIGHTS
    }

    industry_bias = dict(_ACTIVE_FACTOR_STATE.industry_bias)
    style_bias = dict(_ACTIVE_FACTOR_STATE.style_bias)
    if industry_bias_delta:
        for key, value in industry_bias_delta.items():
            industry_bias[key] = industry_bias.get(key, 0.0) + learning_rate * float(value)
    if style_bias_delta:
        for key, value in style_bias_delta.items():
            style_bias[key] = style_bias.get(key, 0.0) + learning_rate * float(value)

    return set_factor_weights(
        candidate,
        industry_bias=industry_bias,
        style_bias=style_bias,
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_directional_score(
    factors: Dict[str, FactorInput],
    weights: Dict[str, float] | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Return (S, scaled_signals) for the base directional composite."""

    scaled: Dict[str, float] = {}
    score = 0.0
    active_weights = weights or get_factor_weights()
    for name, weight in active_weights.items():
        if name not in factors:
            raise KeyError(f"Missing factor '{name}' for DCI computation.")
        scaled_value = factors[name].scaled()
        scaled[name] = scaled_value
        score += weight * scaled_value
    return score, scaled


def _apply_scale(p_value: float, scale: float) -> float:
    scale_clamped = max(0.0, min(1.0, scale))
    return 0.5 + (p_value - 0.5) * scale_clamped


def _evaluate_gating(inputs: DCIInputs) -> DCIGatingDecision:
    reasons: List[str] = []

    if inputs.expected_move_pct < _GATING_CFG.min_em_pct:
        reasons.append(
            f"预期波动 {inputs.expected_move_pct:.2f}% 低于阈值 {_GATING_CFG.min_em_pct:.2f}%"
        )

    weekly_cfg = _GATING_CFG.weekly
    if weekly_cfg.require_weekly:
        has_weekly = bool(inputs.has_weekly_option)
        expiry_type = (inputs.option_expiry_type or "").strip().upper()
        if not has_weekly and expiry_type != "W":
            reasons.append("缺少周度到期合约")

        max_days = weekly_cfg.weekly_max_days
        expiry_days = inputs.option_expiry_days
        if expiry_days is None:
            expiry_days = _DEFAULTS_CFG.option_expiry_days
        if max_days is not None and (expiry_days is None or expiry_days > max_days):
            if expiry_days is None:
                reasons.append(f"周度合约剩余天数未知，需≤{max_days}天")
            else:
                reasons.append(
                    f"周度合约剩余 {expiry_days} 天，超过限制 {max_days} 天"
                )

    shock_cfg = _GATING_CFG.shock_ci
    if shock_cfg and inputs.shock_flag:
        crowding = inputs.crowding_index
        if shock_cfg.min_ci is not None and crowding < shock_cfg.min_ci:
            reasons.append(
                f"冲击期 crowding {crowding:.2f} 低于 {shock_cfg.min_ci:.2f}"
            )
        if shock_cfg.max_ci is not None and crowding > shock_cfg.max_ci:
            reasons.append(
                f"冲击期 crowding {crowding:.2f} 高于 {shock_cfg.max_ci:.2f}"
            )

    return DCIGatingDecision(passed=not reasons, reasons=tuple(reasons))


def compute_dci(inputs: DCIInputs) -> DCIResult:
    """Compute the Directional Certainty Index as specified."""

    weights = inputs.factor_weights or get_factor_weights()
    S, scaled = compute_directional_score(inputs.factors, weights)

    gating_decision = _evaluate_gating(inputs)

    beta = get_beta_for_symbol(inputs.symbol)
    p_raw = _sigmoid(beta.beta0 + beta.beta1 * S)

    eg = 0.7 * inputs.z_cons + 0.3 * inputs.z_narr
    shrink_eg = 1.0 / (1.0 + math.exp(_MODEL_CFG.gamma * eg))
    p_after_eg = _apply_scale(p_raw, shrink_eg)

    ci_scale = evaluate_ci_scale(inputs.crowding_index, inputs.quality_score)
    p_after_ci = _apply_scale(p_after_eg, ci_scale)

    q_scale = evaluate_quality_scale(inputs.quality_score)
    p_after_quality = _apply_scale(p_after_ci, q_scale)

    disagree_scale = max(0.0, min(1.0, 1.0 - _MODEL_CFG.kappa_d * inputs.disagreement))
    p_after_disagreement = _apply_scale(p_after_quality, disagree_scale)

    if inputs.shock_flag:
        shock_scale = _MODEL_CFG.eta_macro
        p_final = _apply_scale(p_after_disagreement, shock_scale)
    else:
        shock_scale = 1.0
        p_final = p_after_disagreement

    p_up = max(0.0, min(1.0, p_final))
    dci_base = max(p_up, 1.0 - p_up) * 100.0

    penalty = evaluate_penalty(inputs.expected_move_pct)
    penalty = min(0.0, penalty)
    dci_pen = max(0.0, dci_base + penalty)

    dci_final_raw = dci_pen * (0.85 + 0.15 * inputs.stability)

    direction = 1 if p_up > 0.5 else -1

    weight = max(0.0, min(1.0, (dci_final_raw - 60.0) / 40.0))
    if inputs.shock_flag and inputs.crowding_index >= 70.0:
        weight *= 0.5

    if dci_final_raw >= 75:
        bucket = "加码"
    elif dci_final_raw >= 65:
        bucket = "小仓尝试"
    else:
        bucket = "放弃"

    if not gating_decision.passed:
        bucket = _GATING_CFG.skip_bucket
        dci_final = 0.0
        certainty = 0.0
        position_weight = 0.0
    else:
        dci_final = dci_final_raw
        certainty = dci_final
        position_weight = weight

    shrink_map = {
        "shrink_EG": shrink_eg,
        "scale_CI": ci_scale,
        "scale_Q": q_scale,
        "disagreement": disagree_scale,
        "shock": shock_scale,
    }

    if _STRUCTURED_LOG_ENABLED:
        log_payload = {
            "symbol": inputs.symbol,
            "base_score": S,
            "p_raw": p_raw,
            "p_final": p_up,
            "dci_final": dci_final,
            "gating_passed": gating_decision.passed,
            "gating_reasons": list(gating_decision.reasons),
        }
        try:
            logger.info(json.dumps(log_payload, ensure_ascii=False, sort_keys=True))
        except Exception:  # pragma: no cover - logging must not break computation
            logger.info("%s", log_payload)

    return DCIResult(
        symbol=inputs.symbol,
        direction=direction,
        p_up=p_up,
        dci_base=dci_base,
        dci_penalised=dci_pen,
        dci_final=dci_final,
        position_weight=position_weight,
        certainty=certainty,
        position_bucket=bucket,
        shrink_factors=shrink_map,
        base_score=S,
        scaled_factors=scaled,
        factor_weights=dict(weights),
        gating_passed=gating_decision.passed,
        gating_reasons=gating_decision.reasons,
    )





def _coerce_factor(value: float | Dict[str, float]) -> FactorInput:
    if isinstance(value, dict):
        if "z" in value:
            return FactorInput(z=float(value["z"]))
        if {"value", "median", "mad"} <= value.keys():
            return FactorInput(
                value=float(value["value"]),
                median=float(value["median"]),
                mad=float(value["mad"]),
            )
        raise ValueError("Unsupported factor schema.")
    return FactorInput(z=float(value))


def parse_factor_inputs(
    raw: Dict[str, float | Dict[str, float]] | None,
    *,
    allow_defaults: bool,
) -> Dict[str, FactorInput]:
    """Parse raw factor inputs using configuration defaults for missing entries."""

    factors: Dict[str, FactorInput] = {}
    raw_map = raw or {}
    missing_default = _DEFAULTS_CFG.missing_factor_z
    for name in BASE_FACTOR_WEIGHTS:
        if name in raw_map:
            factors[name] = _coerce_factor(raw_map[name])
        elif allow_defaults:
            factors[name] = FactorInput(z=missing_default)
        else:
            raise KeyError(f"Missing factor '{name}' and auto baseline disabled")
    return factors


def build_inputs(symbol: str, payload: Dict[str, float | Dict[str, float]]) -> DCIInputs:
    """Construct DCIInputs from a JSON-style payload."""

    factors_raw = payload.get("factors")
    if not isinstance(factors_raw, dict):
        if _AUTO_BASELINE_ENABLED:
            factors_raw = {}
        else:
            raise ValueError("Payload must include 'factors' map.")

    factors = parse_factor_inputs(factors_raw, allow_defaults=_AUTO_BASELINE_ENABLED)

    def _resolve_numeric(
        name: str,
        *,
        aliases: Iterable[str] = (),
        cast: type = float,
        required: bool = True,
    ) -> float:
        keys = [name, *aliases]
        for key in keys:
            if key in payload and payload[key] is not None:
                value = payload[key]
                if cast is int:
                    return int(float(value))
                return float(value)
        default = _DEFAULTS_CFG.get_field(name, 0.0)
        if required and not _AUTO_BASELINE_ENABLED:
            raise ValueError(f"Missing field '{name}' and auto baseline disabled")
        if cast is int:
            return int(default)
        return float(default)

    z_cons = _resolve_numeric("z_cons")
    z_narr = _resolve_numeric("z_narr")
    crowding = _resolve_numeric("CI")
    quality = _resolve_numeric("Q")
    disagreement = _resolve_numeric("D")
    expected_move = _resolve_numeric("EM_pct", aliases=("EM",))
    stability = _resolve_numeric("S_stab", required=False)
    shock_flag = _resolve_numeric("shock_flag", cast=int, required=False)

    def _resolve_bool(keys: Iterable[str], default: bool) -> bool:
        for key in keys:
            if key in payload:
                value = payload[key]
                if isinstance(value, str):
                    return value.strip().lower() not in {"", "0", "false", "no", "off"}
                return bool(value)
        return default

    has_weekly = _resolve_bool(["has_weekly", "has_weekly_option"], _DEFAULTS_CFG.has_weekly_option)

    def _resolve_option_days() -> int | None:
        for key in ("option_expiry_days", "expiry_days"):
            if key in payload and payload[key] is not None:
                try:
                    return int(float(payload[key]))
                except (TypeError, ValueError):
                    continue
        return _DEFAULTS_CFG.option_expiry_days

    option_days = _resolve_option_days()

    def _resolve_option_type() -> str | None:
        for key in ("option_expiry_type", "expiry_type"):
            if key in payload and payload[key]:
                return str(payload[key]).strip()
        return _DEFAULTS_CFG.option_expiry_type

    option_type = _resolve_option_type()

    return DCIInputs(
        symbol=symbol,
        factors=factors,
        z_cons=z_cons,
        z_narr=z_narr,
        crowding_index=crowding,
        quality_score=quality,
        disagreement=disagreement,
        expected_move_pct=expected_move,
        stability=stability,
        shock_flag=int(shock_flag),
        has_weekly_option=has_weekly,
        option_expiry_days=option_days,
        option_expiry_type=option_type,
    )

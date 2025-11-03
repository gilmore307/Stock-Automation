"""Directional Certainty Index (DCI) computation utilities."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Tuple

from .config import (
    evaluate_ci_scale,
    evaluate_penalty,
    evaluate_quality_scale,
    get_beta_for_symbol,
    load_config,
)


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


FACTOR_BUCKET_SPECS: Dict[str, BucketSpec] = {
    "A": BucketSpec(code="A", label="预期/基本面动能", target=0.45),
    "B": BucketSpec(code="B", label="经营质量/利润动能", target=0.25),
    "C": BucketSpec(code="C", label="价格/成交动能", target=0.20),
    "D": BucketSpec(code="D", label="相对估值", target=0.10),
}


FACTOR_DESCRIPTORS: Dict[str, FactorDescriptor] = {
    "EPS_Rev_30d": FactorDescriptor(bucket="A", prior=0.20),
    "Sales_Rev_30d": FactorDescriptor(bucket="A", prior=0.10),
    "Guide_Drift": FactorDescriptor(bucket="A", prior=0.10),
    "Backlog_Bookings_Delta": FactorDescriptor(bucket="A", prior=0.05),
    "GM_YoY_Delta": FactorDescriptor(bucket="B", prior=0.10),
    "OPM_YoY_Delta": FactorDescriptor(bucket="B", prior=0.10),
    "FCF_Margin_Slope": FactorDescriptor(bucket="B", prior=0.05),
    "Ret20_rel": FactorDescriptor(bucket="C", prior=0.10),
    "Ret60_rel": FactorDescriptor(bucket="C", prior=0.05),
    "UDVol20": FactorDescriptor(bucket="C", prior=0.05),
    "Value_vs_Sector": FactorDescriptor(bucket="D", prior=0.07),
    "EarningsYield_vs_Sector": FactorDescriptor(bucket="D", prior=0.03),
}


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


FACTOR_WEIGHT_L2_LIMIT = 0.05
MAX_INTERCEPT_MAGNITUDE = 0.25


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


def compute_dci(inputs: DCIInputs) -> DCIResult:
    """Compute the Directional Certainty Index as specified."""

    cfg = load_config()
    weights = inputs.factor_weights or get_factor_weights()
    S, scaled = compute_directional_score(inputs.factors, weights)

    beta = get_beta_for_symbol(inputs.symbol)
    p_raw = _sigmoid(beta.beta0 + beta.beta1 * S)

    eg = 0.7 * inputs.z_cons + 0.3 * inputs.z_narr
    shrink_eg = 1.0 / (1.0 + math.exp(cfg.gamma * eg))
    p_after_eg = _apply_scale(p_raw, shrink_eg)

    ci_scale = evaluate_ci_scale(inputs.crowding_index, inputs.quality_score)
    p_after_ci = _apply_scale(p_after_eg, ci_scale)

    q_scale = evaluate_quality_scale(inputs.quality_score)
    p_after_quality = _apply_scale(p_after_ci, q_scale)

    disagree_scale = max(0.0, min(1.0, 1.0 - cfg.kappa_d * inputs.disagreement))
    p_after_disagreement = _apply_scale(p_after_quality, disagree_scale)

    if inputs.shock_flag:
        shock_scale = cfg.eta_macro
        p_final = _apply_scale(p_after_disagreement, shock_scale)
    else:
        shock_scale = 1.0
        p_final = p_after_disagreement

    p_up = max(0.0, min(1.0, p_final))
    dci_base = max(p_up, 1.0 - p_up) * 100.0

    penalty = evaluate_penalty(inputs.expected_move_pct)
    penalty = min(0.0, penalty)
    dci_pen = max(0.0, dci_base + penalty)

    dci_final = dci_pen * (0.85 + 0.15 * inputs.stability)

    direction = 1 if p_up > 0.5 else -1

    weight = max(0.0, min(1.0, (dci_final - 60.0) / 40.0))
    if inputs.shock_flag and inputs.crowding_index >= 70.0:
        weight *= 0.5

    if dci_final >= 75:
        bucket = "加码"
    elif dci_final >= 65:
        bucket = "小仓尝试"
    else:
        bucket = "放弃"

    certainty = dci_final

    shrink_map = {
        "shrink_EG": shrink_eg,
        "scale_CI": ci_scale,
        "scale_Q": q_scale,
        "disagreement": disagree_scale,
        "shock": shock_scale,
    }

    return DCIResult(
        symbol=inputs.symbol,
        direction=direction,
        p_up=p_up,
        dci_base=dci_base,
        dci_penalised=dci_pen,
        dci_final=dci_final,
        position_weight=weight,
        certainty=certainty,
        position_bucket=bucket,
        shrink_factors=shrink_map,
        base_score=S,
        scaled_factors=scaled,
        factor_weights=dict(weights),
    )


def parse_factor_inputs(raw: Dict[str, float | Dict[str, float]]) -> Dict[str, FactorInput]:
    """Parse raw JSON-friendly factor inputs into FactorInput objects."""

    out: Dict[str, FactorInput] = {}
    for name, value in raw.items():
        if isinstance(value, dict):
            if "z" in value:
                out[name] = FactorInput(z=float(value["z"]))
            elif {"value", "median", "mad"} <= value.keys():
                out[name] = FactorInput(
                    value=float(value["value"]),
                    median=float(value["median"]),
                    mad=float(value["mad"]),
                )
            else:
                raise ValueError(f"Unsupported factor schema for '{name}'.")
        else:
            out[name] = FactorInput(z=float(value))
    return out


def build_inputs(symbol: str, payload: Dict[str, float | Dict[str, float]]) -> DCIInputs:
    """Construct DCIInputs from a JSON-style payload."""

    factors_raw = payload.get("factors")
    if not isinstance(factors_raw, dict):
        raise ValueError("Payload must include 'factors' map.")
    factors = parse_factor_inputs(factors_raw)

    return DCIInputs(
        symbol=symbol,
        factors=factors,
        z_cons=float(payload.get("z_cons", 0.0)),
        z_narr=float(payload.get("z_narr", 0.0)),
        crowding_index=float(payload.get("CI", 0.0)),
        quality_score=float(payload.get("Q", 0.0)),
        disagreement=float(payload.get("D", 0.0)),
        expected_move_pct=float(payload.get("EM_pct", payload.get("EM", 0.0))),
        stability=float(payload.get("S_stab", 0.0)),
        shock_flag=int(payload.get("shock_flag", 0)),
    )

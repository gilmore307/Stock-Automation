"""Directional Certainty Index (DCI) computation utilities."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Tuple


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


BASE_FACTOR_WEIGHTS: Dict[str, float] = {
    # A. 预期/基本面动能 0.45
    "EPS_Rev_30d": 0.20,
    "Sales_Rev_30d": 0.10,
    "Guide_Drift": 0.10,
    "Backlog_Bookings_Delta": 0.05,
    # B. 经营质量/利润动能 0.25
    "GM_YoY_Delta": 0.10,
    "OPM_YoY_Delta": 0.10,
    "FCF_Margin_Slope": 0.05,
    # C. 价格/成交动能 0.20
    "Ret20_rel": 0.10,
    "Ret60_rel": 0.05,
    "UDVol20": 0.05,
    # D. 相对估值 0.10
    "Value_vs_Sector": 0.07,
    "EarningsYield_vs_Sector": 0.03,
}

# Backward compatibility alias for legacy imports in other modules.
FACTOR_WEIGHTS: Dict[str, float] = BASE_FACTOR_WEIGHTS

_ACTIVE_FACTOR_WEIGHTS: Dict[str, float] = dict(BASE_FACTOR_WEIGHTS)


def _normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Project weights onto the simplex while keeping unknown keys untouched."""

    filtered: Dict[str, float] = {}
    for name in BASE_FACTOR_WEIGHTS:
        value = float(weights.get(name, 0.0))
        filtered[name] = max(0.0, value)

    total = sum(filtered.values())
    if total <= 0.0:
        return dict(BASE_FACTOR_WEIGHTS)

    return {name: filtered[name] / total for name in BASE_FACTOR_WEIGHTS}


def set_factor_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Set the active factor weights and return the normalised copy."""

    global _ACTIVE_FACTOR_WEIGHTS
    normalised = _normalise_weights(weights)
    _ACTIVE_FACTOR_WEIGHTS = dict(normalised)
    return dict(_ACTIVE_FACTOR_WEIGHTS)


def get_factor_weights() -> Dict[str, float]:
    """Return a copy of the currently active factor weights."""

    return dict(_ACTIVE_FACTOR_WEIGHTS)


def adjust_factor_weights(
    deltas: Dict[str, float],
    *,
    learning_rate: float = 1.0,
) -> Dict[str, float]:
    """Apply incremental updates to factor weights and return the result.

    Parameters
    ----------
    deltas:
        Partial map of factor -> delta value to apply.
    learning_rate:
        Scaling multiplier for the delta contribution.
    """

    if not deltas:
        return get_factor_weights()

    current = get_factor_weights()
    updated: Dict[str, float] = dict(current)
    changed = False
    for name, delta in deltas.items():
        if name not in updated:
            continue
        updated[name] = max(0.0, updated[name] + learning_rate * float(delta))
        changed = True

    if not changed:
        return current

    return set_factor_weights(updated)


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


def compute_dci(inputs: DCIInputs) -> DCIResult:
    """Compute the Directional Certainty Index as specified."""

    weights = inputs.factor_weights or get_factor_weights()
    S, scaled = compute_directional_score(inputs.factors, weights)
    p_raw = _sigmoid(S)

    eg = 0.7 * inputs.z_cons + 0.3 * inputs.z_narr
    shrink_eg = 1.0 / (1.0 + math.exp(0.35 * eg))
    p1 = 0.5 + (p_raw - 0.5) * shrink_eg

    ci_eff = min(inputs.crowding_index, 55.0) if inputs.quality_score >= 60.0 else inputs.crowding_index
    shrink_ci = 1.0 - 0.6 * _sigmoid((ci_eff - 50.0) / 12.0)
    p2 = 0.5 + (p1 - 0.5) * shrink_ci

    p3 = 0.5 + (p2 - 0.5) * (1.0 - 0.4 * inputs.disagreement)

    if inputs.shock_flag:
        p4 = 0.5 + (p3 - 0.5) * 0.8
    else:
        p4 = p3

    p_up = max(0.0, min(1.0, p4))
    dci_base = max(p_up, 1.0 - p_up) * 100.0

    penalty = 8.0 * max(0.0, (5.0 - inputs.expected_move_pct) / 5.0)
    dci_pen = max(0.0, dci_base - penalty)

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
        "shrink_CI": shrink_ci,
        "disagreement": 1.0 - 0.4 * inputs.disagreement,
        "shock": 0.8 if inputs.shock_flag else 1.0,
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

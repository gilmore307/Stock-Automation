from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_CORE_PATH = PROJECT_ROOT / "app_core"


def _import_module(name: str, location: Path):
    spec = importlib.util.spec_from_file_location(name, location)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if "app_core" not in sys.modules:
    pkg = types.ModuleType("app_core")
    pkg.__path__ = [str(APP_CORE_PATH)]
    sys.modules["app_core"] = pkg

dci_module = _import_module("app_core.dci", APP_CORE_PATH / "dci" / "__init__.py")
config_module = _import_module("app_core.dci.config", APP_CORE_PATH / "dci" / "config.py")

BASE_FACTOR_WEIGHTS = dci_module.BASE_FACTOR_WEIGHTS
DCIInputs = dci_module.DCIInputs
FactorInput = dci_module.FactorInput
FACTOR_BUCKET_FACTORS = dci_module.FACTOR_BUCKET_FACTORS
FACTOR_BUCKET_SPECS = dci_module.FACTOR_BUCKET_SPECS
FACTOR_WEIGHT_L2_LIMIT = dci_module.FACTOR_WEIGHT_L2_LIMIT
MAX_INTERCEPT_MAGNITUDE = dci_module.MAX_INTERCEPT_MAGNITUDE
compute_dci = dci_module.compute_dci
evaluate_ci_scale = config_module.evaluate_ci_scale
evaluate_quality_scale = config_module.evaluate_quality_scale
evaluate_penalty = config_module.evaluate_penalty
get_factor_weights = dci_module.get_factor_weights
set_factor_weights = dci_module.set_factor_weights
adjust_factor_weights = dci_module.adjust_factor_weights
get_last_weight_report = dci_module.get_last_weight_report
get_factor_biases = dci_module.get_factor_biases


def _build_base_inputs(**overrides: float) -> DCIInputs:
    factors = {
        name: FactorInput(z=0.5) for name in BASE_FACTOR_WEIGHTS.keys()
    }
    payload = dict(
        symbol="TEST",
        factors=factors,
        z_cons=0.0,
        z_narr=0.0,
        crowding_index=50.0,
        quality_score=50.0,
        disagreement=0.1,
        expected_move_pct=5.0,
        stability=0.6,
        shock_flag=0,
    )
    payload.update(overrides)
    return DCIInputs(**payload)


def test_eg_shrink_monotonic():
    base = _build_base_inputs()
    bearish = compute_dci(base)

    bullish_inputs = _build_base_inputs(z_cons=3.0, z_narr=3.0)
    bullish = compute_dci(bullish_inputs)

    dovish_inputs = _build_base_inputs(z_cons=-3.0, z_narr=-3.0)
    dovish = compute_dci(dovish_inputs)

    assert bullish.dci_final < bearish.dci_final
    assert dovish.dci_final > bearish.dci_final


def test_ci_scale_monotonic_increasing():
    low = evaluate_ci_scale(10.0, 40.0)
    high = evaluate_ci_scale(90.0, 40.0)
    assert high >= low

    low_ci_inputs = _build_base_inputs(crowding_index=10.0)
    high_ci_inputs = _build_base_inputs(crowding_index=90.0)
    low_ci = compute_dci(low_ci_inputs)
    high_ci = compute_dci(high_ci_inputs)
    assert abs(high_ci.p_up - 0.5) >= abs(low_ci.p_up - 0.5)


def test_quality_scale_monotonic_decreasing():
    low_quality = evaluate_quality_scale(20.0)
    high_quality = evaluate_quality_scale(80.0)
    assert high_quality <= low_quality

    low_inputs = _build_base_inputs(quality_score=20.0)
    high_inputs = _build_base_inputs(quality_score=80.0)
    assert compute_dci(high_inputs).dci_final <= compute_dci(low_inputs).dci_final


def test_disagreement_shrink_factor():
    base = compute_dci(_build_base_inputs(disagreement=0.0))
    high = compute_dci(_build_base_inputs(disagreement=0.8))
    assert high.dci_final <= base.dci_final


def test_penalty_curve_concave_non_positive():
    low_move = evaluate_penalty(1.0)
    mid_move = evaluate_penalty(5.0)
    high_move = evaluate_penalty(12.0)

    assert low_move <= 0.0
    assert mid_move <= 0.0
    assert high_move <= 0.0

    # Concavity check: incremental slope should decrease
    slope_low_mid = (mid_move - low_move) / (5.0 - 1.0)
    slope_mid_high = (high_move - mid_move) / (12.0 - 5.0)
    assert slope_mid_high <= slope_low_mid + 1e-9

    penalised = compute_dci(_build_base_inputs(expected_move_pct=1.0))
    neutral = compute_dci(_build_base_inputs(expected_move_pct=8.0))
    assert penalised.dci_final <= neutral.dci_final


def test_shock_shrink_uses_macro_eta():
    normal = compute_dci(_build_base_inputs(shock_flag=0))
    shocked = compute_dci(_build_base_inputs(shock_flag=1))
    assert shocked.dci_final <= normal.dci_final


def _reset_factor_state() -> None:
    biases = get_factor_biases()
    zero_industry = {key: 0.0 for key in biases["industry"]}
    zero_style = {key: 0.0 for key in biases["style"]}
    set_factor_weights(BASE_FACTOR_WEIGHTS, industry_bias=zero_industry, style_bias=zero_style)


def test_factor_bucket_totals_locked() -> None:
    _reset_factor_state()
    raw_update = {"EPS_Rev_30d": 0.35, "Sales_Rev_30d": 0.05}
    set_factor_weights(raw_update)
    weights = get_factor_weights()
    assert all(value >= 0.0 for value in weights.values())

    for bucket, spec in FACTOR_BUCKET_SPECS.items():
        total = sum(weights[name] for name in FACTOR_BUCKET_FACTORS[bucket])
        assert math.isclose(total, spec.target, rel_tol=1e-6, abs_tol=1e-6)

    report = get_last_weight_report()
    assert report is not None
    assert report.max_drift >= 0.0
    assert report.mean_drift >= 0.0


def test_weight_adjustment_obeys_norm_limit() -> None:
    _reset_factor_state()
    base = get_factor_weights()
    exaggerated = {name: value + 0.2 for name, value in base.items()}
    set_factor_weights(exaggerated)
    report = get_last_weight_report()
    assert report is not None
    assert report.l2_norm <= FACTOR_WEIGHT_L2_LIMIT + 1e-9


def test_biases_are_clipped_and_persisted() -> None:
    _reset_factor_state()
    current = get_factor_weights()
    set_factor_weights(
        current,
        industry_bias={"Tech": 1.0, "Energy": -0.4},
        style_bias={"Growth": 0.7},
    )
    biases = get_factor_biases()
    assert math.isclose(biases["industry"]["Tech"], MAX_INTERCEPT_MAGNITUDE)
    assert math.isclose(biases["industry"]["Energy"], -MAX_INTERCEPT_MAGNITUDE)
    assert math.isclose(biases["style"]["Growth"], MAX_INTERCEPT_MAGNITUDE)


def test_adjust_factor_weights_projects_to_bucket_targets() -> None:
    _reset_factor_state()
    adjust_factor_weights({"Ret20_rel": 0.02, "Ret60_rel": -0.03})
    weights = get_factor_weights()
    total_c = sum(weights[name] for name in FACTOR_BUCKET_FACTORS["C"])
    assert math.isclose(total_c, FACTOR_BUCKET_SPECS["C"].target, rel_tol=1e-6, abs_tol=1e-6)
    assert all(value >= 0.0 for value in weights.values())


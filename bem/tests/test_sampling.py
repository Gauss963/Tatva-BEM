from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.case import load_case
from finiteblock_bem.cq import BoundaryHistory, OnlineConvolution, convolution_weights, plan_cq
from finiteblock_bem.dynamics import LoadingSchedule, initialize_restart, simulate
from finiteblock_bem.sampling import SamplingOptions, parsimonious_weights


def test_adaptive_cq_matches_independent_rational_weights_and_reduces_calls():
    plan = plan_cq(2, 0.01, 2048)
    matrix = np.array([[1.0, 2.0], [-3.0, 4.0]])
    weights, info = parsimonious_weights(lambda s: matrix / (s + 2), plan, workers=2)
    dense = convolution_weights(lambda s: matrix / (s + 2), plan)
    np.testing.assert_allclose(weights, dense, rtol=1e-6, atol=2e-10)
    assert info["transfer_evaluations"] < info["dense_grid_evaluations"]
    assert info["interpolated_grid_points"] > 0
    assert info["direct_grid_points"] + info["interpolated_grid_points"] == plan.fft_size // 2 + 1


def test_narrow_resonances_force_exact_grid_fallback_and_preserve_history():
    plan = plan_cq(2, 0.001, 1024)

    def wave(s):
        # Poles on the imaginary axis: a sectorial-only acceleration is invalid.
        return np.eye(2) * s / (-np.expm1(-0.07 * s))

    approximate, info = parsimonious_weights(wave, plan)
    dense = convolution_weights(wave, plan)
    np.testing.assert_allclose(approximate, dense, rtol=2e-5, atol=2e-6)
    assert info["rejected_intervals"] > 0
    history = BoundaryHistory(plan, approximate)
    online = OnlineConvolution(history)
    inputs = np.random.default_rng(14).normal(size=(plan.steps + 1, 2))
    for index in range(plan.steps + 1):
        np.testing.assert_allclose(
            online.past(index), history.past_traction(inputs, index), atol=1e-8
        )
        online.append(inputs[index])
    retained = sum(x.nbytes for x in online.transforms.values())
    assert (
        plan.estimated_peak_bytes
        >= approximate.nbytes + retained + online.inputs.nbytes + online.pending.nbytes
    )


def test_sampling_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        SamplingOptions(relative_tolerance=-1)
    with pytest.raises(ValueError):
        SamplingOptions(degree=3)
    plan = plan_cq(2, 0.01, 32)
    with pytest.raises(FloatingPointError):
        parsimonious_weights(lambda s: np.full((2, 2), np.nan), plan)
    with pytest.raises(ValueError):
        parsimonious_weights(lambda s: np.eye(2), plan, row_scale=[-1, 1])


def test_resource_guard_includes_scipy_padding_while_fft_cache_is_live():
    plan = plan_cq(50, 0.01, 70)
    weights = np.random.default_rng(3).normal(size=(71, 50, 50))
    online = OnlineConvolution(BoundaryHistory(plan, weights))
    for _ in range(64):
        online.append(np.zeros(50))
    # The length-128 transform of 71 real matrices first pads a real copy.
    padded = np.empty((128, 50, 50))
    live = weights.nbytes + padded.nbytes + sum(a.nbytes for a in online.transforms.values())
    live += online.pending.nbytes + online.inputs.nbytes
    assert plan.estimated_peak_bytes >= live


def test_adaptive_sampling_preserves_a_nonlinear_two_block_sliding_solution(tmp_path):
    path = (
        Path(__file__).resolve().parents[1]
        / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    )
    case = load_case(path, max_panel_length=80)
    lift = initialize_restart(case)
    schedule = LoadingSchedule(
        lambda _: case.normal_pressure,
        lambda t: case.restart_actuator + 4000 * t,
        "frequency_sampling_validation",
    )
    results = [
        simulate(
            case,
            8e-6,
            128,
            lift=lift,
            schedule=schedule,
            formulation="interface",
            cache_directory=tmp_path,
            sampling=sampling,
        )
        for sampling in [None, SamplingOptions(degree=8)]
    ]
    assert all(r.diagnostics["status"] == "completed_requested_horizon" for r in results)
    assert results[0].cumulative_slip[-1].max() > 0.1
    for name in ["pressure", "shear", "cumulative_slip", "platen"]:
        np.testing.assert_allclose(
            getattr(results[0], name), getattr(results[1], name), rtol=2e-6, atol=2e-7
        )
    cached = simulate(
        case,
        8e-6,
        128,
        lift=lift,
        schedule=schedule,
        formulation="interface",
        cache_directory=tmp_path,
        sampling=SamplingOptions(degree=8),
    )
    assert cached.diagnostics["frequency_sampling"]["cache_hit"]
    np.testing.assert_array_equal(cached.cumulative_slip, results[1].cumulative_slip)

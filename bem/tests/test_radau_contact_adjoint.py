"""Complete RK VJP/JVP duality and independent original-Z finite differences."""

import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
from finiteblock_bem.cq import BoundaryHistory, invert_spectrum, plan_cq
from finiteblock_bem.friction import ContactState, SlipWeakeningContact

TOOLS = Path(__file__).resolve().parents[1] / "tools"
AD = runpy.run_path(str(TOOLS / "radau_contact_adjoint.py"))
SIM = runpy.run_path(str(TOOLS / "simulate_compact_radau.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))
STORAGE = runpy.run_path(str(TOOLS / "compact_radau_storage.py"))


@pytest.fixture(scope="module")
def problem():
    plan = plan_cq(3, 0.02, 64, max_memory_mb=64)
    mass = np.array([[0.3, 0.02, 0], [0.02, 0.2, 0.01], [0, 0.01, 0.4]])
    stiffness = np.array([[30.0, 0.2, 0.1], [0.2, 30.0, -20], [0.1, -20, 30]])
    law = SlipWeakeningContact(100, 40, 0.8, 0.5, 0.2)
    reference = SimpleNamespace(
        base_dn=np.array([0.1]),
        base_ds=np.zeros(1),
        base_platen=0.0,
        spring=2.0,
        case=SimpleNamespace(contact=law, normal_pressure=10.0),
        constraints=SimpleNamespace(
            size=3,
            normal_jump=np.array([[1.0, 0, 0]]),
            tangential_jump=np.array([[0.0, 1, 0]]),
            contact_weights=np.ones(1),
            reduced_weights=np.ones(3),
        ),
    )

    def original(s):
        return s * mass + stiffness

    def compliance(s):
        return np.linalg.inv(original(s) + np.diag([100.0, 40.0, 2.0]))

    nodes = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    pairs = [RADAU["compact_matrix_function"](compliance, z, plan.dt) for z in nodes]
    f0, f1 = np.array([v[0] for v in pairs]), np.array([v[1] for v in pairs])
    w0, w1, wrap = RADAU["invert_compact_spectrum"](f0, f1, plan)
    first, second = BoundaryHistory(plan, w0), BoundaryHistory(plan, w1)
    stage_time = np.zeros((plan.steps + 1, 2))
    stage_time[1:] = (np.arange(plan.steps)[:, None] + RADAU["C"]) * plan.dt
    ramp = 20 * stage_time
    elastic = STORAGE["causal_compact_startup"](
        f0[:, :, -1], f1[:, :, -1], wrap[:, -1], plan, 2 * ramp
    )
    impedance = RADAU["matrix_function"](original, RADAU["radau_symbol"](0) / plan.dt).real
    _, stages = SIM["simulate_stages"](
        reference,
        ContactState(np.zeros(1), np.zeros(1)),
        first,
        second,
        wrap,
        impedance,
        elastic,
        stage_time,
        ramp,
        diagnostic=True,
        segment_substeps=(2, 4),
        initial_residual_force=np.array([0.1, -0.05, 0.02]),
    )
    spectrum = np.array(
        [RADAU["matrix_function"](original, RADAU["radau_symbol"](z) / plan.dt) for z in nodes]
    )
    raw_plan = replace(plan, dofs=6)
    raw = BoundaryHistory(raw_plan, invert_spectrum(spectrum, raw_plan))
    system = AD["CONTACT"]["RadauContactEquilibrium"](
        reference, raw.instantaneous + np.diag([0, 0, 2, 0, 0, 2]), segment_substeps=(2, 4)
    )
    return reference, first, second, wrap, impedance, stages, raw, system


def original_objective(problem, parameters, actuator):
    reference, _, _, _, _, stages, raw, system = problem
    import jax.numpy as jnp

    system.parameters = jnp.asarray(parameters)
    q = np.zeros((len(actuator), 6))
    old = ContactState(np.zeros(1), np.zeros(1))
    for k in range(1, len(actuator)):
        forcing = raw.past_traction(q, k) - np.tile(stages["initial_contact_force"], 2)
        forcing += np.tile(stages["initial_residual_force"], 2)
        forcing[[2, 5]] -= reference.spring * actuator[k]
        answer = system.solve(old, forcing, np.tile(q[k - 1, -3:], 2), q[k - 1, -3:])
        assert answer.converged
        q[k] = answer.displacement
        old = ContactState(answer.response[:, 1, 2], answer.response[:, 1, 3])
    return float(old.cumulative_slip[0])


@pytest.mark.parametrize("cache_mb", [0, 0.02])
def test_all_parameter_and_stage_loading_cotangents_match_batched_jvp(problem, cache_mb):
    ref, first, second, wrap, impedance, stages, _, _ = problem
    dp = np.c_[np.eye(5), np.zeros(5)]
    da = np.zeros((len(stages["actuator"]), 2, 6))
    da[1:, :, 5] = np.random.default_rng(214).normal(size=da[1:, :, 5].shape)
    reverse = AD["reverse_gradient"](
        ref,
        first,
        second,
        wrap,
        impedance,
        stages,
        segment_substeps=(2, 4),
        cache_mb=cache_mb,
        fft_rows=1,
    )
    forward = AD["forward_directions"](
        ref,
        first,
        second,
        wrap,
        impedance,
        stages,
        dp,
        da,
        segment_substeps=(2, 4),
        cache_mb=cache_mb,
        fft_rows=1,
    )
    dual = reverse["gradient"] @ dp + np.einsum("ts,tsp->p", reverse["actuator_gradient"], da)
    np.testing.assert_allclose(
        dual, forward["objective_directional_derivative"], rtol=2e-10, atol=2e-11
    )
    np.testing.assert_array_equal(reverse["actuator_gradient"][0], 0)
    assert forward["history"]["shared_family_ffts_across_stages_and_directions"]
    assert reverse["every_history_source_included"]
    assert not reverse["full_initialized_gradient"]
    assert stages["response"][-1, 0, 1, 3] > 0.2


def test_vjp_matches_independent_original_impedance_finite_difference(problem):
    ref, first, second, wrap, impedance, stages, _, system = problem
    baseline = np.asarray(system.parameters).copy()
    reverse = AD["reverse_gradient"](
        ref, first, second, wrap, impedance, stages, segment_substeps=(2, 4), fft_rows=1
    )
    objective = original_objective(problem, baseline, stages["actuator"])
    assert objective == pytest.approx(reverse["value_mm"], abs=1e-8)
    dp = np.array([0.3, -0.2, 0.02, -0.01, 0.01])
    da = np.zeros_like(stages["actuator"])
    da[1:] = 0.1 * np.sin(np.arange(2 * (len(da) - 1)).reshape(-1, 2))
    expected = float(reverse["gradient"] @ dp + np.sum(reverse["actuator_gradient"] * da))
    epsilon = 1e-4
    plus = original_objective(problem, baseline + epsilon * dp, stages["actuator"] + epsilon * da)
    minus = original_objective(problem, baseline - epsilon * dp, stages["actuator"] - epsilon * da)
    finite = (plus - minus) / (2 * epsilon)
    assert finite == pytest.approx(expected, rel=2e-4, abs=2e-7)


def test_stage_clock_must_match_the_causal_history_plan(problem):
    ref, first, second, wrap, impedance, stages, _, _ = problem
    invalid = {**stages, "stage_time": stages["stage_time"].copy()}
    invalid["stage_time"][3, 0] += 1e-5
    with pytest.raises(ValueError, match="stage clock"):
        AD["context"](ref, first, second, wrap, impedance, invalid, (2, 4))

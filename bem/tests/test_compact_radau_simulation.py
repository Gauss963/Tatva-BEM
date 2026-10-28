"""Full nonlinear compact recurrence agrees with an independent original-Z history."""

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
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))
STORAGE = runpy.run_path(str(TOOLS / "compact_radau_storage.py"))
SIMULATE = runpy.run_path(str(TOOLS / "simulate_compact_radau.py"))
CONTACT = runpy.run_path(str(TOOLS / "radau_contact_experiment.py"))
HISTORY = runpy.run_path(str(TOOLS / "compact_radau_history.py"))


@pytest.mark.parametrize("actuator_plateau", [None, 18.1])
@pytest.mark.parametrize("initial_bias", [False, True])
def test_nonlinear_compact_stages_match_original_impedance_history(actuator_plateau, initial_bias):
    plan = plan_cq(3, 0.02, 64, max_memory_mb=64)
    mass = np.array([[0.3, 0.02, 0], [0.02, 0.2, 0.01], [0, 0.01, 0.4]])
    stiffness = np.array([[30.0, 0.2, 0.1], [0.2, 30.0, -20], [0.1, -20, 30]])
    law = SlipWeakeningContact(100, 40, 0.8, 0.5, 0.2)
    tangent = np.diag([100.0, 40.0, 2.0])
    ref = SimpleNamespace(
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

    def zfun(s):
        return s * mass + stiffness

    def hfun(s):
        return np.linalg.inv(zfun(s) + tangent)

    nodes = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    pairs = [RADAU["compact_matrix_function"](hfun, z, plan.dt) for z in nodes]
    f0, f1 = np.array([v[0] for v in pairs]), np.array([v[1] for v in pairs])
    w0, w1, wrap = RADAU["invert_compact_spectrum"](f0, f1, plan)
    stage_time = np.zeros((plan.steps + 1, 2))
    stage_time[1:] = (np.arange(plan.steps)[:, None] + RADAU["C"]) * plan.dt
    ramp = 20 * stage_time
    elastic = STORAGE["causal_compact_startup"](
        f0[:, :, -1], f1[:, :, -1], wrap[:, -1], plan, 2 * ramp
    )
    impedance = RADAU["matrix_function"](zfun, RADAU["radau_symbol"](0) / plan.dt).real
    old = ContactState(np.zeros(1), np.zeros(1))
    initial_force = np.array([0.1, -0.05, 0.02]) if initial_bias else None
    result, stages = SIMULATE["simulate_stages"](
        ref,
        old,
        BoundaryHistory(plan, w0),
        BoundaryHistory(plan, w1),
        wrap,
        impedance,
        elastic,
        stage_time,
        ramp,
        segment_substeps=(2, 4),
        cache_mb=0,
        fft_rows=1,
        actuator_plateau=actuator_plateau,
        initial_residual_force=initial_force,
    )
    # Assemble and invert Z_RK separately. No elastic baseline, compliance,
    # compact coefficients or zero-prefix rebasing is used in this reference.
    spectrum = np.array(
        [RADAU["matrix_function"](zfun, RADAU["radau_symbol"](z) / plan.dt) for z in nodes]
    )
    raw = BoundaryHistory(replace(plan, dofs=6), invert_spectrum(spectrum, replace(plan, dofs=6)))
    springs = np.diag([0.0, 0.0, 2.0, 0.0, 0.0, 2.0])
    system = CONTACT["RadauContactEquilibrium"](
        ref, raw.instantaneous + springs, segment_substeps=(2, 4)
    )
    q = np.zeros((plan.steps + 1, 6))
    response = np.zeros_like(stages["response"])
    held = None
    raw_actuator = ramp.copy() if actuator_plateau is None else np.minimum(ramp, actuator_plateau)
    for k in range(1, plan.steps + 1):
        if held is not None:
            raw_actuator[k] = held
        force = np.tile([-10.0, 0, 0], 2) + raw.past_traction(q, k)
        if initial_force is not None:
            force += np.tile(initial_force, 2)
        force[[2, 5]] -= 2 * raw_actuator[k]
        answer = system.solve(old, force, np.tile(q[k - 1, -3:], 2), q[k - 1, -3:])
        assert answer.converged
        q[k], response[k] = answer.displacement, answer.response
        old = ContactState(answer.response[:, 1, 2], answer.response[:, 1, 3])
        if (
            actuator_plateau is None
            and held is None
            and old.cumulative_slip[0] >= law.characteristic_slip
        ):
            held = raw_actuator[k, 1]
    if actuator_plateau is None:
        assert held is not None
    else:
        assert result.diagnostics["actuator_hold_time_s"] == pytest.approx(actuator_plateau / 20)
        assert not result.diagnostics["loading_diagnostic"]["contact_coverage_feedback_enabled"]
    assert result.diagnostics["first_correction_step"] is not None
    np.testing.assert_allclose(stages["displacement"].reshape(-1, 6), q, rtol=1e-7, atol=2e-7)
    np.testing.assert_allclose(stages["response"][1:], response[1:], rtol=1e-7, atol=2e-6)
    np.testing.assert_array_equal(stages["actuator"], raw_actuator)
    np.testing.assert_allclose(result.cumulative_slip, response[:, :, 1, 3], rtol=1e-7, atol=2e-7)
    assert result.diagnostics["max_boundary_contact_residual_mpa"] < 1e-7
    assert result.diagnostics["micro_friction_work_mpa_mm2"] >= 0


def test_prefix_rebasing_does_not_drop_active_terms_or_accept_wrong_shapes():
    plan = plan_cq(2, 0.1, 8, max_memory_mb=32)
    rng = np.random.default_rng(29)
    families = [BoundaryHistory(plan, rng.normal(size=(9, 2, 2))) for _ in range(2)]
    wrap = rng.normal(size=(2, 2))
    rebased = HISTORY["CompactCorrectionHistory"](*families, wrap, cache_mb=0)
    for k in range(5):
        np.testing.assert_array_equal(rebased.past(k), 0)
        rebased.append(k, np.zeros(4), active=False)
    with pytest.raises(ValueError, match="Invalid"):
        rebased.append(5, np.zeros(2), active=False)
    reference = HISTORY["CompactRadauConvolution"](*families, wrap, cache_mb=0)
    for k in range(5, 14):
        np.testing.assert_array_equal(rebased.past(k), reference.past(k - 5))
        value = rng.normal(size=4)
        rebased.append(k, value, active=True)
        reference.append(value)
    with pytest.raises(ValueError, match="too early"):
        rebased.past(14)

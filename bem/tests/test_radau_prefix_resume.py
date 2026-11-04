"""Resume the original nonlinear time loop across history and loading events."""

import runpy
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
from finiteblock_bem.cq import BoundaryHistory, plan_cq
from finiteblock_bem.friction import ContactState, SlipWeakeningContact

TOOLS = Path(__file__).parents[1] / "tools"
RESUME = runpy.run_path(str(TOOLS / "radau_prefix_resume.py"))
VALIDATE = runpy.run_path(str(TOOLS / "validate_captured_radau_prefix.py"))
RESTORE = runpy.run_path(str(TOOLS / "restore_compact_history.py"))
BASE = runpy.run_path(str(TOOLS / "simulate_compact_radau.py"))
STORAGE = runpy.run_path(str(TOOLS / "compact_radau_storage.py"))


@pytest.fixture(scope="module")
def full_case():
    plan = plan_cq(3, 0.02, 64, max_memory_mb=64)
    mass = np.array([[0.3, 0.02, 0], [0.02, 0.2, 0.01], [0, 0.01, 0.4]])
    stiffness = np.array([[30.0, 0.2, 0.1], [0.2, 30.0, -20], [0.1, -20, 30]])
    ref = SimpleNamespace(
        base_dn=np.array([0.1]),
        base_ds=np.zeros(1),
        base_platen=0.0,
        spring=2.0,
        case=SimpleNamespace(
            contact=SlipWeakeningContact(100, 40, 0.8, 0.5, 0.2), normal_pressure=10.0
        ),
        constraints=SimpleNamespace(
            size=3,
            normal_jump=np.array([[1.0, 0, 0]]),
            tangential_jump=np.array([[0.0, 1, 0]]),
            contact_weights=np.ones(1),
            reduced_weights=np.ones(3),
        ),
    )
    tangent = np.diag([100.0, 40.0, 2.0])

    def hfun(s):
        return np.linalg.inv(s * mass + stiffness + tangent)

    radau = BASE["RADAU"]
    nodes = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    pairs = [radau["compact_matrix_function"](hfun, z, plan.dt) for z in nodes]
    f0, f1 = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    w0, w1, wrap = radau["invert_compact_spectrum"](f0, f1, plan)
    time = np.zeros((65, 2))
    time[1:] = (np.arange(64)[:, None] + radau["C"]) * plan.dt
    ramp = 20 * time
    elastic = STORAGE["causal_compact_startup"](
        f0[:, :, -1], f1[:, :, -1], wrap[:, -1], plan, 2 * ramp
    )
    impedance = radau["matrix_function"](
        lambda s: s * mass + stiffness, radau["radau_symbol"](0) / plan.dt
    ).real
    state = ContactState(np.zeros(1), np.zeros(1))
    bias = np.array([0.1, -0.05, 0.02])
    histories = [BoundaryHistory(plan, w) for w in [w0, w1]]
    forcing_records = []

    def record_system(*args, **kwargs):
        system = BASE["CONTACT"]["RadauContactEquilibrium"](*args, **kwargs)
        solve = system.solve

        def record_solve(*a, **k):
            forcing_records.append(np.array(a[1], copy=True))
            return solve(*a, **k)

        system.solve = record_solve
        return system

    original = BASE["simulate_stages"]
    recording = FunctionType(
        original.__code__,
        {
            **original.__globals__,
            "CONTACT": {**BASE["CONTACT"], "RadauContactEquilibrium": record_system},
        },
        original.__name__,
        original.__defaults__,
    )
    recording.__kwdefaults__ = original.__kwdefaults__
    args = (ref, state, *histories, wrap, impedance, elastic, time, ramp)
    kwargs = dict(segment_substeps=(2, 4), cache_mb=0, fft_rows=1, initial_residual_force=bias)
    result, stages = recording(*args, **kwargs)
    assert result.diagnostics["actuator_hold_time_s"] is not None
    assert len(forcing_records) == 64
    return SimpleNamespace(
        args=args,
        kwargs=kwargs,
        result=result,
        stages=stages,
        forcings=forcing_records,
        ref=ref,
        state=state,
        histories=histories,
        wrap=wrap,
        impedance=impedance,
        time=time,
        ramp=ramp,
        bias=bias,
    )


def snapshot(case, count):
    stages = case.stages
    q, values = stages["displacement"][:count].copy(), stages["response"][:count].copy()
    prefix = VALIDATE["reconstruct_prefix"](case.ref, case.state, q, values, case.time, case.ramp)
    memory = BASE["HISTORY"]["CompactCorrectionHistory"](
        *case.histories, case.wrap, cache_mb=0, fft_rows=1
    )
    RESTORE["restore_correction_prefix"](memory, prefix["sources"], prefix["active"])
    record = dict(
        next_step=count,
        first_correction_step=memory.start,
        held_actuator=prefix["held"],
        hold_time_s=prefix["hold_time"],
        maximum_inactive_source_force=memory.maximum_inactive_source_force,
        local_history_next_step=None,
        pending_stop=0,
        maximum_prefix_residual_mpa=float(case.result.residual_mpa[:count].max()),
    )
    payload = dict(
        actuator_prefix=prefix["actuator"],
        prefix_residual_mpa=case.result.residual_mpa[:count].copy(),
        prefix_newton_iterations=np.full(count, -1),
        initial_contact_force=prefix["fc0"],
        initial_residual_force=case.bias,
    )
    if memory.online is not None:
        online = memory.online
        length, stop = online.next_step, len(online.streams[0].pending)
        record.update(local_history_next_step=length, pending_stop=stop)
        payload.update(
            active_inputs=online.streams[0].inputs[:length].copy(),
            previous_f1=online.previous_f1.copy(),
            future_pending_0=online.streams[0].pending[length:].copy(),
            future_pending_1=online.streams[1].pending[length:].copy(),
        )
    matrix = case.impedance + np.diag([0.0, 0.0, 2.0, 0.0, 0.0, 2.0])
    capture = dict(
        accepted_displacement=q,
        accepted_response=values,
        matrix=matrix,
        base_dn=case.ref.base_dn,
        base_ds=case.ref.base_ds,
        contact_weights=np.ones(1),
        reduced_weights=np.ones(3),
        parameters=np.array([100.0, 40.0, 0.8, 0.5, 0.2]),
        forcing=case.forcings[count - 1].copy(),
    )
    return RESUME["PrefixSnapshot"](record, capture, payload)


@pytest.mark.parametrize("after_hold", [False, True])
def test_uninterrupted_and_resumed_all_fields_and_future_history_match(full_case, after_hold):
    case = full_case
    hold = round(case.result.diagnostics["actuator_hold_time_s"] / 0.02)
    count = min(63, hold + 4) if after_hold else min(17, hold)
    saved = snapshot(case, count)
    assert (saved.record["held_actuator"] is not None) == after_hold
    if after_hold:
        saved.root_seed = case.stages["displacement"][count].ravel().copy()
    simulate = RESUME["resumable_simulator"](BASE["simulate_stages"], saved)
    result, stages = simulate(*case.args, **case.kwargs)
    assert saved.applied and saved.boundary_forcing_error_mpa == 0
    np.testing.assert_array_equal(stages["displacement"], case.stages["displacement"])
    np.testing.assert_array_equal(stages["response"], case.stages["response"])
    np.testing.assert_array_equal(stages["actuator"], case.stages["actuator"])
    np.testing.assert_array_equal(result.cumulative_slip, case.result.cumulative_slip)
    assert (
        result.diagnostics["actuator_hold_time_s"]
        == case.result.diagnostics["actuator_hold_time_s"]
    )
    assert (
        result.diagnostics["micro_friction_work_mpa_mm2"]
        == case.result.diagnostics["micro_friction_work_mpa_mm2"]
    )
    assert result.diagnostics["max_boundary_contact_residual_mpa"] < 1e-7
    np.testing.assert_array_equal(result.newton_iterations[:count], -1)
    assert "accepted_displacement" not in saved.capture


def test_changed_boundary_forcing_is_rejected_before_time_advance(full_case):
    saved = snapshot(full_case, 17)
    saved.capture["forcing"][0] += 1e-5
    simulate = RESUME["resumable_simulator"](BASE["simulate_stages"], saved)
    with pytest.raises(ValueError, match="boundary forcing"):
        simulate(*full_case.args, **full_case.kwargs)
    assert not saved.applied


def test_original_loop_shape_change_requires_reaudit():
    def altered(steps):
        for k in range(2, steps + 1):
            pass

    with pytest.raises(ValueError, match="physical loop changed"):
        RESUME["resumable_simulator"](altered, None)

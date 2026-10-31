"""Observation noninterference and path loss on fixed physical time grids."""

import runpy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

TOOLS = Path(__file__).resolve().parents[1] / "tools"
OBS = runpy.run_path(str(TOOLS / "fem_contact_path_observations.py"))


def test_observed_scan_keeps_original_carry_and_scalar_bits():
    model = dict(
        master_nodes=np.array([0, 1]),
        slave_nodes=np.array([0, 1]),
        moving_offset=4,
        interface_initial_gap=np.zeros(2),
        penalty_n=100.0,
        penalty_t=40.0,
    )
    carry = tuple(jnp.zeros(8) if i < 2 else jnp.zeros(2) for i in range(12))
    carry = (*carry[:6], jnp.full(2, 0.8), carry[7], jnp.ones(2), *carry[9:])

    def step(state, load, zero, prescribed, allow, relax):
        assert allow and not relax
        u = state[0] + load * jnp.arange(8)
        new = (u, state[1] + load, *state[2:])
        return new, jnp.r_[jnp.sum(u), state[2]]

    loading = jnp.array([0.01, 0.03, -0.02])
    baseline = jax.jit(
        lambda c, s: jax.lax.scan(lambda c, a: step(c, a, [], [], True, False), c, s)
    )
    expected, scalar = baseline(carry, loading)
    actual, (observed_scalar, fields) = OBS["observed_scan"](step, model, [], [])(carry, loading)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(observed_scalar, scalar)
    assert fields.shape == (len(loading), 8, 2)
    np.testing.assert_array_equal(fields[-1], OBS["contact_observer"](model)(expected))


def test_stage_clock_is_global_and_retains_both_original_window_boundaries():
    clock = OBS["radau_path_clock"](0.4, 3.4, 1.0)
    np.testing.assert_array_equal(clock, [0.4, 1.0, 4 / 3, 2.0, 7 / 3, 3.0, 10 / 3, 3.4])
    for args in [(-1, 2, 1), (1, 1, 1), (0, 2, 0), (0, np.inf, 1), (0, 100002, 1)]:
        with pytest.raises(ValueError):
            OBS["radau_path_clock"](*args)


def test_p1_projection_is_exact_for_affine_fields_and_never_extrapolates():
    time = np.linspace(0.4, 3.4, 41)
    values = 2 + time[:, None, None] * np.arange(6).reshape(1, 2, 3)
    target = OBS["radau_path_clock"](time[0], time[-1], 1)
    sampled = OBS["interpolate_time"](time, values, target)
    actual = OBS["interpolate_time"](target, sampled, time)
    np.testing.assert_allclose(actual, values, rtol=3e-16, atol=2e-15)
    with pytest.raises(ValueError, match="without extrapolation"):
        OBS["interpolate_time"](time, values, np.array([0.3, 1.0]))


def test_unresolved_reversal_loses_cumulative_slip_even_with_native_local_sampling():
    # The same unmodified return law sees either the full reversal or a path
    # reconstructed only from its endpoints; the local sampling density matches.
    parameters = np.array([100.0, 40.0, 0.8, 0.6, 0.1])
    native = np.array([[[0.1, 0.0]], [[0.1, 1.0]], [[0.1, 0.0]]])
    projected = np.zeros_like(native)
    projected[:, :, 0] = 0.1
    _, exact = OBS["replay_local_path"](native[1:], np.zeros(1), np.zeros(1), parameters)
    _, reduced = OBS["replay_local_path"](projected[1:], np.zeros(1), np.zeros(1), parameters)
    assert exact[-1, 0, 3] > 1
    assert reduced[-1, 0, 3] == 0
    assert np.all(np.diff(exact[:, 0, 3]) >= 0)


def test_local_gate_failure_keeps_field_diagnostics_and_rejects_constitutive_projection():
    time, y = np.linspace(0, 6e-6, 7), np.array([0.0, 1.0])
    parameters = np.array([100.0, 40.0, 0.8, 0.6, 0.1])
    jump = np.zeros((len(time), 2, 2))
    jump[:, :, 0] = 0.1
    jump[:, :, 1] = np.array([0, 0.5, 1, 0.5, 0, -0.5, 0])[:, None]
    _, values = OBS["replay_local_path"](jump[1:], np.zeros(2), np.zeros(2), parameters)
    _, source_order = OBS["replay_source_order_path"](
        jump[1:], np.zeros(2), np.zeros(2), parameters
    )
    np.testing.assert_allclose(source_order, values, rtol=1e-13, atol=1e-13)
    initial = np.tile([10.0, 0.0, 0.0, 0.0, 0.8], (1, 2, 1))
    v = np.concatenate((initial, values))
    fields = np.stack(
        (
            jump[:, :, 0],
            jump[:, :, 1],
            v[:, :, 0],
            v[:, :, 1],
            v[:, :, 2],
            v[:, :, 3],
            v[:, :, 4],
            v[:, :, 0] * v[:, :, 4],
        ),
        axis=1,
    )
    fields[-1, 5] += 0.1
    report = OBS["projection_report"](time, y, fields, parameters, steps=(2e-6,))
    assert not report["constitutive_projection_accepted"]
    error = report["production_bem_native_path_replay"]["quantities"]["cumulative"]
    assert error["maximum_absolute_error"] == pytest.approx(0.1)
    assert error["actual_shear_time_s"] == time[-1]
    for projection in report["projections"].values():
        assert projection["replayed_constitutive_path_errors"] is None
        assert (
            projection["saved_field_projection_errors"]["pressure"]["relative_space_time_p1_l2"]
            < 1e-14
        )

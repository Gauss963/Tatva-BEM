import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.fft import irfft, rfft

from finiteblock_bem.cq import bdf2_symbol, plan_cq

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/original_history_audit.py"))


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("workers", [1, 2])
def test_original_action_matches_independent_bdf2_matrix_recurrence(rows, workers):
    rng = np.random.default_rng(154)
    d, steps, dt, decay = 5, 17, 0.01, 30.0
    plan = replace(plan_cq(d, dt, steps, max_memory_mb=16), radius=0.8)
    a, b, c = [rng.normal(size=(d, d)) for _ in range(3)]
    mass, stiffness, memory = a.T @ a + np.eye(d), b.T @ b + np.eye(d), c.T @ c
    line = np.geomspace(0.1, 3.0, d)
    scale = np.sqrt(line)
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    s = bdf2_symbol(zeta) / dt
    physical = s[:, None, None] * mass + stiffness + memory / (s[:, None, None] + decay)
    source = np.linalg.inv(physical) * scale[None, :, None] * scale[None, None, :]
    target = np.empty_like(source)
    inversion = TOOL["invert_work_spectrum"](source, target, block_matrices=7, workers=workers)
    assert inversion["maximum_inverse_composition_inf_error"] < 1e-12
    q = rng.normal(size=(steps + 1, d, 3)) * 0.001
    actual, diagnostics = TOOL["causal_original_action"](
        target, plan, q, line, fft_rows=rows, max_workspace_mb=16
    )
    # Analytic BDF2 recurrence for the rational memory 1/(s+decay),
    # independent of either frequency inversion or FFT coefficient recovery.
    denominator = np.array([1.5 / dt + decay, -2 / dt, 0.5 / dt])
    impulse = np.zeros(steps + 1)
    impulse[0] = 1 / denominator[0]
    for n in range(1, steps + 1):
        impulse[n] = (
            -sum(denominator[k] * impulse[n - k] for k in range(1, min(n, 2) + 1)) / denominator[0]
        )
    coefficients = impulse[:, None, None] * memory
    coefficients[0] += stiffness + 1.5 / dt * mass
    coefficients[1] -= 2 / dt * mass
    coefficients[2] += 0.5 / dt * mass
    expected = np.stack(
        [
            sum((coefficients[n - j] @ q[j] for j in range(n + 1)), start=np.zeros((d, 3)))
            for n in range(steps + 1)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-11)
    assert diagnostics["complete_lags"] == steps + 1
    assert diagnostics["future_input_wrap_excluded"]


def test_future_displacement_cannot_create_an_early_original_force():
    d, steps = 3, 20
    plan = replace(plan_cq(d, 0.01, steps, max_memory_mb=16), radius=0.97)
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    original = np.eye(d)[None] / (1 - 0.99 * zeta[:, None, None])
    line = np.array([0.2, 0.8, 2.3])
    q = np.zeros((steps + 1, d))
    q[-1] = 1
    actual, _ = TOOL["causal_original_action"](original, plan, q, line, fft_rows=2)
    np.testing.assert_allclose(actual[:-1], 0, atol=2e-14)
    # The untruncated damped frequency product has a measurable future wrap.
    damped = q * plan.radius ** np.arange(steps + 1)[:, None] * np.sqrt(line)
    product = np.einsum("kij,kj->ki", original, rfft(damped, n=plan.fft_size, axis=0))
    circular = irfft(product, n=plan.fft_size, axis=0)[: steps + 1] * np.sqrt(line)
    circular /= plan.radius ** np.arange(steps + 1)[:, None]
    assert np.max(abs(circular[0])) > 0.01
    np.testing.assert_allclose(
        actual[-1, :, 0], line / (1 - (0.99 * plan.radius) ** plan.fft_size), rtol=2e-14
    )


def test_original_audit_rejects_singular_spectra_and_insufficient_workspace():
    source = np.zeros((5, 3, 3), dtype=complex)
    with pytest.raises(np.linalg.LinAlgError):
        TOOL["invert_work_spectrum"](source, np.empty_like(source))
    plan = plan_cq(3, 0.01, 4, max_memory_mb=16)
    with pytest.raises(ValueError, match="workspace"):
        TOOL["causal_original_action"](
            np.broadcast_to(np.eye(3), (plan.fft_size // 2 + 1, 3, 3)).astype(complex),
            plan,
            np.zeros((5, 3)),
            np.ones(3),
            max_workspace_mb=0.00001,
        )


def test_selected_port_work_matches_independent_bdf2_energy_identity():
    rng = np.random.default_rng(166)
    q = rng.normal(size=(25, 3)) * 0.02
    q[0] = 0
    dt = 0.03
    a, b = rng.normal(size=(2, 3, 3))
    mass, stiffness = a.T @ a + np.eye(3), b.T @ b + np.eye(3)

    def derivative(value):
        padded = np.pad(value, ((2, 0), (0, 0)))
        return (1.5 * padded[2:] - 2 * padded[1:-1] + 0.5 * padded[:-2]) / dt

    velocity = derivative(q)
    force = derivative(velocity) @ mass.T + q @ stiffness.T
    work, weighted, diagnostics = TOOL["selected_bulk_port_work"](force, q, dt, 1.0)
    # Exact discrete G-energy plus the two BDF2 numerical dissipation sums.
    # This is independent of the dot(force, velocity) accumulation under test.
    expected = 0.0
    for values, matrix in ((q, stiffness), (velocity, mass)):
        last = values[-1]
        predictor = 2 * values[-1] - values[-2]
        expected += (last @ matrix @ last + predictor @ matrix @ predictor) / 4
        padded = np.pad(values, ((2, 0), (0, 0)))
        difference = padded[2:] - 2 * padded[1:-1] + padded[:-2]
        expected += np.einsum("ti,ij,tj->", difference, matrix, difference) / 4
    np.testing.assert_allclose(work[-1], expected, rtol=3e-15)
    np.testing.assert_array_equal(work, weighted)
    assert work.min() >= 0
    assert diagnostics["minimum_prefix_discrete_bulk_port_work_mpa_mm2"] == 0

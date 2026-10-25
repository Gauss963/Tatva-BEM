"""Exact two-family Radau histories must include finite-circle aliasing."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import invert_spectrum, plan_cq

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/radau_experiment.py"))


def transfer(s):
    mass = np.array([[2.0, 0.4], [0.4, 1.0]])
    stiffness = np.array([[0.7, -0.2], [-0.2, 0.5]])
    return np.linalg.inv(s * mass + stiffness)


def test_compact_transfer_matches_independent_rk_coupled_ode():
    a = np.array([[5 / 12, -1 / 12], [3 / 4, 1 / 4]])
    b = np.array([3 / 4, 1 / 4])
    mass = np.array([[2.0, 0.4], [0.4, 1.0]])
    stiffness = np.array([[0.7, -0.2], [-0.2, 0.5]])
    for z in 0.94 * np.exp(-1j * np.array([0, 0.3, 1.7, np.pi])):
        f0, f1 = TOOL["compact_matrix_function"](transfer, z, 0.04)
        delta = np.linalg.inv(a + z / (1 - z) * np.outer(np.ones(2), b))
        actual = np.kron(np.eye(2), f0) + np.kron(delta, f1)
        expected = np.linalg.inv(np.kron(delta / 0.04, mass) + np.kron(np.eye(2), stiffness))
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)


def test_compact_weights_and_causal_response_preserve_damped_dft_wrap():
    # Deliberately visible aliasing tests the exact finite CQ algebra, rather
    # than relying on a small tail falling below a loose equality threshold.
    plan = plan_cq(2, 0.1, 15, max_memory_mb=32)
    # Manufactured circle for this algebra test only. Production planner's
    # alias-tolerance bounds remain unchanged.
    plan = replace(plan, radius=0.05 ** (1 / plan.fft_size))
    stage_plan = replace(plan, dofs=4)
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    compact = [TOOL["compact_matrix_function"](transfer, z, plan.dt) for z in zeta]
    f0, f1 = np.array([v[0] for v in compact]), np.array([v[1] for v in compact])
    original0, original1 = f0.copy(), f1.copy()
    w0, w1, wrap = TOOL["invert_compact_spectrum"](f0, f1, plan)
    np.testing.assert_array_equal(f0, original0)
    np.testing.assert_array_equal(f1, original1)
    dense = np.array(
        [TOOL["matrix_function"](transfer, TOOL["radau_symbol"](z) / plan.dt) for z in zeta]
    )
    expected = invert_spectrum(dense, stage_plan)
    actual = TOOL["expand_compact_weights"](w0, w1, wrap)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-13)
    assert np.max(abs(wrap)) > 1e-7
    without_wrap = TOOL["expand_compact_weights"](w0, w1, np.zeros_like(wrap))
    assert np.max(abs(without_wrap[0] - expected[0])) > 1e-7
    forcing = np.random.default_rng(98).normal(size=(plan.steps + 1, 4))
    for k in range(plan.steps + 1):
        left = np.einsum("kij,kj->i", actual[: k + 1], forcing[k::-1])
        right = np.einsum("kij,kj->i", expected[: k + 1], forcing[k::-1])
        np.testing.assert_allclose(left, right, rtol=2e-11, atol=2e-13)


def test_nonhermitian_compact_endpoint_rejected():
    plan = plan_cq(2, 0.1, 3, max_memory_mb=32)
    spectrum = np.ones((plan.fft_size // 2 + 1, 2, 2), complex)
    spectrum[-1] += 1j
    with pytest.raises(ValueError, match="conjugate symmetry"):
        TOOL["invert_compact_spectrum"](spectrum, spectrum, plan)

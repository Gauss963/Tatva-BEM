"""RK resolvent modes must recover the independent coupled ODE stage solve."""

import runpy
from pathlib import Path

import numpy as np
import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/audit_cauchy_reuse.py"))


def test_radau_resolvent_reconstruction_matches_coupled_ode_stages():
    # Independent Butcher-tableau reference; noncommuting mass and stiffness
    # prevent this check from reducing to scalar diagonal channels.
    tableau = np.array([[5 / 12, -1 / 12], [3 / 4, 1 / 4]])
    b = np.array([3 / 4, 1 / 4])
    mass = np.array([[2.0, 0.4], [0.4, 1.0]])
    stiffness = np.array([[7.0, -2.0], [-2.0, 5.0]])
    zeta, dt = 0.97 * np.exp(-1j * np.array([0.0, 0.37, 1.9, np.pi])), 0.04
    frequencies, vectors = TOOL["contour_modes"](zeta, dt, "radau2")
    values = np.array([[np.linalg.inv(s * mass + stiffness) for s in row] for row in frequencies])
    actual = TOOL["compose_modes"](values, vectors)
    for i, z in enumerate(zeta):
        # Eliminate the prior endpoint directly from the RK stage equations.
        delta = np.linalg.inv(tableau + z / (1 - z) * np.outer(np.ones(2), b))
        expected = np.linalg.inv(np.kron(delta / dt, mass) + np.kron(np.eye(2), stiffness))
        np.testing.assert_allclose(actual[i], expected, rtol=2e-12, atol=2e-13)
    # The two eigenmodes at real z may be complex conjugates; projecting each
    # individual mode to real would corrupt the correct real stage transfer.
    assert abs(frequencies[-1].imag).max() > 1
    np.testing.assert_allclose(actual[[0, -1]].imag, 0, atol=2e-13)


def test_bdf2_audit_keeps_the_single_stage_matrix():
    values = np.arange(18.0).reshape(2, 1, 3, 3)
    np.testing.assert_array_equal(TOOL["compose_modes"](values, None), values[:, 0])


@pytest.mark.parametrize("method", ["backward-euler", "radau3"])
def test_unsupported_method_is_rejected(method):
    with pytest.raises(ValueError, match="Unknown"):
        TOOL["contour_modes"](np.array([0.9]), 0.1, method)

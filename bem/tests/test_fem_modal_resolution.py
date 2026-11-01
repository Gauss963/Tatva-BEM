"""Independent constrained energy, oscillator and spatial Fourier controls."""

import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh

F = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/fem_modal_resolution.py"))


def test_mass_constraint_and_modes_match_independent_dense_generalized_problem():
    mass = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    transform, reduced = F["constrained_mass_map"](mass, [0], [4, 5])
    expected = np.array(
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]]
    )
    np.testing.assert_array_equal(transform.toarray(), expected)
    np.testing.assert_array_equal(reduced, [3, 5, 7, 24])
    a = np.random.default_rng(31).normal(size=(6, 6))
    k = a.T @ a + np.eye(6)
    kr, mr = expected.T @ k @ expected, expected.T @ np.diag(mass) @ expected
    values, q, audit = F["largest_elastic_modes"](kr, reduced, count=3)
    wanted = eigh(kr, mr, eigvals_only=True)[::-1][:3]
    np.testing.assert_allclose(values, wanted, rtol=3e-15)
    np.testing.assert_allclose(q.T @ mr @ q, np.eye(3), atol=2e-15)
    assert max(audit["relative_ritz_residuals"]) < 1e-14
    for invalid in [[0, -1, 5, 7, 11, 13], [2, 3, 5, 7, np.nan, 13]]:
        with pytest.raises(ValueError):
            F["constrained_mass_map"](invalid, [0], [4, 5])
    with pytest.raises(ValueError):
        F["constrained_mass_map"](mass, [4], [4, 5])
    with pytest.raises(ValueError, match="symmetric"):
        F["largest_elastic_modes"](kr + np.triu(np.ones_like(kr)), reduced, count=2)


def test_contact_jump_energy_matches_direct_two_block_displacements():
    normal, tangent = F["interface_jump_matrices"](12, [0, 2], [1, 0], 6)
    u = np.arange(12, dtype=float) ** 2
    np.testing.assert_array_equal(normal @ u, [u[0] - u[8], u[4] - u[6]])
    np.testing.assert_array_equal(tangent @ u, [u[1] - u[9], u[5] - u[7]])
    weights = np.array([3.0, 7.0])
    matrix = normal.T @ np.diag(weights) @ normal
    assert u @ matrix @ u == pytest.approx(np.sum(weights * (normal @ u) ** 2))


def test_phase_formula_matches_actual_drift_kick_matrix_eigenangle():
    omega, dt, duration = 3.0, 0.2, 17.0
    # State [u_n, v_(n+1/2)] is drifted then kicked.
    matrix = np.array([[1, dt], [-dt * omega**2, 1 - dt**2 * omega**2]])
    angle = max(np.angle(np.linalg.eigvals(matrix)))
    result = F["leapfrog_phase_control"](omega, dt, duration)
    assert result["numerical_omega_rad_per_s"] == pytest.approx(angle / dt, rel=2e-15)
    assert result["phase_error_over_duration_rad"] == pytest.approx((angle / dt - omega) * duration)
    assert not result["nonlinear_friction_stability_certificate"]
    assert F["leapfrog_phase_control"](omega, 1, duration)["numerical_omega_rad_per_s"] is None
    assert (
        F["leapfrog_phase_control"](omega, 2 / omega, duration)["numerical_omega_rad_per_s"] is None
    )


def test_spatial_mass_matches_independent_gauss_integration():
    y, values = np.array([0.0, 0.03, 0.17, 1.0]), np.array([1.0, -3.0, 5.0, 2.0])
    integral = 0.0
    for i, h in enumerate(np.diff(y)):
        for xi in [-1 / np.sqrt(3), 1 / np.sqrt(3)]:
            value = (1 - xi) / 2 * values[i] + (1 + xi) / 2 * values[i + 1]
            integral += h / 2 * value**2
    assert values @ F["spatial_p1_mass"](y) @ values == pytest.approx(integral, rel=3e-15)


@pytest.mark.parametrize("size", [255, 256])
def test_native_spectrum_resolves_known_peak_and_passes_independent_parseval(size):
    dt, y = 0.003, np.array([0.0, 0.17, 1.0])
    time = np.arange(size) * dt
    field = np.sin(2 * np.pi * 23 / (size * dt) * time[:, None]) * np.array([1, 2, 0.5])
    field += 12 + y
    frequency, power, audit = F["native_normal_jump_spectrum"](time, y, field)
    assert frequency[np.argmax(power)] == pytest.approx(23 / (size * dt), rel=1e-15)
    window = np.hanning(size)
    sample = window[:, None] * (field - np.average(field, axis=0, weights=window))
    direct = 0.0
    for i, h in enumerate(np.diff(y)):
        for xi in [-1 / np.sqrt(3), 1 / np.sqrt(3)]:
            point = (1 - xi) / 2 * sample[:, i] + (1 + xi) / 2 * sample[:, i + 1]
            direct += h / 2 * np.sum(point**2) / size
    assert sum(power) == pytest.approx(direct, rel=3e-15)
    assert audit["relative_parseval_error"] < 1e-14
    with pytest.raises(ValueError, match="uniform"):
        F["native_normal_jump_spectrum"](time**2, y, field)

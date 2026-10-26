import runpy
from pathlib import Path

import numpy as np
import pytest

CHECK = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/compare_linear_bulk_response.py")
)


def test_support_transform_preserves_platen_mass_and_constraint_work():
    p = CHECK["support_transform"](8, [0, 5], [3, 7])
    q = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    np.testing.assert_allclose(p @ q, [0, 1, 2, 10, 3, 0, 4, 10])
    mass = np.arange(1, 9)
    assert (p.T @ np.diag(mass) @ p)[-1, -1] == 12
    force = np.arange(8)
    assert np.dot(force, p @ q) == pytest.approx(np.dot(p.T @ force, q))


def test_contact_jump_and_penalty_energy_independently():
    jump = CHECK["contact_jump"](12, [0, 2], [0, 1], 6)
    u = np.arange(12, dtype=float) ** 2
    expected = np.array([u[0] - u[6], u[4] - u[8], u[1] - u[7], u[5] - u[9]])
    np.testing.assert_array_equal(jump @ u, expected)
    penalty = np.array([2, 3, 5, 7])
    k = jump.T @ np.diag(penalty) @ jump
    assert u @ k @ u == pytest.approx(np.dot(penalty, expected**2))


def test_complex_p1_profile_error_is_positive_and_mesh_independent():
    y = np.array([0.0, 1.0, 3.0])
    target = (1 + 2j) * y + 2j
    assert CHECK["profile_relative_error"](y, 2 * target, y, target) == pytest.approx(1)
    fine = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    assert CHECK["profile_relative_error"](fine, (1 + 2j) * fine + 2j, y, target) < 1e-15


def test_invalid_platen_constraints():
    with pytest.raises(ValueError):
        CHECK["support_transform"](6, [0], [0, 5])

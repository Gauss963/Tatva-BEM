import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import ResourceBudgetError

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/interior_radau.py"))


def test_rectangular_radau_matrix_matches_analytic_polynomial_and_strictly_causal_apply():
    dt, zeta = 0.02, 0.7 * np.exp(-0.8j)
    constant = np.array([[1.0, 2], [0.4, -1], [-0.2, 1]])
    linear = np.array([[0.1, 0], [0.2, 0.4], [0.5, -0.7]])
    matrix = TOOL["stage_matrix"](lambda s: constant + s * linear, zeta, dt)
    symbol = TOOL["RK"]["radau_symbol"](zeta) / dt
    np.testing.assert_allclose(
        matrix, np.kron(np.eye(2), constant) + np.kron(symbol, linear), rtol=1e-12, atol=1e-12
    )
    weights, metadata = TOOL["rectangular_weights"](lambda s: constant, dt, 8, 2, 3)
    forcing = np.zeros((9, 4))
    forcing[3] = [1, 2, 3, 4]
    actual = TOOL["apply_weights"](weights, forcing)
    np.testing.assert_allclose(actual[3], np.kron(np.eye(2), constant) @ forcing[3], atol=1e-12)
    np.testing.assert_allclose(actual[:3], 0, atol=1e-12)
    np.testing.assert_allclose(actual[4:], 0, atol=1e-10)
    assert metadata["every_requested_lag_retained"]


def test_rectangular_cq_rejects_nonzero_prehistory_and_oversized_dense_allocation():
    with pytest.raises(ValueError, match="zero artificial"):
        TOOL["apply_weights"](np.zeros((3, 4, 2)), np.ones((3, 2)))
    with pytest.raises(ResourceBudgetError, match="bounded history"):
        TOOL["rectangular_weights"](
            lambda s: np.ones((1000, 1000)), 0.01, 100, 1000, 1000, max_memory_mb=10
        )


def test_delayed_displacement_and_velocity_converge_without_fitted_phase():
    pulse = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools/check_interior_fields.py")
    )["pulse"]
    errors = []
    for dt in [0.04, 0.02, 0.01]:
        steps = round(1.2 / dt)
        weights, _ = TOOL["rectangular_weights"](
            lambda s: np.array([[np.exp(-0.235 * s)], [s * np.exp(-0.235 * s)]]), dt, steps, 1, 2
        )
        time = np.arange(steps + 1) * dt
        stage_time = np.zeros((steps + 1, 2))
        stage_time[1:] = (np.arange(steps)[:, None] + TOOL["RK"]["C"]) * dt
        forcing = pulse(stage_time)[0]
        forcing[0] = 0
        actual = TOOL["apply_weights"](weights, forcing).reshape(steps + 1, 2, 2)[:, 1]
        exact = np.column_stack(pulse(time - 0.235))
        errors.append(np.linalg.norm(actual - exact, axis=0) / np.linalg.norm(exact, axis=0))
    assert np.all(errors[1] < errors[0] / 3)
    assert np.all(errors[2] < errors[1] / 3)
    assert np.max(errors[-1]) < 0.01

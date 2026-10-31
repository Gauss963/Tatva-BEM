"""Independent mass assembly, orthogonality and representation lower bounds."""

import runpy
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
TOOLS = Path(__file__).resolve().parents[1] / "tools"
PROJECTION = runpy.run_path(str(TOOLS / "temporal_path_projection.py"))
COMPARE = runpy.run_path(str(TOOLS / "compare_refined_fem_history.py"))["compare_saved_p1_fields"]


def test_affine_and_identical_clock_fields_are_reproduced():
    t = np.array([0.4, 0.7, 1.3, 2.1, 3.4])
    values = np.c_[2 + 3 * t, 5 - 2 * t]
    knots = np.array([t[0], 1.0, 2.7, t[-1]])
    actual, proof = PROJECTION["best_p1_time_projection"](t, values, knots)
    np.testing.assert_allclose(actual, np.c_[2 + 3 * knots, 5 - 2 * knots], rtol=2e-15, atol=3e-15)
    actual, _ = PROJECTION["best_p1_time_projection"](t, values, t)
    np.testing.assert_allclose(actual, values, rtol=2e-15, atol=3e-15)
    assert proof["endpoint_coefficients_unconstrained"]


def test_projection_matches_independent_dense_gauss_mass_and_rhs():
    time = np.array([0.0, 0.12, 0.51, 0.83, 1.0])
    knots = np.array([0.0, 0.3, 0.71, 1.0])
    values = np.random.default_rng(16).normal(size=(len(time), 3))
    mass, rhs = np.zeros((len(knots), len(knots))), np.zeros((len(knots), 3))
    union = np.union1d(time, knots)
    for a, b in pairwise(union):
        for q in [-1 / np.sqrt(3), 1 / np.sqrt(3)]:
            x = (a + b) / 2 + (b - a) / 2 * q
            phi = np.array([np.interp(x, knots, np.eye(len(knots))[i]) for i in range(len(knots))])
            field = np.array([np.interp(x, time, v) for v in values.T])
            mass += (b - a) / 2 * np.outer(phi, phi)
            rhs += (b - a) / 2 * phi[:, None] * field
    actual, _ = PROJECTION["best_p1_time_projection"](time, values, knots)
    np.testing.assert_allclose(actual, np.linalg.solve(mass, rhs), rtol=3e-15, atol=3e-15)


def test_best_projection_beats_interpolation_and_arbitrary_coefficients_in_space_time_norm():
    time = np.linspace(0.0, 1.0, 33)
    knots = np.array([0.0, 0.21, 0.68, 1.0])
    y = np.array([0.0, 0.01, 0.3, 1.0])
    value = np.sin(19 * time[:, None] + y) + time[:, None] ** 2
    best, _ = PROJECTION["best_p1_time_projection"](time, value, knots)
    error = COMPARE((time, y, value), (knots, y, best))["error_squared_space_time_integral"]
    sampled = np.column_stack([np.interp(knots, time, v) for v in value.T])
    interpolation_error = COMPARE((time, y, value), (knots, y, sampled))[
        "error_squared_space_time_integral"
    ]
    assert error < interpolation_error
    perturb = np.random.default_rng(31).normal(size=best.shape)
    candidate_error = COMPARE((time, y, value), (knots, y, best + perturb))[
        "error_squared_space_time_integral"
    ]
    perturbation_norm = COMPARE((knots, y, np.zeros_like(best)), (knots, y, perturb))[
        "error_squared_space_time_integral"
    ]
    assert candidate_error == pytest.approx(error + perturbation_norm, rel=2e-14, abs=1e-14)


def test_invalid_or_truncated_projection_clocks_are_rejected():
    time, fields = np.array([0.0, 0.5, 1.0]), np.ones((3, 2))
    for knots in [[0.1, 1.0], [0.0, 0.9], [0.0, 0.5, 0.5, 1.0], [0.0, np.inf]]:
        with pytest.raises(ValueError, match="identical boundaries"):
            PROJECTION["best_p1_time_projection"](time, fields, knots)

"""Independent force-coordinate controls for the nonsymmetric sliding tangent."""

import runpy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
from finiteblock_bem.friction import SlipWeakeningContact

TOOL = runpy.run_path(str(Path(__file__).parents[1] / "tools/compare_kinetic_bulk_response.py"))


def test_weighted_cached_sliding_response_matches_direct_nodal_force_equation():
    n, kn, kt, mu, spring = 3, 5.0, 3.0, 0.6, 12.0
    weights = np.array([0.013, 3.1, 270.0])
    line = np.r_[weights, weights, 200.0]
    scale = np.sqrt(line)
    rng = np.random.default_rng(703)
    raw = rng.normal(size=(2 * n + 1, 2 * n + 1))
    bulk = raw.T @ raw + 100 * np.eye(2 * n + 1)
    bulk = bulk + 0.3j * np.eye(2 * n + 1)
    stick = bulk + np.diag(np.r_[kn * weights, kt * weights, spring])
    resolvent = scale[:, None] * np.linalg.inv(stick) * scale[None, :]
    rhs = rng.normal(size=2 * n + 1)
    correction = TOOL["correction_matrix"](n, kn, kt, mu)
    work_response, residual = TOOL["resolvent_response"](resolvent, correction, rhs / scale)

    # Assemble the physical force law independently, without reference subtraction.
    direct = bulk.copy()
    for i, weight in enumerate(weights):
        direct[i, i] += weight * kn
        direct[n + i, i] += weight * mu * kn
    direct[-1, -1] += spring
    expected = np.linalg.solve(direct, rhs)
    np.testing.assert_allclose(work_response / scale, expected, rtol=2e-13, atol=1e-15)
    assert residual < 1e-13


def test_sliding_and_stick_tangents_match_original_local_ad_and_numpy_maps():
    checks = TOOL["audit_local_tangents"](SlipWeakeningContact(76620, 7662, 0.8, 0.6, 0.0125959))
    assert set(checks) == {"closed_stick", "positive_kinetic_sliding"}
    for value in checks.values():
        assert value["tangent_max_absolute_error"] < 1e-11
        assert value["numpy_value_max_absolute_error"] < 1e-12

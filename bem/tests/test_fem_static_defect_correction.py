import runpy
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps

TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/fem_static_defect_correction.py")
)


def problem():
    # Two tied face coordinates share one unknown. Test the split physical
    # force and support transformation independently of the solver matrix.
    p = sps.csr_matrix([[1, 0], [0, 1], [0, 1]])
    elastic = sps.csr_matrix([[3.0, -1, 0], [-1, 4, -1], [0, -1, 2]])
    contact = sps.diags([100.0, 0, 10])
    spring = np.array([0.0, 5.0])
    target = np.array([0.25, 2.0])
    load = np.asarray(p.T @ (elastic @ (p @ target) + contact @ (p @ target)))
    load += spring * target

    def residual(q):
        u = p @ q
        return np.asarray(p.T @ (elastic @ u + contact @ u)) + spring * q - load

    matrix = p.T @ (elastic + contact) @ p + sps.diags(spring)
    return p, matrix, target, load, residual


def test_corrects_original_split_forces_and_preserves_tied_face():
    p, matrix, target, load, residual = problem()
    q, records = TOOL["correct_fixed_set"](
        matrix,
        target + np.array([0.01, -0.03]),
        residual,
        lambda q: np.all(q > 0),
        np.linalg.norm(load),
    )
    np.testing.assert_allclose(q, target, rtol=1e-14, atol=1e-14)
    assert (p @ q)[1] == (p @ q)[2]
    assert np.linalg.norm(residual(q)) / np.linalg.norm(load) < 1e-9
    assert records[0]["original_relative_force_residual"] > 1e-4


def test_rejects_correction_that_changes_the_prescribed_contact_set():
    _, matrix, target, load, residual = problem()
    with pytest.raises(ValueError, match="changed the original contact active set"):
        TOOL["correct_fixed_set"](
            matrix,
            target + np.array([0.01, 0]),
            residual,
            lambda q: q[0] > 0.255,
            np.linalg.norm(load),
        )


def test_refuses_unconverged_or_relaxed_correction():
    _, matrix, target, load, residual = problem()
    with pytest.raises(ValueError, match="did not reach"):
        TOOL["correct_fixed_set"](
            matrix, target + 0.01, residual, lambda q: True, np.linalg.norm(load), steps=0
        )
    with pytest.raises(ValueError, match="stricter target"):
        TOOL["correct_fixed_set"](
            matrix, target, residual, lambda q: True, np.linalg.norm(load), target=1e-7
        )

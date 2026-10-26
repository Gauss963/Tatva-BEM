import runpy
from pathlib import Path

import numpy as np
import pytest

TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/audit_restart_projection.py")
)


def test_seed_projection_measures_between_node_error_against_gauss_integrals():
    y = np.array([0.0, 0.2, 0.7, 1.0])
    z = np.array([0.1, 0.15, 0.12, 0.14])
    retained = np.array([0.0, 0.4, 1.0])
    report, fields = TOOL["seed_field_difference"](y, z, retained, 40.0)
    gauss, weights = np.polynomial.legendre.leggauss(3)
    coarse_z = np.interp(retained, y, z)
    integral = 0.0
    for a, b in zip(fields["y"][:-1], fields["y"][1:], strict=True):
        points = a + (b - a) * (gauss + 1) / 2
        difference = np.interp(points, y, z) - np.interp(points, retained, coarse_z)
        integral += (b - a) * np.dot(weights, difference**2) / 2
    np.testing.assert_allclose(
        report["p1_l2_seed_difference_mm_sqrt_mm"] ** 2, integral, rtol=2e-14
    )
    assert report["maximum_seed_difference_mm"] > 0.01
    np.testing.assert_array_equal(
        np.interp(retained, fields["y"], fields["original_plastic_slip"]), coarse_z
    )
    np.testing.assert_allclose(
        report["maximum_frozen_jump_trial_shear_difference_mpa"],
        40 * report["maximum_seed_difference_mm"],
    )


def test_linear_seed_is_preserved_and_extrapolation_is_rejected():
    y = np.array([0.0, 0.3, 0.9, 1.0])
    report, _ = TOOL["seed_field_difference"](y, 0.02 * y - 1, [0, 0.5, 1], 40)
    assert report["maximum_seed_difference_mm"] < 1e-15
    with pytest.raises(ValueError, match="same interval"):
        TOOL["seed_field_difference"](y, 0.02 * y, [0, 0.8], 40)

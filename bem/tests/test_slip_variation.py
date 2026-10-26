import runpy
from pathlib import Path

import numpy as np
import pytest

AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/audit_slip_variation.py"))


def test_accumulated_slip_detects_reversals_hidden_between_frames():
    # Fine solver path 0 -> 2 -> 1 -> 3, with only its endpoints exported.
    fine = np.array([0.0, 2.0, 1.0, 3.0])
    accumulated = np.r_[0, np.cumsum(abs(np.diff(fine)))]
    fine_z = np.column_stack([fine, 2 * fine])
    fine_c = np.column_stack([accumulated, 2 * accumulated])
    for ids in [np.arange(4), [0, 3]]:
        p, m = AUDIT["variation_profiles"]([0, 2], fine_z[ids], fine_c[ids])
        np.testing.assert_allclose(p["net_plastic_slip_mm"], [3, 6])
        np.testing.assert_allclose(p["accumulated_minus_net_mm"], [2, 4])
        assert m["accumulated_minus_net_mm"]["p1_weighted_mean"] == 3
    np.testing.assert_allclose(p["accumulated_minus_sampled_variation_mm"], [2, 4])


def test_monotone_signed_slip_and_nonuniform_p1_weights():
    z = np.array([[10, 20, 30], [9, 22, 34], [7, 25, 38]])
    c = abs(z - z[0])
    p, m = AUDIT["variation_profiles"]([0, 1, 4], z, c)
    np.testing.assert_array_equal(p["accumulated_minus_net_mm"], 0)
    assert m["net_plastic_slip_mm"]["p1_weighted_mean"] == pytest.approx(23.5 / 4)


def test_rounding_deficits_are_reported_without_clipping():
    p, m = AUDIT["variation_profiles"]([0, 1], [[0, 0], [1, 1]], [[0, 0], [0.999, 1]])
    assert p["accumulated_minus_net_mm"][0] < 0
    assert m["minimum_saved_cumulative_increment_mm"] == 0.999


@pytest.mark.parametrize(
    "y,z,c",
    [
        ([0, 0], [[0, 0], [1, 1]], [[0, 0], [1, 1]]),
        ([0, 1], [[0, 0]], [[0, 0]]),
        ([0, 1], [[0, 0], [1, np.nan]], [[0, 0], [1, 1]]),
        ([0, 1], [[0, 0], [1, 1]], [[0], [1]]),
    ],
)
def test_invalid_variation_data(y, z, c):
    with pytest.raises(ValueError):
        AUDIT["variation_profiles"](y, z, c)

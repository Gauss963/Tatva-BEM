"""Static comparisons integrate the physical P1 functions across both grids."""

import runpy
from pathlib import Path

import numpy as np
import pytest

TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/report_density_refinement.py")
)


def test_reference_peak_between_candidate_knots_is_not_missed():
    integrate = TOOL["profile_integrals"]
    error, norm = integrate([0, 2], [0, 0], [0, 1, 2], [0, 1, 0], lower=0, upper=2)
    assert error == pytest.approx(2 / 3)
    assert norm == pytest.approx(2 / 3)
    parts = [
        integrate([0, 2], [0, 0], [0, 1, 2], [0, 1, 0], lower=a, upper=b)[0]
        for a, b in [(0, 0.25), (0.25, 1.75), (1.75, 2)]
    ]
    assert sum(parts) == pytest.approx(error)
    assert parts[0] > 0  # Contact ends remain included in the total.


def test_nonuniform_affine_profile_has_analytic_l2_integral():
    y = np.array([0, 0.3, 1.7, 3])
    error, norm = TOOL["profile_integrals"](
        y, 2 * y + 1, [0, 1, 3], [0, 0, 0], lower=0.25, upper=2.75
    )

    def primitive(x):
        return 4 * x**3 / 3 + 2 * x**2 + x

    assert error == pytest.approx(primitive(2.75) - primitive(0.25))
    assert norm == 0


@pytest.mark.parametrize(
    "y,values", [([0, 0, 1], [1, 2, 3]), ([0, 1], [1, np.nan]), ([1, 0], [1, 2])]
)
def test_invalid_profile_rejected(y, values):
    with pytest.raises(ValueError, match="ordered finite"):
        TOOL["profile_integrals"](y, values, [0, 1], [1, 1], lower=0, upper=1)


def test_interval_outside_either_profile_rejected():
    with pytest.raises(ValueError, match="covered"):
        TOOL["profile_integrals"]([0, 1], [1, 1], [0, 2], [1, 1], lower=0, upper=2)

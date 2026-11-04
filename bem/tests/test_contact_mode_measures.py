"""Exact geometric controls for sign-changing spatial P1 contact fields."""

import runpy
from pathlib import Path

import numpy as np
import pytest

TOOL = runpy.run_path(str(Path(__file__).parents[1] / "tools/diagnose_contact_modes.py"))


@pytest.mark.parametrize("sign", [-1, 1])
def test_affine_zero_crossing_has_exact_half_length_and_absolute_integral(sign):
    # The zero is inside an irregular panel; a nodal sign count is incorrect.
    y = np.array([0.0, 0.2, 0.7, 1.0])
    values = sign * (2 * y - 1)[None, :]
    np.testing.assert_allclose(TOOL["negative_length_fraction"](y, values), 0.5, rtol=0, atol=1e-15)
    np.testing.assert_allclose(TOOL["absolute_p1_mean"](y, values), 0.5, rtol=0, atol=1e-15)


def test_zero_intervals_and_isolated_zero_points_have_distinct_measures():
    y = np.array([0.0, 0.13, 0.8, 1.0])
    values = np.stack([y, -y, np.zeros_like(y), np.full_like(y, -3.0)])
    np.testing.assert_allclose(
        TOOL["negative_length_fraction"](y, values), [0, 1, 0, 1], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        TOOL["negative_length_fraction"](y, values, include_zero=True),
        [0, 1, 1, 1],
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        TOOL["absolute_p1_mean"](y, values), [0.5, 0.5, 0, 3], rtol=0, atol=1e-15
    )


def test_actual_writer_flat_jump_columns_are_read_without_component_interleaving():
    data = dict(
        time=np.array([0.0, 1.0]),
        contact_y=np.array([0.0, 3.0]),
        contact_jumps=np.array([[1.0, 2.0, 21.0, 22.0], [3.0, 4.0, 23.0, 24.0]]),
        shear=np.zeros((2, 2)),
        plastic_slip=np.zeros((2, 2)),
        cumulative_slip=np.zeros((2, 2)),
        friction_coefficient=np.ones((2, 2)),
        pressure=np.ones((2, 2)),
        actuator=np.array([5.0, 6.0]),
        platen=np.array([4.0, 7.0]),
    )
    _, _, fields, gap, _ = TOOL["bem_arrays"](data)
    np.testing.assert_array_equal(fields[0], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(gap, [1, -1])

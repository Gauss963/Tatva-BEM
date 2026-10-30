import runpy
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
WINDOW = runpy.run_path(str(TOOLS / "fem_timestep_window.py"))


def test_refinement_preserves_center_velocity_and_exact_constant_acceleration():
    old_dt, new_dt = 0.04, 0.01
    position, velocity, acceleration = (
        np.array([2.0, -3.0]),
        np.array([5.0, -7.0]),
        np.array([11.0, 13.0]),
    )
    half = velocity + old_dt * acceleration / 2
    original = (position, half, np.array([9.0]))
    refined = WINDOW["rephase_half_velocity"](original, acceleration, old_dt, new_dt)
    np.testing.assert_array_equal(refined[0], position)
    assert refined[2] is original[2]
    np.testing.assert_allclose(refined[1] - new_dt * acceleration / 2, velocity, atol=1e-15)
    x, vhalf = refined[:2]
    for _ in range(4):
        x = x + new_dt * vhalf
        vhalf = vhalf + new_dt * acceleration
    np.testing.assert_allclose(x, position + old_dt * velocity + old_dt**2 * acceleration / 2)
    assert WINDOW["rephase_half_velocity"](original, acceleration, old_dt, old_dt) is original


def test_loading_refinement_preserves_physical_endpoints_and_original_bits():
    model = {k: 3.0 + np.arange(12, dtype=float) / 5 for k in WINDOW["SCHEDULES"]}
    expected = np.column_stack([model[k][2:8] for k in WINDOW["SCHEDULES"]])
    np.testing.assert_array_equal(WINDOW["window_schedule"](model, 2, 8, 1), expected)
    for factor in [2, 4]:
        actual = WINDOW["window_schedule"](model, 2, 8, factor)
        np.testing.assert_array_equal(actual[factor - 1 :: factor], expected)
        # A ramp is sampled at the refined physical ENDPOINT, not at its start.
        np.testing.assert_allclose(actual[0], 3.0 + (1 + 1 / factor) / 5)


@pytest.mark.parametrize("new_dt", [0, -1, float("nan"), float("inf"), 2])
def test_invalid_refinement_cannot_reinterpret_a_checkpoint(new_dt):
    with pytest.raises(ValueError):
        WINDOW["rephase_half_velocity"]((0, 1), 0, 1, new_dt)


def test_comparator_requires_every_fine_scalar_step_on_the_canonical_clock():
    compare = runpy.run_path(str(TOOLS / "compare_fem_timestep_window.py"))
    summary = dict(
        first_shear_step=100, last_shear_step=103, physical_shear_interval_s=[0.1, 0.103]
    )
    record = dict(factor=4, dt_s=0.00025)
    rows = np.ones((12, 17))
    clock, result = compare["scalar_clock"]({"all_scalar_history": rows}, summary, record)
    np.testing.assert_array_equal(clock, 0.1 + np.arange(1, 13) * 0.00025)
    assert result is rows
    with pytest.raises(ValueError, match="every"):
        compare["scalar_clock"]({"all_scalar_history": rows[::2]}, summary, record)

import runpy
from pathlib import Path

import numpy as np
import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/compare_fronts.py"))


def test_saved_frame_ties_are_preserved_instead_of_claiming_an_exact_first_location():
    time = np.array([0.0, 1.0, 2.0])
    cumulative = np.array([[0, 0, 0], [0, 0, 0], [0.8, 1, 0]])
    passages = TOOL["arrivals"](time, cumulative, 0.4)
    record = TOOL["earliest_locations"](np.array([0.0, 2, 4]), passages)
    assert record["first_saved_locations_mm"] == [0, 2]
    assert record["first_saved_bracket_s"] == [1, 2]
    assert record["minimum_interpolated_locations_mm"] == [2]
    assert record["minimum_interpolated_time_s"] == 1.4


def test_common_grid_can_miss_an_earliest_native_interior_node():
    time, y = np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])
    cumulative = np.array([[0, 0, 0], [0, 0.6, 0]])
    native = TOOL["earliest_locations"](y, TOOL["arrivals"](time, cumulative, 0.5))
    common = TOOL["earliest_locations"](
        y[[0, 2]], TOOL["arrivals"](time, cumulative[:, [0, 2]], 0.5)
    )
    assert native["first_saved_locations_mm"] == [1]
    assert common["first_saved_time_s"] is None
    assert common["first_saved_locations_mm"] == []


def test_nonfinite_cumulative_field_is_not_reported_as_an_unreached_front():
    with pytest.raises(ValueError, match="saved cumulative"):
        TOOL["arrivals"]([0, 1], np.array([[0.0], [np.nan]]), 0.5)

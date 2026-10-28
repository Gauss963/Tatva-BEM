import runpy
from pathlib import Path

import h5py
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
OBS = runpy.run_path(str(TOOLS / "event_observables.py"))
COMPARE = runpy.run_path(str(TOOLS / "compare_event_observables.py"))


def interval(lower, upper):
    return dict(lower_s=lower, upper_s=upper)


def test_crossings_keep_strict_threshold_and_censoring_without_fitting():
    passages = OBS["crossing_brackets"]([1, 2, 3], [[0, 0.6, 0], [0.5, 0.8, 0], [1, 1, 0]], 0.5)
    assert passages == [
        dict(state="bounded", lower_s=2, upper_s=3),
        dict(state="censored", lower_s=None, upper_s=1),
        dict(state="unreached", lower_s=3, upper_s=None),
    ]
    assert OBS["relative_interval_gate"](passages[0], passages[1], 0.02)["status"] == "inconclusive"


def test_float32_rounding_expands_crossing_bracket_when_threshold_is_inside_cell():
    values = np.array([[0], [0.5], [1]], dtype=np.float32)
    bounds = OBS["storage_bounds"](values)
    assert bounds[0][1, 0] < 0.5 < bounds[1][1, 0]
    quantized = OBS["crossing_brackets"]([0, 1, 2], values, 0.5, bounds=bounds)
    exact = OBS["crossing_brackets"]([0, 1, 2], values.astype(float), 0.5)
    assert quantized[0] == dict(state="bounded", lower_s=0, upper_s=2)
    assert exact[0] == dict(state="bounded", lower_s=1, upper_s=2)
    # A last row inside the rounding cell has no certain crossing.
    last = OBS["crossing_brackets"]([0, 1], values[:2], 0.5, bounds=tuple(v[:2] for v in bounds))
    assert last[0] == dict(state="censored", lower_s=0, upper_s=None)


def test_saved_frame_ties_keep_all_possible_native_first_locations():
    p = OBS["crossing_brackets"]([0, 1, 2], [[0, 0, 0], [0, 0, 0], [0.8, 1, 0]], 0.4)
    first = OBS["first_event"]([0, 2, 4], p)
    assert first["possible_first_native_locations_mm"] == [0, 2]
    assert [first[k] for k in ["lower_s", "upper_s"]] == [1, 2]
    # Do not ignore a location that could have crossed before the first sample.
    p[2] = dict(state="censored", lower_s=None, upper_s=2)
    assert OBS["first_event"]([0, 2, 4], p)["lower_s"] is None


def test_native_first_event_is_not_replaced_by_a_coarse_common_grid():
    time, y = [0, 1], [0, 1, 2]
    field = [[0, 0, 0], [0, 0.6, 0]]
    native = OBS["first_event"](y, OBS["crossing_brackets"](time, field, 0.5))
    common = OBS["sample_columns"](y, field, [0, 2])
    assert native["possible_first_native_locations_mm"] == [1]
    assert (
        OBS["first_event"]([0, 2], OBS["crossing_brackets"](time, common, 0.5))["state"]
        == "unreached"
    )


def test_location_disagreement_retains_ties_instead_of_choosing_matching_node():
    result = OBS["location_difference"](
        dict(possible_first_native_locations_mm=[0, 2]),
        dict(possible_first_native_locations_mm=[2, 4]),
    )
    assert result == dict(minimum_possible_distance_mm=0, maximum_possible_distance_mm=4)


def test_relative_gate_does_not_accept_midpoint_agreement_or_zero_denominator():
    gate = OBS["relative_interval_gate"]
    assert gate(interval(9, 11), interval(9, 11), 0.02)["status"] == "inconclusive"
    assert gate(interval(10, 10.01), interval(10.02, 10.03), 0.02)["status"] == "pass"
    assert gate(interval(10, 10.01), interval(10.5, 10.6), 0.02)["status"] == "fail"
    assert gate(interval(0, 0), interval(0, 0), 0.02)["status"] == "inconclusive"


def test_loading_latency_agreement_cannot_hide_propagation_duration_error():
    gate, travel = OBS["relative_interval_gate"], OBS["gauge_travel"]
    r = [interval(0.020, 0.020), interval(0.0201, 0.0201)]
    a = [interval(0.020, 0.020), interval(0.0202, 0.0202)]
    assert all(gate(rr, aa, 0.02)["status"] == "pass" for rr, aa in zip(r, a, strict=True))
    rt, at = (travel(0, 100, *p) for p in [r, a])
    assert rt["speed_interval_m_per_s"] == pytest.approx([1000, 1000])
    assert at["speed_interval_m_per_s"] == pytest.approx([500, 500])
    assert COMPARE["duration_gate"](rt, at, 0.05)["status"] == "fail"


def test_travel_reports_unresolved_and_opposite_directions_without_false_speed():
    travel = OBS["gauge_travel"]
    unresolved = travel(0, 100, interval(1, 2), interval(1, 2))
    assert unresolved["speed_interval_m_per_s"] is None
    forward = travel(0, 100, interval(1, 1), interval(2, 2))
    backward = travel(0, 100, interval(2, 2), interval(1, 1))
    assert backward["speed_interval_m_per_s"] == [0.1, 0.1]
    assert COMPARE["duration_gate"](forward, backward, 0.05)["status"] == "fail"


def test_waveform_norms_retain_absolute_offset_and_zero_reference_behavior():
    error = OBS["waveform_error"]([0, 1, 2], [0, 1, 2], [0, 0.5, 2], [2, 2.5, 4])
    assert error["absolute_signal"]["absolute_time_p1_l2"] == pytest.approx(np.sqrt(8))
    assert error["absolute_signal"]["relative_time_p1_l2"] == pytest.approx(np.sqrt(3))
    assert error["increment_from_first_shared_time"]["absolute_time_p1_l2"] == 0
    zero = OBS["waveform_error"]([0, 1], [0, 0], [0, 1], [1, 1])
    assert zero["absolute_signal"]["relative_time_p1_l2"] is None
    assert zero["absolute_signal"]["absolute_time_p1_l2"] == 1


@pytest.mark.parametrize("field", [[[0], [np.nan]], [[0], [-1]], [[1], [0.9]]])
def test_invalid_cumulative_state_is_rejected_instead_of_clipped(field):
    with pytest.raises(ValueError):
        OBS["crossing_brackets"]([0, 1], field, 0.5)


def test_no_gauge_extrapolation_or_nonfinite_waveform():
    with pytest.raises(ValueError, match="extrapolation"):
        OBS["sample_columns"]([0, 1], [[0, 1]], [2])
    with pytest.raises(ValueError, match="finite"):
        OBS["waveform_error"]([0, 1], [0, np.nan], [0, 1], [0, 1])


def test_complete_synthetic_event_exercises_thresholds_gauges_and_waveforms():
    time = np.arange(101) / 1000
    y = np.array([0, 120, 240, 360, 480.0])
    field = np.maximum(time[:, None] - 0.01 - y / 1e5, 0)
    event = dict(
        time=time,
        y=y,
        cumulative=field,
        fields=dict(pressure=10 + field, elastic_shear=2 + field),
        stop=dict(state="bounded", lower_s=0.03, upper_s=0.03),
    )
    record, tables = COMPARE["compare_events"](event, event)
    assert set(tables) == {"micro", "Dc10", "Dc"}
    assert record["loading_stop"]["comparison"]["status"] == "pass"
    assert record["gauge_positions_mm"] == [48, 240, 432]
    assert (
        record["contact_gauge_waveforms"]["pressure"][0]["absolute_signal"]["absolute_time_p1_l2"]
        == 0
    )
    # Identical saved traces still cannot resolve sub-frame propagation to 5%.
    propagation = record["thresholds"]["micro"]["gauge_pair_propagation"]
    assert propagation[0]["duration_comparison"]["status"] == "inconclusive"


def test_legacy_reader_uses_integer_shear_clock_and_categorical_stop_only(tmp_path):
    path = tmp_path / "event.h5"
    summary = dict(summary=dict(dt=0.1, pressure_steps=3, shear_steps=4))
    with h5py.File(path, "w") as stream:
        stream.attrs["saved_interface_frames"] = 3
        group = stream.create_group("interface_high_rate")
        group["history_columns"] = np.array(["time", "shear_loading_stopped"], dtype="S")
        group["history"] = np.array([[0.3, 0], [0.5, 0], [0.7, 1]], dtype=np.float32)
        group["phase_id"] = np.array([1, 2, 2], dtype=np.int32)
        group["step_id"] = np.array([3, 2, 4], dtype=np.int32)
        group["contact_line_y"] = np.array([0, 480], dtype=np.float32)
        group["cumulative_slip"] = np.array([[0, 0], [0, 0], [1, 1]], dtype=np.float32)
        event = COMPARE["read_fem_event"](stream, summary, legacy=True)
        np.testing.assert_array_equal(event["time"], [0.2, 0.4])
        assert event["stop"] == dict(state="bounded", lower_s=0.2, upper_s=0.4)
        assert event["fields"] == {}
        assert event["cumulative"].dtype == np.float32
        # A rounded clock is verified; changing an actual saved time is rejected.
        group["history"][1, 0] = 0.6
        with pytest.raises(ValueError, match="timestamps"):
            COMPARE["read_fem_event"](stream, summary, legacy=True)


@pytest.mark.parametrize("float32_shear", [False, True])
def test_float64_reader_keeps_signed_endpoint_shear_and_excludes_normal_handoff(
    tmp_path, float32_shear
):
    base = runpy.run_path(str(Path(__file__).with_name("test_float64_contact_history.py")))
    with h5py.File(tmp_path / "endpoint.h5", "w") as stream:
        summary = base["observations"](stream)
        high = stream["interface_high_rate"]
        high["history_columns"] = np.array(["time", "shear_loading_stopped"], dtype="S")
        high["history"] = np.array([[0, 0], [0.3, 0], [0.4, 0], [0.7, 1]], dtype=np.float32)
        if float32_shear:
            group = stream["contact_observations"]
            del group["elastic_shear"]
            group["elastic_shear"] = np.full((4, 2), -3, dtype=np.float32)
            with pytest.raises(ValueError, match="true float64"):
                COMPARE["read_fem_event"](stream, summary, legacy=False)
        else:
            event = COMPARE["read_fem_event"](stream, summary, legacy=False)
            np.testing.assert_array_equal(event["time"], [0.1, 0.4])
            np.testing.assert_array_equal(event["cumulative"], np.full((2, 2), 0.2))
            np.testing.assert_array_equal(event["fields"]["elastic_shear"], np.full((2, 2), -3.0))
            assert event["stop"] == dict(state="bounded", lower_s=0.1, upper_s=0.4)

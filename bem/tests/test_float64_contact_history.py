import runpy
from pathlib import Path

import h5py
import numpy as np
import pytest

TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/compare_float64_contact_history.py")
)


def observations(stream):
    phases = np.array([0, 1, 2, 2], dtype="i4")
    steps = np.array([0, 3, 1, 4], dtype="i8")
    absolute = steps + np.where(phases == 2, 3, 0)
    group = stream.create_group("contact_observations")
    group.attrs.update(schema_version=1, written_frames=4)
    stream.attrs["saved_interface_frames"] = 4
    for name, value in dict(
        phase_id=phases,
        step_id=steps,
        absolute_step=absolute,
        time_s=absolute * 0.1,
        y=np.array([0.0, 1.0]),
    ).items():
        group[name] = value
    high = stream.create_group("interface_high_rate")
    high["phase_id"], high["step_id"] = phases, steps
    for name, value in dict(
        pressure=10.0,
        elastic_shear=-3.0,
        plastic_slip=0.1,
        cumulative_slip=0.2,
        friction_coefficient=0.8,
        friction_strength=8.0,
    ).items():
        group[name] = np.full((4, 2), value)
    # This pre-handoff value must never be used as the shear initial memory.
    group["cumulative_slip"][1] = 100.0
    return dict(summary=dict(dt=0.1, pressure_steps=3, shear_steps=4))


def test_integer_shear_clock_ignores_normal_rounding_and_pre_handoff_state(tmp_path):
    with h5py.File(tmp_path / "observations.h5", "w") as stream:
        summary = observations(stream)
        indices, time, y = TOOL["observation_grid"](stream, summary)
        np.testing.assert_array_equal(indices, [2, 3])
        np.testing.assert_array_equal(time, [0.1, 0.4])
        np.testing.assert_array_equal(y, [0.0, 1.0])


def test_signed_shear_is_compared_independently_of_strength(tmp_path):
    with h5py.File(tmp_path / "comparison.h5", "w") as stream:
        fem = observations(stream)
        data = dict(time=np.arange(5) * 0.1, contact_y=np.array([0.0, 1.0]))
        for name, value in dict(
            pressure=10.0,
            shear=-2.0,
            plastic_slip=0.1,
            cumulative_slip=0.2,
            friction_coefficient=0.8,
        ).items():
            data[name] = np.full((5, 2), value)
        result = TOOL["compare_fields"](
            stream, fem, data, dict(completed_steps=4, dt_s=0.1), row_chunk=1
        )
        assert result["elastic_shear"]["relative_space_time_p1_l2"] == pytest.approx(1 / 3)
        assert result["elastic_shear"]["maximum_saved_interpolant_absolute_difference"] == 1
        for key in TOOL["FIELDS"]:
            assert result[key]["common_time_interval_s"] == [0.1, 0.4]
            assert result[key]["extrapolation_used"] is False
            if key != "elastic_shear":
                assert result[key]["relative_space_time_p1_l2"] == pytest.approx(0, abs=1e-15)


@pytest.mark.parametrize("corruption", ["float32", "time", "incomplete", "last_step"])
def test_rejects_rounded_clock_or_incomplete_contact_observations(tmp_path, corruption):
    with h5py.File(tmp_path / "invalid.h5", "w") as stream:
        summary = observations(stream)
        group = stream["contact_observations"]
        if corruption == "float32":
            del group["elastic_shear"]
            group["elastic_shear"] = np.zeros((4, 2), dtype="f4")
        elif corruption == "time":
            group["time_s"][2] += 1e-8
        elif corruption == "incomplete":
            group.attrs["written_frames"] = 3
        else:
            summary["summary"]["shear_steps"] = 5
        with pytest.raises(ValueError):
            TOOL["observation_grid"](stream, summary)

import runpy
from pathlib import Path

import h5py
import numpy as np
import pytest

READ = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/fem_reference_io.py"))[
    "read_fem_trace"
]


def create_trace(group, phase, *, slip):
    group["history_columns"] = np.array([b"time", b"max_slip"])
    group["history"] = np.c_[np.linspace(0, 0.003, len(phase)), slip]
    group["phase_id"] = phase
    group["step_id"] = np.arange(len(phase))


def test_high_rate_trace_selected_with_its_own_columns_and_fields(tmp_path):
    path = tmp_path / "fem.h5"
    with h5py.File(path, "w") as f:
        create_trace(f, [0, 1, 2], slip=[0, 9, 0.1])
        f.attrs["saved_frames"] = 3
        f.create_group("interface")
        high = f.create_group("interface_high_rate")
        create_trace(high, [0, 1, 2, 2, 2], slip=[0, 9, 0, 0.05, 0.1])
        high["contact_line_y"] = [0, 1]
        high["cumulative_slip"] = np.arange(10).reshape(5, 2)
        f.attrs["saved_interface_frames"] = 5
    trace = READ(path, fields=["cumulative_slip"])
    assert trace["provenance"]["history_group"] == "interface_high_rate"
    np.testing.assert_array_equal(trace["history"]["max_slip"], [0, 9, 0, 0.05, 0.1])
    np.testing.assert_array_equal(
        trace["interface_fields"]["cumulative_slip"], np.arange(10).reshape(5, 2)
    )


def test_legacy_trace_excludes_rounded_normal_end_by_phase_not_time(tmp_path):
    path = tmp_path / "fem.h5"
    with h5py.File(path, "w") as f:
        create_trace(f, [1, 2], slip=[9, 0])
        f["history"][0, 0] = np.float32(0.002)  # Slightly above float64 0.002.
        f.attrs["saved_frames"] = 2
        f.create_group("interface")
    trace = READ(path)
    assert trace["history"]["time"][0] > 0.002
    np.testing.assert_array_equal(trace["history"]["max_slip"][trace["phase_id"] == 2], [0])
    assert trace["provenance"]["history_group"] == "/"


def test_unwritten_allocated_frames_are_never_treated_as_complete(tmp_path):
    path = tmp_path / "partial.h5"
    with h5py.File(path, "w") as f:
        create_trace(f, [1, 2], slip=[0, 0])
        f.attrs["saved_frames"] = 1
    with pytest.raises(ValueError, match="complete"):
        READ(path)

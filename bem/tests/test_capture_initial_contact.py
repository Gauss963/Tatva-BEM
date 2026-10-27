import hashlib
import json
import runpy
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
TOOL = runpy.run_path(str(TOOLS / "capture_initial_contact.py"))


@pytest.mark.parametrize("compressed", [False, True])
def test_initial_row_keeps_float_bits_without_loading_all_time_rows(tmp_path, compressed):
    a = np.arange(120, dtype=float).reshape(10, 12)
    a[0, 0], a[0, 1] = -0.0, np.nextafter(0.0, 1.0)
    path = tmp_path / "trajectory.npz"
    save = np.savez_compressed if compressed else np.savez
    save(path, pressure=a)
    actual = TOOL["first_npz_row"](path, "pressure")
    np.testing.assert_array_equal(actual.view(np.uint64), a[0].view(np.uint64))


def test_initial_row_rejects_an_unexpected_layout_or_incomplete_member(tmp_path):
    path = tmp_path / "trajectory.npz"
    np.savez(path, pressure=np.asfortranarray(np.ones((4, 3))))
    with pytest.raises(ValueError, match="C-order float64"):
        TOOL["first_npz_row"](path, "pressure")
    np.savez(path, pressure=np.ones((4, 3)))
    with zipfile.ZipFile(path) as source:
        data = source.read("pressure.npy")
    with zipfile.ZipFile(path, "w") as target:
        target.writestr("pressure.npy", data[:-8])
    with pytest.raises(ValueError, match="member size"):
        TOOL["first_npz_row"](path, "pressure")


def static_file(path, dtype=np.float32):
    """Hand-constructed p=10, tau=2, plastic offset=3 on two contact pairs."""
    with h5py.File(path, "w") as stream:
        stream["phase_id"], stream["step_id"] = [0, 1, 2], [0, 4, 1]
        stream["history_columns"] = np.array(
            ["time", "loading_face_displacement", "applied_shear_displacement"], dtype="S32"
        )
        stream["history"] = np.array([[0, 0, 0], [0.004, 3.4, 3.5], [0.005, 3.4, 3.5]], dtype=dtype)
        for name, x, y in [("moving", 1, 3.4), ("stationary", 0, 0)]:
            group = stream.create_group(name)
            group["coords"] = np.array([[1, 0], [1, 1]], dtype=dtype)
            group["displacement"] = np.tile(np.array([x, y], dtype=dtype), (3, 2, 1))
        group = stream.create_group("interface")
        group["contact_line_y"] = np.array([0, 1], dtype=dtype)
        group["master_nodes"], group["slave_nodes"] = [0, 1], [0, 1]
        group["plastic_slip"] = np.full((3, 2), 3, dtype=dtype)
        high = stream.create_group("interface_high_rate")
        high["phase_id"], high["step_id"] = [1, 2], [4, 1]
        high["contact_line_y"] = np.array([0, 1], dtype=dtype)
        for key, value in [
            ("plastic_slip", 3),
            ("cumulative_slip", 0),
            ("friction_coefficient", 0.8),
            ("friction_strength", 8),
        ]:
            high[key] = np.full((2, 2), value, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_static_frame_reconstructs_the_hand_computed_contact_law_with_storage_bounds(
    tmp_path, dtype
):
    path = tmp_path / "simulation.h5"
    static_file(path, dtype)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    fields, record = TOOL["read_static_frame"](path, 10, 5, expected_step=4)
    assert record["frame"] == 1 and record["high_rate_frame"] == 0
    # Subtracting rounded displacement/offset values amplifies float32
    # storage error; compare to the independently known law using its bound.
    assert np.all(abs(fields["pressure"] - 10) <= fields["pressure_storage_bound"])
    assert np.all(abs(fields["shear"] - 2) <= fields["shear_storage_bound"])
    budget = 1e-5 if dtype == np.float32 else 1e-12
    assert np.max(fields["pressure_storage_bound"]) < budget
    assert np.max(fields["shear_storage_bound"]) < budget
    assert record["stored_dtypes"]["moving_u"] == np.dtype(dtype).name
    assert fields["pressure"].dtype == np.float64
    assert before == hashlib.sha256(path.read_bytes()).hexdigest()


def test_static_frame_catches_wrong_phase_endpoint_and_inconsistent_pressure(tmp_path):
    path = tmp_path / "simulation.h5"
    static_file(path)
    with pytest.raises(ValueError, match="normal-phase endpoint"):
        TOOL["read_static_frame"](path, 10, 5, expected_step=3)
    with h5py.File(path, "r+") as stream:
        stream["interface_high_rate/friction_strength"][0] = 20
    with pytest.raises(ValueError, match="exceed storage bounds"):
        TOOL["read_static_frame"](path, 10, 5)


def test_cli_captures_portable_saved_static_artifacts(tmp_path, monkeypatch):
    fem, output = tmp_path / "fem", tmp_path / "capture"
    fem.mkdir()
    static_file(fem / "simulation.h5")
    manifest = fem / "reference_manifest.json"
    manifest.write_text("{}")
    summary = dict(
        status="completed",
        full_original_horizon=True,
        source_and_protected_input_unchanged=True,
        computational_dtype="float64",
        summary=dict(pressure_steps=4),
        reference_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    config = dict(
        dtype="float64",
        normal_phase_mode="static",
        healing_fraction=1,
        reset_slip_weakening_at_shear_start=True,
        normal_penalty=10,
        tangential_penalty=5,
    )
    (fem / "summary.json").write_text(json.dumps(summary))
    (fem / "resolved_run_config.json").write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["capture", str(fem), "--output", str(output)])
    TOOL["main"]()
    result = json.loads((output / "summary.json").read_text())
    assert result["status"] == "captured_saved_static_frames_with_storage_bounds"
    assert result["records"][0]["maximum_pressure_storage_bound_mpa"] > 0
    assert not result["same_accuracy_fem_advantage_verified"]
    with np.load(result["records"][0]["snapshot"], allow_pickle=False) as data:
        assert data["pressure"].dtype == np.float64

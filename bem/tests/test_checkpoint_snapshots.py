import json
import runpy
from pathlib import Path

import h5py
import numpy as np
import pytest

TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/run_checkpointed_fem_reference.py")
)


def output_fixture(path):
    metadata = dict(dt=0.25, pressure_steps=10, frame_count=2, interface_frame_count=2)
    with h5py.File(path, "w") as stream:
        for group in [stream, stream.create_group("interface_high_rate")]:
            group.create_dataset("phase_id", data=[2, 2])
            group.create_dataset("step_id", data=[1, 3])
            group.create_dataset("history", data=[[2.75, 1.0], [3.25, 2.0]], compression="lzf")
        group = stream.create_group("contact_observations")
        group.create_dataset("time_s", data=[2.75, 3.25])
        group.create_dataset("absolute_step", data=[11, 13])
        for name in [
            "pressure",
            "elastic_shear",
            "plastic_slip",
            "cumulative_slip",
            "friction_coefficient",
            "friction_strength",
        ]:
            group.create_dataset(name, data=np.ones((2, 2)))
    return metadata


def test_snapshot_survives_later_compressed_output_corruption(tmp_path):
    path = tmp_path / "simulation.h5"
    metadata = output_fixture(path)
    checkpoint = tmp_path / "checkpoint.npz"
    np.savez(checkpoint, metadata=json.dumps(metadata), carry_0=np.arange(7.0))
    root = tmp_path / "snapshots"
    with h5py.File(path, "r+") as stream:
        record = TOOL["paired_snapshot"](stream, checkpoint, root, metadata)
        stream["interface_high_rate/history"].id.write_direct_chunk((0, 0), b"\x01")
        stream.flush()
        with pytest.raises(OSError):
            stream["interface_high_rate/history"][:]
    snapshot = Path(record["directory"])
    with h5py.File(snapshot / "simulation.h5", "r") as stream:
        TOOL["validate_clocks"](stream, metadata)
        np.testing.assert_array_equal(stream["history"][:], [[2.75, 1], [3.25, 2]])
    assert TOOL["digest"](snapshot / "checkpoint.npz") == TOOL["digest"](checkpoint)
    assert json.loads((root / "latest.json").read_text())["current"] == snapshot.name


def test_silent_scalar_clock_damage_is_rejected_before_snapshot_publication(tmp_path):
    path = tmp_path / "simulation.h5"
    metadata = output_fixture(path)
    checkpoint = tmp_path / "checkpoint.npz"
    np.savez(checkpoint, metadata=json.dumps(metadata))
    root = tmp_path / "snapshots"
    with h5py.File(path, "r+") as stream:
        stream["history"][:] = 0.00033
        with pytest.raises(ValueError, match="damaged"):
            TOOL["paired_snapshot"](stream, checkpoint, root, metadata)
    assert not (root / "latest.json").exists()


def test_complete_state_equivalence_includes_velocity_and_every_internal_state(tmp_path):
    values = {"carry_" + str(i): np.array([float(i), i + 0.5]) for i in range(12)}
    expected, actual = tmp_path / "expected.npz", tmp_path / "actual.npz"
    np.savez(expected, **values)
    np.savez(actual, **values)
    assert all(TOOL["compare_state"](actual, expected).values())
    values["carry_1"][0] += 1e-14
    np.savez(actual, **values)
    with pytest.raises(ValueError, match="state differs"):
        TOOL["compare_state"](actual, expected)


def test_forced_capture_occurs_after_both_output_writers_and_preserves_math(tmp_path):
    source = """def run_simulation_dumped(value):
    checkpoint_path = "checkpoint"
    interval_due = deadline_reached = signal_due = phase_complete = False
    phase_id, stop = 2, 3
    model, interface_plot_mask = {}, []
    def save_high_rate_interface(h5, frame_idx, carry, history_row, *, phase_id, step_id):
        h5.append("high")
    events = []
    value = value * 1.25 + 3
    save_high_rate_interface(events, 0, value, None, phase_id=phase_id, step_id=stop)
    events.append("bulk")
    if checkpoint_path is not None and (interval_due or deadline_reached or signal_due or phase_complete):
        _write_simulation_checkpoint(checkpoint_path, value, {})
    return value, events
"""
    path = tmp_path / "writer_order.py"
    path.write_text(source)
    namespace = {}
    exec(compile(source, str(path), "exec"), namespace)
    original = namespace["run_simulation_dumped"]
    streams = []

    def observe(stream, *args):
        streams.append(stream)
        stream.append("observe")

    adapted = TOOL["instrument_checkpoint_writer"](
        original, observe, lambda *args: streams[-1].append("checkpoint"), lambda *args: True
    )
    assert original(4) == (8, ["high", "bulk"])
    assert adapted(4) == (8, ["high", "observe", "bulk", "checkpoint"])
    assert namespace["run_simulation_dumped"] is original

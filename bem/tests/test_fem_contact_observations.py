import runpy
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/fem_contact_observations.py")
)


def source_state():
    model = dict(
        dimension=2,
        friction_law="slip-weakening",
        moving_offset=4,
        master_nodes=np.array([0, 1]),
        slave_nodes=np.array([0, 1]),
        interface_initial_gap=np.array([0.05, 0]),
        penalty_n=1000.0,
        penalty_t=100.0,
        interface_weights=np.array([0.5, 0.5]),
        pressure_steps=3,
        dt=0.125000000000003,
        moving=SimpleNamespace(mesh=SimpleNamespace(coords=np.array([[0.0, 0.0], [0.0, 1.0]]))),
    )
    u = np.array([0.2, 0.5, -0.1, 2.0, 0.0, 0.1, 0.0, 3.0])
    z, c, mu, strength = (np.array(v) for v in [[0.2, 0.1], [0.1, 0.2], [0.8, 0.8], [120.0, 0.0]])
    carry = (u, None, z, c, None, None, mu, None, strength)
    for value in [u, z, c, mu, strength]:
        value.flags.writeable = False
    return model, carry


def test_exact_endpoint_observations_keep_state_and_zero_open_contact(tmp_path):
    model, carry = source_state()
    before = [a.copy() for a in carry if a is not None]
    with h5py.File(tmp_path / "observations.h5", "w") as h5:
        h5.create_dataset("interface_high_rate/history", shape=(3, 17), dtype="f4")
        observer = TOOL["ContactObserver"]()
        observer(h5, 0, carry, 2, 5, model, np.ones(2, bool))
        group = h5["contact_observations"]
        np.testing.assert_allclose(group["pressure"][0], [150, 0], rtol=1e-14)
        np.testing.assert_allclose(group["elastic_shear"][0], [20, 0], rtol=1e-14)
        assert group["time_s"][0] == 8 * model["dt"]
        assert group["absolute_step"][0] == 8
        assert all(group[k].dtype == np.float64 for k in TOOL["FIELDS"])
        with pytest.raises(ValueError, match="duplicated or out of order"):
            observer(h5, 0, carry, 2, 5, model, np.ones(2, bool))
        assert group.attrs["written_frames"] == 1
    for a, b in zip(before, [a for a in carry if a is not None], strict=True):
        np.testing.assert_array_equal(a, b)


def test_resume_rewinds_uncommitted_observation_rows_to_source_checkpoint(tmp_path):
    model, carry = source_state()
    with h5py.File(tmp_path / "resume.h5", "w") as h5:
        h5.create_dataset("interface_high_rate/history", shape=(3, 17), dtype="f4")
        observer = TOOL["ContactObserver"]()
        for i in range(3):
            observer(h5, i, carry, 2, i + 1, model, np.ones(2, bool))
        before = {k: h5["contact_observations/" + k][:].copy() for k in TOOL["FIELDS"]}
        h5.attrs.update(resumed_from_checkpoint=1, saved_interface_frames=1)
        resumed = TOOL["ContactObserver"]()
        resumed(h5, 1, carry, 2, 2, model, np.ones(2, bool))
        assert h5["contact_observations"].attrs["written_frames"] == 2
        resumed(h5, 2, carry, 2, 3, model, np.ones(2, bool))
        for key, value in before.items():
            np.testing.assert_array_equal(h5["contact_observations/" + key][:], value)


def test_observer_rejects_upcast_float32_state(tmp_path):
    model, carry = source_state()
    invalid = (carry[0].astype(np.float32), *carry[1:])
    with h5py.File(tmp_path / "invalid.h5", "w") as h5:
        with pytest.raises(ValueError, match="true float64"):
            TOOL["ContactObserver"]()(h5, 0, invalid, 2, 1, model, np.ones(2, bool))
        assert "contact_observations" not in h5


def test_writer_instrumentation_changes_only_the_observer_side_effect(tmp_path):
    source = """def run_simulation_dumped(value, *, marker=7):
    model = {"value": value}
    interface_plot_mask = [True]
    def save_high_rate_interface(h5, frame_idx, carry, history_row, *, phase_id, step_id):
        h5.append(("original", frame_idx))
    output = []
    save_high_rate_interface(output, 0, (), None, phase_id=2, step_id=3)
    return marker, output
"""
    path = tmp_path / "writer_fixture.py"
    path.write_text(source)
    namespace = {}
    exec(compile(source, str(path), "exec"), namespace)
    original = namespace["run_simulation_dumped"]
    seen = []
    observed = TOOL["_instrument_writer"](original, lambda *args: seen.append(args))
    assert observed(3) == original(3) == (7, [("original", 0)])
    assert len(seen) == 1 and seen[0][3:5] == (2, 3)
    assert namespace["run_simulation_dumped"] is original
    with pytest.raises(ValueError, match="pinned 933ae1f"):
        TOOL["make_observed_runner"](original)

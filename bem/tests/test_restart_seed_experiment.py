import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/restart_seed_experiment.py"))


def seed(tmp_path, **changes):
    case = ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    model = load_nodal_model(case, max_panel_length=100)
    data = np.genfromtxt(
        ROOT / "src/finiteblock_bem/resources/data/ev16_restart_interface.csv",
        delimiter=",",
        names=True,
    )
    arrays = dict(
        y=data["y"],
        plastic_slip=data["plastic_slip"] + 1e-7,
        cumulative_slip=np.zeros(len(data)),
        actuator=np.asarray(model.reference.restart_actuator),
    )
    arrays.update(changes)
    path = tmp_path / "seed.npz"
    np.savez(path, **arrays)
    return model, path


def test_fixed_knot_seed_changes_state_but_preserves_actual_bulk_transfer(tmp_path):
    model, path = seed(tmp_path)
    updated, record = TOOL["apply_restart_seed"](model, path)
    assert updated.reference is model.reference and updated.constraints is model.constraints
    assert record["maximum_plastic_change_from_packaged_mm"] > 0.9e-7
    np.testing.assert_allclose(
        updated.restart.plastic_slip - model.restart.plastic_slip, 1e-7, atol=1e-14
    )
    original = NodalInterfaceTransfer(model, quadrature=6).evaluate(100 + 2e4j)
    actual = NodalInterfaceTransfer(updated, quadrature=6).evaluate(100 + 2e4j)
    np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(updated.restart.cumulative_slip, 0)


@pytest.mark.parametrize(
    "change,match",
    [("knots", "knots"), ("memory", "healed"), ("actuator", "actuator"), ("dtype", "float64")],
)
def test_seed_cannot_change_geometry_memory_actuator_or_precision(tmp_path, change, match):
    model, path = seed(tmp_path)
    with np.load(path, allow_pickle=False) as stream:
        data = {k: stream[k] for k in stream.files}
    if change == "knots":
        data["y"][1] += 1e-10
    elif change == "memory":
        data["cumulative_slip"][0] = 1e-12
    elif change == "actuator":
        data["actuator"] = data["actuator"] + 1e-12
    else:
        data["plastic_slip"] = data["plastic_slip"].astype(np.float32)
    np.savez(path, **data)
    with pytest.raises(ValueError, match=match):
        TOOL["apply_restart_seed"](model, path)


def test_seed_provenance_must_identify_the_same_arrays(tmp_path):
    model, path = seed(tmp_path)
    path.with_suffix(".json").write_text(json.dumps(dict(seed_file_sha256="another-file")))
    with pytest.raises(ValueError, match="different seed"):
        TOOL["apply_restart_seed"](model, path)


def test_export_rejects_the_wrong_precision_reference_before_opening_source_hdf5(
    tmp_path, monkeypatch
):
    function = TOOL["export_original_restart"]
    monkeypatch.setitem(function.__globals__, "ROOT", tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            dict(
                status="completed",
                full_original_horizon=True,
                source_and_protected_input_unchanged=True,
                computational_dtype="float32",
            )
        )
    )
    (run / "resolved_run_config.json").write_text(json.dumps(dict(dtype="float32")))
    output = tmp_path / "outputs/seed"
    with pytest.raises(ValueError, match="original-precision"):
        function(run, output)
    assert not output.exists()

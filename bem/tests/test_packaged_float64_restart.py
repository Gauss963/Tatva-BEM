import hashlib
import io
import json
import runpy
from importlib.resources import files
from pathlib import Path

import numpy as np

from finiteblock_bem.nodal import load_nodal_model


def test_packaged_original_restart_preserves_float64_fields_and_provenance():
    folder = files("finiteblock_bem").joinpath("resources/data")
    source = folder.joinpath("ev16_restart_float64.npz")
    record = json.loads(folder.joinpath("ev16_restart_float64.json").read_text())
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record["seed_file_sha256"]
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        assert set(data.files) == {"y", "plastic_slip", "cumulative_slip", "actuator"}
        assert all(data[key].dtype == np.float64 for key in data.files)
        np.testing.assert_array_equal(data["y"], np.arange(0, 482, 2))
        assert np.all(data["cumulative_slip"] == 0)
        assert (
            hashlib.sha256(data["plastic_slip"].tobytes() + data["actuator"].tobytes()).hexdigest()
            == record["source_plastic_and_actuator_sha256"]
        )
        root = Path(__file__).resolve().parents[1]
        model = load_nodal_model(
            root / "src/finiteblock_bem/resources/cases/pmma_event16.toml",
            max_panel_length=160,
            contact_panel_length=160,
        )
        updated, provenance = runpy.run_path(str(root / "tools/restart_seed_experiment.py"))[
            "apply_restart_seed"
        ](model, Path(str(source)))
        y = model.reference.moving.starts[model.moving_contact, 1]
        np.testing.assert_array_equal(
            updated.restart.plastic_slip, np.interp(y, data["y"], data["plastic_slip"])
        )
        assert provenance["sha256"] == record["seed_file_sha256"]

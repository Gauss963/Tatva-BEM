import json
import runpy
import sys
import tomllib
from pathlib import Path

import numpy as np

from finiteblock_bem.io import environment_record

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def test_fresh_static_control_never_reads_source_dynamic_weights(tmp_path, monkeypatch):
    tool = runpy.run_path(str(TOOLS / "bem_graded_static.py"))
    source, output = tmp_path / "source", tmp_path / "static"
    source.mkdir()
    options = dict(
        case=str(CASE),
        h=160,
        contact_h=160,
        contact_end_h=160,
        corner_h=160,
        quadrature=4,
        restart_knot_window=None,
        restart_knot_subdivisions=1,
    )
    (source / "summary.json").write_text(
        json.dumps(
            dict(
                status="completed_resolvent_kernel_only",
                source_mesh_arguments=options,
                environment=environment_record(CASE),
            )
        )
    )
    (source / "weights.npy").write_bytes(b"not numerical data")
    (source / "work-resolvent.npy").write_bytes(b"not numerical data")
    case = tomllib.loads(CASE.read_text())
    data = np.genfromtxt(
        CASE.parent / case["source"]["restart_interface"], delimiter=",", names=True
    )
    seed = tmp_path / "seed.npz"
    np.savez(
        seed,
        y=data["y"],
        plastic_slip=data["plastic_slip"],
        cumulative_slip=np.zeros(len(data)),
        actuator=np.array(case["loading"]["restart_actuator"], dtype=float),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "static",
            str(source),
            "--restart-seed",
            str(seed),
            "--outer-h",
            "160",
            "--contact-h",
            "160",
            "--end-sizes",
            "80",
            "--output",
            str(output),
        ],
    )
    tool["main"]()
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "completed_graded_bem_static_controls"
    record = report["records"][0]
    assert record["initialization"]["original_full_balance_mpa"] < 1e-7
    assert record["initialization"]["static_initialization"]["weakening_memory_unchanged"]
    assert record["source_p1_projection"]["maximum_saved_absolute_difference"] >= 0
    with np.load(record["snapshot"], allow_pickle=False) as fields:
        assert np.all(fields["cumulative_slip"] == 0)
        assert np.all(np.isfinite(fields["pressure"]))
    assert (source / "weights.npy").read_bytes() == b"not numerical data"
    assert not report["same_accuracy_fem_advantage_verified"]

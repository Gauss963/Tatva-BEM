import json
import runpy
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import CQPlan, bdf2_symbol, invert_spectrum
from finiteblock_bem.io import environment_record
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def source_metadata(path):
    path.mkdir()
    options = dict(
        case=str(CASE),
        h=160,
        contact_h=160,
        contact_end_h=160,
        corner_h=160,
        quadrature=4,
        restart_knot_window=None,
        restart_knot_subdivisions=1,
        diagnostic_actuator_increment=None,
    )
    (path / "summary.json").write_text(
        json.dumps(
            dict(
                status="completed_resolvent_kernel_only",
                source_mesh_arguments=options,
                environment=environment_record(CASE),
                dt_s=1e-4,
                full_steps=4,
            )
        )
    )
    # These are intentionally not numerical data. The fresh producer must
    # never try to obtain an answer from an old spectrum or time weights.
    (path / "weights.npy").write_bytes(b"do not read source numerical data")
    (path / "work-resolvent.npy").write_bytes(b"do not read source numerical data")
    return options


@pytest.mark.parametrize("layout", ["c", "fortran"])
def test_fresh_raw_kernel_outputs_all_lags_and_causal_startup_against_dense_bem(
    tmp_path, monkeypatch, layout
):
    tool = runpy.run_path(str(TOOLS / "prepare_raw_layer_resolvent.py"))
    source, output = tmp_path / "source", tmp_path / "fresh"
    options = source_metadata(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "raw",
            str(source),
            "--output",
            str(output),
            "--max-workspace-mb",
            "65",
            "--weight-layout",
            layout,
            "--max-memory-mb",
            "1024",
            "--workers",
            "2",
            "--fft-rows",
            "2",
        ],
    )
    tool["main"]()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "completed_resolvent_kernel_only"
    plan = CQPlan(**summary["plan"])
    model = load_nodal_model(
        CASE,
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
    )
    operator = NodalInterfaceTransfer(model, quadrature=4)
    law = model.reference.contact
    weights = model.constraints.contact_weights
    line = np.r_[weights, weights, model.constraints.platen_width]
    diagonal = np.r_[weights * law.normal_penalty, weights * law.tangential_penalty, 0]
    spring = model.reference.spring_stiffness * line[-1]
    diagonal[-1] += spring
    root = np.sqrt(line)
    spectrum = []
    for s in (
        bdf2_symbol(plan.radius * np.exp(-1j * np.linspace(0, np.pi, plan.fft_size // 2 + 1)))
        / plan.dt
    ):
        z = operator.evaluate(s)
        spectrum.append(root[:, None] * np.linalg.solve(z + np.diag(diagonal), np.diag(root)))
    spectrum = np.array(spectrum)
    expected = invert_spectrum(spectrum.copy(), plan) / (root[:, None] * root[None, :])
    actual = np.load(output / "weights.npy", allow_pickle=False)
    assert summary["time_weights_layout"] == layout
    assert actual.flags.f_contiguous == (layout == "fortran")
    assert not (output / "weights-fortran.npy").exists()
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
    helper = runpy.run_path(str(TOOLS / "replay_resolvent.py"))
    time = np.arange(plan.steps + 1) * plan.dt
    elastic = helper["causal_elastic_baseline"](
        spectrum[:, :, -1] / np.sqrt(line * line[-1]),
        plan,
        spring * model.reference.actuator_rate * time,
    )
    with np.load(output / "elastic.npz") as data:
        np.testing.assert_array_equal(data["time"], time)
        np.testing.assert_allclose(data["elastic"], elastic, rtol=1e-8, atol=1e-12)
    assert summary["frequency_sampling"]["all_fft_grid_points_emitted"]
    assert summary["initial_composition_inf_error"] < 1e-6


@pytest.mark.parametrize("layout", ["c", "fortran"])
def test_preflight_uses_complete_storage_and_does_not_create_numerical_spectrum(
    tmp_path, monkeypatch, layout
):
    tool = runpy.run_path(str(TOOLS / "prepare_raw_layer_resolvent.py"))
    source, output = tmp_path / "source", tmp_path / "preflight"
    source_metadata(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "raw",
            str(source),
            "--output",
            str(output),
            "--preflight-only",
            "--max-workspace-mb",
            "65",
            "--weight-layout",
            layout,
            "--max-memory-mb",
            "1024",
        ],
    )
    tool["main"]()
    preflight = json.loads((output / "preflight.json").read_text())
    plan = CQPlan(**preflight["plan"])
    assert asdict(plan) == preflight["plan"]
    needed = (plan.fft_size // 2 + 1) * plan.dofs**2 * 16 + (plan.steps + 1) * plan.dofs**2 * 8
    assert preflight["required_disk_bytes"] > needed
    if layout == "fortran":
        assert preflight["required_disk_bytes"] > needed + (plan.steps + 1) * plan.dofs**2 * 8
        assert preflight["storage_conversion_array_mib"] > 0
    assert not (output / "summary.json").exists()
    assert not (output / "weights.npy").exists()
    assert not (output / "work-resolvent.npy").exists()

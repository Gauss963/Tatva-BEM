"""The complete fresh producer is checked against independent dense RK-CQ."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import CQPlan, plan_cq
from finiteblock_bem.io import environment_record
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
PRODUCER = runpy.run_path(str(TOOLS / "prepare_raw_radau_resolvent.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))


def source_metadata(tmp_path):
    path = tmp_path / "mesh.json"
    options = dict(
        case=str(CASE),
        h=160,
        contact_h=160,
        contact_end_h=160,
        corner_h=160,
        quadrature=12,
        restart_knot_window=None,
        restart_knot_subdivisions=1,
        diagnostic_actuator_increment=0.5,
    )
    path.write_text(
        json.dumps(
            dict(
                status="completed_static_mesh_control_not_a_dynamic_kernel",
                source_mesh_arguments=options,
                environment=environment_record(CASE),
                spectrum_storage=dict(path=str(tmp_path / "must-not-read-source.npy")),
            )
        )
    )
    return path


@pytest.mark.parametrize("layout,advisory", [("c", False), ("fortran", True)])
def test_complete_fresh_kernel_weights_and_causal_startup_match_dense_original_bem(
    tmp_path, monkeypatch, layout, advisory
):
    source = source_metadata(tmp_path)
    output = tmp_path / "fresh"
    argv = [
        "prepare",
        str(source),
        "--output",
        str(output),
        "--dt",
        "2e-6",
        "--duration",
        "32e-6",
        "--workers",
        "1",
        "--fft-rows",
        "2",
        "--combine-matrices",
        "3",
        "--startup-rows",
        "1",
        "--weight-layout",
        layout,
        "--max-workspace-mb",
        "512",
        "--max-memory-mb",
        "1024",
    ]
    if advisory:
        argv.append("--primitive-partition")
    monkeypatch.setattr(sys, "argv", argv)
    original = NodalInterfaceTransfer.evaluate
    PRODUCER["main"]()
    assert NodalInterfaceTransfer.evaluate is original
    metadata = json.loads((output / "summary.json").read_text())
    assert metadata["status"] == "completed_compact_radau_resolvent_kernel_only"
    assert metadata["assembly"] == "fresh_raw_radau_before_trace_schur"
    assert metadata["correction_steps"] == metadata["full_steps"] == 16
    assert metadata["original_tatva_readonly"]
    assert not metadata["full_original_event_reproduced"]
    assert not metadata["same_accuracy_fem_advantage_verified"]
    assert all(r["all_fft_grid_points_emitted"] for r in metadata["frequency_sampling"])
    assert all(not r["source_frequency_values_reused"] for r in metadata["frequency_sampling"])
    assert bool(metadata["frequency_sampling"][0]["primitive_partition_screen"]) == advisory
    model = load_nodal_model(
        CASE,
        max_panel_length=160,
        contact_panel_length=160,
        contact_end_length=160,
        corner_panel_length=160,
    )
    transfer = NodalInterfaceTransfer(model, quadrature=12)
    con, law = model.constraints, model.reference.contact
    spring = model.reference.spring_stiffness * con.platen_width
    diagonal = np.r_[
        con.contact_weights * law.normal_penalty,
        con.contact_weights * law.tangential_penalty,
        spring,
    ]
    d = transfer.dofs

    def hfun(s):
        return np.linalg.solve(transfer.evaluate(s) + np.diag(diagonal), np.eye(d))

    plan = CQPlan(**metadata["plan"])
    stage_plan = plan_cq(2 * d, plan.dt, plan.steps, max_memory_mb=1024)
    expected = RADAU["radau_weights"](hfun, stage_plan)
    weights = [np.load(output / f"weights{i}.npy") for i in range(2)]
    wrap = np.load(output / "shift_wrap.npy")
    actual = RADAU["expand_compact_weights"](*weights, wrap)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
    assert all(
        w.flags.f_contiguous if layout == "fortran" else w.flags.c_contiguous for w in weights
    )
    with np.load(output / "elastic.npz") as data:
        forcing = np.zeros((plan.steps + 1, 2 * d))
        rate = 0.5 / (plan.dt * plan.steps)
        forcing[:, d - 1] = spring * rate * data["stage_time"][:, 0]
        forcing[:, -1] = spring * rate * data["stage_time"][:, 1]
        dense = np.stack(
            [sum(expected[k - j] @ forcing[j] for j in range(k + 1)) for k in range(plan.steps + 1)]
        )
        np.testing.assert_allclose(
            data["elastic"].reshape(plan.steps + 1, 2 * d), dense, rtol=1e-8, atol=1e-12
        )
    for check in metadata["sampled_fresh_bem_mode_checks"]:
        assert check["relative_frobenius"] <= 1e-6
        assert check["solve_composition_inf_error"] <= 1e-6


def test_preflight_constructs_no_frequency_or_weight_arrays_and_budget_failure_is_early(
    tmp_path, monkeypatch
):
    source = source_metadata(tmp_path)
    output = tmp_path / "preflight"
    args = [
        "prepare",
        str(source),
        "--output",
        str(output),
        "--duration",
        "32e-6",
        "--preflight-only",
    ]
    monkeypatch.setattr(sys, "argv", args)
    PRODUCER["main"]()
    assert not list(output.glob("*.npy"))
    assert not (output / "summary.json").exists()
    data = json.loads((output / "preflight.json").read_text())
    assert data["branches_sampled_sequentially"]
    assert data["status"] == "preflight_only_not_a_complete_kernel"
    rejected = tmp_path / "rejected"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            str(source),
            "--output",
            str(rejected),
            "--duration",
            "32e-6",
            "--max-memory-mb",
            "1",
        ],
    )
    with pytest.raises(ValueError, match="array estimate exceeds budgets"):
        PRODUCER["main"]()
    assert not rejected.exists()


def test_underintegrated_kernel_is_rejected_at_the_unchanged_composition_tolerance(
    tmp_path, monkeypatch
):
    source = source_metadata(tmp_path)
    data = json.loads(source.read_text())
    data["source_mesh_arguments"]["quadrature"] = 4
    source.write_text(json.dumps(data))
    output = tmp_path / "underintegrated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            str(source),
            "--output",
            str(output),
            "--duration",
            "32e-6",
            "--weight-layout",
            "c",
        ],
    )
    with pytest.raises(ValueError, match="lag-zero inverse composition failed"):
        PRODUCER["main"]()
    assert not (output / "summary.json").exists()


def test_case_toml_entry_point_needs_no_previously_generated_kernel(tmp_path, monkeypatch):
    output = tmp_path / "from-case"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            str(CASE),
            "--output",
            str(output),
            "--h",
            "160",
            "--contact-h",
            "160",
            "--contact-end-h",
            "160",
            "--corner-h",
            "160",
            "--duration",
            "32e-6",
            "--preflight-only",
        ],
    )
    PRODUCER["main"]()
    metadata = json.loads((output / "preflight.json").read_text())
    assert metadata["source_mesh_arguments"]["case"] == str(CASE.resolve())
    assert metadata["source_mesh_arguments"]["h"] == 160
    assert metadata["plan"]["steps"] == 16
    assert not list(output.glob("*.npy"))

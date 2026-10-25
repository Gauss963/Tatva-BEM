import json
import runpy
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from finiteblock_bem.cq import CQPlan, bdf2_symbol, invert_spectrum, plan_cq
from finiteblock_bem.io import environment_record, fingerprint
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def discretization(**updates):
    values = dict.fromkeys(
        [
            "h",
            "contact_h",
            "contact_end_h",
            "corner_h",
            "restart_knot_window",
            "restart_knot_subdivisions",
            "quadrature",
        ]
    )
    return SimpleNamespace(**{**values, "clear_restart_knots": False, **updates})


def test_explicit_mesh_refinement_does_not_mutate_source_or_physical_arguments():
    tool = runpy.run_path(str(TOOLS / "prepare_native_resolvent.py"))
    original = dict(
        case="physical-case",
        h=20,
        contact_h=5,
        quadrature=12,
        restart_knot_window=8,
        restart_knot_subdivisions=4,
    )
    actual, overrides = tool["discretization_arguments"](
        original, discretization(contact_h=2, clear_restart_knots=True)
    )
    assert actual == {
        **original,
        "contact_h": 2,
        "restart_knot_window": None,
        "restart_knot_subdivisions": 1,
    }
    assert overrides == dict(contact_h=2, restart_knot_window=None, restart_knot_subdivisions=1)
    assert original["contact_h"] == 5 and original["restart_knot_window"] == 8


@pytest.mark.parametrize(
    "updates",
    [
        dict(h=0),
        dict(contact_h=np.nan),
        dict(quadrature=5),
        dict(restart_knot_subdivisions=0),
        dict(clear_restart_knots=True, restart_knot_subdivisions=1),
    ],
)
def test_invalid_refinement_is_rejected_before_construction(updates):
    tool = runpy.run_path(str(TOOLS / "prepare_native_resolvent.py"))
    with pytest.raises(ValueError):
        tool["discretization_arguments"](dict(h=20, quadrature=12), discretization(**updates))


def test_disk_physical_conversion_matches_original_and_retains_all_lags(tmp_path):
    original = runpy.run_path(str(TOOLS / "build_resolvent_history.py"))
    native = runpy.run_path(str(TOOLS / "prepare_native_resolvent.py"))
    rng = np.random.default_rng(20261024)
    expected = rng.normal(size=(131, 3, 3))
    path = tmp_path / "weights.npy"
    np.save(path, expected, allow_pickle=False)
    line_weights = np.array([0.5, 3.7, 80.0])
    original["physical_weights_inplace"](expected, line_weights)
    actual = native["physical_weights_file_inplace"](path, line_weights)
    assert isinstance(actual, np.memmap) and not actual.flags.writeable
    np.testing.assert_array_equal(actual, expected)


def test_time_storage_preflight_covers_actual_serialized_outputs(tmp_path, monkeypatch):
    native = runpy.run_path(str(TOOLS / "prepare_native_resolvent.py"))
    plan = plan_cq(3, 0.01, 5, max_memory_mb=32)
    np.save(tmp_path / "weights.npy", np.zeros((6, 3, 3)), allow_pickle=False)
    np.save(tmp_path / "initial_impedance.npy", np.eye(3), allow_pickle=False)
    np.savez(
        tmp_path / "elastic.npz",
        time=np.arange(13),
        elastic=np.zeros((13, 3)),
        uninterrupted_actuator=np.zeros(13),
    )
    required = native["preflight_time_storage"](tmp_path, plan, 12)
    assert required >= sum(path.stat().st_size for path in tmp_path.iterdir())
    monkeypatch.setattr(
        native["shutil"], "disk_usage", lambda _: SimpleNamespace(free=required - 1)
    )
    with pytest.raises(ValueError, match="complete time weights"):
        native["preflight_time_storage"](tmp_path, plan, 12)


def test_fresh_refined_kernel_matches_dense_bem_and_streams_output_hashes(tmp_path, monkeypatch):
    """A real finite-block refinement, independently reassembled at every CQ node."""
    producer = runpy.run_path(str(TOOLS / "prepare_native_resolvent.py"))
    original = runpy.run_path(str(TOOLS / "build_resolvent_history.py"))
    helper = runpy.run_path(str(TOOLS / "replay_resolvent.py"))
    source, output = tmp_path / "source", tmp_path / "refined"
    source.mkdir()
    options = dict(
        case=str(CASE),
        h=160,
        contact_h=160,
        contact_end_h=160,
        corner_h=160,
        quadrature=4,
        restart_knot_window=8,
        restart_knot_subdivisions=4,
        diagnostic_actuator_increment=None,
    )
    (source / "run_manifest.json").write_text(json.dumps(dict(arguments=options)))
    (source / "summary.json").write_text(
        json.dumps(
            dict(
                status="completed_requested_horizon",
                source_changed_during_run=False,
                environment=environment_record(CASE),
                dt_s=1e-4,
                duration_s=4e-4,
            )
        )
    )
    before = {p.name: fingerprint(p) for p in source.iterdir()}
    arguments = [
        "prepare",
        str(source),
        "--output",
        str(output),
        "--h",
        "80",
        "--contact-h",
        "80",
        "--contact-end-h",
        "80",
        "--corner-h",
        "80",
        "--clear-restart-knots",
        "--quadrature",
        "4",
        "--correction-duration",
        "2e-4",
        "--workers",
        "1",
        "--max-memory-mb",
        "1024",
        "--max-spectrum-mb",
        "32",
    ]
    read_bytes = Path.read_bytes

    def guarded_read(path):
        if path.name in {"weights.npy", "elastic.npz"}:
            raise AssertionError("Large outputs must use bounded streaming digests")
        return read_bytes(path)

    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    producer["main"]()
    metadata = json.loads((output / "summary.json").read_text())
    assert before == {p.name: fingerprint(p) for p in source.iterdir()}
    assert metadata["source_mesh_arguments"]["case"] == str(CASE)
    assert metadata["numerical_discretization_overrides"]["contact_h"] == 80
    assert metadata["numerical_discretization_overrides"]["restart_knot_window"] is None
    assert metadata["history_dofs"] == 15
    assert metadata["correction_steps"] == 2 and metadata["full_steps"] == 4
    assert metadata["weights_sha256"] == helper["digest_file"](output / "weights.npy")
    assert metadata["elastic_sha256"] == helper["digest_file"](output / "elastic.npz")
    assert metadata["total_seconds"] > 0 and metadata["process_cpu_seconds"] > 0
    assert metadata["peak_process_rss_mib"] > 0

    model = load_nodal_model(
        CASE,
        max_panel_length=80,
        contact_panel_length=80,
        contact_end_length=80,
        corner_panel_length=80,
    )
    weights = np.r_[
        model.constraints.contact_weights,
        model.constraints.contact_weights,
        model.constraints.platen_width,
    ]
    law = model.reference.contact
    tangent = np.diag(
        weights
        * np.r_[
            np.full(7, law.normal_penalty),
            np.full(7, law.tangential_penalty),
            model.reference.spring_stiffness,
        ]
    )
    plan = CQPlan(**metadata["plan"])
    circle = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    transfer = NodalInterfaceTransfer(model, quadrature=4)
    dense = np.stack(
        [
            original["compliance_in_work_coordinates"](transfer.evaluate(s), tangent, weights)
            for s in bdf2_symbol(circle) / plan.dt
        ]
    )
    expected = invert_spectrum(dense.copy(), plan)
    original["physical_weights_inplace"](expected, weights)
    np.testing.assert_allclose(np.load(output / "weights.npy"), expected, rtol=1e-10, atol=1e-14)
    time = np.arange(5) * plan.dt
    spring = model.reference.spring_stiffness * model.constraints.platen_width
    expected_startup = helper["causal_elastic_baseline"](
        dense[:, :, -1] / np.sqrt(weights * weights[-1]),
        CQPlan(**{**asdict(plan), "steps": 4}),
        spring * model.reference.actuator_rate * time,
    )
    with np.load(output / "elastic.npz") as data:
        np.testing.assert_allclose(data["elastic"], expected_startup, rtol=1e-10, atol=1e-12)
    assert not metadata["full_original_event_reproduced"]

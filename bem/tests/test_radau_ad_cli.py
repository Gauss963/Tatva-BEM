"""Bounded complete-RK AD planning, exact restart identity and a real CLI run."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import BoundaryHistory, plan_cq
from finiteblock_bem.io import environment_record

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CLI = runpy.run_path(str(TOOLS / "check_compact_radau_ad.py"))
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
SEED = TOOLS.parent / "src/finiteblock_bem/resources/data/ev16_restart_float64.npz"


def test_history_estimate_counts_both_stage_channels_and_both_family_arrays():
    plan = plan_cq(5, 0.1, 17, max_memory_mb=32)
    history = runpy.run_path(str(TOOLS / "compact_radau_history.py"))
    streams = [
        history["BatchedOnlineConvolution"](BoundaryHistory(plan, np.zeros((18, 5, 5))), 6)
        for _ in range(2)
    ]
    allocated = sum(s.pending.nbytes + s.inputs.nbytes for s in streams)
    estimate = CLI["estimate_ad_arrays"](plan, 2, batch_size=3)
    assert estimate["histories"] == allocated
    assert estimate["peak"] > allocated


def test_actual_fine_horizons_need_bounded_direction_batches():
    budget = 12288 * 1024**2
    two = plan_cq(531, 2e-6, 22500, max_memory_mb=1e9, disk_spectrum=True)
    one = plan_cq(531, 1e-6, 45000, max_memory_mb=1e9, disk_spectrum=True)
    estimate = CLI["estimate_ad_arrays"]
    assert (
        estimate(two, 265, batch_size=3)["peak"] < budget < estimate(two, 265, batch_size=7)["peak"]
    )
    assert (
        estimate(one, 265, batch_size=1)["peak"] < budget < estimate(one, 265, batch_size=3)["peak"]
    )


def test_portable_restart_fallback_is_allowed_only_for_identical_primal_bytes(tmp_path):
    digest = CLI["READ"]["digest_file"](SEED)
    record = dict(path=str(tmp_path / "original-path-unavailable.npz"), sha256=digest)
    assert CLI["restart_path"](record, None) == SEED.resolve()
    changed = tmp_path / "changed.npz"
    changed.write_bytes(b"not the captured seed")
    record["path"] = str(changed)
    with pytest.raises(ValueError, match="exact restart"):
        CLI["restart_path"](record, None)
    with pytest.raises(ValueError, match="stock restart"):
        CLI["restart_path"](None, SEED)


def test_real_fresh_kernel_static_body_original_audit_and_initialized_ad_cli(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    source = tmp_path / "mesh.json"
    kernel = tmp_path / "kernel"
    run = tmp_path / "run"
    audit = tmp_path / "audit"
    output = tmp_path / "ad"
    source.write_text(
        json.dumps(
            dict(
                source_mesh_arguments=dict(
                    case=str(CASE),
                    h=160,
                    contact_h=160,
                    contact_end_h=160,
                    corner_h=160,
                    quadrature=12,
                    restart_knot_window=None,
                    restart_knot_subdivisions=1,
                    diagnostic_actuator_increment=0.5,
                ),
                environment=environment_record(CASE),
            )
        )
    )
    producer = runpy.run_path(str(TOOLS / "prepare_raw_radau_resolvent.py"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            str(source),
            "--output",
            str(kernel),
            "--dt",
            "2e-6",
            "--duration",
            "32e-6",
            "--workers",
            "1",
            "--fft-rows",
            "2",
            "--max-memory-mb",
            "1024",
            "--max-workspace-mb",
            "512",
        ],
    )
    producer["main"]()
    simulation = runpy.run_path(str(TOOLS / "simulate_compact_radau.py"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simulate",
            str(kernel),
            "--output",
            str(run),
            "--restart-seed",
            str(SEED),
            "--static-initialization",
            "fem-body-active-set",
            "--segment-substeps",
            "1",
            "2",
            "--contact-root-recovery",
            "--kinetic-root-seed",
            "--fft-cache-mb",
            "0",
            "--max-memory-mb",
            "2048",
        ],
    )
    simulation["main"]()
    with np.load(run / "stages.npz", allow_pickle=False) as f:
        assert np.max(f["response"][:, :, :, 3]) > 0
    operator = runpy.run_path(str(TOOLS / "audit_compact_radau_history.py"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            str(run),
            "--output",
            str(audit),
            "--workers",
            "1",
            "--fft-rows",
            "2",
            "--max-memory-mb",
            "2048",
            "--max-workspace-mb",
            "1024",
            "--residual-tolerance-mpa",
            "1e-7",
        ],
    )
    operator["main"]()
    arguments = [
        "check",
        str(run),
        "--operator-audit",
        str(audit),
        "--output",
        str(output),
        "--jvp-batch-size",
        "3",
        "--fft-cache-mb",
        "0",
        "--max-memory-mb",
        "4096",
    ]
    preflight = tmp_path / "preflight"
    planned = arguments.copy()
    planned[planned.index(str(output))] = str(preflight)
    monkeypatch.setattr(sys, "argv", [*planned, "--preflight-only"])
    CLI["main"]()
    assert (preflight / "preflight.json").is_file()
    assert not (preflight / "gradients.npz").exists()
    assert not (preflight / "summary.json").exists()
    limited = arguments.copy()
    limited[limited.index("4096")] = "1"
    monkeypatch.setattr(sys, "argv", limited)
    with pytest.raises(ValueError, match="array estimate exceeds memory"):
        CLI["main"]()
    assert not output.exists()
    monkeypatch.setattr(sys, "argv", arguments)
    CLI["main"]()
    result = json.loads((output / "summary.json").read_text())
    assert result["status"] == "passed_complete_radau_adjoint_jvp_control"
    assert result["complete_prepared_horizon_checked"]
    assert result["full_initialized_gradient"]
    assert not result["full_original_event_reproduced"]
    assert [r["directions"] for r in result["jvp_batches"]] == [[0, 1, 2], [3, 4, 5], [6]]
    with np.load(output / "gradients.npz", allow_pickle=False) as f:
        np.testing.assert_allclose(f["jvp"], f["vjp_dot_products"], rtol=1e-7, atol=1e-18)
    # A changed source stage must not inherit a prior original-equation pass.
    with (run / "stages.npz").open("ab") as stream:
        stream.write(b"changed after the causal audit")
    next_output = tmp_path / "changed-ad"
    arguments[arguments.index(str(output))] = str(next_output)
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(ValueError, match="differ from the original causal"):
        CLI["main"]()

"""End-to-end CQ reuse with analytic matrices and a dense Cauchy test backend."""

import json
import runpy
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from finiteblock_bem.cq import bdf2_symbol, invert_spectrum, plan_cq
from finiteblock_bem.friction import SlipWeakeningContact
from finiteblock_bem.io import fingerprint

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def test_reused_kernel_preserves_weights_startup_and_checks_independent_audit(tmp_path):
    producer = runpy.run_path(str(TOOLS / "prepare_reused_resolvent.py"))
    cauchy = runpy.run_path(str(TOOLS / "cauchy_reuse_experiment.py"))
    helper = runpy.run_path(str(TOOLS / "replay_resolvent.py"))
    environment = dict(source_sha256="test", case_sha256="test", restart_data_sha256="test")
    model = SimpleNamespace(
        constraints=SimpleNamespace(contact_weights=np.ones(1), platen_width=1.0),
        reference=SimpleNamespace(
            contact=SlipWeakeningContact(100, 40, 0.8, 0.5, 0.2),
            spring_stiffness=2.0,
            actuator_rate=15.0,
            restart_actuator=0.0,
        ),
    )
    stiffness = np.array([[30.0, 0.2, 0.1], [0.2, 30, -20], [0.1, -20, 30]])
    tangent = np.diag([100.0, 40, 2])

    def zfun(s):
        return 0.2 * s * np.eye(3) + stiffness

    def hfun(s):
        return np.linalg.inv(zfun(s) + tangent)

    source_plan = plan_cq(3, 0.02, 40, alias_tolerance=1e-10, max_memory_mb=32)
    plan = plan_cq(3, 0.04, 20, alias_tolerance=1e-10, max_memory_mb=32)
    source_circle = source_plan.radius * np.exp(
        -2j * np.pi * np.arange(source_plan.fft_size // 2 + 1) / source_plan.fft_size
    )
    source_values = np.stack([hfun(s) for s in bdf2_symbol(source_circle) / source_plan.dt])
    source = tmp_path / "source"
    source.mkdir()
    source_path = source / "work-resolvent.npy"
    np.save(source_path, source_values, allow_pickle=False)
    source_digest = fingerprint(source_path)
    options = dict(
        case="test", h=1.0, contact_h=1.0, contact_end_h=1.0, corner_h=1.0, quadrature=12
    )
    metadata = dict(
        status="completed_resolvent_kernel_only",
        environment=environment,
        source_mesh_arguments=options,
        source_run=str(source),
        full_steps=40,
        correction_steps=40,
        plan=asdict(source_plan),
        spectrum_storage=dict(
            path=str(source_path),
            sha256=source_digest,
            coordinates="diag(sqrt(w)) * L * diag(sqrt(w))",
        ),
    )
    (source / "summary.json").write_text(json.dumps(metadata))
    target_circle = plan.radius * np.exp(
        -2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size
    )
    frequencies = bdf2_symbol(target_circle) / plan.dt
    fresh = np.stack([hfun(s) for s in frequencies])
    mapped = cauchy["bdf2_preimage"](frequencies, source_plan.dt)
    reused = cauchy["CircleCauchy"](source_plan.fft_size, source_plan.radius).evaluate(
        source_values, mapped
    )
    audit = tmp_path / "audit"
    audit.mkdir()
    indices = [0, 1, 8, plan.fft_size // 2]
    np.savez(audit / "operators.npz", fresh=fresh[indices], reused=reused[indices])
    audit_record = dict(
        status="passed_sampled_operator_comparison",
        source_kernel=str(source),
        source_spectrum_sha256=source_digest,
        target_dt_s=plan.dt,
        target_fft_size=plan.fft_size,
        environment=environment,
        indices=indices,
        operator_arrays_sha256=fingerprint(audit / "operators.npz"),
        relative_tolerance=1e-6,
    )
    (audit / "summary.json").write_text(json.dumps(audit_record))
    library = tmp_path / "fake-library-test-only"
    library.write_text("The test injects a direct Cauchy backend; no native ABI is tested here.")

    def load_tool(path):
        loaded = runpy.run_path(str(path))
        if Path(path).name == "cauchy_reuse_experiment.py":
            loaded["fortran_fmm2d_backend"] = lambda _: cauchy["_direct"]
        return loaded

    replacements = dict(
        environment_record=lambda _: environment,
        load_nodal_model=lambda *a, **kw: model,
        NodalInterfaceTransfer=lambda *a, **kw: SimpleNamespace(evaluate=zfun),
        runpy=SimpleNamespace(run_path=load_tool),
    )
    output = tmp_path / "reused"
    arguments = [
        "prepare",
        str(source),
        "--operator-audit",
        str(audit),
        "--fmm-library",
        str(library),
        "--dt",
        "0.04",
        "--output",
        str(output),
    ]
    read_bytes = Path.read_bytes

    def guarded_read(path):
        if path.name in {"work-resolvent.npy", "weights.npy"}:
            raise AssertionError("Large numerical files must be hashed through bounded reads")
        return read_bytes(path)

    with (
        patch.dict(producer["main"].__globals__, replacements),
        patch.object(sys, "argv", arguments),
        patch.object(Path, "read_bytes", guarded_read),
    ):
        producer["main"]()
    actual = np.load(output / "weights.npy", allow_pickle=False)
    expected = invert_spectrum(fresh.copy(), plan)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
    with np.load(output / "elastic.npz", allow_pickle=False) as data:
        startup = helper["causal_elastic_baseline"](fresh[:, :, -1], plan, 2 * 15 * data["time"])
        np.testing.assert_allclose(data["elastic"], startup, rtol=1e-8, atol=1e-10)
    completed = json.loads((output / "summary.json").read_text())
    assert completed["status"] == "completed_resolvent_kernel_only"
    assert completed["initial_composition_inf_error"] < 1e-6
    assert fingerprint(source_path) == source_digest
    audit_record["target_time_method"] = "radau2"
    (audit / "summary.json").write_text(json.dumps(audit_record))
    arguments[-1] = str(tmp_path / "wrong-time-method")
    with (
        patch.dict(producer["main"].__globals__, replacements),
        patch.object(sys, "argv", arguments),
    ):
        with pytest.raises(ValueError, match="matching passed independent operator audit"):
            producer["main"]()
    audit_record["target_time_method"] = "bdf2"
    audit_record["status"] = "failed_sampled_operator_comparison"
    (audit / "summary.json").write_text(json.dumps(audit_record))
    arguments[-1] = str(tmp_path / "rejected")
    with (
        patch.dict(producer["main"].__globals__, replacements),
        patch.object(sys, "argv", arguments),
    ):
        with pytest.raises(ValueError, match="matching passed independent"):
            producer["main"]()
    assert not (tmp_path / "rejected").exists()

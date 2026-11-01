"""Independent dense kernels, genuine reference audits and durable JIT resume."""

import json
import runpy
import sys
from pathlib import Path
from types import FunctionType

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
TOOLS = Path(__file__).resolve().parents[1] / "tools"
JAX = runpy.run_path(str(TOOLS / "prepare_jax_radau_resolvent.py"))
FIXTURE = runpy.run_path(str(Path(__file__).with_name("test_raw_radau_producer.py")))


@pytest.mark.parametrize("layout,advisory", [("c", False), ("fortran", True)])
def test_full_jax_kernel_and_startup_against_independent_dense_original(
    tmp_path, monkeypatch, layout, advisory
):
    private = JAX["private_producer"]()
    original_main = private["main"]

    def durable_main():
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--durable-build"])
        original_main()

    control = FIXTURE[
        "test_complete_fresh_kernel_weights_and_causal_startup_match_dense_original_bem"
    ]
    run = FunctionType(
        control.__code__, {**control.__globals__, "PRODUCER": {**private, "main": durable_main}}
    )
    run(tmp_path, monkeypatch, layout, advisory)
    metadata = json.loads((tmp_path / "fresh/summary.json").read_text())
    assert metadata["jax_backend"]["original_audit_uses_scipy"]
    assert metadata["jax_backend"]["runtime"]["x64"]
    assert metadata["estimated_construction_array_peak_mib"] >= 512
    assert str(TOOLS / "prepare_jax_radau_resolvent.py") in metadata["frozen_inputs_sha256"]


def test_independent_audit_unwraps_jax_even_if_wrapper_assembly_is_unusable(monkeypatch):
    private = JAX["private_producer"]()
    model = FIXTURE["load_nodal_model"](
        FIXTURE["CASE"],
        max_panel_length=160,
        contact_panel_length=160,
        contact_end_length=160,
        corner_panel_length=160,
    )
    approximate = private["TRANSFER"](model, quadrature=12)

    def forbidden(s):
        raise AssertionError("Independent audit called the approximate backend")

    for prepared in approximate.prepared:
        monkeypatch.setattr(prepared, "assemble", forbidden)
    audit = JAX["OriginalAuditTransfer"](model, quadrature=12, prepared=False)
    audit.prepared = approximate.prepared
    assert all(type(a) is JAX["PreparedGalerkinOperators"] for a in audit.prepared)
    expected = FIXTURE["NodalInterfaceTransfer"](model, quadrature=12).evaluate(175 + 31400j)
    np.testing.assert_array_equal(audit.evaluate(175 + 31400j), expected)


@pytest.mark.parametrize("field", ["runtime", "sources_sha256", "original_audit_uses_scipy"])
def test_backend_context_tampering_is_rejected_without_rewriting(tmp_path, field):
    frozen = {}
    JAX["initialize_context"](tmp_path, frozen)
    JAX["verify_context"](tmp_path, frozen)
    path = tmp_path / JAX["CONTEXT"]
    record = json.loads(path.read_text())
    record[field] = "tampered"
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="source, runtime or policy changed"):
        JAX["verify_context"](tmp_path)
    assert path.read_bytes() == before


def test_original_cache_requires_explicit_handoff_context(tmp_path):
    (tmp_path / "preflight.json").write_text("{}\n")
    with pytest.raises(ValueError, match="explicit handoff"):
        JAX["verify_context"](tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["preflight.json"]


def test_durable_jax_resume_preserves_every_array_bit_and_readonly_verification(
    tmp_path, monkeypatch
):
    producer = JAX["private_producer"]()
    source = FIXTURE["source_metadata"](tmp_path)
    flags = [
        str(source),
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
        "--spectrum-buffer-matrices",
        "4",
        "--startup-rows",
        "1",
        "--weight-layout",
        "fortran",
        "--max-workspace-mb",
        "512",
        "--max-memory-mb",
        "1024",
        "--durable-build",
    ]

    def execute(output, extra=()):
        monkeypatch.setattr(sys, "argv", ["jax-prepare", *flags, "--output", str(output), *extra])
        producer["main"]()

    baseline, continued = tmp_path / "baseline", tmp_path / "continued"
    execute(baseline)
    journal = producer["CHECKPOINT"]["SamplingJournal"]
    original_commit = journal.commit

    def interrupted(self, name, *args):
        original_commit(self, name, *args)
        if name == "work-family0.npy":
            raise RuntimeError("injected committed JAX interruption")

    with monkeypatch.context() as patch:
        patch.setattr(journal, "commit", interrupted)
        with pytest.raises(RuntimeError, match="injected committed"):
            execute(continued)
    digest = producer["CHECKPOINT"]["digest_range"]
    before = {p.name: digest(p) for p in continued.iterdir() if p.is_file()}
    execute(continued, ["--resume", "--verify-resume-only"])
    assert {p.name: digest(p) for p in continued.iterdir() if p.is_file()} == before
    execute(continued, ["--resume"])
    for path in baseline.glob("*.npy"):
        actual, expected = np.load(continued / path.name), np.load(path)
        np.testing.assert_array_equal(
            actual.ravel().view(np.uint64), expected.ravel().view(np.uint64)
        )
    with (
        np.load(baseline / "elastic.npz") as expected,
        np.load(continued / "elastic.npz") as actual,
    ):
        assert actual.files == expected.files
        for name in actual.files:
            np.testing.assert_array_equal(actual[name], expected[name])
    record = json.loads((continued / "summary.json").read_text())
    assert [a["status"] for a in record["durable_construction"]["attempts"]] == [
        "interrupted_or_failed",
        "completed",
    ]

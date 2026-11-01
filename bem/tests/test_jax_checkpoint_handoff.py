"""Real mixed-backend continuation and read-only rejection of invalid copies."""

import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
TOOLS = Path(__file__).resolve().parents[1] / "tools"
CLONE = runpy.run_path(str(TOOLS / "clone_jax_radau_checkpoint.py"))
JAX = CLONE["JAX"]
FIXTURE = runpy.run_path(str(Path(__file__).with_name("test_raw_radau_producer.py")))
DIGEST = CLONE["DIGEST"]


@pytest.fixture(scope="module")
def original_builds(tmp_path_factory):
    root = tmp_path_factory.mktemp("original-jax-handoff")
    source = FIXTURE["source_metadata"](root)
    flags = [
        str(source),
        "--dt",
        "2e-6",
        "--duration",
        ".000256",
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
        "--primitive-partition",
    ]
    original = runpy.run_path(str(TOOLS / "prepare_raw_radau_resolvent.py"))
    baseline, parent = root / "baseline", root / "parent"
    prior_argv = sys.argv
    try:
        sys.argv = ["prepare", *flags, "--output", str(baseline)]
        original["main"]()
        journal = original["CHECKPOINT"]["SamplingJournal"]
        save = journal.save_sampler_state

        def stop_at_boundary(self, name, left, right, stack, writer):
            save(self, name, left, right, stack, writer)
            if name == "work-family0.npy" and left > 0:
                raise RuntimeError("injected exact interval boundary stop")

        journal.save_sampler_state = stop_at_boundary
        try:
            sys.argv = ["prepare", *flags, "--output", str(parent)]
            with pytest.raises(RuntimeError, match="exact interval"):
                original["main"]()
        finally:
            journal.save_sampler_state = save
    finally:
        sys.argv = prior_argv
    return root, baseline, parent, flags


def prepare_target(path, flags, monkeypatch, extra=()):
    monkeypatch.setattr(
        sys, "argv", ["jax-prepare", *flags, "--output", str(path), "--preflight-only", *extra]
    )
    JAX["private_producer"]()["main"]()
    return path


def files(folder):
    return {p.name: DIGEST(p) for p in folder.iterdir() if p.is_file()}


def test_actual_original_to_jax_resume_keeps_prefix_and_all_causal_arrays(
    original_builds, tmp_path, monkeypatch
):
    _, baseline, parent, flags = original_builds
    before = files(parent)
    target = prepare_target(tmp_path / "mixed", flags, monkeypatch)
    receipt = CLONE["clone_checkpoint"](parent, target)
    assert receipt["sampler_boundaries_exact"]
    assert receipt["prefixes"]["work-family0.npy"] > 0
    assert files(parent) == before
    clone_before = files(target)
    monkeypatch.setattr(
        sys,
        "argv",
        ["jax-prepare", *flags, "--output", str(target), "--resume", "--verify-resume-only"],
    )
    JAX["private_producer"]()["main"]()
    assert files(target) == clone_before
    monkeypatch.setattr(sys, "argv", ["jax-prepare", *flags, "--output", str(target), "--resume"])
    JAX["private_producer"]()["main"]()
    assert files(parent) == before
    for name in [
        "weights0.npy",
        "weights1.npy",
        "shift_wrap.npy",
        "initial_impedance.npy",
        "compact-family0.npy",
        "compact-family1.npy",
    ]:
        np.testing.assert_allclose(
            np.load(target / name), np.load(baseline / name), rtol=1e-8, atol=1e-12
        )
    np.testing.assert_array_equal(
        np.load(target / "initial_impedance.npy"), np.load(baseline / "initial_impedance.npy")
    )
    with np.load(target / "elastic.npz") as actual, np.load(baseline / "elastic.npz") as expected:
        assert actual.files == expected.files
        for name in actual.files:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-8, atol=1e-12)
    record = json.loads((target / "summary.json").read_text())
    assert record["frequency_sampling"][0]["same_configuration_prefix_points_reused"] > 0
    assert record["frequency_sampling"][0]["recomputed_prefix_points_verified_bitwise"] == 0
    for name, entry in json.loads((target / "parent-sampling-journal.json").read_text())[
        "channels"
    ].items():
        for chunk in entry["chunks"]:
            count = (chunk["last"] - chunk["first"]) * record["history_dofs"] ** 2 * 16
            offset = entry["offset"] + chunk["first"] * record["history_dofs"] ** 2 * 16
            assert DIGEST(target / name, offset, count) == chunk["sha256"]
    frozen = {}
    JAX["verify_context"](target, frozen)
    assert str(target / "handoff.json") in frozen
    assert str(target / "parent-sampling-journal.json") in frozen


def test_active_original_lease_refuses_copy_without_target_changes(
    original_builds, tmp_path, monkeypatch
):
    _, _, parent, flags = original_builds
    target = prepare_target(tmp_path / "locked", flags, monkeypatch)
    before = files(target)
    lease = CLONE["CHECK"]["exclusive_lease"](parent)
    try:
        with pytest.raises(RuntimeError, match="still owns"):
            CLONE["clone_checkpoint"](parent, target)
    finally:
        lease.close()
    assert files(target) == before


def test_changed_cq_plan_refuses_copy_without_target_changes(
    original_builds, tmp_path, monkeypatch
):
    _, _, parent, flags = original_builds
    target = prepare_target(tmp_path / "changed", flags, monkeypatch, ["--dt", "1e-6"])
    before = files(target)
    with pytest.raises(ValueError, match="physical problem or CQ plan"):
        CLONE["clone_checkpoint"](parent, target)
    assert files(target) == before


@pytest.mark.parametrize("damage", ["unaligned", "corrupt"])
def test_invalid_original_prefix_is_rejected_before_target_writes(
    original_builds, tmp_path, monkeypatch, damage
):
    _, _, source, flags = original_builds
    parent = tmp_path / "invalid-parent"
    shutil.copytree(source, parent)
    preflight = json.loads((parent / "preflight.json").read_text())
    preflight["arguments"]["output"] = str(parent)
    (parent / "preflight.json").write_text(json.dumps(preflight))
    journal = json.loads((parent / "sampling-journal.json").read_text())
    journal["preflight_sha256"] = DIGEST(parent / "preflight.json")
    if damage == "unaligned":
        count = preflight["plan"]["fft_size"] // 2 + 1
        journal["sampler_states"]["work-family0.npy"] = dict(start=0, stack=[[0, count]])
    else:
        with (parent / "work-family0.npy").open("r+b") as stream:
            stream.seek(journal["channels"]["work-family0.npy"]["offset"])
            original = stream.read(1)
            stream.seek(-1, 1)
            stream.write(bytes([original[0] ^ 1]))
    (parent / "sampling-journal.json").write_text(json.dumps(journal))
    target = prepare_target(tmp_path / "target", flags, monkeypatch)
    before, source_before = files(target), files(parent)
    with pytest.raises(ValueError, match=r"sampler boundary|bytes changed"):
        CLONE["clone_checkpoint"](parent, target)
    assert files(target) == before
    assert files(parent) == source_before

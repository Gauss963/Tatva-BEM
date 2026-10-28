import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import plan_cq

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CHECK = runpy.run_path(str(TOOLS / "radau_sampling_checkpoint.py"))
RAW = runpy.run_path(str(TOOLS / "raw_layer_sampling.py"))
FIXTURE = runpy.run_path(str(Path(__file__).with_name("test_raw_radau_producer.py")))


def test_mid_interpolated_interval_resume_is_bitwise_and_keeps_every_guard(tmp_path):
    plan = plan_cq(2, 0.1, 96, max_memory_mb=128, disk_spectrum=True)
    count = plan.fft_size // 2 + 1
    options = RAW["RawSamplingOptions"](degree=16)

    def provider(s):
        value = np.eye(2) * (1 + 0.01 * s)
        return RAW["RawLayerPacket"]((value, value, value), (np.eye(2),))

    baseline = []
    baseline_record = RAW["sample_raw_layers"](
        provider,
        lambda p, s: p.families[0],
        plan,
        lambda i, s, value: baseline.append(value.copy()),
        options=options,
    )
    assert baseline_record["interpolated_grid_points"] > 0
    (tmp_path / "preflight.json").write_text('{"model":"same"}\n')

    def execute(journal, interrupt):
        spectrum = journal.spectrum("work-family0.npy")
        writer = CHECK["JournaledSpectrumWriter"](journal, spectrum, matrices=7)
        stack, start = journal.sampler_stack("work-family0.npy", [(0, count)])
        sampler = CHECK["checkpointed_sampler"](
            RAW["sample_raw_layers"],
            stack,
            count,
            start,
            lambda left, right, rest: journal.save_sampler_state(
                "work-family0.npy", left, right, rest, writer
            ),
        )
        original = journal.commit

        def commit(*args):
            original(*args)
            if interrupt:
                raise RuntimeError("simulated interruption after durable chunk")

        journal.commit = commit
        result = sampler(
            provider,
            lambda p, s: p.families[0],
            plan,
            lambda i, s, value: writer.accept(i, value),
            options=options,
        )
        writer.finish()
        return spectrum, result

    journal = CHECK["SamplingJournal"](tmp_path, plan)
    journal.begin()
    with pytest.raises(RuntimeError, match="simulated interruption"):
        execute(journal, True)
    assert journal.prefix("work-family0.npy") == 7
    saved = CHECK["digest_range"](tmp_path / "work-family0.npy", 128, 7 * 2**2 * 16)
    continued = CHECK["SamplingJournal"](tmp_path, plan, resume=True)
    continued.begin()
    spectrum, result = execute(continued, False)
    assert result["all_fft_grid_points_emitted"]
    actual = spectrum[:]
    np.testing.assert_array_equal(actual.view(np.uint64), np.stack(baseline).view(np.uint64))
    assert CHECK["digest_range"](tmp_path / "work-family0.npy", 128, 7 * 2**2 * 16) == saved
    assert continued.data["attempts"][1]["inherited_unfinished_attempts"] == 1


def test_committed_byte_corruption_is_rejected_before_any_rewrite(tmp_path):
    plan = plan_cq(2, 0.1, 4, max_memory_mb=128, disk_spectrum=True)
    (tmp_path / "preflight.json").write_text("{}\n")
    journal = CHECK["SamplingJournal"](tmp_path, plan)
    spectrum = journal.spectrum("work-family0.npy")
    writer = CHECK["JournaledSpectrumWriter"](journal, spectrum, matrices=2)
    for i in range(2):
        writer.append(i, np.eye(2) * (i + 1))
    with spectrum.path.open("r+b") as stream:
        stream.seek(spectrum.offset)
        stream.write(b"\xff")
    before = spectrum.path.read_bytes()
    with pytest.raises(ValueError, match="bytes changed"):
        CHECK["SamplingJournal"](tmp_path, plan, resume=True)
    assert spectrum.path.read_bytes() == before


@pytest.mark.parametrize("failure", ["raw", "combination", "postprocess", "legacy"])
def test_real_full_kernel_resumes_to_uninterrupted_weights_and_startup(
    tmp_path, monkeypatch, failure
):
    producer = runpy.run_path(str(TOOLS / "prepare_raw_radau_resolvent.py"))
    source = FIXTURE["source_metadata"](tmp_path)
    baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
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
    ]

    def execute(folder, extra=()):
        monkeypatch.setattr(sys, "argv", ["prepare", *flags, "--output", str(folder), *extra])
        producer["main"]()

    execute(baseline)
    journal_class = producer["CHECKPOINT"]["SamplingJournal"]
    original_commit = journal_class.commit
    native = producer["NATIVE"]
    original_weights = native["physical_weights_file_inplace"]

    def interrupted_commit(self, name, *args):
        original_commit(self, name, *args)
        if (failure in {"raw", "legacy"} and name == "work-family0.npy") or (
            failure == "combination" and name == "compact-family0.npy"
        ):
            raise RuntimeError("injected durable stage interruption")

    def interrupted_weights(*args):
        result = original_weights(*args)
        result._mmap.close()
        raise RuntimeError("injected durable stage interruption")

    with monkeypatch.context() as patch:
        if failure == "postprocess":
            patch.setitem(native, "physical_weights_file_inplace", interrupted_weights)
        else:
            patch.setattr(journal_class, "commit", interrupted_commit)
        with pytest.raises(RuntimeError, match="injected durable"):
            execute(resumed, ["--durable-build"])
    journal_path = resumed / "sampling-journal.json"
    journal = json.loads(journal_path.read_text())
    if failure == "raw":
        assert journal["channels"]["work-family0.npy"]["chunks"][-1]["last"] == 4
        before = journal_path.read_bytes()
        with pytest.raises(ValueError, match="recorded argument: dt"):
            execute(resumed, ["--durable-build", "--resume", "--dt", "1e-6"])
        assert journal_path.read_bytes() == before
    resume_flags = ["--durable-build", "--resume"]
    if failure == "legacy":
        preflight_path = resumed / "preflight.json"
        old = json.loads(preflight_path.read_text())
        old["frozen_input_sha256"][str(TOOLS / "prepare_raw_radau_resolvent.py")] = producer[
            "LEGACY_PRODUCER_SHA256"
        ]
        old["frozen_input_sha256"].pop(str(TOOLS / "radau_sampling_checkpoint.py"))
        for key in ["resume", "resume_receipt", "durable_build", "verify_resume_only"]:
            old["arguments"].pop(key, None)
        preflight_path.write_text(json.dumps(old))
        journal_path.unlink()
        entry = journal["channels"]["work-family0.npy"]
        prefix = entry["chunks"][-1]["last"]
        receipt = tmp_path / "legacy-receipt.json"
        receipt.write_text(
            json.dumps(
                dict(
                    path=str(resumed / "work-family0.npy"),
                    source_preflight_sha256=CHECK["digest_range"](preflight_path),
                    committed_matrices=prefix,
                    committed_bytes=prefix * 11**2 * 16,
                    committed_bytes_sha256=CHECK["digest_range"](
                        resumed / "work-family0.npy", entry["offset"], prefix * 11**2 * 16
                    ),
                    offset=entry["offset"],
                    shape=entry["shape"],
                    dtype="complex128",
                    fortran=False,
                )
            )
        )
        resume_flags += ["--resume-receipt", str(receipt)]
        before = {p.name: CHECK["digest_range"](p) for p in resumed.iterdir() if p.is_file()}
        execute(resumed, [*resume_flags, "--verify-resume-only"])
        assert {
            p.name: CHECK["digest_range"](p) for p in resumed.iterdir() if p.is_file()
        } == before
        assert not journal_path.exists()
        for key, value in [("source_sha256", {}), ("scipy", "0.invalid")]:
            changed = json.loads(preflight_path.read_text())
            changed["environment"][key] = value
            preflight_path.write_text(json.dumps(changed))
            with pytest.raises(ValueError, match="numerical package, case"):
                execute(resumed, resume_flags)
            preflight_path.write_text(json.dumps(old))
    execute(resumed, resume_flags)
    for name in ["weights0.npy", "weights1.npy", "shift_wrap.npy", "initial_impedance.npy"]:
        a, b = np.load(baseline / name), np.load(resumed / name)
        np.testing.assert_array_equal(
            a.view(np.uint64) if a.flags.c_contiguous else a.ravel(order="C").view(np.uint64),
            b.view(np.uint64) if b.flags.c_contiguous else b.ravel(order="C").view(np.uint64),
        )
    for i in range(2):
        np.testing.assert_array_equal(
            np.load(baseline / f"work-family{i}.npy"), np.load(resumed / f"compact-family{i}.npy")
        )
    with np.load(baseline / "elastic.npz") as a, np.load(resumed / "elastic.npz") as b:
        assert a.files == b.files
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])
    metadata = json.loads((resumed / "summary.json").read_text())
    assert metadata["initial_composition_inf_error"] <= 1e-6
    assert metadata["durable_construction"]["raw_branch_spectra_retained"]
    assert [row["status"] for row in metadata["durable_construction"]["attempts"]] == (
        ["completed"] if failure == "legacy" else ["interrupted_or_failed", "completed"]
    )
    assert not metadata["full_original_event_reproduced"]
    assert not metadata["same_accuracy_fem_advantage_verified"]


def test_exclusive_lease_rejects_a_second_constructor(tmp_path):
    first = CHECK["exclusive_lease"](tmp_path)
    try:
        with pytest.raises(RuntimeError, match="Another process"):
            CHECK["exclusive_lease"](tmp_path)
    finally:
        first.close()

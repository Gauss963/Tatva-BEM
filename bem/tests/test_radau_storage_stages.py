import inspect
import json
import os
import runpy
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import plan_cq
from finiteblock_bem.cq_storage import DiskSpectrum

TOOLS = Path(__file__).resolve().parents[1] / "tools"
RAW = runpy.run_path(str(TOOLS / "raw_radau_stage.py"))
HANDOFF = runpy.run_path(str(TOOLS / "radau_stage_handoff.py"))


def owned_raw(tmp_path):
    source = tmp_path / "raw-stage"
    source.mkdir()
    plan = plan_cq(3, 2e-6, 5, max_memory_mb=128)
    records = []
    for i in range(2):
        spectrum = DiskSpectrum(source / f"work-family{i}.npy", plan)
        spectrum[:] = np.full(spectrum.shape, i + 0.25, dtype=np.complex128)
        records.append({**spectrum.record(), "role": HANDOFF["TX"]["ROLE"], "mode_branch": i})
    return source, tmp_path / "compact-stage", records, plan, tmp_path / "handoff.json"


def test_raw_adapter_preserves_original_main_globals_and_ast_contract():
    original = runpy.run_path(str(RAW["ORIGINAL"]))
    before = original["_main"].__globals__["CHECKPOINT"]
    namespace = RAW["adapted_namespace"]()
    assert original["_main"].__globals__["CHECKPOINT"] is before
    assert namespace["CHECKPOINT"] is not before
    assert namespace["main"].__globals__ is namespace
    assert namespace["_main"].__globals__ is namespace
    assert RAW["raw_required_bytes"](plan_cq(3, 2e-6, 5, max_memory_mb=128)) > 0


def test_raw_adapter_rejects_changed_original_storage_preflight(monkeypatch):
    original = inspect.getsource

    def changed(function):
        text = original(function)
        if function.__name__ == "_main":
            return text.replace(
                "weight_bytes = (steps + 1) * d * d * 8", "weight_bytes = (steps + 1) * d * d * 16"
            )
        return text

    monkeypatch.setattr(inspect, "getsource", changed)
    with pytest.raises(ValueError, match="disk preflight"):
        RAW["adapted_namespace"]()


def test_handoff_moves_both_files_without_copying_or_replacing_inodes(tmp_path):
    source, target, records, plan, journal = owned_raw(tmp_path)
    inodes = [Path(r["path"]).stat().st_ino for r in records]
    result, state = HANDOFF["handoff_raw"](source, target, records, plan, journal)
    assert state["phase"] == "completed"
    assert not list(source.glob("*.npy"))
    for i, record in enumerate(result):
        assert Path(record["path"]).parent == target
        assert Path(record["path"]).stat().st_ino == inodes[i]
        assert record["sha256"] == records[i]["sha256"]


def test_actual_exit_after_first_rename_is_resumed_from_the_same_two_inodes(tmp_path):
    source, target, records, plan, journal = owned_raw(tmp_path)
    config = tmp_path / "test-config.json"
    config.write_text(
        json.dumps(
            dict(
                source=str(source),
                target=str(target),
                records=records,
                plan=asdict(plan),
                journal=str(journal),
            )
        )
    )
    script = """
import json,os,runpy,sys
from finiteblock_bem.cq import CQPlan
d=json.load(open(sys.argv[1]));m=runpy.run_path(sys.argv[2])
def stop(e):
    if e['phase']=='renamed' and e['branch']==0: os._exit(77)
m['handoff_raw'](d['source'],d['target'],d['records'],CQPlan(**d['plan']),d['journal'],progress=stop)
"""
    child = subprocess.run(
        [sys.executable, "-B", "-c", script, str(config), str(TOOLS / "radau_stage_handoff.py")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 77, child.stderr
    assert (target / "work-family0.npy").exists()
    assert (source / "work-family1.npy").exists()
    result, state = HANDOFF["handoff_raw"](source, target, records, plan, journal, resume=True)
    assert state["phase"] == "completed"
    assert all(Path(r["path"]).is_file() for r in result)


@pytest.mark.parametrize("damage", ["bytes", "hardlink", "duplicate", "role"])
def test_invalid_second_branch_is_rejected_before_moving_first(tmp_path, damage):
    source, target, records, plan, journal = owned_raw(tmp_path)
    second = Path(records[1]["path"])
    if damage == "bytes":
        a = np.load(second, mmap_mode="r+")
        a.flat[-1] += 1
        a.flush()
        a._mmap.close()
    elif damage == "hardlink":
        os.link(second, tmp_path / "other-reference.npy")
    elif damage == "duplicate":
        target.mkdir()
        (target / second.name).write_bytes(second.read_bytes())
    else:
        records[1]["role"] = "compact_work_F1"
    with pytest.raises(ValueError):
        HANDOFF["handoff_raw"](source, target, records, plan, journal)
    assert Path(records[0]["path"]).is_file()
    assert second.is_file()
    assert not journal.exists()


def test_changed_handoff_context_after_partial_rename_preserves_both_files(tmp_path):
    source, target, records, plan, journal = owned_raw(tmp_path)

    def stop(event):
        if event["branch"] == 0:
            raise InterruptedError("after rename")

    with pytest.raises(InterruptedError):
        HANDOFF["handoff_raw"](source, target, records, plan, journal, progress=stop)
    saved = journal.read_bytes()
    records[0]["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="context"):
        HANDOFF["handoff_raw"](source, target, records, plan, journal, resume=True)
    assert journal.read_bytes() == saved
    assert (target / "work-family0.npy").is_file() and (source / "work-family1.npy").is_file()


def test_complete_new_producer_matches_original_bem_frequencies_weights_and_startup(tmp_path):
    case = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    common = [
        str(case),
        "--h",
        "160",
        "--contact-h",
        "160",
        "--contact-end-h",
        "160",
        "--corner-h",
        "160",
        "--dt",
        "2e-6",
        "--duration",
        "3.2e-5",
        "--workers",
        "1",
        "--quadrature",
        "12",
        "--fft-rows",
        "2",
        "--combine-matrices",
        "3",
        "--startup-rows",
        "1",
        "--max-workspace-mb",
        "512",
        "--max-memory-mb",
        "1024",
    ]
    environment = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "FDBEM_CORES": "1",
    }
    for name, tool, extra in [
        ("reference", "prepare_raw_radau_resolvent.py", ["--durable-build"]),
        ("candidate", "prepare_storage_radau.py", []),
    ]:
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(TOOLS / tool),
                *common,
                "--output",
                str(tmp_path / name),
                *extra,
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    check = runpy.run_path(str(TOOLS / "compare_storage_radau_kernels.py"))["compare"]
    result = check(tmp_path / "reference", tmp_path / "candidate")
    assert result["complete_raw_branch_receipts_equal"]
    assert result["complete_physical_weight_file_bytes_equal"]
    assert result["complete_causal_startup_fields_bitwise_equal"]
    assert result["candidate_attempt_timings_accounted"]
    assert not list((tmp_path / "candidate/raw-stage").glob("work-family*.npy"))
    assert len(list((tmp_path / "candidate/compact-stage").glob("*.npy"))) == 2
    interrupted = subprocess.run(
        [
            sys.executable,
            "-B",
            str(TOOLS.parent / "tests/fixtures/storage_radau_interrupt.py"),
            "--stop-stage",
            "weights",
            *common,
            "--output",
            str(tmp_path / "recovered"),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert interrupted.returncode == 81, interrupted.stdout + interrupted.stderr
    resumed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(TOOLS / "prepare_storage_radau.py"),
            *common,
            "--output",
            str(tmp_path / "recovered"),
            "--resume",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    check(tmp_path / "reference", tmp_path / "recovered")
    recovered = json.loads((tmp_path / "recovered/summary.json").read_text())
    assert recovered["all_attempt_timings_accounted"] is False
    assert recovered["timing_seconds"]["total_before_reporting"] is None
    assert recovered["timing_seconds"]["known_main_interval_seconds"] > 0


def test_interrupt_fixture_forwards_model_h_and_help_to_the_real_parser():
    child = subprocess.run(
        [
            sys.executable,
            "-B",
            str(TOOLS.parent / "tests/fixtures/storage_radau_interrupt.py"),
            "--stop-stage",
            "handoff",
            "--h",
            "160",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stderr
    assert "--contact-h" in child.stdout

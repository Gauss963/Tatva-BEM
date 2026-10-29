import hashlib
import json
import mmap
import os
import runpy
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import plan_cq
from finiteblock_bem.cq_storage import DiskSpectrum

TOOLS = Path(__file__).resolve().parents[1] / "tools"
MODULE = runpy.run_path(str(TOOLS / "inplace_radau_compaction.py"))
RAW = MODULE["RAW"]


def stage(tmp_path, *, d=5, steps=17):
    folder = tmp_path / "stage"
    folder.mkdir()
    plan = plan_cq(d, 2e-6, steps, max_memory_mb=128)
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    modes = RAW["lower_half_radau_modes"](zeta, 1.0)
    rng = np.random.default_rng(451)
    f = rng.normal(size=(2, len(zeta), d, d)) + 1j * rng.normal(size=(2, len(zeta), d, d))
    f[:, 0] = f[:, 0].real
    f[:, -1] = f[:, -1].real
    arrays = [f[0] + modes[:, i, None, None] * f[1] for i in range(2)]
    records = []
    for i, values in enumerate(arrays):
        disk = DiskSpectrum(folder / f"work-family{i}.npy", plan)
        disk[:] = values
        records.append({**disk.record(), "role": MODULE["ROLE"], "mode_branch": i})
    # Use the original producer's combination and endpoint operation order.
    expected = [np.empty_like(a) for a in arrays]
    for first in range(0, len(zeta), 3):
        last = min(first + 3, len(zeta))
        families = RAW["compact_coefficients"](zeta[first:last], *(a[first:last] for a in arrays))
        for i, values in enumerate(families):
            for endpoint in [0, len(zeta) - 1]:
                if first <= endpoint < last:
                    values[endpoint - first] = values[endpoint - first].real
            expected[i][first:last] = values
    return folder, plan, records, expected


@pytest.mark.parametrize("batch,d,steps", [(1, 3, 17), (3, 5, 33), (8, 9, 65)])
def test_complete_compaction_keeps_all_original_coefficient_bits_and_inodes(
    tmp_path, batch, d, steps
):
    folder, plan, records, expected = stage(tmp_path, d=d, steps=steps)
    inodes = [Path(r["path"]).stat().st_ino for r in records]
    compact, state = MODULE["compact_inplace"](
        records, plan, folder, batch_matrices=batch, max_workspace_mb=128
    )
    assert state["phase"] == "completed"
    assert state["raw_paths_consumed"] and state["raw_sampler_resume_forbidden"]
    assert state["all_attempt_timings_accounted"]
    assert state["resources"]["extra_complete_spectrum_files"] == 0
    assert len(list(folder.glob("*.npy"))) == 2
    assert not list(folder.glob("*.npz"))
    for i, r in enumerate(compact):
        p = Path(r["path"])
        assert p.stat().st_ino == inodes[i]
        np.testing.assert_array_equal(np.load(p).view(np.uint64), expected[i].view(np.uint64))
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"]
        assert r["sha256"] != records[i]["sha256"]
        assert r["all_frequency_matrices_written"]


@pytest.mark.parametrize(
    "phase,family,first,tear_pending",
    [
        ("pending_committed", None, 3, False),
        ("family_written", 0, 3, True),
        ("chunk_committed", None, 0, False),
    ],
)
def test_real_abrupt_process_exit_resumes_mixed_raw_compact_and_torn_pending_range(
    tmp_path, phase, family, first, tear_pending
):
    folder, plan, records, expected = stage(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(
                plan=asdict(plan),
                records=records,
                folder=str(folder),
                target=dict(phase=phase, family=family, first=first),
            )
        )
    )
    script = """
import json,os,runpy,sys
from finiteblock_bem.cq import CQPlan
d=json.load(open(sys.argv[1]))
m=runpy.run_path(sys.argv[2])
def stop(event):
    t=d['target']
    if all(event.get(k)==v for k,v in t.items()): os._exit(73)
m['compact_inplace'](d['records'],CQPlan(**d['plan']),d['folder'],
    batch_matrices=3,max_workspace_mb=128,progress=stop)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            str(config),
            str(TOOLS / "inplace_radau_compaction.py"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 73, result.stderr
    if tear_pending:
        data = np.load(records[1]["path"], mmap_mode="r+")
        data[first, 0, 0] = complex(1e100, -1e100)
        data.flush()
        data._mmap.close()
    _, state = MODULE["compact_inplace"](
        records, plan, folder, batch_matrices=3, max_workspace_mb=128, resume=True
    )
    for r, values in zip(records, expected, strict=True):
        np.testing.assert_array_equal(np.load(r["path"]).view(np.uint64), values.view(np.uint64))
    assert not state["all_attempt_timings_accounted"]
    assert state["attempts"][0]["status"] == "running"
    assert "seconds" not in state["attempts"][0]


@pytest.mark.parametrize("damage", ["compact", "raw_suffix", "backup"])
def test_unaffected_committed_raw_or_backup_corruption_rejected_before_any_mutation(
    tmp_path, damage
):
    folder, plan, records, _ = stage(tmp_path)

    def stop(event):
        if event["phase"] == "family_written" and event["family"] == 0 and event["first"] == 3:
            raise InterruptedError("one family of second chunk written")

    with pytest.raises(InterruptedError):
        MODULE["compact_inplace"](
            records, plan, folder, batch_matrices=3, max_workspace_mb=128, progress=stop
        )
    if damage == "backup":
        p = folder / "raw-write-ahead.npz"
        b = bytearray(p.read_bytes())
        b[-1] ^= 1
        p.write_bytes(b)
    else:
        a = np.load(records[0]["path"], mmap_mode="r+")
        a[0 if damage == "compact" else 9, 0, 0] += 1
        a.flush()
        a._mmap.close()
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.name != "construction.lock"}
    with pytest.raises(ValueError, match="changed"):
        MODULE["compact_inplace"](
            records, plan, folder, batch_matrices=3, max_workspace_mb=128, resume=True
        )
    assert all((folder / k).read_bytes() == v for k, v in before.items())


@pytest.mark.parametrize("change", ["dt", "receipt", "runtime", "batch"])
def test_changed_context_is_rejected_without_reinterpreting_compact_as_raw(
    tmp_path, change, monkeypatch
):
    folder, plan, records, _ = stage(tmp_path)
    MODULE["compact_inplace"](records, plan, folder, batch_matrices=3, max_workspace_mb=128)
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.name != "construction.lock"}
    batch = 3
    if change == "dt":
        plan = replace(plan, dt=plan.dt / 2)
    elif change == "receipt":
        records[0]["sha256"] = "a" * 64
    elif change == "runtime":
        monkeypatch.setattr(np, "__version__", "changed")
    else:
        batch = 4
    with pytest.raises(ValueError, match="changes"):
        MODULE["compact_inplace"](
            records, plan, folder, batch_matrices=batch, max_workspace_mb=128, resume=True
        )
    assert all((folder / k).read_bytes() == v for k, v in before.items())


@pytest.mark.parametrize("bad", ["incomplete", "producer", "hardlink", "compact_role", "raw_bytes"])
def test_refuses_incomplete_inputs_original_producer_and_aliased_files(tmp_path, bad):
    folder, plan, records, _ = stage(tmp_path)
    if bad == "incomplete":
        records[0]["all_frequency_matrices_written"] = False
    elif bad == "producer":
        (folder / "sampling-journal.json").write_text("{}")
    elif bad == "hardlink":
        os.link(records[0]["path"], tmp_path / "other-reference.npy")
    elif bad == "raw_bytes":
        records[0]["sha256"] = "a" * 64
    else:
        records[0]["role"] = "compact_work_F0"
    before = {p.name: p.read_bytes() for p in folder.glob("*.npy")}
    with pytest.raises(ValueError):
        MODULE["compact_inplace"](records, plan, folder, max_workspace_mb=128)
    assert all((folder / k).read_bytes() == v for k, v in before.items())
    assert not (folder / "inplace-compaction-journal.json").exists()


def test_final_whole_file_receipt_also_checks_every_previously_committed_chunk(tmp_path):
    folder, plan, records, _ = stage(tmp_path)
    damaged = False

    def damage(event):
        nonlocal damaged
        if not damaged and event["phase"] == "family_written" and event["first"] == 3:
            a = np.load(records[0]["path"], mmap_mode="r+")
            a[0, 0, 0] += 1
            a.flush()
            a._mmap.close()
            damaged = True

    with pytest.raises(ValueError, match="final receipt"):
        MODULE["compact_inplace"](
            records, plan, folder, batch_matrices=3, max_workspace_mb=128, progress=damage
        )
    state = json.loads((folder / "inplace-compaction-journal.json").read_text())
    assert state["phase"] != "completed"
    assert "compact_records" not in state


def test_memory_preflight_precedes_square_arrays_and_write_mappings_stay_bounded(
    tmp_path, monkeypatch
):
    folder, plan, records, _ = stage(tmp_path)
    with pytest.raises(ValueError, match="workspace"):
        MODULE["compact_inplace"](records, replace(plan, dofs=100000), folder, max_workspace_mb=128)
    with pytest.raises(ValueError, match="workspace"):
        MODULE["compact_inplace"](
            records, replace(plan, fft_size=2**34), folder, max_workspace_mb=128
        )
    original = mmap.mmap
    lengths = []

    def mapping(fd, length, *args, **kwargs):
        if kwargs.get("access") == mmap.ACCESS_WRITE:
            lengths.append(length)
        return original(fd, length, *args, **kwargs)

    monkeypatch.setattr(mmap, "mmap", mapping)
    MODULE["compact_inplace"](records, plan, folder, batch_matrices=3, max_workspace_mb=128)
    assert max(lengths) <= 3 * plan.dofs**2 * 16 + mmap.ALLOCATIONGRANULARITY

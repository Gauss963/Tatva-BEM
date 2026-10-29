import hashlib
import json
import mmap
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import plan_cq
from finiteblock_bem.cq_storage import DiskSpectrum

TOOLS = Path(__file__).resolve().parents[1] / "tools"
DIRECT = runpy.run_path(str(TOOLS / "direct_fortran_radau.py"))
OLD = runpy.run_path(str(TOOLS / "compact_radau_storage.py"))
PHYSICAL = runpy.run_path(str(TOOLS / "bounded_weight_conversion.py"))["convert_physical_weights"]
FORTRAN = runpy.run_path(str(TOOLS / "fortran_weight_storage.py"))["convert_weights_fortran"]


def sources(tmp_path, d=5, steps=17):
    plan = plan_cq(d, 0.02, steps, max_memory_mb=128)
    rng = np.random.default_rng(951)
    arrays = []
    records = []
    for family in range(2):
        values = rng.normal(size=(plan.fft_size // 2 + 1, d, d)) + 1j * rng.normal(
            size=(plan.fft_size // 2 + 1, d, d)
        )
        values[0] = values[0].real
        values[-1] = values[-1].real
        disk = DiskSpectrum(tmp_path / f"family{family}.npy", plan)
        disk[:] = values
        records.append(disk.record())
        arrays.append(values)
    return plan, arrays, records, np.geomspace(0.1, 200, d)


def reference(arrays, plan, line, path, rows):
    paths, wrap, _ = OLD["invert_families_to_files"](
        *arrays, plan, path, fft_rows=rows, max_workspace_mb=128
    )
    for p in paths:
        a = PHYSICAL(p, line)
        a._mmap.close()
        target = p.with_name(p.stem + "-fortran.npy")
        FORTRAN(p, target, max_workspace_mb=128, max_rows=2)
        target.replace(p)
    wrap /= np.sqrt(line[:, None] * line[None, :])
    return paths, wrap


@pytest.mark.parametrize("d,steps,rows", [(3, 17, 1), (9, 65, 4)])
def test_direct_ifft_physical_fortran_files_and_wrap_preserve_original_bits(
    tmp_path, d, steps, rows
):
    plan, arrays, receipts, line = sources(tmp_path, d, steps)
    old, old_wrap = reference(arrays, plan, line, tmp_path / "old", rows)
    paths, wrap, state = DIRECT["invert_physical_fortran"](
        receipts, plan, line, tmp_path / "new", fft_rows=rows, time_tile=7, max_workspace_mb=128
    )
    for a, b in zip(old, paths, strict=True):
        assert a.read_bytes() == b.read_bytes()
        values = np.load(b)
        assert values.flags.f_contiguous
    np.testing.assert_array_equal(wrap.view(np.uint64), old_wrap.view(np.uint64))
    assert state["phase"] == "completed"
    assert state["all_attempt_timings_accounted"]
    assert state["resources"]["extra_full_time_weight_files"] == 0
    assert {p.name for p in (tmp_path / "new").glob("*.npy")} == {
        "weights0.npy",
        "weights1.npy",
        "shift_wrap.npy",
    }
    for receipt in receipts:
        assert hashlib.sha256(Path(receipt["path"]).read_bytes()).hexdigest() == receipt["sha256"]
    # The wrap's sqrt(product) must not be silently replaced by product(sqrt).
    assert np.any(
        np.sqrt(line[:, None] * line[None, :]).view(np.uint64)
        != (np.sqrt(line)[:, None] * np.sqrt(line)[None, :]).view(np.uint64)
    )


@pytest.mark.parametrize(
    "phase,family,first",
    [
        ("column_written", 0, 0),
        ("column_written", 0, 2),
        ("stripe_committed", 1, 0),
        ("column_written", 1, 2),
    ],
)
def test_interrupted_columns_or_committed_stripes_resume_bitwise(tmp_path, phase, family, first):
    plan, _, receipts, line = sources(tmp_path)
    full = tmp_path / "full"
    output = tmp_path / "interrupted"
    expected, wrap, _ = DIRECT["invert_physical_fortran"](
        receipts, plan, line, full, fft_rows=2, max_workspace_mb=128
    )

    def stop(event):
        if event["phase"] == phase and event["family"] == family and event["first"] == first:
            raise InterruptedError("controlled interruption")

    with pytest.raises(InterruptedError):
        DIRECT["invert_physical_fortran"](
            receipts, plan, line, output, fft_rows=2, max_workspace_mb=128, progress=stop
        )
    actual, restored, state = DIRECT["invert_physical_fortran"](
        receipts, plan, line, output, fft_rows=2, max_workspace_mb=128, resume=True
    )
    for a, b in zip(expected, actual, strict=True):
        assert a.read_bytes() == b.read_bytes()
    np.testing.assert_array_equal(restored.view(np.uint64), wrap.view(np.uint64))
    assert len(state["attempts"]) == 2
    assert state["attempts"][0]["status"] == "interrupted_or_failed"


def test_interrupted_array_allocation_is_rebuilt_without_touching_input(tmp_path, monkeypatch):
    plan, _, receipts, line = sources(tmp_path)
    create = DIRECT["invert_physical_fortran"].__globals__["create_array"]

    def stop(*args, **kwargs):
        create(*args, **kwargs)
        raise InterruptedError("allocation interrupted")

    monkeypatch.setitem(DIRECT["invert_physical_fortran"].__globals__, "create_array", stop)
    output = tmp_path / "partial"
    with pytest.raises(InterruptedError):
        DIRECT["invert_physical_fortran"](receipts, plan, line, output, max_workspace_mb=128)
    monkeypatch.setitem(DIRECT["invert_physical_fortran"].__globals__, "create_array", create)
    _, _, state = DIRECT["invert_physical_fortran"](
        receipts, plan, line, output, max_workspace_mb=128, resume=True
    )
    assert state["phase"] == "completed"


@pytest.mark.parametrize("target", ["weights0.npy", "shift_wrap.npy"])
def test_committed_corruption_is_rejected_without_repairing_data_or_journal(tmp_path, target):
    plan, _, receipts, line = sources(tmp_path)
    output = tmp_path / "data"
    DIRECT["invert_physical_fortran"](receipts, plan, line, output, max_workspace_mb=128)
    array = np.load(output / target, mmap_mode="r+")
    array.flat[0] += 1
    array.flush()
    array._mmap.close()
    before = (output / target).read_bytes()
    journal = (output / "direct-fortran-journal.json").read_bytes()
    with pytest.raises(ValueError, match="changed"):
        DIRECT["invert_physical_fortran"](
            receipts, plan, line, output, max_workspace_mb=128, resume=True
        )
    assert (output / target).read_bytes() == before
    assert (output / "direct-fortran-journal.json").read_bytes() == journal


@pytest.mark.parametrize("change", ["line", "dt", "input", "runtime"])
def test_changed_mathematical_input_or_runtime_rejected_before_output_mutation(
    tmp_path, change, monkeypatch
):
    plan, _, receipts, line = sources(tmp_path)
    output = tmp_path / "data"
    DIRECT["invert_physical_fortran"](receipts, plan, line, output, max_workspace_mb=128)
    before = {p.name: p.read_bytes() for p in output.glob("*.npy")}
    journal = (output / "direct-fortran-journal.json").read_bytes()
    if change == "line":
        line = line * 2
    elif change == "dt":
        plan = replace(plan, dt=plan.dt / 2)
    elif change == "runtime":
        monkeypatch.setattr(np, "__version__", "different-runtime")
    else:
        p = Path(receipts[0]["path"])
        data = np.load(p, mmap_mode="r+")
        data.flat[-1] += 1
        data.flush()
        data._mmap.close()
    with pytest.raises(ValueError):
        DIRECT["invert_physical_fortran"](
            receipts, plan, line, output, max_workspace_mb=128, resume=True
        )
    assert all((output / name).read_bytes() == value for name, value in before.items())
    assert (output / "direct-fortran-journal.json").read_bytes() == journal


def test_writable_mappings_are_bounded_and_second_owner_is_rejected(tmp_path, monkeypatch):
    plan, _, receipts, line = sources(tmp_path, d=17, steps=77)
    original = mmap.mmap
    lengths = []
    output = tmp_path / "data"
    attempted = False

    def mapping(fd, length, *args, **kwargs):
        if kwargs.get("access") == mmap.ACCESS_WRITE:
            lengths.append(length)
        return original(fd, length, *args, **kwargs)

    monkeypatch.setattr(mmap, "mmap", mapping)

    def competing(event):
        nonlocal attempted
        if not attempted:
            attempted = True
            with pytest.raises(RuntimeError, match="owns"):
                DIRECT["invert_physical_fortran"](
                    receipts, plan, line, output, fft_rows=3, max_workspace_mb=128, resume=True
                )

    DIRECT["invert_physical_fortran"](
        receipts, plan, line, output, fft_rows=3, max_workspace_mb=128, progress=competing
    )
    assert (
        lengths
        and max(lengths) <= max(plan.steps + 1, plan.dofs) * 3 * 8 + mmap.ALLOCATIONGRANULARITY
    )


def test_workspace_guard_precedes_square_array_allocations_and_incomplete_receipts(
    tmp_path, monkeypatch
):
    plan = plan_cq(2, 0.02, 17, max_memory_mb=128)
    large = replace(plan, dofs=100000)

    def forbidden(*args, **kwargs):
        raise AssertionError("Allocated normalization before preflight")

    monkeypatch.setattr(np, "sqrt", forbidden)
    with pytest.raises(ValueError, match="workspace"):
        DIRECT["invert_physical_fortran"](
            [], large, np.ones(large.dofs), tmp_path / "oversized", max_workspace_mb=128
        )
    assert not (tmp_path / "oversized").exists()


def test_incomplete_receipt_creates_no_output_and_unknown_prior_end_is_not_fabricated(tmp_path):
    plan, _, receipts, line = sources(tmp_path)
    receipts[0]["all_frequency_matrices_written"] = False
    with pytest.raises(ValueError, match="complete"):
        DIRECT["invert_physical_fortran"](
            receipts, plan, line, tmp_path / "incomplete", max_workspace_mb=128
        )
    assert not (tmp_path / "incomplete").exists()
    receipts[0]["all_frequency_matrices_written"] = True
    output = tmp_path / "data"
    DIRECT["invert_physical_fortran"](receipts, plan, line, output, max_workspace_mb=128)
    path = output / "direct-fortran-journal.json"
    state = json.loads(path.read_text())
    state["attempts"].append(dict(status="running", started_utc="unknown-terminal-time"))
    path.write_text(json.dumps(state))
    _, _, restored = DIRECT["invert_physical_fortran"](
        receipts, plan, line, output, max_workspace_mb=128, resume=True
    )
    assert not restored["all_attempt_timings_accounted"]
    assert "seconds" not in restored["attempts"][-2]

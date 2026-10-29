"""Consume owned complete raw branches into compact F0/F1 without a full copy.

The staging folder initially contains only work-family0/1.npy. It must not be
an original producer directory. A bounded raw write-ahead backup permits
roll-forward after an abrupt stop between writes of the two compact families.
Original raw receipts remain provenance; their paths now contain COMPACT data.
"""

import hashlib
import json
import mmap
import os
import runpy
import shutil
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

TOOLS = Path(__file__).resolve().parent
CHECKPOINT = runpy.run_path(str(TOOLS / "radau_sampling_checkpoint.py"))
RAW = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))
NAMES = [f"work-family{i}.npy" for i in range(2)]
ROLE = "complete_raw_radau_mode_branch"


def sync_directory(folder):
    fd = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(path, state):
    CHECKPOINT["atomic_json"](path, state)
    sync_directory(path.parent)


def compaction_resources(plan, batch_matrices=8):
    count = plan.fft_size // 2 + 1
    batch = min(batch_matrices, count)
    raw_pair = 2 * batch * plan.dofs**2 * 16
    arrays = 6 * raw_pair + 64 * 1024**2
    # Reserve Python dictionaries, hashes, ranges and JSON publication copies.
    metadata = 8192 * ((count + batch - 1) // batch)
    return dict(
        batch_matrices=batch,
        array_peak_bytes=arrays,
        journal_metadata_reserve_bytes=metadata,
        estimated_workspace_bytes=arrays + metadata,
        raw_backup_payload_bytes=raw_pair,
        maximum_extra_disk_bytes=2 * raw_pair + 64 * 1024**2,
        extra_complete_spectrum_files=0,
        all_frequency_matrices_retained=True,
    )


def block_hash(path, offset, first, last, matrix_bytes):
    return CHECKPOINT["digest_range"](
        path, offset + first * matrix_bytes, (last - first) * matrix_bytes
    )


def scan_file(path, offset, ranges, matrix_bytes, expected=None):
    """Bind chunk receipts and the complete file receipt in one bounded read."""
    whole, chunks = hashlib.sha256(), []
    with path.open("rb") as stream:
        whole.update(stream.read(offset))
        for index, (first, last) in enumerate(ranges):
            if stream.tell() != offset + first * matrix_bytes:
                raise ValueError("Scan ranges omit or overlap frequency matrices")
            chunk = hashlib.sha256()
            remaining = (last - first) * matrix_bytes
            while remaining:
                data = stream.read(min(4 * 1024**2, remaining))
                if not data:
                    raise ValueError("Truncated spectrum during receipt scan")
                whole.update(data)
                chunk.update(data)
                remaining -= len(data)
            value = chunk.hexdigest()
            if expected is not None and value != expected[index]:
                raise ValueError("Compact chunk changed during final receipt scan")
            chunks.append(value)
        if stream.read(1):
            raise ValueError("Unexpected trailing spectrum data")
    return whole.hexdigest(), chunks


def write_block(path, offset, first, values):
    """Map only this complete-matrix batch, flush, then fsync its file."""
    start = offset + first * values.shape[1] ** 2 * 16
    aligned = start // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
    with path.open("r+b") as stream:
        mapping = mmap.mmap(
            stream.fileno(),
            start - aligned + values.nbytes,
            offset=aligned,
            access=mmap.ACCESS_WRITE,
        )
        target = np.ndarray(values.shape, np.complex128, buffer=mapping, offset=start - aligned)
        try:
            target[:] = values
            mapping.flush()
        finally:
            del target
            mapping.close()
        os.fsync(stream.fileno())


def read_backup(path, pending, chunk, d):
    if CHECKPOINT["digest_range"](path) != pending["sha256"]:
        raise ValueError("Raw write-ahead backup changed")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"branch0", "branch1"}:
            raise ValueError("Invalid raw write-ahead backup arrays")
        arrays = [saved[f"branch{i}"] for i in range(2)]
    for i, values in enumerate(arrays):
        if (
            values.dtype != np.complex128
            or values.shape != (chunk["last"] - chunk["first"], d, d)
            or not np.all(np.isfinite(values))
            or hashlib.sha256(values.tobytes()).hexdigest() != chunk["raw_sha256"][i]
        ):
            raise ValueError("Raw backup does not match the original committed chunk")
    return arrays


def compact_inplace(
    raw_records,
    plan,
    folder,
    *,
    batch_matrices=8,
    max_workspace_mb=1024,
    resume=False,
    progress=None,
):
    """Return complete compact receipts and the transaction journal.

    This consumes two singly linked files owned by an independent staging
    folder. Raw receipts require role=ROLE and mode_branch=0/1. The original
    raw sampler must never resume these paths after ownership is transferred.
    No input copy, numerical interpolation or frequency assembly is performed.
    """
    begun = perf_counter()
    folder = Path(folder).resolve()
    if folder.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Staging must be outside protected Tatva")
    if (
        not isinstance(batch_matrices, int)
        or isinstance(batch_matrices, bool)
        or batch_matrices < 1
        or not np.isfinite(max_workspace_mb)
        or max_workspace_mb <= 0
        or not np.isfinite(plan.dt)
        or plan.dt <= 0
        or not 0 < plan.radius < 1
        or not 1 <= plan.steps < plan.fft_size
        or plan.dofs < 2
        or plan.fft_size % 2
    ):
        raise ValueError("Require a valid CQ plan and positive batch/workspace limits")
    resources = compaction_resources(plan, batch_matrices)
    if resources["estimated_workspace_bytes"] > max_workspace_mb * 1024**2:
        raise ValueError("In-place compaction exceeds its array workspace budget")
    if not folder.is_dir() or len(raw_records) != 2:
        raise ValueError("Require an existing owned staging folder and both raw receipts")
    state_path = folder / "inplace-compaction-journal.json"
    if (folder / "sampling-journal.json").exists() or (folder / "preflight.json").exists():
        raise ValueError("Refuse original producer directories; transfer ownership to new staging")
    if resume:
        if not state_path.is_file():
            raise ValueError("Resume requires this component's existing journal")
    elif any(p.name not in [*NAMES, "construction.lock"] for p in folder.iterdir()):
        raise ValueError("Fresh staging may contain only its two owned raw branches")
    paths, sources, headers = [], [], []
    for i, item in enumerate(raw_records):
        path = folder / NAMES[i]
        stat = path.stat()
        info = CHECKPOINT["header"](path, plan)
        if (
            path.is_symlink()
            or stat.st_nlink != 1
            or Path(item["path"]).resolve() != path
            or item.get("role") != ROLE
            or item.get("mode_branch") != i
            or item.get("all_frequency_matrices_written") is not True
            or item.get("shape") != info["shape"]
            or item.get("bytes") != stat.st_size
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
        ):
            raise ValueError("Require complete singly linked raw branches owned by this staging")
        paths.append(path)
        headers.append(info)
        sources.append({**item, "device": stat.st_dev, "inode": stat.st_ino})
    if sources[0]["inode"] == sources[1]["inode"]:
        raise ValueError("Raw branch files must be distinct")
    context = dict(
        plan=dict(
            dt=plan.dt, steps=plan.steps, dofs=plan.dofs, fft_size=plan.fft_size, radius=plan.radius
        ),
        original_raw_sources=sources,
        batch_matrices=resources["batch_matrices"],
        numpy=np.__version__,
        scipy=scipy.__version__,
        source_sha256={
            str(p): CHECKPOINT["digest_range"](p)
            for p in [
                Path(__file__).resolve(),
                TOOLS / "radau_raw_sampling.py",
                TOOLS / "radau_sampling_checkpoint.py",
            ]
        },
    )
    lease = CHECKPOINT["exclusive_lease"](folder)
    state, attempt = None, None
    wal = folder / "raw-write-ahead.npz"
    temporary = folder / "raw-write-ahead.tmp"
    matrix_bytes = plan.dofs**2 * 16
    count = plan.fft_size // 2 + 1
    batch = resources["batch_matrices"]
    expected_ranges = [(i, min(i + batch, count)) for i in range(0, count, batch)]

    def emit(phase, **kwargs):
        if progress is not None:
            progress(dict(phase=phase, **kwargs))

    def actual_hashes(chunk):
        return [
            block_hash(p, h["offset"], chunk["first"], chunk["last"], matrix_bytes)
            for p, h in zip(paths, headers, strict=True)
        ]

    try:
        if resume:
            state = json.loads(state_path.read_text())
            if (
                state["schema_version"] != 1
                or state["context"] != context
                or state["phase"] not in ["compacting", "completed"]
            ):
                raise ValueError(
                    "In-place resume changes mathematical inputs, ownership or runtime"
                )
        else:
            chunks = [dict(first=a, last=b, raw_sha256=[]) for a, b in expected_ranges]
            for source, path, h in zip(sources, paths, headers, strict=True):
                whole, digests = scan_file(path, h["offset"], expected_ranges, matrix_bytes)
                if whole != source["sha256"]:
                    raise ValueError("Complete raw branch bytes changed")
                for chunk, value in zip(chunks, digests, strict=True):
                    chunk["raw_sha256"].append(value)
            state = dict(
                schema_version=1,
                context=context,
                phase="compacting",
                chunks=chunks,
                entries=[],
                pending=None,
                attempts=[],
                resources=resources,
                raw_paths_consumed=True,
                raw_sampler_resume_forbidden=True,
            )
        if [(c["first"], c["last"]) for c in state["chunks"]] != expected_ranges:
            raise ValueError("Raw chunk manifest omits or overlaps frequencies")
        done = len(state["entries"])
        if done > len(expected_ranges):
            raise ValueError("Invalid committed compact prefix")
        pending = state["pending"]
        backup = None
        if pending is not None:
            if pending["chunk"] != done or done == len(expected_ranges):
                raise ValueError("Write-ahead backup is outside the uncommitted chunk")
            backup = read_backup(wal, pending, state["chunks"][done], plan.dofs)
        # Validate ALL unaffected data before modifying a pending range or journal.
        for index, chunk in enumerate(state["chunks"] if resume else []):
            if index < done:
                entry = state["entries"][index]
                if (entry["first"], entry["last"]) != expected_ranges[index] or actual_hashes(
                    chunk
                ) != entry["compact_sha256"]:
                    raise ValueError("Committed compact chunk changed")
            elif pending is None or index != done:
                if actual_hashes(chunk) != chunk["raw_sha256"]:
                    raise ValueError("Unconsumed raw chunk changed")
        if state["phase"] == "completed" and (done != len(expected_ranges) or pending is not None):
            raise ValueError("Incomplete completed compaction journal")
        if shutil.disk_usage(folder).free < resources["maximum_extra_disk_bytes"]:
            raise ValueError("Insufficient disk for bounded raw write-ahead backups")
        attempt = dict(
            started_utc=datetime.now(UTC).isoformat(),
            initial_completed_chunks=done,
            preparation_including_verification_seconds=perf_counter() - begun,
            status="running",
        )
        state["attempts"].append(attempt)
        publish(state_path, state)
        for index in range(done, len(state["chunks"])):
            chunk = state["chunks"][index]
            first, last = chunk["first"], chunk["last"]
            if backup is None:
                arrays = []
                for path, h in zip(paths, headers, strict=True):
                    with path.open("rb") as stream:
                        stream.seek(h["offset"] + first * matrix_bytes)
                        values = np.fromfile(
                            stream, np.complex128, count=(last - first) * plan.dofs**2
                        ).reshape(last - first, plan.dofs, plan.dofs)
                    arrays.append(values)
                for i, values in enumerate(arrays):
                    if hashlib.sha256(values.tobytes()).hexdigest() != chunk["raw_sha256"][i]:
                        raise ValueError("Raw chunk changed before write-ahead backup")
                with temporary.open("wb") as stream:
                    np.savez(stream, branch0=arrays[0], branch1=arrays[1])
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(wal)
                sync_directory(folder)
                state["pending"] = dict(chunk=index, sha256=CHECKPOINT["digest_range"](wal))
                publish(state_path, state)
                del arrays, values
                backup = read_backup(wal, state["pending"], chunk, plan.dofs)
            emit("pending_committed", first=first, last=last)
            zeta = plan.radius * np.exp(-2j * np.pi * np.arange(first, last) / plan.fft_size)
            families = RAW["compact_coefficients"](zeta, *backup)
            backup = None
            output_hashes = []
            for family, values in enumerate(families):
                for endpoint in [0, plan.fft_size // 2]:
                    if first <= endpoint < last:
                        value = values[endpoint - first]
                        if np.linalg.norm(value.imag) > 1e-10 * max(np.linalg.norm(value.real), 1):
                            raise ValueError("Compact family violates real endpoint symmetry")
                        values[endpoint - first] = value.real
                        del value
                if not np.all(np.isfinite(values)):
                    raise ValueError("Nonfinite compact family")
                expected = hashlib.sha256(values.tobytes()).hexdigest()
                write_block(paths[family], headers[family]["offset"], first, values)
                if (
                    block_hash(paths[family], headers[family]["offset"], first, last, matrix_bytes)
                    != expected
                ):
                    raise ValueError("Compact output changed numerical bits")
                output_hashes.append(expected)
                emit("family_written", family=family, first=first, last=last)
            del families, values
            state["entries"].append(dict(first=first, last=last, compact_sha256=output_hashes))
            state["pending"] = None
            publish(state_path, state)
            emit("chunk_committed", first=first, last=last)
            wal.unlink(missing_ok=True)
            sync_directory(folder)
        for path, source in zip(paths, sources, strict=True):
            if (path.stat().st_dev, path.stat().st_ino) != (source["device"], source["inode"]):
                raise ValueError("Owned spectrum was replaced during compaction")
        if any(CHECKPOINT["digest_range"](p) != v for p, v in context["source_sha256"].items()):
            raise ValueError("Compaction numerical source changed")
        records = []
        for i, path in enumerate(paths):
            whole, _ = scan_file(
                path,
                headers[i]["offset"],
                expected_ranges,
                matrix_bytes,
                expected=[e["compact_sha256"][i] for e in state["entries"]],
            )
            records.append(
                dict(
                    path=str(path),
                    shape=headers[i]["shape"],
                    bytes=path.stat().st_size,
                    sha256=whole,
                    all_frequency_matrices_written=True,
                    role="compact_work_F0" if i == 0 else "compact_work_F1",
                )
            )
        state.update(
            phase="completed",
            compact_records=records,
            all_attempt_timings_accounted=all(
                a is attempt or "seconds" in a for a in state["attempts"]
            ),
        )
        attempt.update(status="completed", seconds=perf_counter() - begun)
        publish(state_path, state)
        # A stop after commit may leave this obsolete backup; it is never reused.
        wal.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        sync_directory(folder)
        return records, state
    except BaseException as error:
        if attempt is not None:
            attempt.update(
                status="interrupted_or_failed",
                seconds=perf_counter() - begun,
                error=f"{type(error).__name__}: {error}",
            )
            publish(state_path, state)
        raise
    finally:
        lease.close()

"""Durable direct IFFT -> physical Fortran-order compact Radau weights.

No C-order time-weight copy or later storage conversion is needed. Every lag
and the finite-DFT wrap are retained. Preserve the original operation order:
weights divide by sqrt(w_i)*sqrt(w_j), wrap by sqrt(w_i*w_j).
Inputs are completed WORK-coordinate compact families with SHA256 receipts.
This independent component does not change the running kernel producer.
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
from scipy.fft import irfft

TOOLS = Path(__file__).resolve().parent
JOURNAL = runpy.run_path(str(TOOLS / "radau_sampling_checkpoint.py"))


def array_header(path):
    with Path(path).open("rb") as stream:
        if np.lib.format.read_magic(stream) != (2, 0):
            raise ValueError("Require version-2 NPY arrays")
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        offset = stream.tell()
    if Path(path).stat().st_size != offset + int(np.prod(shape)) * dtype.itemsize:
        raise ValueError("NPY array length differs from its header")
    return shape, fortran, dtype, offset


def create_array(path, shape, *, fortran):
    with Path(path).open("xb") as stream:
        np.lib.format.write_array_header_2_0(
            stream, dict(shape=tuple(shape), fortran_order=fortran, descr=np.dtype("float64").str)
        )
        stream.truncate(stream.tell() + int(np.prod(shape)) * 8)
        stream.flush()
        os.fsync(stream.fileno())


def mapped_write(stream, offset, shape, values, *, fortran):
    aligned = offset // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
    length = offset - aligned + int(np.prod(shape)) * 8
    mapping = mmap.mmap(stream.fileno(), length, offset=aligned, access=mmap.ACCESS_WRITE)
    target = np.ndarray(
        shape, np.float64, buffer=mapping, offset=offset - aligned, order="F" if fortran else "C"
    )
    try:
        target[:] = values
        mapping.flush()
    finally:
        del target
        mapping.close()
    return length


def stripe_digest(path, header, n, d, first, last):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for column in range(d):
            stream.seek(header + (column * d + first) * n * 8)
            remaining = (last - first) * n * 8
            while remaining:
                block = stream.read(min(4 * 1024**2, remaining))
                if not block:
                    raise ValueError("Truncated committed Fortran stripe")
                digest.update(block)
                remaining -= len(block)
    return digest.hexdigest()


def expected_stripe_digest(tile):
    digest = hashlib.sha256()
    for column in range(tile.shape[2]):
        digest.update(tile[:, :, column].tobytes(order="F"))
    return digest.hexdigest()


def inversion_resources(plan, fft_rows=2):
    rows = min(plan.dofs, fft_rows)
    n = plan.steps + 1
    half = plan.fft_size // 2 + 1
    d = plan.dofs
    peak = (
        4 * plan.fft_size * rows * d * 8
        + 2 * half * rows * d * 16
        + 2 * n * rows * d * 8
        + 4 * d * d * 8
        + 64 * 1024**2
    )
    return dict(
        array_peak_bytes=peak,
        weight_data_bytes=2 * n * d * d * 8,
        wrap_data_bytes=d * d * 8,
        fft_rows=rows,
        extra_full_time_weight_files=0,
        all_requested_lags_retained=True,
    )


def invert_physical_fortran(
    family_records,
    plan,
    line_weights,
    folder,
    *,
    fft_rows=2,
    time_tile=128,
    max_workspace_mb=2048,
    resume=False,
    progress=None,
):
    """Return physical family paths, physical wrap and a verified journal.

    Receipts must describe complete C-order complex128 F0/F1 arrays. Check
    their actual bytes before writing. A resume verifies every committed
    output stripe before recomputing any uncommitted stripe.
    """
    begun = perf_counter()
    folder = Path(folder).resolve()
    if folder.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Output must be outside protected Tatva")
    if (
        not isinstance(fft_rows, int)
        or isinstance(fft_rows, bool)
        or fft_rows < 1
        or not isinstance(time_tile, int)
        or isinstance(time_tile, bool)
        or time_tile < 1
        or not np.isfinite(max_workspace_mb)
        or max_workspace_mb <= 0
        or not np.isfinite(plan.dt)
        or plan.dt <= 0
        or not 0 < plan.radius < 1
        or not 1 <= plan.steps < plan.fft_size
        or plan.dofs < 2
        or plan.fft_size % 2
    ):
        raise ValueError("Require a valid CQ plan and positive row/tile/workspace limits")
    line = np.asarray(line_weights)
    if (
        np.iscomplexobj(line)
        or line.shape != (plan.dofs,)
        or not np.all(np.isfinite(line))
        or np.any(line <= 0)
    ):
        raise ValueError("Require positive finite real work-coordinate line weights")
    line = line.astype(np.float64)
    resources = inversion_resources(plan, fft_rows)
    if resources["array_peak_bytes"] > max_workspace_mb * 1024**2:
        raise ValueError("Direct Fortran IFFT exceeds its array workspace budget")
    root = np.sqrt(line)
    with np.errstate(over="ignore"):
        wrap_scale = np.sqrt(line[:, None] * line[None, :])
    if not np.all(np.isfinite(wrap_scale)):
        raise ValueError("Wrap normalization must remain finite")
    d, n, rows = plan.dofs, plan.steps + 1, resources["fft_rows"]
    shape = (plan.fft_size // 2 + 1, d, d)
    if len(family_records) != 2:
        raise ValueError("Require both completed compact family receipts")
    sources = []
    for item in family_records:
        path = Path(item["path"]).resolve()
        saved_shape, fortran, dtype, _ = array_header(path)
        if (
            item.get("all_frequency_matrices_written") is not True
            or saved_shape != shape
            or fortran
            or dtype != np.complex128
            or item.get("shape") != list(shape)
            or item.get("bytes") != path.stat().st_size
            or JOURNAL["digest_range"](path) != item.get("sha256")
        ):
            raise ValueError("Require complete unmodified compact WORK-coordinate spectra")
        sources.append(dict(path=str(path), sha256=item["sha256"], bytes=item["bytes"]))
    if sources[0]["path"] == sources[1]["path"]:
        raise ValueError("The two compact source families must be distinct")
    context = dict(
        plan=dict(dt=plan.dt, steps=plan.steps, dofs=d, fft_size=plan.fft_size, radius=plan.radius),
        line_weights=line.tolist(),
        sources=sources,
        fft_rows=rows,
        time_tile=time_tile,
        numpy=np.__version__,
        scipy=scipy.__version__,
        source_sha256={
            str(p): JOURNAL["digest_range"](p)
            for p in [Path(__file__).resolve(), TOOLS / "radau_sampling_checkpoint.py"]
        },
    )
    state_path = folder / "direct-fortran-journal.json"
    if resume:
        if not folder.is_dir() or not state_path.is_file():
            raise ValueError("Resume needs this component's existing output journal")
    elif folder.exists():
        raise ValueError("Choose a fresh direct Fortran output directory")
    else:
        folder.mkdir(parents=True)
    lease = JOURNAL["exclusive_lease"](folder)
    state = None
    attempt = None
    paths = [folder / f"weights{i}.npy" for i in range(2)]
    wrap_path = folder / "shift_wrap.npy"
    maps = []
    try:
        if resume:
            state = json.loads(state_path.read_text())
            if (
                state["schema_version"] != 1
                or state["context"] != context
                or state["phase"] not in ["allocating", "inverting", "completed"]
            ):
                raise ValueError(
                    "Direct Fortran resume changes mathematical inputs, source or runtime"
                )
        else:
            state = dict(
                schema_version=1,
                context=context,
                phase="allocating",
                entries=[],
                attempts=[],
                resources=resources,
            )
            JOURNAL["atomic_json"](state_path, state)
        if state["phase"] == "allocating":
            if state["entries"]:
                raise ValueError("Invalid allocation journal with committed rows")
            needed = resources["weight_data_bytes"] + resources["wrap_data_bytes"] + 64 * 1024**2
            if shutil.disk_usage(folder).free < needed:
                raise ValueError("Insufficient disk for complete Fortran weights and wrap")
            # Only uncommitted files owned by this fresh allocation phase.
            for path in [*paths, wrap_path]:
                path.unlink(missing_ok=True)
            for path in paths:
                create_array(path, (n, d, d), fortran=True)
            create_array(wrap_path, (d, d), fortran=False)
            state["phase"] = "inverting"
            JOURNAL["atomic_json"](state_path, state)
        headers = []
        for path in paths:
            actual, fortran, dtype, offset = array_header(path)
            if actual != (n, d, d) or not fortran or dtype != np.float64:
                raise ValueError("Fortran time-weight header changed")
            headers.append(offset)
        actual, fortran, dtype, wrap_header = array_header(wrap_path)
        if actual != (d, d) or fortran or dtype != np.float64:
            raise ValueError("Physical shift-wrap header changed")
        completed = [0, 0]
        for entry in state["entries"]:
            family, first, last = (entry[k] for k in ["family", "first", "last"])
            if (
                family not in (0, 1)
                or first != completed[family]
                or not first < last <= d
                or (family == 1 and completed[0] != d)
            ):
                raise ValueError("Committed Fortran stripes omit or overlap rows")
            if stripe_digest(paths[family], headers[family], n, d, first, last) != entry["sha256"]:
                raise ValueError("Committed Fortran weight stripe changed")
            if (
                family == 1
                and JOURNAL["digest_range"](
                    wrap_path, wrap_header + first * d * 8, (last - first) * d * 8
                )
                != entry["wrap_sha256"]
            ):
                raise ValueError("Committed physical shift wrap changed")
            completed[family] = last
        if state["phase"] == "completed" and completed != [d, d]:
            raise ValueError("Incomplete completed inversion journal")
        committed = (sum(completed) * n * d + completed[1] * d) * 8
        needed = (
            resources["weight_data_bytes"] + resources["wrap_data_bytes"] - committed + 64 * 1024**2
        )
        if shutil.disk_usage(folder).free < needed:
            raise ValueError("Insufficient disk for remaining complete Fortran weights")
        attempt = dict(
            started_utc=datetime.now(UTC).isoformat(),
            initial_completed_rows=completed.copy(),
            preparation_including_source_hashes_seconds=perf_counter() - begun,
            status="running",
        )
        state["attempts"].append(attempt)
        JOURNAL["atomic_json"](state_path, state)
        amplification = plan.radius ** (-np.arange(n))
        for family, source in enumerate(sources):
            values_map = np.load(source["path"], mmap_mode="r", allow_pickle=False)
            maps.append(values_map)
            with paths[family].open("r+b") as target, wrap_path.open("r+b") as shift:
                for first in range(completed[family], d, rows):
                    last = min(first + rows, d)
                    values = np.array(values_map[:, first:last], copy=True)
                    if hasattr(values_map._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                        values_map._mmap.madvise(mmap.MADV_DONTNEED)
                    if not np.all(np.isfinite(values)):
                        raise ValueError("Nonfinite compact spectrum")
                    for endpoint in [0, len(values) - 1]:
                        if np.linalg.norm(values[endpoint].imag) > 1e-10 * max(
                            np.linalg.norm(values[endpoint].real), 1
                        ):
                            raise ValueError("Compact endpoint violates real symmetry")
                        values[endpoint] = values[endpoint].real
                    transformed = irfft(values, n=plan.fft_size, axis=0)
                    work = transformed[:n] * amplification[:, None, None]
                    work /= root[first:last, None] * root[None, :]
                    if not np.all(np.isfinite(work)):
                        raise ValueError("Nonfinite physical time weights")
                    tile = np.empty_like(work, order="F")
                    for t in range(0, n, time_tile):
                        tile[t : t + time_tile] = work[t : t + time_tile]
                    if family == 1:
                        wrap = plan.radius * transformed[-1] / wrap_scale[first:last]
                        if not np.all(np.isfinite(wrap)):
                            raise ValueError("Nonfinite physical shift wrap")
                    del work, values, transformed
                    expected = expected_stripe_digest(tile)
                    for column in range(d):
                        offset = headers[family] + (column * d + first) * n * 8
                        mapped_write(
                            target, offset, (n, last - first), tile[:, :, column], fortran=True
                        )
                        if progress is not None:
                            progress(
                                dict(
                                    phase="column_written",
                                    family=family,
                                    first=first,
                                    column=column,
                                )
                            )
                    os.fsync(target.fileno())
                    if stripe_digest(paths[family], headers[family], n, d, first, last) != expected:
                        raise ValueError("Fortran output changed weight bits")
                    entry = dict(family=family, first=first, last=last, sha256=expected)
                    if family == 1:
                        mapped_write(
                            shift, wrap_header + first * d * 8, wrap.shape, wrap, fortran=False
                        )
                        os.fsync(shift.fileno())
                        entry["wrap_sha256"] = hashlib.sha256(wrap.tobytes()).hexdigest()
                        if (
                            JOURNAL["digest_range"](
                                wrap_path, wrap_header + first * d * 8, wrap.nbytes
                            )
                            != entry["wrap_sha256"]
                        ):
                            raise ValueError("Physical shift-wrap output changed bits")
                        del wrap
                    del tile
                    state["entries"].append(entry)
                    completed[family] = last
                    JOURNAL["atomic_json"](state_path, state)
                    if progress is not None:
                        progress(
                            dict(phase="stripe_committed", family=family, first=first, last=last)
                        )
        for source in sources:
            if JOURNAL["digest_range"](source["path"]) != source["sha256"]:
                raise ValueError("Input spectra changed during direct inversion")
        for path, expected in context["source_sha256"].items():
            if JOURNAL["digest_range"](path) != expected:
                raise ValueError("Direct inversion source changed during execution")
        state["files"] = [
            dict(path=str(p), bytes=p.stat().st_size, sha256=JOURNAL["digest_range"](p))
            for p in [*paths, wrap_path]
        ]
        state["phase"] = "completed"
        state["all_attempt_timings_accounted"] = all(
            a is attempt or "seconds" in a for a in state["attempts"]
        )
        attempt.update(status="completed", seconds=perf_counter() - begun)
        JOURNAL["atomic_json"](state_path, state)
        return paths, np.load(wrap_path, allow_pickle=False), state
    except BaseException as error:
        if attempt is not None:
            attempt.update(
                status="interrupted_or_failed",
                seconds=perf_counter() - begun,
                error=f"{type(error).__name__}: {error}",
            )
            JOURNAL["atomic_json"](state_path, state)
        raise
    finally:
        for values in maps:
            values._mmap.close()
        lease.close()

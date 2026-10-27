"""Check every fresh raw-layer frequency against an existing complete CQ.

Reference data is used only after fresh construction. Streaming hashes cover
the same bytes as the stored NPY files during this single comparison pass.
The reference original operator is the independently audited inverse of the
old work resolvent; this tests added error, not an all-frequency BEM theorem.
"""

import argparse
import hashlib
import json
import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import CQPlan
from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent


def compare_emitted_prefix(
    reference,
    candidate,
    *,
    count,
    reference_written,
    candidate_written,
    block=4,
    relative_tolerance=1e-9,
    absolute_tolerance=1e-14,
    composition_tolerance=1e-6,
    candidate_live=False,
    max_workspace_mb=256,
):
    """Early rejection check only; caller must prove the emitted prefix/grid.

    No incomplete file is promoted to complete. Reference solve composition
    catches weak directions hidden by a global norm, but is not an independent
    original BEM operator audit. A live candidate prefix is hashed a second time.
    """
    if (
        any(
            isinstance(v, bool) or not isinstance(v, int)
            for v in [count, reference_written, candidate_written, block]
        )
        or not 1 <= count <= min(reference_written, candidate_written)
        or block < 1
        or any(
            not np.isfinite(v) or v <= 0
            for v in [
                relative_tolerance,
                absolute_tolerance,
                composition_tolerance,
                max_workspace_mb,
            ]
        )
    ):
        raise ValueError("Require a proven emitted prefix, positive chunks and finite tolerances")
    paths = [Path(reference), Path(candidate)]
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError("The early prefix control requires two separate artifacts")
    before = [p.stat() for p in paths]
    streams = []
    headers = []
    hashes = []
    begun = perf_counter()
    maximum_normalized, maximum_composition = 0.0, 0.0
    worst = None
    try:
        for path in paths:
            stream = path.open("rb")
            streams.append(stream)
            version = np.lib.format.read_magic(stream)
            reader = {
                (1, 0): np.lib.format.read_array_header_1_0,
                (2, 0): np.lib.format.read_array_header_2_0,
            }.get(version)
            if reader is None:
                raise ValueError("Unsupported prefix NPY format")
            shape, fortran, dtype = reader(stream)
            offset = stream.tell()
            if (
                len(shape) != 3
                or shape[1] != shape[2]
                or fortran
                or dtype != np.dtype("complex128")
            ):
                raise ValueError("Require C-order complex128 matrix half spectra")
            if (
                count > shape[0]
                or max(reference_written, candidate_written) > shape[0]
                or path.stat().st_size != offset + int(np.prod(shape)) * dtype.itemsize
            ):
                raise ValueError("Prefix exceeds the declared complete file storage")
            headers.append((shape, offset, dtype))
            stream.seek(0)
            h = hashlib.sha256()
            h.update(stream.read(offset))
            hashes.append(h)
        if headers[0][0] != headers[1][0]:
            raise ValueError("Prefix physical grid shapes differ")
        d = headers[0][0][1]
        if (3 * block + 8) * d * d * 16 > max_workspace_mb * 1024**2:
            raise ValueError("Prefix chunks exceed the conservative array budget")
        with threadpool_limits(limits=1, user_api="blas"):
            for first in range(0, count, block):
                rows = min(block, count - first)
                data = []
                for stream, h in zip(streams, hashes, strict=True):
                    value = stream.read(rows * d * d * 16)
                    if len(value) != rows * d * d * 16:
                        raise ValueError("Truncated emitted prefix")
                    h.update(value)
                    data.append(np.frombuffer(value, dtype="complex128").reshape(rows, d, d))
                if any(not np.all(np.isfinite(a)) for a in data):
                    raise ValueError("Nonfinite emitted prefix")
                for j, (old, new) in enumerate(zip(*data, strict=True)):
                    normalized = float(
                        np.linalg.norm(new - old)
                        / (absolute_tolerance + relative_tolerance * np.linalg.norm(old))
                    )
                    try:
                        composition = float(
                            np.max(np.sum(abs(np.linalg.solve(old, new) - np.eye(d)), axis=1))
                        )
                    except np.linalg.LinAlgError as error:
                        raise ValueError(f"Singular reference prefix at {first + j}") from error
                    if not np.isfinite(normalized + composition):
                        raise ValueError("Nonfinite prefix metric")
                    maximum_normalized = max(maximum_normalized, normalized)
                    if worst is None or composition > maximum_composition:
                        maximum_composition = composition
                        worst = first + j
    finally:
        for stream in streams:
            stream.close()
    after = [p.stat() for p in paths]
    for i in [0, 1]:
        if before[i].st_size != after[i].st_size or (
            (i == 0 or not candidate_live) and before[i].st_mtime_ns != after[i].st_mtime_ns
        ):
            raise ValueError("A frozen prefix artifact changed during comparison")
    digests = [h.hexdigest() for h in hashes]
    if candidate_live:
        again = hashlib.sha256()
        remaining = headers[1][1] + count * d * d * 16
        with paths[1].open("rb") as stream:
            while remaining:
                value = stream.read(min(4 * 1024**2, remaining))
                if not value:
                    raise ValueError("Live prefix was truncated")
                again.update(value)
                remaining -= len(value)
        if again.hexdigest() != digests[1]:
            raise ValueError("Live candidate prefix changed after it was compared")
    return dict(
        passed=maximum_normalized <= 1 and maximum_composition <= composition_tolerance,
        frequency_prefix_matrices=count,
        complete_half_spectrum_matrices=headers[0][0][0],
        claimed_written_matrices=[reference_written, candidate_written],
        maximum_normalized_prefix_difference=maximum_normalized,
        maximum_reference_solve_composition_inf=maximum_composition,
        worst_reference_solve_index=worst,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        reference_solve_composition_tolerance=composition_tolerance,
        streamed_npy_prefix_sha256=digests,
        hash_scope="NPY header and the compared prefix only; never a complete-file hash",
        candidate_live=candidate_live,
        seconds=perf_counter() - begun,
        scope="Only an already emitted immutable prefix; reference inverse is derived from its approximant",
        caller_must_verify_same_physical_grid_and_work_coordinates=True,
        whole_kernel_verified=False,
        original_operator_audit_verified=False,
    )


def compare_complete_spectra(
    reference,
    candidate,
    original_operator,
    *,
    workers=4,
    block=32,
    relative_tolerance=1e-7,
    composition_tolerance=1e-6,
):
    readers = [reference, candidate, original_operator]
    if (
        len({r.shape for r in readers}) != 1
        or any(r.dtype != np.complex128 for r in readers)
        or not isinstance(workers, int)
        or workers < 1
        or not isinstance(block, int)
        or block < 1
    ):
        raise ValueError("Matching complete complex matrix spectra and positive workers required")
    hashes = []
    before = []
    for r in readers:
        stat = r.path.stat()
        before.append((stat.st_size, stat.st_mtime_ns))
        h = hashlib.sha256()
        with r.path.open("rb") as f:
            h.update(f.read(r.offset))
        hashes.append(h)
    maximum_relative = 0.0
    maximum_composition = 0.0
    maximum_symmetry = 0.0
    worst = None
    begun = perf_counter()
    n, d, _ = reference.shape

    def diagnostic(args):
        index, old, new, a = args
        scale = np.linalg.norm(old)
        relative = (
            float(np.linalg.norm(new - old) / scale)
            if scale > 0
            else (0.0 if not np.any(new) else float("inf"))
        )
        residual = a @ new - np.eye(d)
        composition = float(np.max(np.sum(abs(residual), axis=1)))
        norm = np.linalg.norm(new)
        symmetry = float(np.linalg.norm(new - new.T) / max(norm, np.finfo(float).tiny))
        if not all(np.isfinite(v) for v in [relative, composition, symmetry]):
            raise ValueError("Nonfinite complete frequency comparison metric")
        return index, relative, composition, symmetry

    with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(workers) as pool:
        for first in range(0, n, block):
            last = min(first + block, n)
            data = [r[first:last] for r in readers]
            if any(not np.all(np.isfinite(a)) for a in data):
                raise ValueError("Nonfinite complete spectrum")
            for h, value in zip(hashes, data, strict=True):
                h.update(value.tobytes(order="C"))
            for index, relative, composition, symmetry in pool.map(
                diagnostic,
                ((first + j, data[0][j], data[1][j], data[2][j]) for j in range(last - first)),
            ):
                maximum_relative = max(maximum_relative, relative)
                maximum_symmetry = max(maximum_symmetry, symmetry)
                if worst is None or composition > maximum_composition:
                    maximum_composition = composition
                    worst = dict(
                        index=index,
                        relative_work_frobenius=relative,
                        original_inverse_composition_inf=composition,
                    )
    for r, stat in zip(readers, before, strict=True):
        current = r.path.stat()
        if (current.st_size, current.st_mtime_ns) != stat:
            raise ValueError("Spectrum changed during complete comparison")
    return dict(
        frequency_matrices=n,
        maximum_relative_work_frobenius_difference=maximum_relative,
        maximum_original_inverse_composition_inf=maximum_composition,
        maximum_bilinear_symmetry_relative=maximum_symmetry,
        worst_inverse_composition=worst,
        relative_work_tolerance=relative_tolerance,
        inverse_composition_tolerance=composition_tolerance,
        passed=maximum_relative <= relative_tolerance
        and maximum_composition <= composition_tolerance,
        streamed_npy_sha256=[h.hexdigest() for h in hashes],
        seconds_including_streamed_hashes=perf_counter() - begun,
        no_reference_data_used_for_fresh_construction=True,
        scope="Every saved frequency; added error versus prior audited operator, not an all-frequency fresh BEM certificate",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_kernel", type=Path)
    parser.add_argument("candidate_kernel", type=Path)
    parser.add_argument("--original-equation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--block", type=int, default=32)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a new comparison directory outside protected Tatva")
    paths = [
        args.reference_kernel / "summary.json",
        args.candidate_kernel / "summary.json",
        args.original_equation_audit,
        Path(__file__),
        TOOLS / "replay_resolvent.py",
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    old, new, audit = [json.loads(p.read_text()) for p in paths[:3]]
    if (
        old["status"] != "completed_resolvent_kernel_only"
        or new["status"] != "completed_resolvent_kernel_only"
        or audit["status"] != "passed_original_causal_equation_audit"
    ):
        raise ValueError("Completed kernels and passing original equation audit required")
    for key in ["case_sha256", "restart_data_sha256", "source_sha256"]:
        if old["environment"][key] != new["environment"][key]:
            raise ValueError("Physical input or bulk package differs")
    for key in [
        "h",
        "contact_h",
        "contact_end_h",
        "corner_h",
        "quadrature",
        "restart_knot_window",
        "restart_knot_subdivisions",
    ]:
        if old["source_mesh_arguments"].get(key) != new["source_mesh_arguments"].get(key):
            raise ValueError("Spatial discretization differs")
    a, b = CQPlan(**old["plan"]), CQPlan(**new["plan"])
    for key in ["dofs", "dt", "steps", "fft_size", "radius"]:
        if getattr(a, key) != getattr(b, key):
            raise ValueError("Physical CQ grid differs")
    if audit["source_resolvent_sha256"] != old["spectrum_storage"]["sha256"]:
        raise ValueError("Original audit was built from another reference spectrum")
    records = [old["spectrum_storage"], new["spectrum_storage"], audit["inverse_operator_storage"]]
    if any(r["coordinates"] != "diag(sqrt(w)) * L * diag(sqrt(w))" for r in records[:2]):
        raise ValueError("Require work-coordinate resolvents")
    reader = runpy.run_path(str(TOOLS / "replay_resolvent.py"))["SavedDiskSpectrum"]
    output.mkdir(parents=True)
    write_json(
        output / "run_manifest.json",
        dict(
            status="prepared_not_completed",
            frozen_input_sha256=frozen,
            files=[r["path"] for r in records],
            workers=args.workers,
        ),
    )
    result = compare_complete_spectra(
        *[reader(Path(r["path"]), a) for r in records], workers=args.workers, block=args.block
    )
    if result["streamed_npy_sha256"] != [r["sha256"] for r in records]:
        raise ValueError("Complete streamed spectrum hashes differ from their provenance")
    if any(fingerprint(p) != v for p, v in frozen.items()):
        raise ValueError("Frozen spectrum comparison input changed")
    write_json(
        output / "summary.json",
        dict(
            status="passed_complete_raw_kernel_spectrum_check"
            if result["passed"]
            else "failed_complete_raw_kernel_spectrum_check",
            frozen_input_sha256=frozen,
            comparison=result,
            original_tatva_readonly=True,
            same_accuracy_fem_advantage_verified=False,
        ),
    )
    print(output / "summary.json", flush=True)
    if not result["passed"]:
        raise ValueError("Complete raw-kernel spectrum comparison failed")


if __name__ == "__main__":
    main()

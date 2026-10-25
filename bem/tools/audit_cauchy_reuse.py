"""Compare contour reuse with fresh BEM matrices at independent frequencies.

This is a sampled accuracy audit, not a new time kernel or nonlinear run.
The direct backend is intentionally limited to five targets per batch.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import CQPlan, bdf2_symbol
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOL_PATH = Path(__file__).with_name("cauchy_reuse_experiment.py")
TOOL = runpy.run_path(str(TOOL_PATH))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--fft-size", type=int, required=True)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    if not np.isfinite(args.dt) or args.dt <= 0 or args.fft_size < 4 or args.fft_size % 2:
        raise ValueError("A positive dt and even target FFT size >= 4 are required")
    if not np.isfinite(args.relative_tolerance) or args.relative_tolerance <= 0:
        raise ValueError("A finite positive operator tolerance is required")
    started = perf_counter()
    metadata_path = args.kernel / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    for name in ["source_sha256", "case_sha256", "restart_data_sha256"]:
        if environment[name] != metadata["environment"][name]:
            raise ValueError(f"Prepared kernel differs in {name}")
    if metadata["status"] != "completed_resolvent_kernel_only":
        raise ValueError("A completed source kernel is required")
    storage = metadata["spectrum_storage"]
    if storage.get("coordinates") != "diag(sqrt(w)) * L * diag(sqrt(w))":
        raise ValueError("Source must contain the work-coordinate elastic resolvent")
    spectrum_path = Path(storage["path"])
    reader_path = Path(__file__).with_name("replay_resolvent.py")
    frozen_paths = [Path(__file__), TOOL_PATH, metadata_path, reader_path]
    frozen = {str(path.resolve()): fingerprint(path) for path in frozen_paths}
    if TOOL["stream_file_sha256"](spectrum_path) != storage["sha256"]:
        raise ValueError("Source spectrum digest differs")
    stat = spectrum_path.stat()
    source_plan = CQPlan(**metadata["plan"])
    reader = runpy.run_path(str(reader_path))
    source = reader["SavedDiskSpectrum"](spectrum_path, source_plan)
    d = source_plan.dofs
    if source.shape != (source_plan.fft_size // 2 + 1, d, d) or source.dtype != np.complex128:
        raise ValueError("Source spectrum dimensions or dtype differ")
    indices = sorted(set(args.indices or [0, 1, 64, args.fft_size // 8, args.fft_size // 2]))
    if min(indices) < 0 or max(indices) > args.fft_size // 2:
        raise ValueError("Target indices must lie on the positive half circle")
    target_radius = 1e-10 ** (1 / args.fft_size)
    zeta = target_radius * np.exp(-2j * np.pi * np.array(indices) / args.fft_size)
    frequency = bdf2_symbol(zeta) / args.dt
    mapped = TOOL["bdf2_preimage"](frequency, source_plan.dt)
    circle = TOOL["CircleCauchy"](source_plan.fft_size, source_plan.radius)
    circle.validate_targets(mapped)
    resampling_started = perf_counter()

    def progress(done, total):
        if done == total or done % 4096 == 0:
            print(f"Cauchy source matrices: {done}/{total}", flush=True)

    with threadpool_limits(limits=1, user_api="blas"):
        evaluated = circle.evaluate_streamed(source, mapped, progress=progress)
    resampling_seconds = perf_counter() - resampling_started
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    wc = model.constraints.contact_weights
    line = np.r_[wc, wc, model.constraints.platen_width]
    law = model.reference.contact
    tangent = np.diag(
        np.r_[
            wc * law.normal_penalty,
            wc * law.tangential_penalty,
            model.reference.spring_stiffness * model.constraints.platen_width,
        ]
    )
    root = np.sqrt(line)
    fresh = np.empty_like(evaluated)
    direct_started = perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        transfer = NodalInterfaceTransfer(model, quadrature=options["quadrature"])
        for index, s in enumerate(frequency):
            impedance = transfer.evaluate(s)
            fresh[index] = root[:, None] * solve(
                impedance + tangent, np.diag(root), check_finite=False
            )
    direct_seconds = perf_counter() - direct_started
    errors = [
        float(np.linalg.norm(a - b) / np.linalg.norm(b))
        for a, b in zip(evaluated, fresh, strict=True)
    ]
    final_stat = spectrum_path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise ValueError("Source spectrum changed during reuse")
    if any(fingerprint(path) != digest for path, digest in frozen.items()):
        raise ValueError("Audit implementation, FMM library or source metadata changed")
    if environment_record(options["case"])["source_sha256"] != environment["source_sha256"]:
        raise ValueError("Numerical package changed during audit")
    passed = max(errors) <= args.relative_tolerance
    args.output.mkdir(parents=True)
    np.savez(
        args.output / "operators.npz",
        frequency=frequency,
        mapped=mapped,
        reused=evaluated,
        fresh=fresh,
    )
    record = dict(
        status="passed_sampled_operator_comparison"
        if passed
        else "failed_sampled_operator_comparison",
        source_kernel=str(args.kernel.resolve()),
        source_spectrum_sha256=storage["sha256"],
        source_dt_s=source_plan.dt,
        target_dt_s=args.dt,
        target_fft_size=args.fft_size,
        indices=indices,
        reuse_evaluation="Frequency-major bounded direct Cauchy blocks; one source pass",
        relative_frobenius_errors=errors,
        relative_tolerance=args.relative_tolerance,
        maximum_relative_error=max(errors),
        operator_arrays_sha256=fingerprint(args.output / "operators.npz"),
        resampling_seconds=resampling_seconds,
        fresh_bem_seconds=direct_seconds,
        total_seconds=perf_counter() - started,
        frozen_inputs_sha256=frozen,
        environment=environment,
        full_original_event_reproduced=False,
        certified_error_bound=False,
        scope="Sampled work-resolvent comparison at fixed geometry/material/quadrature; no nonlinear trajectory or speed advantage established",
    )
    write_json(args.output / "summary.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "environment"}, indent=2))
    if not passed:
        raise ValueError("Reused spectrum failed the independent BEM matrix comparison")


if __name__ == "__main__":
    main()

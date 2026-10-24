"""Build a fresh BEM resolvent without allocating a dense frequency circle.

The original adaptive sampler writes its accepted interpolants directly; row
IFFTs put every requested correction weight on disk. The full causal elastic
startup is preserved. This prepares a kernel, not a claim of FEM equivalence
or a memory bound for the separate nonlinear solver and its FFT caches.
"""

import argparse
import ast
import hashlib
import json
import math
import mmap
import os
import runpy
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model
from finiteblock_bem.nodal_dynamics import plan_nodal_history
from finiteblock_bem.sampling import SamplingOptions

TOOLS = Path(__file__).resolve().parent


def preflight_time_storage(folder, plan, full_steps):
    """Protect space for every time weight, startup vector and small metadata.

    Frequency pieces recheck free space against this reserve while sampling.
    This is a fail-fast check, not an OS quota against concurrent writers.
    """
    required = (plan.steps + 1) * plan.dofs**2 * 8
    required += (full_steps + 1) * (plan.dofs + 2) * 8 + plan.dofs**2 * 8 + 65536
    if shutil.disk_usage(folder).free < required:
        raise ValueError("Insufficient disk space for complete time weights and elastic startup")
    return required


def physical_weights_file_inplace(path, line_weights):
    """Convert the stored prefix from work coordinates in bounded time blocks."""
    weights = np.load(path, mmap_mode="r+", allow_pickle=False)
    root = np.sqrt(line_weights)
    scale = root[:, None] * root[None, :]
    for first in range(0, len(weights), 64):
        weights[first : first + 64] /= scale
        weights.flush()
        if hasattr(weights._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
            weights._mmap.madvise(mmap.MADV_DONTNEED)
    weights._mmap.close()
    return np.load(path, mmap_mode="r", allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Completed run specifying case and geometry")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--correction-duration", type=float)
    parser.add_argument("--sampling-rtol", type=float, default=1e-8)
    parser.add_argument("--sampling-atol", type=float, default=1e-14)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("FDBEM_CORES", "1")))
    parser.add_argument("--fft-rows", type=int, default=4)
    parser.add_argument("--max-memory-mb", type=float, default=4096)
    parser.add_argument("--max-spectrum-mb", type=float, default=4096)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new empty output directory")
    if not np.isfinite(args.max_memory_mb) or args.max_memory_mb <= 0:
        raise ValueError("A finite positive array workspace budget is required")
    if not isinstance(args.fft_rows, int) or args.fft_rows < 1:
        raise ValueError("fft_rows must be a positive integer")
    started = perf_counter()
    frozen_paths = [
        Path(__file__).resolve(),
        TOOLS / "native_spectrum.py",
        TOOLS / "polynomial_spectrum.py",
        TOOLS / "build_resolvent_history.py",
        TOOLS / "replay_resolvent.py",
        args.source / "summary.json",
        args.source / "run_manifest.json",
    ]
    frozen = {str(path): fingerprint(path) for path in frozen_paths}
    source = json.loads((args.source / "summary.json").read_text())
    options = json.loads((args.source / "run_manifest.json").read_text())["arguments"]
    environment = environment_record(options["case"])
    source_keys = ["source_sha256", "case_sha256", "restart_data_sha256"]
    if (
        source["status"] != "completed_requested_horizon"
        or source["source_changed_during_run"]
        or any(source["environment"][name] != environment[name] for name in source_keys)
    ):
        raise ValueError("Require a completed source-frozen run with the current case")

    def check_source():
        current = environment_record(options["case"])
        if any(current[name] != environment[name] for name in source_keys):
            raise ValueError("Numerical package or case changed during kernel construction")
        if any(fingerprint(path) != digest for path, digest in frozen.items()):
            raise ValueError("Native kernel implementation or source manifest changed")

    dt = source["dt_s"] if args.dt is None else args.dt
    duration = source["duration_s"] if args.duration is None else args.duration
    correction = duration if args.correction_duration is None else args.correction_duration
    if (
        not all(np.isfinite(x) and x > 0 for x in [dt, duration, correction])
        or correction > duration
    ):
        raise ValueError("Require positive dt and 0 < correction duration <= full duration")
    full_steps, correction_steps = round(duration / dt), math.ceil(correction / dt)
    if full_steps < 1 or correction_steps > full_steps:
        raise ValueError("Time grid and correction horizon are incompatible")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    sampling = SamplingOptions(
        relative_tolerance=args.sampling_rtol, absolute_tolerance=args.sampling_atol
    )
    # Derive the existing solver/assembly estimates without allocating arrays.
    # The separate online solver still checks its original, larger cache bound.
    short, reference_peak = plan_nodal_history(
        model,
        dt,
        correction_steps,
        workers=args.workers,
        quadrature=options["quadrature"],
        sampling=sampling,
        max_memory_mb=1e9,
        spectrum_directory=args.output,
    )
    fft_size = 1 << math.ceil(math.log2(2 * (full_steps + 1)))
    plan = replace(short, fft_size=fft_size, radius=1e-10 ** (1 / fft_size))
    d, rows = plan.dofs, min(args.fft_rows, plan.dofs)
    assembly = reference_peak - short.estimated_peak_bytes
    row_ifft = 4 * fft_size * rows * d * 8 + (correction_steps + 1) * rows * d * 8
    row_ifft += 64 * d**2 * 16
    interpolation_copies = 4 * (sampling.degree + 1) * d**2 * 16
    startup = 8 * fft_size * d * 16 + 24 * (full_steps + 1) * d * 8
    peak = assembly + interpolation_copies + max(row_ifft, startup) + 128 * 1024**2
    if peak > args.max_memory_mb * 1024**2:
        raise ValueError(f"Native construction array estimate is {peak / 1024**2:.1f} MiB")
    args.output.mkdir(parents=True, exist_ok=True)
    reserved_disk = preflight_time_storage(args.output, plan, full_steps)
    line_weights = np.r_[
        model.constraints.contact_weights,
        model.constraints.contact_weights,
        model.constraints.platen_width,
    ]
    n, law = len(model.moving_contact), model.reference.contact
    tangent = np.diag(
        np.r_[
            line_weights[:n] * law.normal_penalty,
            line_weights[n : 2 * n] * law.tangential_penalty,
            0,
        ]
    )
    spring = model.reference.spring_stiffness * model.constraints.platen_width
    tangent[-1, -1] += spring
    original = runpy.run_path(str(TOOLS / "build_resolvent_history.py"))
    native = runpy.run_path(str(TOOLS / "native_spectrum.py"))
    helper = runpy.run_path(str(TOOLS / "replay_resolvent.py"))
    last_progress = [0.0]

    def progress(done, total):
        now = perf_counter()
        if done == total or now - last_progress[0] >= 30:
            print(f"progress: {done}/{total}", flush=True)
            last_progress[0] = now

    transfer = NodalInterfaceTransfer(model, quadrature=options["quadrature"])

    def evaluate(s, operator=transfer):
        return original["compliance_in_work_coordinates"](
            operator.evaluate(s), tangent, line_weights
        )

    with threadpool_limits(limits=1, user_api="blas"):
        weights, spectrum, records = native["native_parsimonious_weights"](
            evaluate,
            plan,
            args.output / "spectrum",
            options=sampling,
            workers=args.workers,
            progress=progress,
            max_storage_mb=args.max_spectrum_mb,
            source_check=check_source,
            weight_path=args.output / "weights.npy",
            fft_rows=rows,
            max_workspace_mb=args.max_memory_mb,
            reserved_disk_bytes=reserved_disk,
            provenance=dict(plan=asdict(plan), source_run=str(args.source.resolve())),
        )
        impedance = transfer.evaluate(1.5 / dt)
    del evaluate, transfer
    if not np.all(np.isfinite(impedance)) or np.max(abs(impedance.imag)) > 1e-10:
        raise ValueError("Positive-real instantaneous impedance must be finite and real")
    weights._mmap.close()
    weights = physical_weights_file_inplace(args.output / "weights.npy", line_weights)
    composition = float(
        np.linalg.norm(weights[0] @ (impedance.real + tangent) - np.eye(d), ord=np.inf)
    )
    weights._mmap.close()
    if not np.isfinite(composition) or composition > 1e-6:
        raise ValueError(f"Instantaneous CQ inverse composition failed: {composition}")
    np.save(args.output / "initial_impedance.npy", impedance.real, allow_pickle=False)
    time = np.arange(full_steps + 1) * dt
    increment = options.get("diagnostic_actuator_increment")
    rate = model.reference.actuator_rate if increment is None else increment / time[-1]
    column = spectrum[:, :, -1] / np.sqrt(line_weights * line_weights[-1])
    elastic = helper["causal_elastic_baseline"](
        column, replace(plan, steps=full_steps), spring * rate * time
    )
    np.savez(
        args.output / "elastic.npz",
        time=time,
        elastic=elastic,
        uninterrupted_actuator=model.reference.restart_actuator + rate * time,
    )
    helper_node = next(
        node
        for node in ast.parse((TOOLS / "replay_resolvent.py").read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "causal_elastic_baseline"
    )
    check_source()
    record = dict(
        status="completed_resolvent_kernel_only",
        source_run=str(args.source.resolve()),
        source_mesh_arguments=options,
        dt_s=dt,
        full_steps=full_steps,
        correction_steps=correction_steps,
        plan=asdict(plan),
        history_dofs=d,
        frequency_sampling=records,
        sampling_options=asdict(sampling),
        spectrum_storage={**spectrum.record(), "coordinates": "diag(sqrt(w)) * L * diag(sqrt(w))"},
        time_weights_storage="disk_mmap",
        reserved_time_output_bytes=reserved_disk,
        initial_composition_inf_error=composition,
        estimated_construction_array_peak_mib=peak / 1024**2,
        estimated_original_solver_array_peak_mib=short.estimated_peak_bytes / 1024**2,
        total_seconds=perf_counter() - started,
        environment=environment,
        helper_sha256=frozen[str(TOOLS / "replay_resolvent.py")],
        helper_math_sha256=hashlib.sha256(
            ast.dump(helper_node, include_attributes=False).encode()
        ).hexdigest(),
        weights_sha256=fingerprint(args.output / "weights.npy"),
        elastic_sha256=fingerprint(args.output / "elastic.npz"),
        prototype_sha256=frozen[str(Path(__file__).resolve())],
        frozen_inputs_sha256=frozen,
        full_original_event_reproduced=False,
        scope="Fresh native BEM kernel, complete frequency grid, disk matrix prefix, full causal elastic startup; nonlinear trajectory/adjoint/FEM comparison require separate validation",
    )
    write_json(args.output / "summary.json", record)
    print(
        json.dumps(
            {
                key: record[key]
                for key in [
                    "status",
                    "frequency_sampling",
                    "spectrum_storage",
                    "initial_composition_inf_error",
                    "total_seconds",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

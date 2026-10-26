"""Build a two-family Radau kernel from a guarded, audited BDF2 circle.

Large arrays may use a separate new storage directory. The source spectrum
is read-only. This producer does not run friction or establish FEM equivalence.
"""

import argparse
import json
import math
import os
import resource
import runpy
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import CQPlan, plan_cq
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOLS = Path(__file__).resolve().parent
CAUCHY = runpy.run_path(str(TOOLS / "cauchy_reuse_experiment.py"))
AUDIT = runpy.run_path(str(TOOLS / "audit_cauchy_reuse.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))
STORAGE = runpy.run_path(str(TOOLS / "compact_radau_storage.py"))
REPLAY = runpy.run_path(str(TOOLS / "replay_resolvent.py"))
CONVERT = runpy.run_path(str(TOOLS / "prepare_native_resolvent.py"))


def safe_output(path):
    path = path.resolve()
    if path.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("The original Tatva directory must remain read-only")
    if path.exists():
        raise ValueError("Choose new output and storage directories")
    return path


def validate_native_backend(backend):
    circle = CAUCHY["CircleCauchy"](64, 0.9)
    values = (1 / (2 - circle.sources[:33]))[:, None]
    target = np.array([0, 0.7 + 0.2j, 0.8 - 0.1j])
    expected = circle.evaluate(values, target, derivative=True)
    actual = circle.evaluate(values, target, derivative=True, backend=backend)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--operator-audit", type=Path, required=True)
    parser.add_argument("--fmm-library", type=Path, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-directory", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fft-rows", type=int, default=4)
    parser.add_argument("--max-memory-mb", type=float, default=8192)
    parser.add_argument("--fmm-workspace-mb", type=float, default=2048)
    args = parser.parse_args()
    output = safe_output(args.output)
    storage = output if args.storage_directory is None else safe_output(args.storage_directory)
    if (
        not all(
            np.isfinite(v) and v > 0 for v in [args.dt, args.max_memory_mb, args.fmm_workspace_mb]
        )
        or args.threads < 1
        or args.fft_rows < 1
    ):
        raise ValueError("Positive finite time and resource options are required")
    begun, cpu_begun = perf_counter(), process_time()
    metadata_path = args.kernel / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    keys = ["source_sha256", "case_sha256", "restart_data_sha256"]
    if metadata["status"] != "completed_resolvent_kernel_only" or any(
        environment[k] != metadata["environment"][k] for k in keys
    ):
        raise ValueError("Require a completed source kernel matching the current package and case")
    source_storage = metadata["spectrum_storage"]
    if source_storage.get("coordinates") != "diag(sqrt(w)) * L * diag(sqrt(w))":
        raise ValueError("Source must be the work-coordinate elastic resolvent")
    source_path = Path(source_storage["path"])
    source_plan = CQPlan(**metadata["plan"])
    d = source_plan.dofs
    duration = metadata["full_steps"] * source_plan.dt
    steps = round(duration / args.dt)
    if steps < 1 or not np.isclose(steps * args.dt, duration, rtol=0, atol=1e-12):
        raise ValueError("Target grid must cover exactly the source duration")
    correction = math.ceil(metadata["correction_steps"] * source_plan.dt / args.dt)
    nfft = 1 << math.ceil(math.log2(2 * (steps + 1)))
    plan = replace(
        plan_cq(d, args.dt, correction, max_memory_mb=1e9, disk_spectrum=True),
        fft_size=nfft,
        radius=1e-10 ** (1 / nfft),
    )
    audit_path = args.operator_audit / "summary.json"
    array_path = args.operator_audit / "operators.npz"
    audit = json.loads(audit_path.read_text())
    if (
        audit["status"] != "passed_sampled_operator_comparison"
        or audit.get("target_time_method") != "radau2"
        or audit["relative_tolerance"] > 1e-6
        or Path(audit["source_kernel"]).resolve() != args.kernel.resolve()
        or audit["source_spectrum_sha256"] != source_storage["sha256"]
        or (audit["target_dt_s"], audit["target_fft_size"]) != (args.dt, nfft)
        or fingerprint(array_path) != audit["operator_arrays_sha256"]
        or any(audit["environment"][k] != environment[k] for k in keys)
    ):
        raise ValueError("A matching passed independent Radau operator audit is required")
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(nfft // 2 + 1) / nfft)
    frequencies, _ = AUDIT["contour_modes"](zeta, args.dt, "radau2")
    eigenvalues = frequencies * args.dt
    mapped = CAUCHY["bdf2_preimage"](frequencies.ravel(), source_plan.dt)
    circle = CAUCHY["CircleCauchy"](source_plan.fft_size, source_plan.radius)
    circle.validate_targets(mapped)
    frozen_paths = [
        Path(__file__),
        metadata_path,
        audit_path,
        array_path,
        args.fmm_library,
        *[
            TOOLS / name
            for name in [
                "compact_radau_storage.py",
                "radau_experiment.py",
                "cauchy_reuse_experiment.py",
                "audit_cauchy_reuse.py",
                "replay_resolvent.py",
                "prepare_native_resolvent.py",
                "bounded_weight_conversion.py",
            ]
        ],
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in frozen_paths}
    with np.load(array_path, allow_pickle=False) as saved:
        fresh, audited = saved["fresh"], saved["reused"]
        np.testing.assert_allclose(
            saved["mode_frequencies"], frequencies[audit["indices"]], rtol=1e-12
        )
    if (
        not audit["indices"]
        or fresh.shape != (len(audit["indices"]), 2 * d, 2 * d)
        or audited.shape != fresh.shape
    ):
        raise ValueError("Audited stage matrix dimensions differ")
    estimate = 4 * source_plan.fft_size * (d + 1) * 16
    estimate += 4 * len(mapped) * (d + 1) * 16 + 4 * nfft * min(args.fft_rows, d) * d * 8
    estimate += args.fmm_workspace_mb * 1024**2 + 256 * 1024**2
    if estimate > args.max_memory_mb * 1024**2:
        raise ValueError(f"Known arrays plus FMM reserve need {estimate / 1024**2:.1f} MiB")
    required = 2 * len(zeta) * d**2 * 16 + 2 * (correction + 1) * d**2 * 8
    required += (steps + 1) * 8 * d * 8 + 128 * 1024**2
    storage.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(storage.parent).free < required:
        raise ValueError("Insufficient storage for both complete spectra, weights and startup")
    if CAUCHY["stream_file_sha256"](source_path) != source_storage["sha256"]:
        raise ValueError("Source spectrum digest differs")
    source_stat = source_path.stat()
    source = REPLAY["SavedDiskSpectrum"](source_path, source_plan)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    backend = CAUCHY["fortran_fmm2d_backend"](args.fmm_library)
    validate_native_backend(backend)
    output.mkdir(parents=True)
    if storage != output:
        storage.mkdir(parents=True)
    spectrum_paths = [storage / f"work-family{i}.npy" for i in range(2)]
    for path in spectrum_paths:
        STORAGE["create_array_file"](path, (len(zeta), d, d), np.complex128)
    preparation_seconds = perf_counter() - begun
    sampling_start = perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        for row in range(d):
            values = circle.evaluate(source[:, row, :], mapped, backend=backend)
            families = STORAGE["compact_rows"](eigenvalues, values.reshape(len(zeta), 2, d))
            for path, values in zip(spectrum_paths, families, strict=True):
                for endpoint in [0, len(zeta) - 1]:
                    if np.linalg.norm(values[endpoint].imag) > 1e-10 * max(
                        np.linalg.norm(values[endpoint].real), 1
                    ):
                        raise ValueError("Compact real-axis row violates conjugate symmetry")
                    values[endpoint] = values[endpoint].real
                target = np.load(path, mmap_mode="r+", allow_pickle=False)
                try:
                    target[:, row] = values
                    target.flush()
                finally:
                    target._mmap.close()
            print(f"compact rows: {row + 1}/{d}", flush=True)
            write_json(
                output / "progress.json",
                dict(phase="resampling", completed_rows=row + 1, total_rows=d),
            )
    sampling_seconds = perf_counter() - sampling_start
    spectra = [REPLAY["SavedDiskSpectrum"](path, plan) for path in spectrum_paths]
    fresh_errors, fmm_errors = [], []
    for position, index in enumerate(audit["indices"]):
        actual = np.kron(np.eye(2), spectra[0][index]) + np.kron(
            RADAU["radau_symbol"](zeta[index]), spectra[1][index]
        )
        fresh_errors.append(
            float(np.linalg.norm(actual - fresh[position]) / np.linalg.norm(fresh[position]))
        )
        fmm_errors.append(
            float(np.linalg.norm(actual - audited[position]) / np.linalg.norm(audited[position]))
        )
    if max(fresh_errors) > audit["relative_tolerance"] or max(fmm_errors) > 1e-10:
        raise ValueError("Complete compact spectrum differs from the independent stage audit")
    ifft_start = perf_counter()
    weight_paths, wrap, inversion = STORAGE["invert_families_to_files"](
        *spectra, plan, storage, fft_rows=args.fft_rows, max_workspace_mb=args.max_memory_mb
    )
    ifft_seconds = perf_counter() - ifft_start
    startup_start = perf_counter()
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    wc, law = model.constraints.contact_weights, model.reference.contact
    line = np.r_[wc, wc, model.constraints.platen_width]
    if len(line) != d:
        raise ValueError("Source mesh and retained history dimensions differ")
    spring = model.reference.spring_stiffness * line[-1]
    tangent = np.diag(np.r_[wc * law.normal_penalty, wc * law.tangential_penalty, spring])
    for path in weight_paths:
        weights = CONVERT["physical_weights_file_inplace"](path, line)
        weights._mmap.close()
    wrap /= np.sqrt(line[:, None] * line[None, :])
    np.save(output / "shift_wrap.npy", wrap, allow_pickle=False)
    with threadpool_limits(limits=1, user_api="blas"):
        transfer = NodalInterfaceTransfer(model, quadrature=options["quadrature"])
        impedance = RADAU["matrix_function"](transfer.evaluate, RADAU["radau_symbol"](0) / args.dt)
    if np.linalg.norm(impedance.imag) > 1e-10 * max(np.linalg.norm(impedance.real), 1):
        raise ValueError("Instantaneous stage impedance must be real")
    del transfer
    first_weights = [np.load(path, mmap_mode="r", allow_pickle=False) for path in weight_paths]
    instantaneous = RADAU["expand_compact_weights"](
        first_weights[0][:1], first_weights[1][:1], wrap
    )[0]
    for weights in first_weights:
        weights._mmap.close()
    composition = float(
        np.linalg.norm(
            instantaneous @ (impedance.real + np.kron(np.eye(2), tangent)) - np.eye(2 * d),
            ord=np.inf,
        )
    )
    if not np.isfinite(composition) or composition > 1e-6:
        raise ValueError(f"Radau inverse-CQ composition failed: {composition}")
    np.save(output / "initial_impedance.npy", impedance.real, allow_pickle=False)
    time = np.arange(steps + 1) * args.dt
    stage_time = np.zeros((steps + 1, 2))
    stage_time[1:] = (np.arange(steps)[:, None] + RADAU["C"]) * args.dt
    increment = options.get("diagnostic_actuator_increment")
    rate = model.reference.actuator_rate if increment is None else increment / duration
    force = spring * rate * stage_time
    columns = [spectrum[:, :, -1] / np.sqrt(line * line[-1]) for spectrum in spectra]
    elastic = STORAGE["causal_compact_startup"](*columns, wrap[:, -1], plan, force)
    np.savez(
        output / "elastic.npz",
        time=time,
        stage_time=stage_time,
        elastic=elastic,
        uninterrupted_actuator=model.reference.restart_actuator + rate * stage_time,
    )
    startup_seconds = perf_counter() - startup_start
    final_started = perf_counter()
    if any(fingerprint(path) != digest for path, digest in frozen.items()):
        raise ValueError("Compact producer implementation or input changed")
    final_stat = source_path.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (
        final_stat.st_size,
        final_stat.st_mtime_ns,
    ):
        raise ValueError("Source spectrum changed during reuse")
    final_environment = environment_record(options["case"])
    if any(final_environment[k] != environment[k] for k in keys):
        raise ValueError("Numerical package or case changed")

    def records(paths):
        return [
            dict(
                path=str(p.resolve()),
                bytes=p.stat().st_size,
                sha256=CAUCHY["stream_file_sha256"](p),
            )
            for p in paths
        ]

    record = dict(
        status="completed_compact_radau_resolvent_kernel_only",
        time_method="radau2",
        source_kernel=str(args.kernel.resolve()),
        source_spectrum_sha256=source_storage["sha256"],
        source_mesh_arguments=options,
        dt_s=args.dt,
        full_steps=steps,
        correction_steps=correction,
        plan=asdict(plan),
        history_dofs=d,
        stage_dofs=2 * d,
        family_spectra=records(spectrum_paths),
        family_weights=records(weight_paths),
        shift_wrap_sha256=fingerprint(output / "shift_wrap.npy"),
        initial_impedance_sha256=fingerprint(output / "initial_impedance.npy"),
        elastic_sha256=fingerprint(output / "elastic.npz"),
        sampled_fresh_bem_relative_errors=fresh_errors,
        sampled_fmm_relative_errors=fmm_errors,
        initial_composition_inf_error=composition,
        inversion=inversion,
        estimated_array_peak_mib=estimate / 1024**2,
        fmm_workspace_reserve_mib=args.fmm_workspace_mb,
        fmm_threads_requested=args.threads,
        hard_cpu_affinity_enforced=False,
        frozen_inputs_sha256=frozen,
        environment=environment,
        timing_seconds=dict(
            source_verification_and_preparation=preparation_seconds,
            resampling_and_spectrum_writes=sampling_seconds,
            inverse_fft=ifft_seconds,
            physical_weights_and_startup=startup_seconds,
        ),
        process_cpu_seconds=process_time() - cpu_begun,
        full_original_event_reproduced=False,
        scope="Two complete compact RK-CQ families, all causal startup and sampled independent stage checks; no nonlinear trajectory, AD or FEM speed claim",
    )
    record["timing_seconds"]["total_before_reporting"] = perf_counter() - begun
    record["timing_seconds"]["final_checks_and_hashes"] = perf_counter() - final_started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["peak_process_rss_mib"] = rss / (1024**2 if sys.platform == "darwin" else 1024)
    write_json(output / "summary.json", record)
    write_json(
        output / "progress.json",
        dict(phase="completed_kernel_only", completed_rows=d, total_rows=d),
    )
    print(
        json.dumps(
            {k: v for k, v in record.items() if k not in ["environment", "frozen_inputs_sha256"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

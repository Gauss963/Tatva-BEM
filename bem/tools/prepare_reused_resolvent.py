"""Prepare a guarded new CQ grid by streaming an existing resolvent circle.

Requires a passed independent BEM operator audit and an optional FMM2D shared
library. Output remains experimental until the complete trajectory is checked.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import runpy
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import CQPlan, bdf2_symbol, plan_cq
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOLS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--operator-audit", type=Path, required=True)
    parser.add_argument("--fmm-library", type=Path, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-memory-mb", type=float, default=8192)
    parser.add_argument("--fmm-workspace-mb", type=float, default=2048)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    if (
        not np.isfinite(args.dt)
        or args.dt <= 0
        or args.threads < 1
        or not all(np.isfinite(x) and x > 0 for x in [args.max_memory_mb, args.fmm_workspace_mb])
    ):
        raise ValueError("Positive finite time/resource options are required")
    begun = perf_counter()
    metadata_path = args.kernel / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    keys = ["source_sha256", "case_sha256", "restart_data_sha256"]
    if any(environment[k] != metadata["environment"][k] for k in keys):
        raise ValueError("Source kernel does not match the current numerical package/case")
    if metadata["status"] != "completed_resolvent_kernel_only":
        raise ValueError("A completed source kernel is required")
    storage = metadata["spectrum_storage"]
    if storage.get("coordinates") != "diag(sqrt(w)) * L * diag(sqrt(w))":
        raise ValueError("Source must be the work-coordinate elastic resolvent")
    source_path = Path(storage["path"])
    if fingerprint(source_path) != storage["sha256"]:
        raise ValueError("Source spectrum digest differs")
    source_stat = source_path.stat()
    source_plan = CQPlan(**metadata["plan"])
    duration = metadata["full_steps"] * source_plan.dt
    steps = round(duration / args.dt)
    if steps < 1 or not np.isclose(steps * args.dt, duration, rtol=0, atol=1e-12):
        raise ValueError("Target grid must cover exactly the source horizon")
    correction = math.ceil(metadata["correction_steps"] * source_plan.dt / args.dt)
    d = source_plan.dofs
    nfft = 1 << math.ceil(math.log2(2 * (steps + 1)))
    plan = replace(
        plan_cq(d, args.dt, correction, max_memory_mb=1e9, disk_spectrum=True),
        fft_size=nfft,
        radius=1e-10 ** (1 / nfft),
    )
    audit_path = args.operator_audit / "summary.json"
    audit = json.loads(audit_path.read_text())
    audit_array_path = args.operator_audit / "operators.npz"
    if (
        audit["status"] != "passed_sampled_operator_comparison"
        or audit["relative_tolerance"] > 1e-6
        or Path(audit["source_kernel"]).resolve() != args.kernel.resolve()
        or audit["source_spectrum_sha256"] != storage["sha256"]
        or (audit["target_dt_s"], audit["target_fft_size"]) != (args.dt, nfft)
        or fingerprint(audit_array_path) != audit["operator_arrays_sha256"]
    ):
        raise ValueError("A matching passed independent operator audit is required")
    if any(audit["environment"][k] != environment[k] for k in keys):
        raise ValueError("Operator audit numerical package/case differs")
    frozen_paths = [
        Path(__file__),
        metadata_path,
        audit_path,
        audit_array_path,
        args.fmm_library,
        *[
            TOOLS / f
            for f in [
                "cauchy_reuse_experiment.py",
                "native_spectrum.py",
                "prepare_native_resolvent.py",
                "replay_resolvent.py",
            ]
        ],
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in frozen_paths}
    cauchy = runpy.run_path(str(TOOLS / "cauchy_reuse_experiment.py"))
    helper = runpy.run_path(str(TOOLS / "replay_resolvent.py"))
    native = runpy.run_path(str(TOOLS / "native_spectrum.py"))
    conversion = runpy.run_path(str(TOOLS / "prepare_native_resolvent.py"))
    circle = cauchy["CircleCauchy"](source_plan.fft_size, source_plan.radius)
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(nfft // 2 + 1) / nfft)
    mapped = cauchy["bdf2_preimage"](bdf2_symbol(zeta) / args.dt, source_plan.dt)
    circle.validate_targets(mapped)
    # Explicit reserve for external-library workspace; not a measured hard RSS bound.
    known = 4 * source_plan.fft_size * (d + 1) * 16 + 4 * len(mapped) * (d + 1) * 16
    known += 16 * nfft * d * 16 + 24 * (steps + 1) * d * 8
    estimate = known + args.fmm_workspace_mb * 1024**2
    if estimate > args.max_memory_mb * 1024**2:
        raise ValueError(f"Known arrays plus FMM reserve need {estimate / 1024**2:.1f} MiB")
    needed = len(mapped) * d**2 * 16 + (correction + 1) * d**2 * 8
    needed += (steps + 1) * (d + 2) * 8 + 64 * 1024**2
    if shutil.disk_usage(args.output.parent).free < needed:
        raise ValueError("Insufficient disk space for complete spectrum, weights and startup")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    backend = cauchy["fortran_fmm2d_backend"](args.fmm_library)
    # Independent small dense sum checks the actual loaded ABI and normalization.
    probe = cauchy["CircleCauchy"](64, 0.9)
    samples = (1 / (2 - probe.sources[:33]))[:, None]
    targets = np.array([0.0, 0.7 + 0.2j, 0.8 - 0.1j])
    expected = probe.evaluate(samples, targets, derivative=True)
    actual = probe.evaluate(samples, targets, derivative=True, backend=backend)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)
    with np.load(audit_array_path, allow_pickle=False) as arrays:
        fresh, audited = arrays["fresh"], arrays["reused"]
    indices = audit["indices"]
    if fresh.shape != (len(indices), d, d) or audited.shape != fresh.shape:
        raise ValueError("Audit matrix dimensions differ")
    args.output.mkdir()
    spectrum_path = args.output / "work-resolvent.npy"
    with spectrum_path.open("xb") as stream:
        np.lib.format.write_array_header_2_0(
            stream,
            dict(descr=np.dtype(complex).str, fortran_order=False, shape=(len(mapped), d, d)),
        )
        stream.truncate(stream.tell() + len(mapped) * d**2 * 16)
    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if source.shape != (source_plan.fft_size // 2 + 1, d, d) or source.dtype != np.complex128:
        raise ValueError("Source half-circle dimensions or dtype differ")
    source._mmap.close()
    reuse_square, fresh_square = np.zeros(len(indices)), np.zeros(len(indices))
    resampling_started = perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        for row in range(d):
            source = np.load(source_path, mmap_mode="r", allow_pickle=False)
            values = circle.evaluate(source[:, row, :], mapped, backend=backend)
            source._mmap.close()
            reuse_square += np.sum(abs(values[indices] - audited[:, row]) ** 2, axis=1)
            fresh_square += np.sum(abs(values[indices] - fresh[:, row]) ** 2, axis=1)
            for index in [0, len(mapped) - 1]:
                if np.linalg.norm(values[index].imag) > 1e-10 * max(
                    np.linalg.norm(values[index].real), 1
                ):
                    raise ValueError("Reused real-axis row violates conjugate symmetry")
                values[index] = values[index].real
            target = np.load(spectrum_path, mmap_mode="r+", allow_pickle=False)
            target[:, row] = values
            target.flush()
            target._mmap.close()
            print(f"resampled rows: {row + 1}/{d}", flush=True)
    resampling_seconds = perf_counter() - resampling_started
    norm = np.sum(abs(fresh) ** 2, axis=(1, 2))
    reuse_errors = np.sqrt(reuse_square / norm)
    fresh_errors = np.sqrt(fresh_square / norm)
    if max(reuse_errors) > 1e-8 or max(fresh_errors) > audit["relative_tolerance"]:
        raise ValueError("Full target-circle FMM evaluation failed the independent sampled guard")
    spectrum = helper["SavedDiskSpectrum"](spectrum_path, plan)
    weights = native["invert_spectrum_to_file"](
        spectrum, plan, args.output / "weights.npy", max_workspace_mb=args.max_memory_mb
    )
    weights._mmap.close()
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
    spring = model.reference.spring_stiffness * line[-1]
    tangent = np.diag(np.r_[wc * law.normal_penalty, wc * law.tangential_penalty, spring])
    weights = conversion["physical_weights_file_inplace"](args.output / "weights.npy", line)
    with threadpool_limits(limits=1, user_api="blas"):
        impedance = NodalInterfaceTransfer(model, quadrature=options["quadrature"]).evaluate(
            1.5 / args.dt
        )
    if not np.all(np.isfinite(impedance)) or np.max(abs(impedance.imag)) > 1e-10:
        raise ValueError("Invalid positive-real instantaneous impedance")
    composition = float(
        np.linalg.norm(weights[0] @ (impedance.real + tangent) - np.eye(d), ord=np.inf)
    )
    weights._mmap.close()
    if not np.isfinite(composition) or composition > 1e-6:
        raise ValueError(f"CQ inverse composition failed: {composition}")
    np.save(args.output / "initial_impedance.npy", impedance.real, allow_pickle=False)
    time = np.arange(steps + 1) * args.dt
    increment = options.get("diagnostic_actuator_increment")
    rate = model.reference.actuator_rate if increment is None else increment / duration
    column = spectrum[:, :, -1] / np.sqrt(line * line[-1])
    elastic = helper["causal_elastic_baseline"](
        column, replace(plan, steps=steps), spring * rate * time
    )
    np.savez(
        args.output / "elastic.npz",
        time=time,
        elastic=elastic,
        uninterrupted_actuator=model.reference.restart_actuator + rate * time,
    )
    for path, digest in frozen.items():
        if fingerprint(path) != digest:
            raise ValueError("Reused kernel implementation or frozen input changed")
    if (source_path.stat().st_size, source_path.stat().st_mtime_ns) != (
        source_stat.st_size,
        source_stat.st_mtime_ns,
    ):
        raise ValueError("Source spectrum changed during reuse")
    final_environment = environment_record(options["case"])
    if any(final_environment[k] != environment[k] for k in keys):
        raise ValueError("Numerical package/case changed during reuse")
    helper_node = next(
        n
        for n in ast.parse((TOOLS / "replay_resolvent.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "causal_elastic_baseline"
    )
    record = dict(
        status="completed_resolvent_kernel_only",
        source_run=metadata["source_run"],
        source_mesh_arguments=options,
        dt_s=args.dt,
        full_steps=steps,
        correction_steps=correction,
        plan=asdict(plan),
        history_dofs=d,
        frequency_sampling=dict(
            method="guarded_circle_cauchy_reuse", source_spectrum_sha256=storage["sha256"]
        ),
        sampling_options=None,
        spectrum_storage={**spectrum.record(), "coordinates": storage["coordinates"]},
        initial_composition_inf_error=composition,
        estimated_array_peak_mib=estimate / 1024**2,
        fmm_workspace_reserve_mib=args.fmm_workspace_mb,
        fmm_threads_requested=args.threads,
        hard_cpu_affinity_enforced=False,
        resampling_seconds=resampling_seconds,
        total_seconds=perf_counter() - begun,
        environment=environment,
        sampled_fresh_bem_relative_errors=fresh_errors.tolist(),
        sampled_fmm_relative_errors=reuse_errors.tolist(),
        frozen_inputs_sha256=frozen,
        helper_sha256=fingerprint(TOOLS / "replay_resolvent.py"),
        helper_math_sha256=hashlib.sha256(
            ast.dump(helper_node, include_attributes=False).encode()
        ).hexdigest(),
        weights_sha256=fingerprint(args.output / "weights.npy"),
        elastic_sha256=fingerprint(args.output / "elastic.npz"),
        full_original_event_reproduced=False,
        certified_error_bound=False,
        scope="Streamed contour reuse checked against independent sampled BEM matrices; full trajectory and conditional AD require separate verification",
    )
    write_json(args.output / "summary.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "environment"}, indent=2))


if __name__ == "__main__":
    main()

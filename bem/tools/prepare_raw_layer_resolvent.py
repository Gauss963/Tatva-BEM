"""Build every resolvent CQ lag using raw-layer frequency interpolation.

Source kernel metadata supplies the physical case and numerical defaults;
no source frequency matrix or weight is read to generate the fresh kernel.
Complete output remains compatible with the refined-contact forward/AD tools.
"""

import argparse
import ast
import hashlib
import json
import os
import resource
import runpy
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import plan_cq
from finiteblock_bem.cq_storage import DiskSpectrum, spectrum_bytes
from finiteblock_bem.galerkin_fast import prepared_geometry_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_kernel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--duration", type=float)
    for flag in ["h", "contact-h", "contact-end-h", "corner-h", "restart-knot-window"]:
        parser.add_argument(f"--{flag}", type=float)
    parser.add_argument("--restart-knot-subdivisions", type=int)
    parser.add_argument("--clear-restart-knots", action="store_true")
    parser.add_argument("--quadrature", type=int)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--raw-rtol", type=float, default=1e-11)
    parser.add_argument("--output-rtol", type=float, default=1e-9)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("FDBEM_CORES", "1")))
    parser.add_argument("--fft-rows", type=int, default=4)
    parser.add_argument("--weight-layout", choices=["c", "fortran"], default="c")
    parser.add_argument("--max-workspace-mb", type=float, default=3072)
    parser.add_argument("--max-memory-mb", type=float, default=8192)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a fresh kernel directory outside protected Tatva")
    if (
        args.workers < 1
        or args.fft_rows < 1
        or not all(np.isfinite(v) and v > 0 for v in (args.max_workspace_mb, args.max_memory_mb))
    ):
        raise ValueError("Require positive construction workers and budgets")
    begun, cpu_begun = perf_counter(), process_time()
    source_path = args.source_kernel / "summary.json"
    source = json.loads(source_path.read_text())
    if source["status"] != "completed_resolvent_kernel_only":
        raise ValueError("Require completed kernel metadata as physical/numerical input")
    native = runpy.run_path(str(TOOLS / "prepare_native_resolvent.py"))
    options, overrides = native["discretization_arguments"](source["source_mesh_arguments"], args)
    environment = environment_record(options["case"])
    for key in ("source_sha256", "case_sha256", "restart_data_sha256"):
        if source["environment"][key] != environment[key]:
            raise ValueError("Source physical package or case changed")
    dt = source["dt_s"] if args.dt is None else args.dt
    duration = source["full_steps"] * source["dt_s"] if args.duration is None else args.duration
    if not all(np.isfinite(v) and v > 0 for v in (dt, duration)):
        raise ValueError("Require positive dt and duration")
    steps = round(duration / dt)
    if steps < 1 or abs(steps * dt - duration) > 1e-12 * max(duration, dt):
        raise ValueError("Duration must be covered by an integer number of steps")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    d = 2 * len(model.moving_contact) + 1
    plan = plan_cq(d, dt, steps, max_memory_mb=1e9)
    raw = runpy.run_path(str(TOOLS / "raw_layer_sampling.py"))
    sampling = raw["RawSamplingOptions"](
        degree=args.degree, raw_rtol=args.raw_rtol, output_rtol=args.output_rtol
    )
    meshes = (model.reference.moving, model.reference.stationary)
    packet_bytes = sum(3 * mesh.dofs**2 * 16 for mesh in meshes)
    interpolation = (
        4 * (sampling.degree + 1) + 3 * sampling.check_points + 2 * args.workers + 4
    ) * packet_bytes
    geometry = sum(prepared_geometry_bytes(mesh, options["quadrature"]) for mesh in meshes)
    solver = 32 * args.workers * (model.constraints.size**2 + sum(m.dofs**2 for m in meshes)) * 16
    rows = min(d, args.fft_rows)
    ifft = 4 * plan.fft_size * rows * d * 8 + (steps + 1) * rows * d * 8 + 64 * d * d * 16
    startup = 8 * plan.fft_size * d * 16 + 24 * (steps + 1) * d * 8
    peak = max(geometry + interpolation + solver, max(ifft, startup)) + 128 * 1024**2
    weight_bytes = (steps + 1) * d * d * 8
    conversion = 0
    if args.weight_layout == "fortran":
        conversion_rows = min(
            d,
            32,
            int((args.max_workspace_mb * 1024**2 - 64 * 1024**2) // (2 * (steps + 1) * d * 8)),
        )
        if conversion_rows < 1:
            raise ValueError("Fortran conversion budget cannot hold one bounded row tile")
        conversion = 2 * (steps + 1) * conversion_rows * d * 8 + 64 * 1024**2
        peak = max(peak, conversion + 128 * 1024**2)
    if (
        interpolation > args.max_workspace_mb * 1024**2
        or ifft > args.max_workspace_mb * 1024**2
        or peak > args.max_memory_mb * 1024**2
    ):
        raise ValueError(
            f"Raw construction array estimate exceeds budgets: {peak / 1024**2:.1f} MiB"
        )
    names = [
        "raw_layer_sampling.py",
        "schur_condensation_experiment.py",
        "prepare_native_resolvent.py",
        "native_spectrum.py",
        "bounded_weight_conversion.py",
        "replay_resolvent.py",
        "build_resolvent_history.py",
        "fortran_weight_storage.py",
    ]
    paths = [Path(__file__), source_path, *[TOOLS / n for n in names]]
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    required = spectrum_bytes(plan) + weight_bytes + (steps + 1) * (d + 2) * 8 + d * d * 8 + 65536
    if args.weight_layout == "fortran":
        # Preserve the newly computed C-order file until its complete Fortran
        # copy has passed the bitwise read-back check. Source kernels are read-only.
        required += weight_bytes + 8192
    if required > shutil.disk_usage(output.parent).free:
        raise ValueError("Insufficient disk for complete spectrum, all time weights and startup")
    output.mkdir()
    write_json(
        output / "preflight.json",
        dict(
            status="preflight_only_not_a_complete_kernel",
            plan=asdict(plan),
            source_kernel_metadata=str(source_path.resolve()),
            fresh_raw_samples_no_source_frequency_reuse=True,
            required_disk_bytes=required,
            estimated_peak_array_mib=peak / 1024**2,
            interpolation_array_mib=interpolation / 1024**2,
            ifft_array_mib=ifft / 1024**2,
            weight_layout=args.weight_layout,
            storage_conversion_array_mib=conversion / 1024**2,
            numerical_discretization_overrides=overrides,
            frozen_input_sha256=frozen,
            original_tatva_readonly=True,
            same_accuracy_fem_advantage_verified=False,
        ),
    )
    if args.preflight_only:
        print(output / "preflight.json", flush=True)
        return
    read = runpy.run_path(str(TOOLS / "replay_resolvent.py"))
    schur = runpy.run_path(str(TOOLS / "schur_condensation_experiment.py"))[
        "ConstrainedSchurTransfer"
    ]
    transfer = schur(model, quadrature=options["quadrature"])
    con, ref = model.constraints, model.reference
    line = np.r_[con.contact_weights, con.contact_weights, con.platen_width]
    law = ref.contact
    tangent = np.diag(
        np.r_[
            con.contact_weights * law.normal_penalty,
            con.contact_weights * law.tangential_penalty,
            0,
        ]
    )
    spring = ref.spring_stiffness * con.platen_width
    tangent[-1, -1] += spring
    compliance = runpy.run_path(str(TOOLS / "build_resolvent_history.py"))[
        "compliance_in_work_coordinates"
    ]
    spectrum = DiskSpectrum(output / "work-resolvent.npy", plan)
    last = [0.0]

    def diagnostics(record):
        now = perf_counter()
        if now - last[0] >= 30 or record["all_fft_grid_points_emitted"]:
            write_json(
                output / "progress.json",
                dict(phase="raw_layer_sampling", sampling=record, elapsed_seconds=now - begun),
            )
            print(
                f"raw samples={record['source_evaluations']}, grid={record['transformed_grid_points']}/{plan.fft_size // 2 + 1}",
                flush=True,
            )
            last[0] = now

    def provider(s, operator=transfer):
        return raw["RawLayerPacket"].from_operators(p.assemble(s) for p in operator.prepared)

    def transform(packet, s, operator=transfer):
        return compliance(operator.evaluate_operators(packet.operators()), tangent, line)

    sampling_start = perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        records = raw["sample_raw_layers"](
            provider,
            transform,
            plan,
            lambda i, s, value: spectrum.__setitem__(i, value),
            options=sampling,
            workers=args.workers,
            max_workspace_mb=args.max_workspace_mb,
            diagnostics=diagnostics,
        )
        impedance = transfer.evaluate(1.5 / dt)
    sampling_seconds = perf_counter() - sampling_start
    del transfer, provider, transform
    if np.max(abs(impedance.imag)) > 1e-10 or not np.all(np.isfinite(impedance)):
        raise ValueError("Fresh positive-real instantaneous impedance is invalid")
    fft = runpy.run_path(str(TOOLS / "native_spectrum.py"))
    transform_start = perf_counter()
    weights = fft["invert_spectrum_to_file"](
        spectrum,
        plan,
        output / "weights.npy",
        fft_rows=rows,
        max_workspace_mb=args.max_workspace_mb,
    )
    weights._mmap.close()
    weights = native["physical_weights_file_inplace"](output / "weights.npy", line)
    composition = float(
        np.linalg.norm(weights[0] @ (impedance.real + tangent) - np.eye(d), ord=np.inf)
    )
    weights._mmap.close()
    if not np.isfinite(composition) or composition > 1e-6:
        raise ValueError(f"Raw-layer lag-zero inverse composition failed: {composition}")
    conversion_record = None
    if args.weight_layout == "fortran":
        write_json(
            output / "progress.json",
            dict(
                phase="bitwise_fortran_storage_conversion", elapsed_seconds=perf_counter() - begun
            ),
        )
        convert = runpy.run_path(str(TOOLS / "fortran_weight_storage.py"))[
            "convert_weights_fortran"
        ]
        converted = output / "weights-fortran.npy"
        conversion_record = convert(
            output / "weights.npy", converted, max_workspace_mb=args.max_workspace_mb
        )
        # Only replace this fresh build's own temporary file, after validation.
        converted.replace(output / "weights.npy")
    np.save(output / "initial_impedance.npy", impedance.real, allow_pickle=False)
    time = np.arange(steps + 1) * dt
    increment = options.get("diagnostic_actuator_increment")
    rate = ref.actuator_rate if increment is None else increment / time[-1]
    column = spectrum[:, :, -1] / np.sqrt(line * line[-1])
    elastic = read["causal_elastic_baseline"](column, plan, spring * rate * time)
    np.savez(
        output / "elastic.npz",
        time=time,
        elastic=elastic,
        uninterrupted_actuator=ref.restart_actuator + rate * time,
    )
    if any(fingerprint(p) != v for p, v in frozen.items()):
        raise ValueError("Raw-layer preparation source changed")
    final = environment_record(options["case"])
    if any(
        final[k] != environment[k] for k in ("source_sha256", "case_sha256", "restart_data_sha256")
    ):
        raise ValueError("Physical bulk package or inputs changed")
    node = next(
        n
        for n in ast.parse((TOOLS / "replay_resolvent.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "causal_elastic_baseline"
    )
    storage = spectrum.record()
    storage["coordinates"] = "diag(sqrt(w)) * L * diag(sqrt(w))"
    write_json(
        output / "summary.json",
        dict(
            status="completed_resolvent_kernel_only",
            source_run=source.get("source_run", str(args.source_kernel.resolve())),
            source_role="Physical case and numerical defaults only; fresh raw-layer kernel",
            source_mesh_arguments=options,
            numerical_discretization_overrides=overrides,
            dt_s=dt,
            full_steps=steps,
            correction_steps=steps,
            plan=asdict(plan),
            history_dofs=d,
            frequency_sampling=records,
            sampling_options=asdict(sampling),
            spectrum_storage=storage,
            assembly="raw_layers_before_constrained_schur",
            time_weights_storage="disk_mmap",
            time_weights_layout=args.weight_layout,
            storage_conversion=conversion_record,
            initial_composition_inf_error=composition,
            estimated_construction_array_peak_mib=peak / 1024**2,
            environment=environment,
            helper_sha256=frozen[str((TOOLS / "replay_resolvent.py").resolve())],
            helper_math_sha256=hashlib.sha256(
                ast.dump(node, include_attributes=False).encode()
            ).hexdigest(),
            weights_sha256=read["digest_file"](output / "weights.npy"),
            elastic_sha256=read["digest_file"](output / "elastic.npz"),
            frozen_inputs_sha256=frozen,
            total_seconds=perf_counter() - begun,
            process_cpu_seconds=process_time() - cpu_begun,
            peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2 if sys.platform == "darwin" else 1024),
            raw_sampling_and_fresh_instantaneous_seconds=sampling_seconds,
            transforms_startup_hashing_and_output_seconds=perf_counter() - transform_start,
            full_original_event_reproduced=False,
            original_tatva_readonly=True,
            same_accuracy_fem_advantage_verified=False,
            scope="Fresh complete CQ kernel; empirical raw/output checks, not physical convergence or an all-frequency certificate",
        ),
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

"""Prepare complete physical interior Tq Radau spectra with empirical raw sampling.

Body V/H/D and probe potentials/gradients share each raw evaluation. Both modes
are sampled sequentially, audited against the original free-body algebra and
retained separately from compact F0/F1. Every frequency is transformed; no rank,
history, material damping or source trajectory approximation is introduced.
"""

import argparse
import json
import os
import runpy
import shutil
import signal
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import plan_cq
from finiteblock_bem.galerkin_fast import prepared_geometry_bytes
from finiteblock_bem.io import environment_record
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
METHOD = runpy.run_path(str(TOOLS / "interior_raw_sampling.py"))
CHECKPOINT = runpy.run_path(str(TOOLS / "rectangular_sampling_checkpoint.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))
REFERENCE = runpy.run_path(str(TOOLS / "reconstruct_interior_reference.py"))
NATIVE = runpy.run_path(str(TOOLS / "prepare_native_resolvent.py"))
TX = CHECKPOINT["TX"]
DIGEST = CHECKPOINT["BASE"]["digest_range"]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dt", type=float, default=2e-6)
    p.add_argument("--duration", type=float, default=0.045)
    for k in ["h", "contact-h", "contact-end-h", "corner-h", "restart-knot-window"]:
        p.add_argument("--" + k, type=float)
    p.add_argument("--restart-knot-subdivisions", type=int)
    p.add_argument("--clear-restart-knots", action="store_true")
    p.add_argument("--quadrature", type=int)
    p.add_argument(
        "--probe-plan", type=Path, default=TOOLS.parent / "examples/event16_interior_probes.json"
    )
    p.add_argument("--probe-quadrature", type=int, default=16)
    p.add_argument("--degree", type=int, default=16)
    p.add_argument("--raw-rtol", type=float, default=1e-11)
    p.add_argument("--output-rtol", type=float, default=1e-9)
    p.add_argument("--workers", type=int, default=int(os.environ.get("FDBEM_CORES", "1")))
    p.add_argument("--buffer-matrices", type=int, default=64)
    p.add_argument("--combine-matrices", type=int, default=16)
    p.add_argument("--max-workspace-mb", type=float, default=10240)
    p.add_argument("--max-memory-mb", type=float, default=18432)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verify-resume-only", action="store_true")
    return p.parse_args()


def main():
    args = arguments()
    begun = perf_counter()
    output = args.output.resolve()
    if (
        output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve())
        or (args.resume and args.preflight_only)
        or (args.verify_resume_only and not args.resume)
        or args.workers < 1
        or args.buffer_matrices < 1
        or args.combine_matrices < 1
        or args.probe_quadrature < 4
        or args.probe_quadrature % 2
        or any(
            not np.isfinite(v) or v <= 0
            for v in [args.dt, args.duration, args.max_workspace_mb, args.max_memory_mb]
        )
    ):
        raise ValueError(
            "Require valid fresh/resume options, positive budgets and output outside protected Tatva"
        )
    steps = round(args.duration / args.dt)
    if steps < 1 or abs(steps * args.dt - args.duration) > 1e-12 * max(args.duration, args.dt):
        raise ValueError("Duration must be an integer number of complete CQ steps")
    if args.source.suffix.lower() == ".toml":
        defaults = dict(
            case=str(args.source.resolve()),
            h=10,
            contact_h=2,
            contact_end_h=0.015625,
            corner_h=0.015625,
            quadrature=12,
            restart_knot_window=None,
            restart_knot_subdivisions=1,
        )
    else:
        source = json.loads(args.source.read_text())
        defaults = source["source_mesh_arguments"]
        current = environment_record(defaults["case"])
        if any(
            source["environment"][k] != current[k]
            for k in ["source_sha256", "case_sha256", "restart_data_sha256"]
        ):
            raise ValueError("Mesh/case defaults differ from current numerical source")
    options, _ = NATIVE["discretization_arguments"](defaults, args)
    options.update(dt=args.dt, steps=steps, duration=args.duration)
    environment = environment_record(options["case"])
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    probes = json.loads(args.probe_plan.read_text())
    points = [np.asarray(probes[k], float) for k in ["moving", "stationary"]]
    inputs = 2 * len(model.moving_contact) + 1
    outputs = 8 * sum(len(p) for p in points)
    plan = plan_cq(inputs, args.dt, steps, max_memory_mb=1e9, disk_spectrum=True)
    sampling = METHOD["RAW"]["RawSamplingOptions"](
        degree=args.degree, raw_rtol=args.raw_rtol, output_rtol=args.output_rtol
    )
    shape = (plan.fft_size // 2 + 1, outputs, inputs)
    matrix_bytes = outputs * inputs * 16
    meshes = [model.reference.moving, model.reference.stationary]
    packet = sum(
        (3 * m.dofs**2 + 12 * len(p) * m.dofs) * 16 for m, p in zip(meshes, points, strict=True)
    )
    interpolation = (
        4 * (sampling.degree + 1) + 3 * sampling.check_points + 2 * args.workers + 4
    ) * packet
    geometry = sum(prepared_geometry_bytes(m, options["quadrature"]) for m in meshes)
    solver = 32 * args.workers * (model.constraints.size**2 + sum(m.dofs**2 for m in meshes)) * 16
    buffer = min(args.buffer_matrices, shape[0]) * matrix_bytes
    combination = 6 * min(args.combine_matrices, shape[0]) * matrix_bytes + 2 * buffer
    peak = (
        max(
            geometry + interpolation + solver + buffer,
            combination,
        )
        + 256 * 1024**2
    )
    required = 4 * shape[0] * matrix_bytes + 256 * 1024**2
    source_names = [
        "prepare_interior_spectra.py",
        "interior_raw_sampling.py",
        "rectangular_sampling_checkpoint.py",
        "reconstruct_interior_reference.py",
        "rectangular_cq_history.py",
        "interior_fields.py",
        "interior_radau.py",
        "radau_experiment.py",
        "radau_raw_sampling.py",
        "raw_layer_sampling.py",
        "radau_sampling_checkpoint.py",
        "buffered_spectrum.py",
        "trace_schur_experiment.py",
        "schur_condensation_experiment.py",
        "prepare_native_resolvent.py",
        "inplace_radau_compaction.py",
        "direct_fortran_radau.py",
    ]
    frozen = {
        str(p.resolve()): DIGEST(p)
        for p in [args.source, args.probe_plan, *[TOOLS / k for k in source_names]]
    }
    context = dict(
        arguments={
            k: str(v.resolve()) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in ["resume", "verify_resume_only", "preflight_only"]
        },
        environment={
            k: environment[k]
            for k in ["source_sha256", "case_sha256", "restart_data_sha256", "numpy", "scipy"]
        },
        source_sha256=frozen,
    )
    preflight_path = output / "preflight.json"
    if args.resume:
        previous = json.loads(preflight_path.read_text())
        if (
            previous["context"] != context
            or previous["plan"] != asdict(plan)
            or previous["shape"] != list(shape)
            or previous["preflight_only"]
        ):
            raise ValueError("Interior resume changes source, runtime, channels or configuration")
        if (output / "summary.json").exists():
            raise ValueError("Completed interior spectra need no resume")
    elif output.exists():
        raise ValueError("Choose a fresh interior spectral directory")
    allocated = (
        sum(
            (output / n).stat().st_blocks * 512
            for n in CHECKPOINT["BASE"]["CHANNELS"]
            if (output / n).is_file()
        )
        if args.resume
        else 0
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if (
        peak > args.max_memory_mb * 1024**2
        or max(interpolation, combination) > args.max_workspace_mb * 1024**2
        or max(0, required - allocated) > shutil.disk_usage(output.parent).free
    ):
        raise ValueError("Interior spectral construction exceeds array or disk budgets")
    if not args.resume:
        output.mkdir()
        TX["publish"](
            preflight_path,
            dict(
                kind="rectangular_interior_raw_sampling",
                context=context,
                plan=asdict(plan),
                shape=list(shape),
                source_mesh_arguments=options,
                probe_plan=probes,
                probe_quadrature=args.probe_quadrature,
                preflight_only=args.preflight_only,
                resources=dict(
                    array_peak_bytes=peak,
                    interpolation_bytes=interpolation,
                    geometry_bytes=geometry,
                    solver_bytes=solver,
                    required_disk_bytes=required,
                    raw_and_compact_spectra_retained=True,
                    provider_target_geometry_in_256MiB_reserve=True,
                ),
            ),
        )
    if args.preflight_only:
        print(preflight_path, flush=True)
        return
    lease = CHECKPOINT["BASE"]["exclusive_lease"](output)
    journal = None
    prior = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Interior sampling interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)

    def verify():
        if any(DIGEST(p) != v for p, v in frozen.items()):
            raise ValueError("Frozen interior source/input changed")
        now = environment_record(options["case"])
        if any(now[k] != v for k, v in context["environment"].items()):
            raise ValueError("Interior runtime changed")

    try:
        journal = CHECKPOINT["RectangularJournal"](
            output, plan, outputs, inputs, resume=args.resume
        )
        if args.verify_resume_only:
            print(
                json.dumps(
                    dict(
                        status="verified_rectangular_prefix_only",
                        committed={k: journal.prefix(k) for k in CHECKPOINT["BASE"]["CHANNELS"]},
                        build_modified=False,
                    )
                ),
                flush=True,
            )
            return
        journal.begin()
        raw = [journal.spectrum(f"work-family{i}.npy") for i in range(2)]
        audit_path = output / "raw-audit.json"
        if not audit_path.exists():
            with threadpool_limits(limits=1, user_api="blas"):
                transfer = METHOD["StableInteriorTransfer"](
                    model,
                    points,
                    quadrature=options["quadrature"],
                    probe_quadrature=args.probe_quadrature,
                )
                writer = sampler = None

                def evaluate(packet, s):
                    return REFERENCE["pack"](
                        transfer.evaluate_packet(packet, s), sum(len(p) for p in points), inputs
                    )

                for branch in range(2):
                    name = f"work-family{branch}.npy"
                    prefix = journal.prefix(name)
                    if prefix == shape[0]:
                        continue
                    writer = CHECKPOINT["Writer"](
                        journal, raw[branch], matrices=args.buffer_matrices
                    )
                    stack, start = journal.sampler_stack(name, [(0, shape[0])])
                    sampler = METHOD["rectangular_sampler"](
                        outputs,
                        inputs,
                        stack=stack,
                        start=start,
                        interval=lambda left, right, remaining, w=writer, k=name: (
                            journal.save_sampler_state(k, left, right, remaining, w)
                        ),
                    )
                    last = [0.0]

                    def progress(record, b=branch):
                        if perf_counter() - last[0] > 30 or record["all_fft_grid_points_emitted"]:
                            verify()
                            TX["publish"](
                                output / "progress.json",
                                dict(
                                    branch=b,
                                    sampling=record,
                                    committed=journal.prefix(f"work-family{b}.npy"),
                                ),
                            )
                            print(
                                f"interior branch={b}, raw={record['source_evaluations']}, grid={record['transformed_grid_points']}/{shape[0]}",
                                flush=True,
                            )
                            last[0] = perf_counter()

                    record = RADAU["sample_radau_branch"](
                        transfer.raw,
                        evaluate,
                        plan,
                        branch,
                        lambda index, s, value, target=writer: target.accept(index, value),
                        options=sampling,
                        workers=args.workers,
                        max_workspace_mb=args.max_workspace_mb,
                        diagnostics=progress,
                        sampler=sampler,
                    )
                    writer.finish()
                    record.update(
                        restored_prefix=prefix,
                        recomputed_active_interval_start=start,
                        output_check="original Frobenius AND every observation row",
                    )
                    TX["publish"](
                        output / "progress.json",
                        dict(branch=branch, sampling=record, committed=journal.prefix(name)),
                    )
                    journal.data["branch_records"][str(branch)] = record
                    journal.publish()
                direct = METHOD["FIELD"]["InteriorRetainedTransfer"](
                    model,
                    points,
                    quadrature=options["quadrature"],
                    probe_quadrature=args.probe_quadrature,
                    prepared=False,
                )
                direct.blocks = [
                    (*block[:-1], assembler)
                    for block, assembler in zip(direct.blocks, transfer.bulk.prepared, strict=True)
                ]
                indices = sorted(
                    set(
                        [
                            0,
                            1,
                            min(7, shape[0] - 1),
                            plan.fft_size // 16,
                            plan.fft_size // 4,
                            plan.fft_size // 2,
                        ]
                    )
                )
                checks = []
                audit_options = METHOD["RAW"]["RawSamplingOptions"](
                    output_rtol=1e-6, output_atol=1e-12
                )
                for index in indices:
                    z = plan.radius * np.exp(-2j * np.pi * index / plan.fft_size)
                    for branch, s in enumerate(RADAU["lower_half_radau_modes"](z, plan.dt)):
                        fresh = REFERENCE["pack"](
                            direct.evaluate(s), sum(len(p) for p in points), inputs
                        )
                        value = raw[branch][index]
                        relative = float(
                            np.max(
                                np.linalg.norm(value - fresh, axis=1)
                                / np.maximum(np.linalg.norm(fresh, axis=1), 1e-30)
                            )
                        )
                        # Damped high-frequency probe rows can vanish. Retain
                        # the global AND row checks with explicit absolute and
                        # relative tolerances instead of dividing by near-zero.
                        normalized = METHOD["maximum_row_error"]([fresh], [value], audit_options)
                        if not np.isfinite(normalized) or normalized > 1:
                            raise ValueError("Independent original interior transfer audit failed")
                        checks.append(
                            dict(
                                index=index,
                                branch=branch,
                                maximum_row_relative_error=relative,
                                relative_floor=1e-30,
                                normalized_global_and_row_error=normalized,
                                relative_tolerance=audit_options.output_rtol,
                                absolute_tolerance=audit_options.output_atol,
                            )
                        )
                audit = dict(
                    plan=asdict(plan),
                    preflight_sha256=DIGEST(preflight_path),
                    raw_records=[r.record() for r in raw],
                    checks=checks,
                )
                TX["publish"](audit_path, audit)
                del direct, transfer, evaluate, writer, sampler
        audit = json.loads(audit_path.read_text())
        if audit["plan"] != asdict(plan) or audit["preflight_sha256"] != DIGEST(preflight_path):
            raise ValueError("Raw interior audit changed")
        if any(
            journal.prefix(f"work-family{i}.npy") != shape[0]
            or raw[i].record()["sha256"] != audit["raw_records"][i]["sha256"]
            for i in range(2)
        ):
            raise ValueError("Complete raw interior branches changed")
        journal.mark("combination", raw_audit_sha256=DIGEST(audit_path))
        compact = [journal.spectrum(f"compact-family{i}.npy") for i in range(2)]
        writers = [CHECKPOINT["Writer"](journal, s, matrices=args.buffer_matrices) for s in compact]
        first = min(w.committed for w in writers)
        for left in range(first, shape[0], args.combine_matrices):
            right = min(left + args.combine_matrices, shape[0])
            z = plan.radius * np.exp(-2j * np.pi * np.arange(left, right) / plan.fft_size)
            a, b = RADAU["lower_half_radau_modes"](z, 1.0).T
            slow, fast = [r[left:right] for r in raw]
            a, b = a[:, None, None], b[:, None, None]
            families = [(b * slow - a * fast) / (b - a), (fast - slow) / (b - a)]
            for family, values in enumerate(families):
                for endpoint in [0, shape[0] - 1]:
                    if left <= endpoint < right:
                        v = values[endpoint - left]
                        if np.linalg.norm(v.imag) > 1e-10 * max(np.linalg.norm(v.real), 1):
                            raise ValueError("Combined interior endpoint symmetry failed")
                        values[endpoint - left] = v.real
                for index in range(left, right):
                    writers[family].accept(index, values[index - left])
        for writer in writers:
            writer.finish()
        verify()
        journal.mark("completed")
        journal.end("completed")
        accounted = not journal.data.get(
            "prior_attempt_before_journal_creation_unaccounted", False
        ) and all(a["status"] != "running" for a in journal.data["attempts"])
        TX["publish"](
            output / "summary.json",
            dict(
                status="completed_empirical_raw_interior_radau_spectra",
                time_method="radau2",
                plan=asdict(plan),
                source_mesh_arguments=options,
                environment=environment,
                probe_plan=probes,
                probe_quadrature=args.probe_quadrature,
                family_spectra=[s.record() for s in compact],
                raw_branch_spectra=audit["raw_records"],
                family_spectra_coordinates="physical Tq",
                raw_mode_checks=audit["checks"],
                frequency_sampling=journal.data["branch_records"],
                sampling_options=asdict(sampling),
                preflight_sha256=DIGEST(preflight_path),
                raw_audit_sha256=DIGEST(audit_path),
                frozen_inputs_sha256=frozen,
                durable_attempts=journal.data["attempts"],
                all_attempt_timings_accounted=accounted,
                current_main_seconds=perf_counter() - begun,
                timing_scope="Main after parsing/imports; journal after preflight. Outer invocation intervals are required; nested intervals must not be added.",
                every_frequency_transformed=True,
                no_full_time_weights=True,
                empirical_checks_not_error_theorem=True,
                full_original_event_reproduced=False,
                initialized_observation_gradient_verified=False,
                same_accuracy_fem_advantage_verified=False,
                original_tatva_readonly=True,
            ),
        )
        print(output / "summary.json", flush=True)
    except BaseException:
        if journal is not None and journal.attempt is not None:
            journal.end("interrupted_or_failed")
        raise
    finally:
        signal.signal(signal.SIGTERM, prior)
        lease.close()


if __name__ == "__main__":
    main()

"""Build fresh complete Radau CQ families without a source frequency spectrum.

Two raw-layer branches are sampled sequentially. Their disk arrays are then
converted in place to the compact families, retaining every requested lag,
the finite-DFT shift wrap and complete causal actuator startup. This prepares
a kernel; it does not establish nonlinear accuracy or an advantage over FEM.
"""

import argparse
import json
import os
import resource
import runpy
import shutil
import signal
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import plan_cq
from finiteblock_bem.cq_storage import DiskSpectrum, spectrum_bytes
from finiteblock_bem.galerkin_fast import prepared_geometry_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOLS = Path(__file__).resolve().parent
RAW = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))
STORAGE = runpy.run_path(str(TOOLS / "compact_radau_storage.py"))
NATIVE = runpy.run_path(str(TOOLS / "prepare_native_resolvent.py"))
REPLAY = runpy.run_path(str(TOOLS / "replay_resolvent.py"))
WORK = runpy.run_path(str(TOOLS / "build_resolvent_history.py"))["compliance_in_work_coordinates"]
TRANSFER = runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"]
BUFFER = runpy.run_path(str(TOOLS / "buffered_spectrum.py"))["BufferedSpectrumWriter"]
CHECKPOINT = runpy.run_path(str(TOOLS / "radau_sampling_checkpoint.py"))
LEGACY_PRODUCER_SHA256 = "d56a13706c3d78d55958587b84650f268ab08ea056a3224840df5628063aa363"
ACTIVE_JOURNAL = ACTIVE_LEASE = None


def construction_arrays(
    model,
    plan,
    sampling,
    workers,
    quadrature,
    fft_rows,
    combine_matrices,
    startup_rows=8,
    spectrum_buffer_matrices=64,
):
    """Conservative array estimates for sequential phases, excluding OS cache."""
    meshes = [model.reference.moving, model.reference.stationary]
    d, steps, nfft = plan.dofs, plan.steps, plan.fft_size
    packet = sum(3 * m.dofs**2 * 16 for m in meshes)
    interpolation = (
        4 * (sampling.degree + 1) + 3 * sampling.check_points + 2 * workers + 4
    ) * packet
    geometry = sum(prepared_geometry_bytes(m, quadrature) for m in meshes)
    buffer = min(plan.fft_size // 2 + 1, spectrum_buffer_matrices) * d * d * 16
    solver = 32 * workers * (model.constraints.size**2 + sum(m.dofs**2 for m in meshes)) * 16
    rows = min(d, fft_rows)
    inversion = 4 * nfft * rows * d * 8 + 2 * (nfft // 2 + 1) * rows * d * 16
    inversion += (steps + 1) * rows * d * 8 + 4 * d * d * 8
    combine = 12 * combine_matrices * d * d * 16 + 64 * d * d * 16
    startup = 12 * nfft * min(d, startup_rows) * 16 + 8 * (steps + 1) * d * 8
    peak = (
        max(geometry + interpolation + solver + buffer, inversion, combine, startup) + 128 * 1024**2
    )
    return dict(
        peak=peak,
        interpolation=interpolation,
        inversion=inversion,
        combination=combine,
        startup=startup,
        geometry=geometry,
        solver=solver,
        spectrum_buffer=buffer,
    )


def _main():
    global ACTIVE_JOURNAL, ACTIVE_LEASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="Case TOML or small mesh/case summary JSON; no source matrix is read",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=2e-6)
    parser.add_argument("--duration", type=float, default=0.045)
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
    parser.add_argument("--combine-matrices", type=int, default=64)
    parser.add_argument("--spectrum-buffer-matrices", type=int, default=64)
    parser.add_argument("--startup-rows", type=int, default=8)
    parser.add_argument("--primitive-partition", action="store_true")
    parser.add_argument("--weight-layout", choices=["c", "fortran"], default="fortran")
    parser.add_argument("--max-workspace-mb", type=float, default=10240)
    parser.add_argument("--max-memory-mb", type=float, default=18432)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--durable-build",
        action="store_true",
        help="Journal raw/compact matrices and retain original modes",
    )
    parser.add_argument("--resume", action="store_true", help="Continue this exact durable build")
    parser.add_argument(
        "--resume-receipt", type=Path, help="Explicit SHA256 receipt for a legacy raw prefix"
    )
    parser.add_argument(
        "--verify-resume-only",
        action="store_true",
        help="Verify saved source, matrices and resources without modifying the build",
    )
    args = parser.parse_args()
    publish = CHECKPOINT["atomic_json"] if args.durable_build else write_json
    output = args.output.resolve()
    if (output.exists() and not args.resume) or output.is_relative_to(
        Path("/Volumes/Gauss-T7/tatva").resolve()
    ):
        raise ValueError("Choose a fresh kernel directory outside protected Tatva")
    if (args.resume and (not args.durable_build or args.preflight_only or not output.is_dir())) or (
        (args.resume_receipt is not None or args.verify_resume_only) and not args.resume
    ):
        raise ValueError(
            "Resume requires an existing durable build; a legacy receipt is resume-only"
        )
    previous = None
    if args.resume:
        previous = json.loads((output / "preflight.json").read_text())
        if previous["arguments"].get("preflight_only"):
            raise ValueError("A preflight-only directory has no kernel construction to resume")
        if (output / "summary.json").exists():
            raise ValueError("A completed kernel does not require resuming")
        current = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        for key, value in current.items():
            if (
                key
                not in {
                    "resume",
                    "resume_receipt",
                    "durable_build",
                    "preflight_only",
                    "verify_resume_only",
                }
                and previous["arguments"].get(key) != value
            ):
                raise ValueError(f"Resume changes the recorded argument: {key}")
        for path, expected in previous["frozen_input_sha256"].items():
            actual = fingerprint(path)
            if Path(path).resolve() == Path(__file__).resolve():
                if expected not in {actual, LEGACY_PRODUCER_SHA256}:
                    raise ValueError(
                        "Unknown producer source migration; preserve the partial build"
                    )
            elif actual != expected:
                raise ValueError(f"Resume changes a numerical source/input: {path}")
    if (
        args.workers < 1
        or args.fft_rows < 1
        or args.combine_matrices < 1
        or args.startup_rows < 1
        or args.spectrum_buffer_matrices < 1
        or not all(
            np.isfinite(v) and v > 0
            for v in [args.dt, args.duration, args.max_workspace_mb, args.max_memory_mb]
        )
    ):
        raise ValueError("Require positive finite time and resource options")
    steps = round(args.duration / args.dt)
    if steps < 1 or abs(steps * args.dt - args.duration) > 1e-12 * max(args.duration, args.dt):
        raise ValueError("Duration must be covered by an integer number of time steps")
    begun, cpu_begun = perf_counter(), process_time()
    if args.source.suffix.lower() == ".toml":
        case = args.source.resolve()
        source = dict(
            source_mesh_arguments=dict(
                case=str(case),
                h=10,
                contact_h=2,
                contact_end_h=0.015625,
                corner_h=0.015625,
                quadrature=12,
                restart_knot_window=None,
                restart_knot_subdivisions=1,
                dt=args.dt,
                steps=steps,
                duration=args.duration,
            ),
            environment=environment_record(case),
        )
    else:
        source = json.loads(args.source.read_text())
    options, overrides = NATIVE["discretization_arguments"](source["source_mesh_arguments"], args)
    environment = environment_record(options["case"])
    keys = ["source_sha256", "case_sha256", "restart_data_sha256"]
    if any(source["environment"][key] != environment[key] for key in keys):
        raise ValueError("Source metadata differs from the current numerical package or case")
    if previous is not None and any(
        previous["environment"][key] != environment[key] for key in [*keys, "numpy", "scipy"]
    ):
        raise ValueError("Resume changes the numerical package, case, restart data or runtime")
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
    plan = plan_cq(d, args.dt, steps, max_memory_mb=1e9, disk_spectrum=True)
    if previous is not None and (
        previous["plan"] != asdict(plan) or previous["source_mesh_arguments"] != options
    ):
        raise ValueError("Resume changes the CQ plan or mesh configuration")
    sampling = RAW["RAW"]["RawSamplingOptions"](
        degree=args.degree, raw_rtol=args.raw_rtol, output_rtol=args.output_rtol
    )
    arrays = construction_arrays(
        model,
        plan,
        sampling,
        args.workers,
        options["quadrature"],
        args.fft_rows,
        args.combine_matrices,
        args.startup_rows,
        args.spectrum_buffer_matrices,
    )
    conversion = 0
    if args.weight_layout == "fortran":
        rows = min(
            d,
            32,
            int((args.max_workspace_mb * 1024**2 - 64 * 1024**2) // (2 * (steps + 1) * d * 8)),
        )
        if rows < 1:
            raise ValueError("Storage conversion budget cannot hold one time-weight row")
        conversion = 2 * (steps + 1) * rows * d * 8 + 64 * 1024**2
        arrays["peak"] = max(arrays["peak"], conversion + 128 * 1024**2)
    if (
        arrays["peak"] > args.max_memory_mb * 1024**2
        or max(arrays[k] for k in ["interpolation", "inversion", "combination"])
        > args.max_workspace_mb * 1024**2
    ):
        raise ValueError(
            f"Fresh Radau array estimate exceeds budgets: {arrays['peak'] / 1024**2:.1f} MiB"
        )
    names = [
        "radau_raw_sampling.py",
        "raw_layer_sampling.py",
        "seeded_raw_sampling.py",
        "trace_schur_experiment.py",
        "schur_condensation_experiment.py",
        "radau_experiment.py",
        "compact_radau_storage.py",
        "prepare_native_resolvent.py",
        "replay_resolvent.py",
        "build_resolvent_history.py",
        "fortran_weight_storage.py",
        "bounded_weight_conversion.py",
        "buffered_spectrum.py",
        "radau_sampling_checkpoint.py",
    ]
    frozen = {
        str(p.resolve()): fingerprint(p)
        for p in [Path(__file__), args.source, *[TOOLS / n for n in names]]
    }

    def check_source():
        current = environment_record(options["case"])
        if any(fingerprint(p) != v for p, v in frozen.items()) or any(
            current[k] != environment[k] for k in keys
        ):
            raise ValueError("Fresh Radau numerical source or input changed")

    weight_bytes = (steps + 1) * d * d * 8
    required = 2 * spectrum_bytes(plan) + 2 * weight_bytes + (steps + 1) * 8 * d * 8 + 1024**2
    if args.durable_build:
        required += 2 * spectrum_bytes(plan)
    if args.weight_layout == "fortran":
        required += weight_bytes + 8192
    output.parent.mkdir(parents=True, exist_ok=True)
    allocated = 0
    if args.resume:
        for name in [
            *CHECKPOINT["CHANNELS"],
            "weights0.npy",
            "weights1.npy",
            "weights0-fortran.npy",
            "weights1-fortran.npy",
        ]:
            path = output / name
            if path.is_file():
                allocated += path.stat().st_blocks * 512
    additional = max(0, required - allocated)
    if additional > shutil.disk_usage(output.parent).free:
        raise ValueError("Insufficient disk for both complete spectra, time weights and startup")
    if not args.resume:
        output.mkdir()
        publish(
            output / "preflight.json",
            dict(
                status="preflight_only_not_a_complete_kernel",
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                source_metadata=str(args.source.resolve()),
                source_role="Mesh/case defaults only; no frequency values reused",
                source_mesh_arguments=options,
                plan=asdict(plan),
                history_dofs=d,
                stage_dofs=2 * d,
                estimated_array_mib={k: v / 1024**2 for k, v in arrays.items()},
                required_disk_bytes=required,
                additional_disk_bytes=additional,
                storage_conversion_array_mib=conversion / 1024**2,
                branches_sampled_sequentially=True,
                frozen_input_sha256=frozen,
                environment=environment,
                original_tatva_readonly=True,
            ),
        )
    if args.preflight_only:
        print(output / "preflight.json", flush=True)
        return
    if args.verify_resume_only:
        verified = CHECKPOINT["SamplingJournal"](
            output, plan, resume=True, receipt=args.resume_receipt
        )
        print(
            json.dumps(
                dict(
                    status="verified_checkpoint_not_running",
                    plan=asdict(plan),
                    committed_matrices={
                        name: verified.prefix(name) for name in CHECKPOINT["CHANNELS"]
                    },
                    required_disk_bytes=required,
                    additional_disk_bytes=additional,
                    estimated_construction_array_peak_mib=arrays["peak"] / 1024**2,
                    source_and_runtime_verified=True,
                    build_modified=False,
                    full_original_event_reproduced=False,
                    same_accuracy_fem_advantage_verified=False,
                ),
                indent=2,
            ),
            flush=True,
        )
        return
    journal = None
    if args.durable_build:
        ACTIVE_LEASE = CHECKPOINT["exclusive_lease"](output)
        journal = CHECKPOINT["SamplingJournal"](
            output, plan, resume=args.resume, receipt=args.resume_receipt
        )
        ACTIVE_JOURNAL = journal
        journal.begin()
        spectra = [journal.spectrum(f"work-family{i}.npy") for i in range(2)]
    else:
        spectra = [DiskSpectrum(output / f"work-family{i}.npy", plan) for i in range(2)]
    frozen[str((output / "preflight.json").resolve())] = fingerprint(output / "preflight.json")
    if args.resume_receipt is not None:
        frozen[str(args.resume_receipt.resolve())] = fingerprint(args.resume_receipt)
    if args.resume:
        journal.data.setdefault("source_migrations", []).append(
            dict(
                previous_producer_sha256=previous["frozen_input_sha256"][
                    str(Path(__file__).resolve())
                ],
                current_producer_sha256=fingerprint(Path(__file__)),
                numerical_sources_otherwise_unchanged=True,
            )
        )
        journal.publish()
    records, branch_digests = [], []
    sampling_start = perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        transfer = TRANSFER(model, quadrature=options["quadrature"])
        con, ref, law = model.constraints, model.reference, model.reference.contact
        line = np.r_[con.contact_weights, con.contact_weights, con.platen_width]
        spring = ref.spring_stiffness * con.platen_width
        tangent = np.diag(
            np.r_[
                con.contact_weights * law.normal_penalty,
                con.contact_weights * law.tangential_penalty,
                spring,
            ]
        )

        def provider(s, operator=transfer):
            return RAW["RAW"]["RawLayerPacket"].from_operators(
                p.assemble(s) for p in operator.prepared
            )

        def transform(packet, s, operator=transfer):
            return WORK(operator.evaluate_operators(packet.operators()), tangent, line)

        writer = None
        sampled_branch = False
        for branch in range(2):
            check_source()
            name = f"work-family{branch}.npy"
            prefix = 0 if journal is None else journal.prefix(name)
            if prefix == plan.fft_size // 2 + 1:
                record = dict(
                    all_fft_grid_points_emitted=True,
                    transformed_grid_points=prefix,
                    mode_branch=branch,
                    time_method="radau2",
                    source_evaluations=0,
                    source_frequency_values_reused=True,
                    same_configuration_prefix_points_reused=prefix,
                    preserved_sampling_record=journal.data["branch_records"].get(str(branch)),
                    scope="Durably restored raw branch; original independent operator audits still follow",
                )
                records.append(record)
                branch_digests.append(
                    dict(
                        branch=branch,
                        pre_combination_sha256=spectra[branch].record()["sha256"],
                        retained_separately_after_combination=True,
                    )
                )
                continue
            writer = (
                BUFFER(spectra[branch], matrices=args.spectrum_buffer_matrices)
                if journal is None
                else CHECKPOINT["JournaledSpectrumWriter"](
                    journal, spectra[branch], matrices=args.spectrum_buffer_matrices
                )
            )
            sampled_branch = True
            sampler, proxy = None, None
            intervals = [(0, plan.fft_size // 2 + 1)]
            if args.primitive_partition:
                intervals, proxy = RAW["primitive_branch_partition"](
                    plan, branch, ref.material, options=sampling
                )
                sampler = RAW["SEEDED"]["sampler_with_partition"](
                    RAW["RAW"]["sample_raw_layers"], intervals, plan.fft_size // 2 + 1
                )
            start = 0
            if journal is not None:
                stack, start = journal.sampler_stack(name, intervals)
                sampler = CHECKPOINT["checkpointed_sampler"](
                    RAW["RAW"]["sample_raw_layers"],
                    stack,
                    plan.fft_size // 2 + 1,
                    start,
                    lambda left, right, remaining, target=writer, key=name: (
                        journal.save_sampler_state(key, left, right, remaining, target)
                    ),
                )
            last = [0.0]

            def diagnostics(record, target=writer):
                now = perf_counter()
                if now - last[0] >= 30 or record["all_fft_grid_points_emitted"]:
                    check_source()
                    publish(
                        output / "progress.json",
                        dict(
                            phase="raw_radau_branch",
                            branch=branch,
                            sampling=record,
                            frequency_matrices_committed=target.committed,
                            pending_buffer_matrices=target.pending,
                            elapsed_seconds=now - begun,
                        ),
                    )
                    print(
                        f"branch={branch}, raw={record['source_evaluations']}, grid={record['transformed_grid_points']}/{plan.fft_size // 2 + 1}",
                        flush=True,
                    )
                    last[0] = now
                    if journal is not None:
                        journal.publish()

            record = RAW["sample_radau_branch"](
                provider,
                transform,
                plan,
                branch,
                lambda i, s, value, target=writer: (
                    target.append(i, value) if journal is None else target.accept(i, value)
                ),
                options=sampling,
                workers=args.workers,
                max_workspace_mb=args.max_workspace_mb,
                diagnostics=diagnostics,
                sampler=sampler,
            )
            writer.finish()
            record["primitive_partition_screen"] = proxy
            if journal is not None:
                record.update(
                    same_configuration_prefix_points_reused=prefix,
                    recomputed_prefix_points_verified_bitwise=prefix - start,
                    source_frequency_values_reused=bool(prefix),
                    current_invocation_sampling_only=True,
                )
                journal.data["branch_records"][str(branch)] = record
                journal.publish()
            records.append(record)
            branch_digests.append(
                dict(
                    branch=branch,
                    pre_combination_sha256=spectra[branch].record()["sha256"],
                    retained_separately_after_combination=args.durable_build,
                )
            )
        sampling_seconds = perf_counter() - sampling_start
        if journal is not None and journal.data["phase"] == "raw":
            journal.mark("raw_complete")
        audit_start = perf_counter()
        indices = sorted(
            set([0, 1, 7, plan.fft_size // 16, plan.fft_size // 4, plan.fft_size // 2])
        )
        indices = [i for i in indices if i <= plan.fft_size // 2]
        direct = NodalInterfaceTransfer(model, quadrature=options["quadrature"], prepared=False)
        direct.prepared = transfer.prepared
        checks, fresh_stages = [], []
        for index in indices:
            zeta = plan.radius * np.exp(-2j * np.pi * index / plan.fft_size)
            frequencies = RAW["lower_half_radau_modes"](zeta, plan.dt)
            fresh = [WORK(direct.evaluate(s), tangent, line) for s in frequencies]
            for branch in range(2):
                value = spectra[branch][index]
                relative = float(
                    np.linalg.norm(value - fresh[branch]) / np.linalg.norm(fresh[branch])
                )
                composition = float(
                    np.linalg.norm(solve(fresh[branch], value) - np.eye(d), ord=np.inf)
                )
                if (
                    not np.isfinite(relative)
                    or not np.isfinite(composition)
                    or max(relative, composition) > 1e-6
                ):
                    raise ValueError(
                        "Fresh Radau branch failed independent original compliance audit"
                    )
                checks.append(
                    dict(
                        index=index,
                        branch=branch,
                        relative_frobenius=relative,
                        solve_composition_inf_error=composition,
                    )
                )

            def modal_value(s):
                j = int(np.argmin(abs(frequencies - s)))
                if abs(frequencies[j] - s) > 1e-9 * max(1, abs(s)):
                    raise ValueError(
                        "Independent stage eigensystem differs from the branch frequencies"
                    )
                return fresh[j]

            fresh_stages.append(
                RADAU["matrix_function"](modal_value, RADAU["radau_symbol"](zeta) / plan.dt)
            )
        impedance = RADAU["matrix_function"](transfer.evaluate, RADAU["radau_symbol"](0) / plan.dt)
        if np.linalg.norm(impedance.imag) > 1e-10 * max(np.linalg.norm(impedance.real), 1):
            raise ValueError("Instantaneous Radau impedance must be real")
        del direct, transfer, provider, transform, writer
        if sampled_branch:
            del diagnostics, sampler
        audit_seconds = perf_counter() - audit_start
    combination_start = perf_counter()
    raw_spectra = spectra
    compact_writers = None
    first_combination = 0
    if journal is not None:
        spectra = [journal.spectrum(f"compact-family{i}.npy") for i in range(2)]
        compact_writers = [
            CHECKPOINT["JournaledSpectrumWriter"](
                journal, s, matrices=args.spectrum_buffer_matrices
            )
            for s in spectra
        ]
        first_combination = min(w.committed for w in compact_writers)
        if first_combination < plan.fft_size // 2 + 1:
            journal.mark("combination")
    for first in range(first_combination, plan.fft_size // 2 + 1, args.combine_matrices):
        last = min(first + args.combine_matrices, plan.fft_size // 2 + 1)
        zeta = plan.radius * np.exp(-2j * np.pi * np.arange(first, last) / plan.fft_size)
        families = RAW["compact_coefficients"](
            zeta, raw_spectra[0][first:last], raw_spectra[1][first:last]
        )
        for family, values in enumerate(families):
            for index in [0, plan.fft_size // 2]:
                if first <= index < last:
                    endpoint = values[index - first]
                    if np.linalg.norm(endpoint.imag) > 1e-10 * max(
                        np.linalg.norm(endpoint.real), 1
                    ):
                        raise ValueError("Combined Radau family violates real endpoint symmetry")
                    values[index - first] = endpoint.real
            if compact_writers is None:
                spectra[family][first:last] = values
            else:
                target = compact_writers[family]
                for index in range(max(first, target.emitted), last):
                    target.append(index, values[index - first])
    if compact_writers is not None:
        for target in compact_writers:
            target.finish()
        del compact_writers, target
    stage_errors = []
    for index, fresh in zip(indices, fresh_stages, strict=True):
        zeta = plan.radius * np.exp(-2j * np.pi * index / plan.fft_size)
        actual = np.kron(np.eye(2), spectra[0][index]) + np.kron(
            RADAU["radau_symbol"](zeta), spectra[1][index]
        )
        error = float(np.linalg.norm(actual - fresh) / np.linalg.norm(fresh))
        if not np.isfinite(error) or error > 1e-6:
            raise ValueError("Combined families failed the independent complete stage audit")
        stage_errors.append(dict(index=index, relative_frobenius=error))
    combination_seconds = perf_counter() - combination_start
    del fresh_stages
    if journal is not None:
        derived = [
            output / f"weights{i}{suffix}.npy" for i in range(2) for suffix in ["", "-fortran"]
        ]
        if any(p.exists() for p in derived):
            if journal.data["phase"] not in {"ifft_started", "postprocess_started"}:
                raise ValueError(
                    "Unexpected derived weight files before inversion; preserve for inspection"
                )
            CHECKPOINT["discard_derived_weights"](output, journal)
        journal.mark("ifft_started")
    ifft_start = perf_counter()
    weight_paths, wrap, inversion = STORAGE["invert_families_to_files"](
        *spectra, plan, output, fft_rows=args.fft_rows, max_workspace_mb=args.max_workspace_mb
    )
    ifft_seconds = perf_counter() - ifft_start
    startup_start = perf_counter()
    if journal is not None:
        journal.mark("postprocess_started")
    conversion_records = []
    for path in weight_paths:
        weights = NATIVE["physical_weights_file_inplace"](path, line)
        weights._mmap.close()
        if args.weight_layout == "fortran":
            converted = path.with_name(path.stem + "-fortran.npy")
            convert = runpy.run_path(str(TOOLS / "fortran_weight_storage.py"))[
                "convert_weights_fortran"
            ]
            conversion_records.append(
                convert(path, converted, max_workspace_mb=args.max_workspace_mb)
            )
            converted.replace(path)
    wrap /= np.sqrt(line[:, None] * line[None, :])
    maps = [np.load(p, mmap_mode="r", allow_pickle=False) for p in weight_paths]
    instantaneous = RADAU["expand_compact_weights"](maps[0][:1], maps[1][:1], wrap)[0]
    for values in maps:
        values._mmap.close()
    composition = float(
        np.linalg.norm(
            instantaneous @ (impedance.real + np.kron(np.eye(2), tangent)) - np.eye(2 * d),
            ord=np.inf,
        )
    )
    if not np.isfinite(composition) or composition > 1e-6:
        raise ValueError(f"Fresh Radau lag-zero inverse composition failed: {composition}")
    np.save(output / "initial_impedance.npy", impedance.real, allow_pickle=False)
    np.save(output / "shift_wrap.npy", wrap, allow_pickle=False)
    stage_time = np.zeros((steps + 1, 2))
    stage_time[1:] = (np.arange(steps)[:, None] + RADAU["C"]) * plan.dt
    increment = options.get("diagnostic_actuator_increment")
    rate = ref.actuator_rate if increment is None else increment / args.duration
    elastic = np.empty((steps + 1, 2, d))
    for first in range(0, d, args.startup_rows):
        last = min(first + args.startup_rows, d)
        columns = [s[:, first:last, -1] / np.sqrt(line[first:last] * line[-1]) for s in spectra]
        block_plan = replace(plan, dofs=last - first)
        elastic[:, :, first:last] = STORAGE["causal_compact_startup"](
            *columns, wrap[first:last, -1], block_plan, spring * rate * stage_time
        )
    np.savez(
        output / "elastic.npz",
        time=np.arange(steps + 1) * plan.dt,
        stage_time=stage_time,
        elastic=elastic,
        uninterrupted_actuator=ref.restart_actuator + rate * stage_time,
    )
    startup_seconds = perf_counter() - startup_start
    check_source()

    def files(paths):
        return [
            dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=REPLAY["digest_file"](p))
            for p in paths
        ]

    spectrum_records = [s.record() for s in spectra]
    publish(
        output / "summary.json",
        dict(
            status="completed_compact_radau_resolvent_kernel_only",
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            time_method="radau2",
            assembly="fresh_raw_radau_before_trace_schur",
            source_metadata=str(args.source.resolve()),
            source_role="Mesh/case defaults only; same-build raw/compact checkpoints may be restored; no foreign spectrum or weights read",
            source_mesh_arguments=options,
            numerical_discretization_overrides=overrides,
            dt_s=plan.dt,
            full_steps=steps,
            correction_steps=steps,
            plan=asdict(plan),
            history_dofs=d,
            stage_dofs=2 * d,
            workers_requested=args.workers,
            branches_sampled_sequentially=True,
            family_spectra=spectrum_records,
            family_weights=files(weight_paths),
            family_spectra_coordinates="diag(sqrt(w)) * L * diag(sqrt(w))",
            pre_combination_branch_digests=branch_digests,
            frequency_sampling=records,
            sampling_options=asdict(sampling),
            sampled_fresh_bem_mode_checks=checks,
            sampled_fresh_stage_checks=stage_errors,
            shift_wrap_sha256=fingerprint(output / "shift_wrap.npy"),
            initial_impedance_sha256=fingerprint(output / "initial_impedance.npy"),
            elastic_sha256=REPLAY["digest_file"](output / "elastic.npz"),
            initial_composition_inf_error=composition,
            time_weights_layout=args.weight_layout,
            storage_conversion=conversion_records,
            inversion=inversion,
            estimated_construction_array_peak_mib=arrays["peak"] / 1024**2,
            frozen_inputs_sha256=frozen,
            environment=environment,
            timing_seconds=dict(
                raw_sampling_and_branch_hashes=sampling_seconds,
                independent_operator_audit=audit_seconds,
                compact_combination=combination_seconds,
                inverse_fft=ifft_seconds,
                physical_weights_and_startup=startup_seconds,
                total_before_reporting=perf_counter() - begun,
            ),
            process_cpu_seconds=process_time() - cpu_begun,
            peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2 if sys.platform == "darwin" else 1024),
            full_original_event_reproduced=False,
            original_tatva_readonly=True,
            same_accuracy_fem_advantage_verified=False,
            durable_construction=None
            if journal is None
            else dict(
                journal_path=str(journal.path),
                attempts=journal.data["attempts"],
                timing_scope="Stage timings cover this invocation only; all previous attempt records are retained. Missing end records are not exact total times.",
                raw_branch_spectra_retained=True,
                committed_matrix_chunks_verified_on_resume=True,
            ),
            scope="Fresh complete compact RK kernel and causal startup; sampled raw/operator guards, not physical convergence or an all-frequency theorem",
        ),
    )
    publish(
        output / "progress.json",
        dict(phase="completed_kernel_only", elapsed_seconds=perf_counter() - begun),
    )
    print(output / "summary.json", flush=True)
    if journal is not None:
        journal.mark("completed")


def main():
    global ACTIVE_JOURNAL, ACTIVE_LEASE
    ACTIVE_JOURNAL = ACTIVE_LEASE = None
    prior = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Kernel construction received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        _main()
    except BaseException:
        if ACTIVE_JOURNAL is not None:
            ACTIVE_JOURNAL.end("interrupted_or_failed")
        raise
    else:
        if ACTIVE_JOURNAL is not None:
            ACTIVE_JOURNAL.end("completed")
            report = ACTIVE_JOURNAL.folder / "summary.json"
            record = json.loads(report.read_text())
            record["durable_construction"]["attempts"] = ACTIVE_JOURNAL.data["attempts"]
            CHECKPOINT["atomic_json"](report, record)
    finally:
        signal.signal(signal.SIGTERM, prior)
        if ACTIVE_LEASE is not None:
            ACTIVE_LEASE.close()


if __name__ == "__main__":
    main()

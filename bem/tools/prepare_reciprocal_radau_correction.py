"""Sample, audit and invert a short reciprocal Radau correction kernel."""

import argparse
import json
import runpy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import plan_cq
from finiteblock_bem.io import environment_record, fingerprint
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOLS = Path(__file__).resolve().parent
ORIGINAL = runpy.run_path(str(TOOLS / "prepare_raw_radau_resolvent.py"))
JOURNAL = runpy.run_path(str(TOOLS / "reciprocal_radau_checkpoint.py"))
PLAN = runpy.run_path(str(TOOLS / "plan_split_radau_history.py"))
CK, STORAGE = JOURNAL["CHECKPOINT"], JOURNAL["STORAGE"]
RAW, RK, WORK = (ORIGINAL[k] for k in ("RAW", "RADAU", "WORK"))
NAMES = ("branch0.bin", "branch1.bin")
DERIVED = ("weights0.npy", "weights1.npy", "shift_wrap.npy", "initial_impedance.npy")
WEIGHTS_OWNER = "reciprocal-owner.json"


def parse():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("kernel", type=Path, help="Completed kernel specifying the case and trace")
    parser.add_argument("--spectra-output", type=Path, required=True)
    parser.add_argument("--weights-output", type=Path, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--lags", type=int, required=True)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--buffer-matrices", type=int, default=16)
    parser.add_argument("--max-memory-mb", type=float, default=20480)
    parser.add_argument("--max-workspace-mb", type=float, default=12288)
    parser.add_argument("--spectra-reserve-gib", type=float, default=0)
    parser.add_argument("--weights-reserve-gib", type=float, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def check_paths(args):
    if args.spectra_output.is_symlink() or args.weights_output.is_symlink():
        raise ValueError("Outputs must not be symlinks")
    spectra, weights = args.spectra_output.resolve(), args.weights_output.resolve()
    protected = Path("/Volumes/Gauss-T7/tatva").resolve()
    source = args.kernel.resolve()
    if (
        spectra.is_relative_to(weights)
        or weights.is_relative_to(spectra)
        or any(p.is_relative_to(protected) or p.is_relative_to(source) for p in (spectra, weights))
    ):
        raise ValueError("Require distinct owned directories outside Tatva and the source kernel")
    if args.resume and args.preflight_only:
        raise ValueError("Use fresh preflight or resume")
    if not spectra.parent.is_dir() or not weights.parent.is_dir():
        raise ValueError("Choose existing parent directories")
    if not args.resume and (spectra.exists() or weights.exists()):
        raise ValueError("Choose unused output directories")
    return spectra, weights


def physical_configuration(args):
    summary_path = args.kernel.resolve() / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "completed_compact_radau_resolvent_kernel_only" or (
        summary.get("time_method") != "radau2"
    ):
        raise ValueError("Require completed original Radau kernel metadata")
    options = summary["source_mesh_arguments"]
    case = options["case"]
    environment = environment_record(case)
    stable = ("source_sha256", "case_sha256", "restart_data_sha256")
    if any(environment[k] != summary["environment"][k] for k in stable):
        raise ValueError("Prepared source and current physical package differ")
    model = load_nodal_model(
        case,
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    con, ref = model.constraints, model.reference
    line = np.r_[con.contact_weights, con.contact_weights, con.platen_width]
    tangent = np.diag(
        line
        * np.r_[
            np.full(len(con.contact_weights), ref.contact.normal_penalty),
            np.full(len(con.contact_weights), ref.contact.tangential_penalty),
            ref.spring_stiffness,
        ]
    )
    plan = plan_cq(len(line), args.dt, args.lags, max_memory_mb=1e9, disk_spectrum=True)
    if plan.dofs != summary["plan"]["dofs"]:
        raise ValueError("Declared trace size changed")
    frozen = [
        summary_path,
        Path(case),
        Path(__file__),
        *[
            TOOLS / name
            for name in (
                "prepare_raw_radau_resolvent.py",
                "radau_sampling_checkpoint.py",
                "reciprocal_radau_checkpoint.py",
                "reciprocal_radau_storage.py",
                "plan_split_radau_history.py",
                "raw_layer_sampling.py",
                "radau_raw_sampling.py",
                "seeded_raw_sampling.py",
                "build_resolvent_history.py",
                "trace_schur_experiment.py",
                "schur_condensation_experiment.py",
                "radau_experiment.py",
                "compact_radau_storage.py",
            )
        ],
    ]
    hashes = {str(p.resolve()): fingerprint(p) for p in frozen}
    return model, options, line, tangent, plan, environment, hashes


def memory_estimate(model, plan, options, sampling, args):
    arrays = ORIGINAL["construction_arrays"](
        model,
        plan,
        sampling,
        args.workers,
        options["quadrature"],
        1,
        1,
        spectrum_buffer_matrices=args.buffer_matrices,
    )
    arrays["reciprocity_check_extra"] = 3 * arrays["spectrum_buffer"]
    arrays["peak"] += arrays["reciprocity_check_extra"]
    inversion = STORAGE["inversion_workspace_bytes"](plan, args.block)
    arrays["reciprocal_inversion_workspace"] = inversion
    if (
        max(arrays["peak"], inversion) > args.max_memory_mb * 1024**2
        or max(arrays["interpolation"], inversion) > args.max_workspace_mb * 1024**2
    ):
        raise ValueError("Reciprocal correction exceeds the declared array budget")
    return arrays


def prepare_branch(journal, transfer, model, tangent, line, plan, branch, sampling, args):
    name = NAMES[branch]
    count, prefix = journal.count, journal.prefix(name)
    if prefix == count:
        # A crash after the final commit can precede publication of the branch.
        if not (journal.folder / name).is_file():
            with journal.open_branch(name):
                pass
        return

    def provider(s):
        return RAW["RAW"]["RawLayerPacket"].from_operators(p.assemble(s) for p in transfer.prepared)

    def transform(packet, _):
        return WORK(transfer.evaluate_operators(packet.operators()), tangent, line)

    with journal.open_branch(name) as packed:
        intervals, proxy = RAW["primitive_branch_partition"](
            plan, branch, model.reference.material, options=sampling
        )
        stack, start = journal.sampler_stack(name, intervals)
        writer = JOURNAL["JournaledReciprocalWriter"](
            journal, packed, start, matrices=args.buffer_matrices
        )
        sampler = CK["checkpointed_sampler"](
            RAW["RAW"]["sample_raw_layers"],
            stack,
            count,
            start,
            lambda left, right, remaining: journal.save_sampler_state(
                name, left, right, remaining, writer
            ),
        )
        last = 0.0

        def progress(record):
            nonlocal last
            now = perf_counter()
            if now - last >= 30 or record["all_fft_grid_points_emitted"]:
                CK["atomic_json"](
                    journal.folder / "progress.json",
                    dict(
                        branch=branch,
                        committed=writer.committed,
                        pending=writer.pending,
                        sampling=record,
                    ),
                )
                last = now

        record = RAW["sample_radau_branch"](
            provider,
            transform,
            plan,
            branch,
            lambda i, _, value: writer.accept(i, value),
            options=sampling,
            workers=args.workers,
            max_workspace_mb=args.max_workspace_mb,
            diagnostics=progress,
            sampler=sampler,
        )
        writer.finish()
        journal.data["branch_records"][str(branch)] = dict(
            sampling=record,
            restored_prefix=prefix,
            replayed_prefix=prefix - start,
            primitive_partition=proxy,
            invocation_maximum_reciprocity_skew=packed.maximum_relative_skew,
        )
        journal.publish()


def operator_audit(transfer, model, options, plan, tangent, line, paths, block):
    direct = NodalInterfaceTransfer(model, quadrature=options["quadrature"], prepared=False)
    direct.prepared = transfer.prepared
    indices = sorted(
        {
            0,
            1,
            min(7, plan.fft_size // 2),
            plan.fft_size // 16,
            plan.fft_size // 4,
            plan.fft_size // 2,
        }
    )
    result = []
    for index in indices:
        zeta = plan.radius * np.exp(-2j * np.pi * index / plan.fft_size)
        for branch, s in enumerate(RAW["lower_half_radau_modes"](zeta, plan.dt)):
            fresh = WORK(direct.evaluate(s), tangent, line)
            actual = STORAGE["read_frequency_matrix"](paths[branch], plan, index, block=block)
            error = float(np.linalg.norm(actual - fresh) / np.linalg.norm(fresh))
            composition = float(
                np.linalg.norm(solve(fresh, actual) - np.eye(plan.dofs), ord=np.inf)
            )
            if (
                not np.isfinite(error)
                or not np.isfinite(composition)
                or max(error, composition) > 1e-6
            ):
                raise ValueError("Reciprocal branch failed the original fresh compliance gate")
            result.append(
                dict(
                    index=index,
                    branch=branch,
                    relative_frobenius=error,
                    solve_composition_inf_error=composition,
                )
            )
    impedance = RK["matrix_function"](transfer.evaluate, RK["radau_symbol"](0) / plan.dt)
    if np.linalg.norm(impedance.imag) > 1e-10 * max(np.linalg.norm(impedance.real), 1):
        raise ValueError("Instantaneous Radau impedance must be real")
    return result, impedance.real


def inversion_owner(journal):
    return dict(spectrum_build=str(journal.folder), preflight_sha256=journal.preflight)


def discard_incomplete_inversion(journal, weights):
    """Only remove derived files from this recorded, incomplete inversion."""
    if journal.data["phase"] != "inverting" or not weights.is_dir() or weights.is_symlink():
        raise ValueError("Weights path is not an owned incomplete inversion")
    owner = weights / WEIGHTS_OWNER
    if (
        owner.is_symlink()
        or not owner.is_file()
        or json.loads(owner.read_text()) != inversion_owner(journal)
    ):
        raise ValueError("Incomplete inversion belongs to a different build")
    files = list(weights.iterdir())
    if any(
        p.name not in (*DERIVED, WEIGHTS_OWNER)
        or not p.is_file()
        or p.is_symlink()
        or p.stat().st_nlink != 1
        for p in files
    ):
        raise ValueError("Unrecognized file in the incomplete inversion")
    for path in files:
        path.unlink()
    weights.rmdir()


def run(args):
    started = perf_counter()
    spectra, weights = check_paths(args)
    if (
        args.workers < 1
        or args.buffer_matrices < 1
        or args.block < 1
        or not all(np.isfinite(v) and v > 0 for v in (args.max_memory_mb, args.max_workspace_mb))
    ):
        raise ValueError("Require positive worker, block and workspace settings")
    model, options, line, tangent, plan, environment, frozen = physical_configuration(args)
    sampling = RAW["RAW"]["RawSamplingOptions"]()
    arrays = memory_estimate(model, plan, options, sampling, args)
    branch_bytes = STORAGE["reciprocal_size"](plan, args.block)
    data = dict(
        reciprocal_raw_branches=2 * branch_bytes,
        physical_weights=2 * plan.weight_bytes + plan.dofs**2 * 8 + 1024**2,
    )
    volumes = PLAN["inspect_volumes"](
        spectra.parent, weights.parent, data, args.spectra_reserve_gib, args.weights_reserve_gib
    )
    config = dict(
        plan=asdict(plan),
        source_mesh_arguments=options,
        environment=environment,
        source_sha256=frozen,
        spectra_output=str(spectra),
        weights_output=str(weights),
        reciprocal_block=args.block,
        sampling_options=asdict(sampling),
        workers=args.workers,
        buffer_matrices=args.buffer_matrices,
        array_estimate_bytes=arrays,
    )
    if args.preflight_only:
        print(json.dumps(dict(**config, volumes=volumes, scope="Preflight only"), indent=2))
        return 0 if all(v["margin_bytes"] >= 0 for v in volumes) else 1
    if not args.resume:
        if any(v["margin_bytes"] < 0 for v in volumes):
            raise ValueError("Insufficient storage for both reciprocal branches and weights")
        spectra.mkdir()
    preflight = spectra / "preflight.json"
    with CK["exclusive_lease"](spectra):
        if args.resume:
            if json.loads(preflight.read_text()) != config or (spectra / "summary.json").exists():
                raise ValueError("Resume changes the owned build or source is already completed")
        else:
            CK["atomic_json"](preflight, config)
        journal = JOURNAL["ReciprocalSamplingJournal"](
            spectra, plan, args.block, resume=args.resume
        )
        if args.resume:
            if weights.exists():
                if any(journal.prefix(name) != journal.count for name in NAMES):
                    raise ValueError(
                        "Complete verified sources are required to reset the inversion"
                    )
                discard_incomplete_inversion(journal, weights)
            # Subtract only blocks already allocated by this verified source.
            allocated = 0
            for name in NAMES:
                final = spectra / name
                source = final if final.is_file() else final.with_name(name + ".part")
                if source.is_file():
                    allocated += min(branch_bytes, source.stat().st_blocks * 512)
            data["reciprocal_raw_branches"] -= allocated
            volumes = PLAN["inspect_volumes"](
                spectra.parent,
                weights.parent,
                data,
                args.spectra_reserve_gib,
                args.weights_reserve_gib,
            )
            if any(v["margin_bytes"] < 0 for v in volumes):
                raise ValueError("Insufficient space for the remaining reciprocal build")
        if weights.exists():
            raise ValueError("Choose a fresh derived weights path before inversion")
        journal.begin()
        try:
            with threadpool_limits(limits=1, user_api="blas"):
                transfer = ORIGINAL["TRANSFER"](model, quadrature=options["quadrature"])
                for branch in range(2):
                    prepare_branch(
                        journal, transfer, model, tangent, line, plan, branch, sampling, args
                    )
                paths = [spectra / name for name in NAMES]
                audit, impedance = operator_audit(
                    transfer, model, options, plan, tangent, line, paths, args.block
                )
            del transfer
            journal.mark("inverting")
            wrap = STORAGE["invert_reciprocal_branches"](
                paths,
                plan,
                line,
                weights,
                block=args.block,
                max_workspace_mb=args.max_workspace_mb,
                on_create=lambda folder: CK["atomic_json"](
                    folder / WEIGHTS_OWNER, inversion_owner(journal)
                ),
            )
            with threadpool_limits(limits=1, user_api="blas"):
                maps = [
                    np.load(weights / f"weights{i}.npy", mmap_mode="r", allow_pickle=False)
                    for i in range(2)
                ]
                try:
                    instant = RK["expand_compact_weights"](maps[0][:1], maps[1][:1], wrap)[0]
                finally:
                    for mapping in maps:
                        mapping._mmap.close()
                composition = float(
                    np.linalg.norm(
                        instant @ (impedance + np.kron(np.eye(2), tangent)) - np.eye(2 * plan.dofs),
                        ord=np.inf,
                    )
                )
                if not np.isfinite(composition) or composition > 1e-6:
                    raise ValueError("Short correction failed the original lag-zero gate")
            np.save(weights / "initial_impedance.npy", impedance, allow_pickle=False)
            if any(fingerprint(p) != sha for p, sha in frozen.items()) or (
                environment_record(options["case"]) != environment
            ):
                raise ValueError("Reciprocal inputs changed during construction")
            paths += [
                weights / name
                for name in (
                    "weights0.npy",
                    "weights1.npy",
                    "shift_wrap.npy",
                    "initial_impedance.npy",
                )
            ]
            payloads = [
                dict(path=str(p), bytes=p.stat().st_size, sha256=CK["digest_range"](p))
                for p in paths
            ]
            CK["atomic_json"](
                spectra / "summary.json",
                dict(
                    status="completed_reciprocal_correction_only",
                    configuration=config,
                    fresh_mode_audit=audit,
                    initial_composition_inf_error=composition,
                    payloads=payloads,
                    seconds=perf_counter() - started,
                    scope="Full correction lags only. Requires a separate full startup history.",
                ),
            )
            journal.end("completed")
        except BaseException:
            journal.end("interrupted_or_failed")
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse()))

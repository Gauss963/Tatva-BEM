#!/usr/bin/env python3
"""Run the archived FEM implementation locally, never writing to the source run.

Defaults to a short timing experiment on the original 2 mm mesh. Use
--shear-duration .045 --normal-duration .002 for the full original horizon.
This imports an exact local Git snapshot, not the evolved root FEM runner.
"""

import argparse
import hashlib
import io
import json
import math
import os
import runpy
import subprocess
import sys
import tarfile
import tomllib
import uuid
from pathlib import Path
from time import perf_counter, process_time

PROJECT = Path(__file__).resolve().parents[2]
REVISION = "933ae1f3fe1c634391bdbd316e4efa5dbf0e6d5d"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resume_manifest(output, arguments, runner_sha256):
    """Validate a checkpoint continuation before changing any saved run file."""
    output = Path(output)
    record = json.loads((output / "reference_manifest.json").read_text())
    if record["runner_sha256"] != runner_sha256:
        raise ValueError("Resume requires the same FEM runner source")
    for name, value in arguments.items():
        if name not in {"resume", "prepare_only"} and record["arguments"].get(name) != value:
            raise ValueError(f"Resume changes the recorded argument: {name}")
    if not (output / "checkpoint.npz").is_file() or not (output / "simulation.h5").is_file():
        raise ValueError("Resume requires both the original checkpoint and partial HDF5 output")
    if (output / "summary.json").exists():
        raise ValueError("A completed FEM run cannot be resumed")
    return record


def invocation_event(output, event):
    with (Path(output) / "invocations.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")


def invocation_totals(output):
    rows = [
        json.loads(line) for line in (Path(output) / "invocations.jsonl").read_text().splitlines()
    ]
    starts = [row["id"] for row in rows if row["event"] == "started"]
    ends = [row for row in rows if row["event"] == "ended"]
    complete = len(starts) == len(set(starts)) == len(ends) and set(starts) == {
        r["id"] for r in ends
    }
    return dict(
        all_invocations_accounted=complete,
        invocation_count=len(starts),
        segments=ends,
        **{
            key: sum(row[key] for row in ends) if complete else None
            for key in [
                "preparation_seconds",
                "simulation_seconds",
                "simulation_cpu_seconds",
                "total_seconds",
                "total_cpu_seconds",
            ]
        },
    )


def load_reference_input(reference, *, revision=REVISION):
    """Use authored/resolved solver input, not the stored history's dtype label."""
    reference = Path(reference)
    case_path, resolved_path = reference / "input/case.toml", reference / "input/resolved.json"
    resolved = json.loads(resolved_path.read_text())
    if resolved.get("input_sha256") != digest(case_path):
        raise ValueError("Resolved FEM input does not match the authored TOML fingerprint")
    if resolved.get("git_revision") != revision:
        raise ValueError("Resolved FEM input identifies a different source revision")
    if not isinstance(resolved.get("solver"), dict) or not isinstance(resolved.get("case"), dict):
        raise ValueError("Resolved FEM input must contain case and solver mappings")
    if resolved["solver"].get("dtype") not in {"float32", "float64"}:
        raise ValueError("Resolved FEM input must explicitly specify float32 or float64")
    authored_dtype = tomllib.loads(case_path.read_text()).get("solver", {}).get("dtype")
    if authored_dtype is not None and authored_dtype != resolved["solver"]["dtype"]:
        raise ValueError("Authored and resolved FEM solver precision differ")
    return resolved


def restart_seed_record(path):
    """Fingerprint the values actually used by the archived static initializer."""
    if path is None:
        return None
    import h5py
    import numpy as np

    path = Path(path).expanduser().resolve()
    before = path.stat()
    with h5py.File(path, "r") as stream:
        group = stream["final_state"]
        plastic = np.asarray(group["plastic_slip"], dtype=np.float64)
        actuator = np.float64(group.attrs["applied_shear_displacement"])
        u_shape = list(group["u"].shape)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("FEM restart input changed while reading its seed")
    checksum = hashlib.sha256(plastic.tobytes() + actuator.tobytes()).hexdigest()
    return dict(
        path=str(path),
        bytes=after.st_size,
        mtime_ns=after.st_mtime_ns,
        plastic_shape=list(plastic.shape),
        checked_u_shape=u_shape,
        actuator_mm=float(actuator),
        seed_sha256=checksum,
        scope="Static initializer uses plastic slip and actuator; u is checked for size only",
    )


def project_restart_seed(path, target_y, expected_record):
    """Interpolate the original final-state seed, preserving its float64 values."""
    import h5py
    import numpy as np

    if expected_record is None or restart_seed_record(path) != expected_record:
        raise ValueError("Original restart differs from its recorded source")
    with h5py.File(path, "r") as stream:
        plastic = stream["final_state/plastic_slip"][:]
        coordinates = stream["interface/contact_line_y"][:]
    y, target = np.asarray(coordinates, dtype=float), np.asarray(target_y, dtype=float)
    if plastic.dtype != np.float64:
        raise ValueError("Mesh refinement requires the original float64 final-state plastic seed")
    if (
        y.ndim != 1
        or len(y) < 2
        or plastic.shape != y.shape
        or target.ndim != 1
        or len(target) < 2
        or not all(np.all(np.isfinite(a)) for a in [y, plastic, target])
        or np.any(np.diff(y) <= 0)
        or np.any(np.diff(target) <= 0)
        or target[0] != y[0]
        or target[-1] != y[-1]
    ):
        raise ValueError(
            "Restart interpolation must preserve ordered finite nodes and the contact interval"
        )
    if restart_seed_record(path) != expected_record:
        raise ValueError("Original restart changed during projection")
    return (
        np.interp(target, y, plastic),
        float(expected_record["actuator_mm"]),
        dict(
            authority="Original final_state/plastic_slip and recorded actuator; P1 interpolation on original interface knots",
            source_plastic_dtype=str(plastic.dtype),
            source_coordinate_dtype=str(coordinates.dtype),
            source_coordinate_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
            target_coordinate_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
            source_nodes=len(y),
            target_nodes=len(target),
            source_restart=expected_record,
        ),
    )


def verify_snapshot(snapshot, expected):
    """A cached directory must still contain the exact archived Python source."""
    for name, record in expected.items():
        path = Path(snapshot) / name
        if path.is_symlink() or not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"FEM source snapshot differs from Git archive: {name}")
    unexpected = {
        str(path.relative_to(snapshot))
        for path in Path(snapshot).rglob("*.py")
        if str(path.relative_to(snapshot)) not in expected
    }
    if unexpected:
        raise ValueError(f"Unexpected Python modules in FEM snapshot: {sorted(unexpected)}")


def stage_snapshot(project, snapshot, *, revision=REVISION):
    archive = subprocess.run(
        ["git", "-C", str(project), "archive", revision, "tatva"],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        expected = {}
        for member in stream.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("FEM source archive must contain ordinary files only")
            expected[member.name] = {
                "sha256": hashlib.sha256(stream.extractfile(member).read()).hexdigest(),
                "bytes": member.size,
            }
        if not Path(snapshot).exists():
            Path(snapshot).mkdir(parents=True)
            stream.extractall(snapshot, filter="data")
    verify_snapshot(snapshot, expected)
    return dict(
        revision=revision,
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        files=expected,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-run",
        type=Path,
        default=Path("/Volumes/Gauss-T7/tatva/runs/local-mus08-muk06-gap0-v2-ev16"),
    )
    parser.add_argument("--output", type=Path, default=PROJECT / "bem/outputs/fem-4c-timing")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="Repeat the same arguments with a saved checkpoint"
    )
    parser.add_argument("--shear-duration", type=float, default=5e-5)
    parser.add_argument("--normal-duration", type=float, default=5e-8)
    parser.add_argument("--mesh-size", type=float, help="Optional coarse FEM comparison mesh")
    parser.add_argument(
        "--contact-end-size", type=float, help="Explicit chamfer-preserving endpoint grading"
    )
    parser.add_argument("--grading-growth", type=float, default=1.2)
    parser.add_argument(
        "--checkpoint-interval", type=float, help="Seconds between original-source checkpoints"
    )
    parser.add_argument("--normal-frames", type=int, default=4)
    parser.add_argument("--shear-frames", type=int, default=32)
    parser.add_argument("--interface-normal-frames", type=int)
    parser.add_argument("--interface-shear-frames", type=int)
    parser.add_argument("--time-step", type=float, help="0 selects FEM's assembled stability limit")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT / "bem/outputs") or args.cores < 1:
        raise ValueError(
            "FEM outputs must stay in this checkout's bem/outputs; cores must be positive"
        )
    for value in [args.normal_duration, args.shear_duration, args.mesh_size]:
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("Durations and optional mesh size must be finite and positive")
    if (
        not math.isfinite(args.grading_growth)
        or args.grading_growth <= 1
        or (
            args.contact_end_size is not None
            and (not math.isfinite(args.contact_end_size) or args.contact_end_size <= 0)
        )
        or (
            args.checkpoint_interval is not None
            and (not math.isfinite(args.checkpoint_interval) or args.checkpoint_interval <= 0)
        )
    ):
        raise ValueError("Require positive endpoint/checkpoint sizes and grading growth > 1")
    if args.time_step is not None and (not math.isfinite(args.time_step) or args.time_step < 0):
        raise ValueError("Time step must be finite and nonnegative")
    for value in [
        args.normal_frames,
        args.shear_frames,
        args.interface_normal_frames,
        args.interface_shear_frames,
    ]:
        if value is not None and value < 1:
            raise ValueError("Frame targets must be positive")
    if args.resume and (args.prepare_only or args.checkpoint_interval is None):
        raise ValueError("Resume requires checkpointing and cannot be prepare-only")
    if not args.resume and (
        (output / "simulation.h5").exists() or (output / "summary.json").exists()
    ):
        raise ValueError(
            "Choose a new output directory; existing FEM results are never overwritten"
        )
    runner_sha = digest(Path(__file__))
    arguments = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    previous = resume_manifest(output, arguments, runner_sha) if args.resume else None
    output.mkdir(parents=True, exist_ok=True)
    total_started, total_cpu = perf_counter(), process_time()
    invocation = uuid.uuid4().hex
    if not args.prepare_only:
        invocation_event(output, dict(event="started", id=invocation, resume=args.resume))
    protected = {
        str(path): digest(path)
        for path in [
            args.reference_run / "stats/summary.json",
            args.reference_run / "input/case.toml",
            args.reference_run / "input/resolved.json",
        ]
    }
    resolved_input = load_reference_input(args.reference_run)
    if previous is not None and protected != previous["protected_input_sha256"]:
        raise ValueError("Protected FEM inputs changed before resume")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
    ):
        os.environ[name] = str(args.cores)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={args.cores}"
    )
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["JAX_ENABLE_X64"] = (
        "true" if resolved_input["solver"]["dtype"] == "float64" else "false"
    )
    sys.dont_write_bytecode = True
    snapshot = PROJECT / "bem/outputs/fem-source" / REVISION
    archived_source = stage_snapshot(PROJECT, snapshot)
    sys.path.insert(0, str(snapshot))
    # Imports deliberately follow CPU setup and source selection.
    from dataclasses import asdict, fields, replace

    import jax
    import numpy as np
    from tatva.pmma import dynamics
    from tatva.pmma.dynamics import RunConfig, build_case_model, run_simulation_dumped
    from tatva.pmma.model import (
        BlockSpec,
        FrictionReference,
        LoadingReference,
        Material,
        PMMAModelInput,
    )
    from threadpoolctl import threadpool_info

    archived = json.loads((args.reference_run / "stats/summary.json").read_text())["summary"]
    values = dict(resolved_input["solver"])
    unknown = set(values) - {f.name for f in fields(RunConfig)}
    if unknown:
        raise ValueError(
            f"Resolved FEM solver fields are unsupported by the snapshot: {sorted(unknown)}"
        )
    values.update(
        simulation_time=args.shear_duration,
        normal_phase_time=args.normal_duration,
        shear_phase_time=args.shear_duration,
    )
    config = RunConfig(**values)
    original_builder, original_runner = build_case_model, run_simulation_dumped
    mesh_adapter = None
    if args.contact_end_size is not None:
        target_h = config.mesh_size if args.mesh_size is None else args.mesh_size
        if args.contact_end_size >= target_h:
            raise ValueError("Endpoint size must be smaller than the background FEM mesh")
        adapter_path = Path(__file__).with_name("chamfer_mesh_refinement.py")
        adapter = runpy.run_path(str(adapter_path))
        build_case_model = adapter["make_chamfer_refined_builder"](dynamics)
        run_simulation_dumped = adapter["rebind"](
            original_runner, {"build_case_model": build_case_model}
        )
        config = replace(
            config,
            fault_end_refinement_size=args.contact_end_size,
            fault_end_refinement_growth=args.grading_growth,
        )
        mesh_adapter = dict(
            source=str(adapter_path.resolve()),
            sha256=digest(adapter_path),
            scope="Explicit mesh-only adapter; original material, contact, static and transient kernels",
            end_size_mm=args.contact_end_size,
            grading_growth=args.grading_growth,
        )
    source_restart = restart_seed_record(config.restart_from_file)
    restart_projection = None
    if args.mesh_size is not None:
        config = replace(config, mesh_size=args.mesh_size)
    if args.time_step is not None:
        config = replace(config, time_step_override=args.time_step or None)
    legacy = resolved_input["case"]
    moving = BlockSpec(**legacy["moving"])
    stationary = BlockSpec(**legacy["stationary"])
    materials = {key: Material(**item) for key, item in legacy["materials"].items()}
    case = PMMAModelInput(
        moving,
        stationary,
        materials,
        FrictionReference(**legacy["friction"]),
        LoadingReference(**legacy["simulation"]),
    )
    if args.mesh_size is not None or args.contact_end_size is not None:
        import h5py

        # The archived static initializer uses only plastic slip and actuator;
        # its u array is checked for size but is not reused in the static solve.
        model = build_case_model(case, replace(config, restart_from_file=None))
        moving, stationary = model["moving"], model["stationary"]
        yy = np.asarray(moving.mesh.coords)[np.asarray(model["master_nodes"]), 1]
        plastic, actuator, restart_projection = project_restart_seed(
            values["restart_from_file"], yy, source_restart
        )
        restart_path = output / "projected_restart.h5"
        if args.resume:
            with h5py.File(restart_path, "r") as stream:
                np.testing.assert_array_equal(stream["final_state/plastic_slip"][:], plastic)
                if stream["final_state/u"].shape != (2 * (moving.n_nodes + stationary.n_nodes),):
                    raise ValueError("Projected restart mesh changed before resume")
                if stream["final_state"].attrs["applied_shear_displacement"] != actuator:
                    raise ValueError("Projected restart actuator changed before resume")
        else:
            with h5py.File(restart_path, "w") as stream:
                group = stream.create_group("final_state")
                group["u"] = np.zeros(2 * (moving.n_nodes + stationary.n_nodes))
                group["plastic_slip"] = plastic
                group.attrs["applied_shear_displacement"] = actuator
        config = replace(config, restart_from_file=str(restart_path))
        # The runner builds its own model. Avoid retaining this projection-only
        # model during a long fine-mesh transient.
        del model, moving, stationary
    actual_restart = restart_seed_record(config.restart_from_file)
    config_path = output / "resolved_run_config.json"
    if args.resume:
        if json.loads(config_path.read_text()) != asdict(config):
            raise ValueError("Effective FEM configuration changed before resume")
    else:
        config_path.write_text(json.dumps(asdict(config), indent=2) + "\n")
    output_options = dict(
        frames_per_phase=args.normal_frames,
        shear_frames_per_phase=args.shear_frames,
        interface_frames_per_phase=args.interface_normal_frames,
        shear_interface_frames_per_phase=args.interface_shear_frames,
        compression="lzf",
        include_initial_frame=True,
        store_bulk_strain=False,
        store_bulk_velocity=False,
    )
    manifest = dict(
        status="prepared_not_simulated",
        runner_sha256=runner_sha,
        source=archived_source,
        protected_input_sha256=protected,
        configuration_authority="input/resolved.json verified against authored input/case.toml and source revision",
        authored_input_dtype=resolved_input["solver"]["dtype"],
        historical_statistics_dtype=archived.get("dtype"),
        source_restart=source_restart,
        actual_restart=actual_restart,
        restart_projection=restart_projection,
        output_options=output_options,
        mesh_adapter=mesh_adapter,
        checkpoint_interval_seconds=args.checkpoint_interval,
        requested_cores=args.cores,
        source_snapshot_path=str(snapshot),
        arguments=arguments,
    )
    if args.resume:
        for key in ["source", "actual_restart", "source_restart", "mesh_adapter", "output_options"]:
            if previous[key] != manifest[key]:
                raise ValueError(f"FEM resume provenance changed: {key}")
        manifest = previous
    else:
        (output / "reference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(f"Archived FEM source and resolved settings ready: {output}")
        return
    started, cpu = perf_counter(), process_time()
    preparation_seconds = started - total_started
    runtime_options = dict(output_options)
    if args.checkpoint_interval is not None:
        runtime_options.update(
            checkpoint_path=output / "checkpoint.npz",
            checkpoint_interval_seconds=args.checkpoint_interval,
            resume=args.resume,
        )
    try:
        simulation = run_simulation_dumped(
            case, config, output / "simulation.h5", **runtime_options
        )
    except BaseException as error:
        invocation_event(
            output,
            dict(
                event="ended",
                id=invocation,
                status="checkpointed"
                if isinstance(error, dynamics.SimulationCheckpointed)
                else "failed",
                preparation_seconds=preparation_seconds,
                simulation_seconds=perf_counter() - started,
                simulation_cpu_seconds=process_time() - cpu,
                total_seconds=perf_counter() - total_started,
                total_cpu_seconds=process_time() - total_cpu,
            ),
        )
        raise
    solver_seconds, solver_cpu = perf_counter() - started, process_time() - cpu
    invocation_event(
        output,
        dict(
            event="ended",
            id=invocation,
            status="completed",
            preparation_seconds=preparation_seconds,
            simulation_seconds=solver_seconds,
            simulation_cpu_seconds=solver_cpu,
            total_seconds=perf_counter() - total_started,
            total_cpu_seconds=process_time() - total_cpu,
        ),
    )
    accounted = invocation_totals(output)
    verify_snapshot(snapshot, archived_source["files"])
    if (
        dynamics.build_case_model is not original_builder
        or dynamics.run_simulation_dumped is not original_runner
    ):
        raise ValueError("Original FEM function bindings changed")
    if mesh_adapter is not None and digest(mesh_adapter["source"]) != mesh_adapter["sha256"]:
        raise ValueError("Explicit FEM mesh adapter changed during simulation")
    if runner_sha != digest(Path(__file__)) or any(
        digest(path) != checksum for path, checksum in protected.items()
    ):
        raise ValueError("FEM runner or protected input changed during simulation")
    if restart_seed_record(config.restart_from_file) != actual_restart:
        raise ValueError("FEM restart input changed during simulation")
    if restart_seed_record(values["restart_from_file"]) != source_restart:
        raise ValueError("Original FEM restart source changed during simulation")
    summary = simulation["summary"]
    record = {
        "status": "completed",
        "source_revision": REVISION,
        "reference_run": str(args.reference_run),
        "cores_requested": args.cores,
        "hard_cpu_affinity": False,
        "cpu_limit_note": "macOS thread limits are best effort; affinity is not enforced",
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "blas_threadpools": threadpool_info(),
        "wall_seconds": accounted["simulation_seconds"],
        "process_cpu_seconds": accounted["simulation_cpu_seconds"],
        "current_invocation_simulation_seconds": solver_seconds,
        "invocation_timing": accounted,
        "timing_seconds": {
            "source_and_config_preparation": accounted["preparation_seconds"],
            "simulation_including_outputs": accounted["simulation_seconds"],
            "total_before_reporting": accounted["total_seconds"],
        },
        "total_process_cpu_seconds": accounted["total_cpu_seconds"],
        "timing_control": "Not established by this runner; concurrent jobs must be recorded separately",
        "runner_sha256": runner_sha,
        "reference_manifest_sha256": digest(output / "reference_manifest.json"),
        "source_archive_sha256": archived_source["archive_sha256"],
        "source_and_protected_input_unchanged": True,
        "configuration_authority": manifest["configuration_authority"],
        "authored_input_dtype": manifest["authored_input_dtype"],
        "computational_dtype": config.dtype,
        "stored_history_dtype": summary.get("dtype"),
        "historical_statistics_dtype": manifest["historical_statistics_dtype"],
        "actual_restart": actual_restart,
        "restart_projection": restart_projection,
        "mesh_adapter": mesh_adapter,
        "checkpoint_interval_seconds": args.checkpoint_interval,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "output_options": output_options,
        "comparison_history_group": "interface_high_rate"
        if args.interface_normal_frames is not None or args.interface_shear_frames is not None
        else "/",
        "summary": summary,
        "full_original_horizon": args.shear_duration == 0.045 and args.normal_duration == 0.002,
        "output_frames_reduced": True,
        "restart_projected_to_coarse_mesh": args.mesh_size is not None,
        "restart_projected_to_target_mesh": args.mesh_size is not None
        or args.contact_end_size is not None,
    }

    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Unsupported FEM metadata: {type(value)}")

    (output / "summary.json").write_text(json.dumps(record, indent=2, default=encode) + "\n")
    print(
        json.dumps(
            {key: value for key, value in record.items() if key != "summary"},
            indent=2,
            default=encode,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

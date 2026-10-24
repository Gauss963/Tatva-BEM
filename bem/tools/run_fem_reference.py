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
import subprocess
import sys
import tarfile
from pathlib import Path
from time import perf_counter, process_time

PROJECT = Path(__file__).resolve().parents[2]
REVISION = "933ae1f3fe1c634391bdbd316e4efa5dbf0e6d5d"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    parser.add_argument("--shear-duration", type=float, default=5e-5)
    parser.add_argument("--normal-duration", type=float, default=5e-8)
    parser.add_argument("--mesh-size", type=float, help="Optional coarse FEM comparison mesh")
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
    if (output / "simulation.h5").exists() or (output / "summary.json").exists():
        raise ValueError(
            "Choose a new output directory; existing FEM results are never overwritten"
        )
    output.mkdir(parents=True, exist_ok=True)
    total_started, total_cpu = perf_counter(), process_time()
    runner_sha = digest(Path(__file__))
    protected = {
        str(path): digest(path)
        for path in [
            args.reference_run / "stats/summary.json",
            args.reference_run / "input/case.toml",
        ]
    }
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
    sys.dont_write_bytecode = True
    snapshot = PROJECT / "bem/outputs/fem-source" / REVISION
    archived_source = stage_snapshot(PROJECT, snapshot)
    sys.path.insert(0, str(snapshot))
    # Imports deliberately follow CPU setup and source selection.
    from dataclasses import asdict, fields, replace

    import jax
    import numpy as np
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
    values = {f.name: archived[f.name] for f in fields(RunConfig) if f.name in archived}
    values.update(
        simulation_time=args.shear_duration,
        normal_phase_time=args.normal_duration,
        shear_phase_time=args.shear_duration,
        normal_stress_override=archived["normal_stress"],
        mu_k_override=archived["mu_k"],
        critical_slip_override=archived["critical_slip"],
        shear_displacement_k_override=archived["shear_displacement_k"],
        shear_displacement_s_override=archived["shear_displacement_s"],
        tau_k_start_fraction_override=1.0,
        tau_k_full_fraction_override=1.0,
        shear_tau_k_override=archived["tau_k"],
        shear_tau_s_override=archived["tau_s"],
    )
    import tomllib

    source_input = tomllib.loads((args.reference_run / "input/case.toml").read_text())
    values["restart_from_file"] = source_input["solver"]["restart_from_file"]
    config = RunConfig(**values)
    if args.mesh_size is not None:
        config = replace(config, mesh_size=args.mesh_size)
    if args.time_step is not None:
        config = replace(config, time_step_override=args.time_step or None)
    legacy = archived["legacy"]
    geometry = legacy["geometry"]
    moving = BlockSpec(**geometry["moving"])
    stationary = BlockSpec(**geometry["stationary"])
    materials = {key: Material(**item) for key, item in legacy["materials"].items()}
    case = PMMAModelInput(
        moving,
        stationary,
        materials,
        FrictionReference(**legacy["friction"]),
        LoadingReference(**legacy["simulation"]),
    )
    if args.mesh_size is not None:
        import h5py

        # The archived static initializer uses only plastic slip and actuator;
        # its u array is checked for size but is not reused in the static solve.
        model = build_case_model(case, replace(config, restart_from_file=None))
        seed = np.genfromtxt(
            PROJECT / "bem/src/finiteblock_bem/resources/data/ev16_restart_interface.csv",
            delimiter=",",
            names=True,
        )
        moving, stationary = model["moving"], model["stationary"]
        yy = np.asarray(moving.mesh.coords)[np.asarray(model["master_nodes"]), 1]
        restart_path = output / "projected_restart.h5"
        with h5py.File(restart_path, "w") as stream:
            group = stream.create_group("final_state")
            group["u"] = np.zeros(2 * (moving.n_nodes + stationary.n_nodes))
            group["plastic_slip"] = np.interp(yy, seed["y"], seed["plastic_slip"])
            group.attrs["applied_shear_displacement"] = 10.806226665707777
        config = replace(config, restart_from_file=str(restart_path))
    (output / "resolved_run_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
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
        output_options=output_options,
        requested_cores=args.cores,
        source_snapshot_path=str(snapshot),
        arguments={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    (output / "reference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(f"Archived FEM source and resolved settings ready: {output}")
        return
    started, cpu = perf_counter(), process_time()
    preparation_seconds = started - total_started
    simulation = run_simulation_dumped(
        case,
        config,
        output / "simulation.h5",
        **output_options,
    )
    solver_seconds, solver_cpu = perf_counter() - started, process_time() - cpu
    verify_snapshot(snapshot, archived_source["files"])
    if runner_sha != digest(Path(__file__)) or any(
        digest(path) != checksum for path, checksum in protected.items()
    ):
        raise ValueError("FEM runner or protected input changed during simulation")
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
        "wall_seconds": solver_seconds,
        "process_cpu_seconds": solver_cpu,
        "timing_seconds": {
            "source_and_config_preparation": preparation_seconds,
            "simulation_including_outputs": solver_seconds,
            "total_before_reporting": perf_counter() - total_started,
        },
        "total_process_cpu_seconds": process_time() - total_cpu,
        "timing_control": "Not established by this runner; concurrent jobs must be recorded separately",
        "runner_sha256": runner_sha,
        "reference_manifest_sha256": digest(output / "reference_manifest.json"),
        "source_archive_sha256": archived_source["archive_sha256"],
        "source_and_protected_input_unchanged": True,
        "output_options": output_options,
        "comparison_history_group": "interface_high_rate"
        if args.interface_normal_frames is not None or args.interface_shear_frames is not None
        else "/",
        "summary": summary,
        "full_original_horizon": args.shear_duration == 0.045 and args.normal_duration == 0.002,
        "output_frames_reduced": True,
        "restart_projected_to_coarse_mesh": args.mesh_size is not None,
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

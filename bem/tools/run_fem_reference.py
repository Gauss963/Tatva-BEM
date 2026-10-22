#!/usr/bin/env python3
"""Run the archived FEM implementation locally, never writing to the source run.

Defaults to a short timing experiment on the original 2 mm mesh. Use
--shear-duration .045 --normal-duration .002 for the full original horizon.
This imports an exact local Git snapshot, not the evolved root FEM runner.
"""

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from time import perf_counter, process_time

PROJECT = Path(__file__).resolve().parents[2]
REVISION = "933ae1f3fe1c634391bdbd316e4efa5dbf0e6d5d"


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
    parser.add_argument("--shear-frames", type=int, default=32)
    parser.add_argument("--time-step", type=float, help="0 selects FEM's assembled stability limit")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT / "bem/outputs") or args.cores < 1:
        raise ValueError(
            "FEM outputs must stay in this checkout's bem/outputs; cores must be positive"
        )
    output.mkdir(parents=True, exist_ok=True)
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
    if not snapshot.is_dir():
        archive = subprocess.run(
            ["git", "-C", str(PROJECT), "archive", REVISION, "tatva"],
            check=True,
            capture_output=True,
        ).stdout
        snapshot.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(snapshot, filter="data")
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
    if args.prepare_only:
        print(f"Archived FEM source and resolved settings ready: {output}")
        return
    started, cpu = perf_counter(), process_time()
    simulation = run_simulation_dumped(
        case,
        config,
        output / "simulation.h5",
        frames_per_phase=4,
        shear_frames_per_phase=args.shear_frames,
        compression="lzf",
        include_initial_frame=True,
        store_bulk_strain=False,
        store_bulk_velocity=False,
    )
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
        "wall_seconds": perf_counter() - started,
        "process_cpu_seconds": process_time() - cpu,
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

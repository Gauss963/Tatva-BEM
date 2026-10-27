"""Run event-16 with refined contact states and the complete fixed bulk CQ.

The physical parameters, bulk mesh, restart input and loading policy are
fixed. Healed static Coulomb initialization is an explicit separate control;
its plastic offset corrections are recorded. Every lag is required from
step one; a late-start correction prefix is rejected.
"""

import argparse
import json
import os
import resource
import runpy
import sys
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.fft import set_workers

from finiteblock_bem.cq import BoundaryHistory, CQPlan
from finiteblock_bem.friction import ContactState
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model, nodal_body_stiffness

TOOLS = Path(__file__).resolve().parent
HELPER = runpy.run_path(str(TOOLS / "replay_resolvent.py"))
REFINED = runpy.run_path(str(TOOLS / "refined_contact.py"))


def save_trajectory(output, model, space, lift, result, summary):
    q, n = result.bulk_displacement, space.bulk_nodes
    fine_jump = np.concatenate(
        (
            space.interpolate(q[:, :n]) + space.interpolate(lift.retained_displacement[:n]),
            space.interpolate(q[:, n : 2 * n])
            + space.interpolate(lift.retained_displacement[n : 2 * n]),
        ),
        axis=1,
    )
    ref = model.reference
    np.savez(
        output / "trajectory.npz",
        time=result.time,
        pressure=result.pressure,
        shear=result.shear,
        friction_coefficient=result.friction_coefficient,
        plastic_slip=result.plastic_slip,
        cumulative_slip=result.cumulative_slip,
        actuator=result.actuator,
        platen=result.platen,
        normal_load=np.full(len(result.time), ref.normal_pressure),
        contact_y=space.y,
        contact_weights=space.weights,
        contact_interpolation=space.interpolation,
        retained_contact_y=space.bulk_y,
        retained_displacement=q,
        initial_retained_displacement=lift.retained_displacement,
        initial_retained_bulk_force=lift.retained_bulk_force,
        contact_jumps=fine_jump,
        moving_starts=ref.moving.starts,
        moving_ends=ref.moving.ends,
        stationary_starts=ref.stationary.starts,
        stationary_ends=ref.stationary.ends,
        moving_contact=model.moving_contact,
        stationary_contact=model.stationary_contact,
    )
    table = np.column_stack(
        (
            result.time,
            np.full(len(result.time), ref.normal_pressure),
            np.average(result.pressure, weights=space.weights, axis=1),
            np.average(result.shear, weights=space.weights, axis=1),
            result.cumulative_slip.max(axis=1),
            result.actuator,
            result.platen,
            ref.spring_stiffness * (result.actuator - result.platen),
            result.residual,
            result.iterations,
        )
    )
    np.savetxt(
        output / "history.csv",
        table,
        delimiter=",",
        comments="",
        header="time_s,normal_load_mpa,mean_contact_pressure_mpa,mean_contact_shear_mpa,max_cumulative_slip_mm,actuator_mm,platen_mm,applied_shear_mpa,residual_mpa,newton_iterations",
    )
    write_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--restart-seed", type=Path, required=True)
    parser.add_argument("--subdivisions", type=int, default=1)
    parser.add_argument(
        "--seed-interpolation", choices=["coarse-p1", "original-p1"], default="coarse-p1"
    )
    parser.add_argument(
        "--static-initialization",
        choices=["preserve-offset", "fem-active-set", "fem-body-active-set"],
        default="preserve-offset",
    )
    parser.add_argument("--constitutive-substeps", type=int, default=1)
    parser.add_argument("--kinetic-root-seed", action="store_true")
    parser.add_argument("--fft-cache-mb", type=float, default=512)
    parser.add_argument("--fft-rows", type=int, default=4)
    parser.add_argument("--max-memory-mb", type=float, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a new output outside protected Tatva")
    if (
        args.subdivisions < 1
        or args.constitutive_substeps < 1
        or args.fft_rows < 1
        or not all(np.isfinite(v) and v > 0 for v in (args.max_memory_mb, args.fft_cache_mb))
        or (args.kinetic_root_seed and args.constitutive_substeps == 1)
    ):
        raise ValueError("Require positive discretization/budgets; kinetic recovery needs substeps")
    begun = perf_counter()
    metadata_path = args.kernel / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    plan = CQPlan(**metadata["plan"])
    options = metadata["source_mesh_arguments"]
    if (
        metadata["status"] != "completed_resolvent_kernel_only"
        or plan.steps < metadata["full_steps"]
        or metadata["full_steps"] * plan.dt != 0.045
        or options.get("diagnostic_actuator_increment") is not None
    ):
        raise ValueError(
            "Require every lag for the original 45 ms event, not a late-start correction prefix"
        )
    environment = environment_record(options["case"])
    if any(
        environment[k] != metadata["environment"][k]
        for k in ("source_sha256", "case_sha256", "restart_data_sha256")
    ):
        raise ValueError("Prepared bulk package/physical input changed")
    helpers = [
        "refined_contact.py",
        "replay_resolvent.py",
        "restart_seed_experiment.py",
        "bounded_history.py",
        "substepped_contact_experiment.py",
        "robust_contact_experiment.py",
    ]
    if args.static_initialization in ("fem-active-set", "fem-body-active-set"):
        helpers.append("fem_static_contact.py")
    if args.static_initialization == "fem-body-active-set":
        helpers.append("uncondensed_static_contact.py")
    paths = [
        Path(__file__),
        metadata_path,
        args.restart_seed,
        args.kernel / "initial_impedance.npy",
        *[TOOLS / name for name in helpers],
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    for name, key in (("weights.npy", "weights_sha256"), ("elastic.npz", "elastic_sha256")):
        if HELPER["digest_file"](args.kernel / name) != metadata[key]:
            raise ValueError(f"Complete prepared data digest differs: {name}")
    with np.load(args.kernel / "elastic.npz", allow_pickle=False) as data:
        elastic, time, ramp = (data[k] for k in ("elastic", "time", "uninterrupted_actuator"))
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    seed_tool = runpy.run_path(str(TOOLS / "restart_seed_experiment.py"))
    model, seed_record = seed_tool["apply_restart_seed"](model, args.restart_seed)
    np.testing.assert_array_equal(time, np.arange(metadata["full_steps"] + 1) * plan.dt)
    np.testing.assert_allclose(
        ramp,
        model.reference.restart_actuator + model.reference.actuator_rate * time,
        rtol=0,
        atol=1e-12,
    )
    nodes = model.reference.moving.starts[model.moving_contact, 1]
    space = REFINED["ContactQuadrature"].build(nodes, args.subdivisions)
    if args.seed_interpolation == "coarse-p1":
        state = space.seed_from_bulk_p1(model.restart)
    else:
        with np.load(args.restart_seed, allow_pickle=False) as data:
            state = ContactState(
                np.interp(space.y, data["y"], data["plastic_slip"]),
                np.interp(space.y, data["y"], data["cumulative_slip"]),
            )
    bounded = runpy.run_path(str(TOOLS / "bounded_history.py"))
    online_bytes = bounded["estimate_online_arrays"](
        plan, metadata["full_steps"], args.fft_cache_mb, fft_rows=args.fft_rows
    )
    trajectory_bytes = len(time) * (2 * space.dofs + 7 * space.points + 8) * 8
    if online_bytes + trajectory_bytes > args.max_memory_mb * 1024**2:
        raise ValueError("Estimated complete history and fine trajectory exceed the memory budget")
    # Nothing under the prepared kernel or protected source is written.
    output.mkdir(parents=True)
    write_json(
        output / "run_manifest.json",
        dict(
            status="prepared_not_completed",
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            frozen_input_sha256=frozen,
            original_tatva_readonly=True,
            source_recorded_preparation_seconds=metadata["total_seconds"],
        ),
    )
    weights = np.load(args.kernel / "weights.npy", mmap_mode="r", allow_pickle=False)
    if weights.shape != (plan.steps + 1, space.dofs, space.dofs) or weights.dtype != np.float64:
        raise ValueError("Complete physical correction weights do not match the fixed bulk trace")
    stiffness = nodal_body_stiffness(model, quadrature=options["quadrature"])
    reduction = REFINED["reduce_static_trace"](model, stiffness)
    initialize = REFINED["initialize_refined_contact"]
    if args.static_initialization == "fem-active-set":
        initialize = runpy.run_path(str(TOOLS / "fem_static_contact.py"))[
            "initialize_fem_static_contact"
        ]
    elif args.static_initialization == "fem-body-active-set":
        initialize = runpy.run_path(str(TOOLS / "uncondensed_static_contact.py"))[
            "initialize_uncondensed_contact"
        ]
    lift = initialize(model, space, reduction, state)
    state = lift.state
    impedance = np.load(args.kernel / "initial_impedance.npy", allow_pickle=False)
    reference = REFINED["dynamic_refined_reference"](model, space, lift, impedance)
    del stiffness, reduction
    factory = None
    if args.constitutive_substeps > 1:
        cls = runpy.run_path(str(TOOLS / "robust_contact_experiment.py"))[
            "RobustSubsteppedContactEquilibrium"
        ]
        factory = partial(
            cls, substeps=args.constitutive_substeps, kinetic_seed_recovery=args.kinetic_root_seed
        )
    memory = bounded["bounded_correction_class"](
        HELPER["CorrectionHistory"], cache_mb=args.fft_cache_mb, fft_rows=args.fft_rows
    )
    initialization_seconds = perf_counter() - begun

    def failure(index, data):
        np.savez(
            output / "failed_equilibrium.npz",
            **data,
            matrix=reference.matrix,
            base_dn=reference.base_dn,
            base_ds=reference.base_ds,
            contact_weights=space.weights,
            contact_interpolation=space.interpolation,
        )
        write_json(
            output / "failure.json",
            dict(
                status="failed_refined_contact_root",
                step=index,
                time_s=float(data["time"]),
                residual_mpa=float(data["residual_mpa"]),
                resumable_cq_checkpoint=False,
                full_original_event_reproduced=False,
            ),
        )

    with set_workers(int(os.environ.get("FDBEM_CORES", "1"))):
        result = REFINED["simulate_refined_prepared"](
            model,
            space,
            reference,
            state,
            BoundaryHistory(plan, weights),
            elastic,
            time,
            ramp,
            equilibrium_factory=factory,
            history_factory=memory,
            failure_callback=failure,
        )
    if any(fingerprint(path) != value for path, value in frozen.items()):
        raise ValueError("Frozen inputs or solver implementation changed")
    final_environment = environment_record(options["case"])
    if any(
        final_environment[k] != environment[k]
        for k in ("source_sha256", "case_sha256", "restart_data_sha256")
    ):
        raise ValueError("Dynamic package or case changed during the event")
    summary = dict(result.diagnostics)
    summary.update(
        environment=environment,
        case=model.reference.payload,
        initialization=lift.diagnostics,
        prepared_kernel=str(args.kernel.resolve()),
        frozen_input_sha256=frozen,
        restart_seed_override=seed_record,
        seed_interpolation=args.seed_interpolation,
        static_initialization=args.static_initialization,
        point_seed_scope="P1 interpolation of the same coarse seed field"
        if args.seed_interpolation == "coarse-p1"
        else "Original float64 source P1 field sampled on refined state points",
        kernel_summary_sha256=fingerprint(metadata_path),
        quadrature=options["quadrature"],
        contact_integration="Positive composite nodal trapezoid, work-conjugate pullback",
        bulk_trace_unchanged=True,
        fft_cache_mb=args.fft_cache_mb,
        estimated_solver_array_peak_mib=(online_bytes + trajectory_bytes) / 1024**2,
        timing_seconds=dict(
            static_lift_kernel_hashing_and_load=initialization_seconds,
            time_stepping=result.diagnostics["time_stepping_seconds"],
            kernel_recorded_preparation=metadata["total_seconds"],
            total_before_output=perf_counter() - begun,
        ),
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        same_accuracy_fem_advantage_verified=False,
    )
    save_trajectory(output, model, space, lift, result, summary)
    summary["timing_seconds"]["total_including_output"] = perf_counter() - begun
    write_json(output / "summary.json", summary)
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

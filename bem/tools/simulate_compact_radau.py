"""Nonlinear endpoint-committed Radau experiment with exact compact histories."""

import argparse
import json
import resource
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.cq import BoundaryHistory, CQPlan
from finiteblock_bem.dynamics import DynamicResult
from finiteblock_bem.friction import ContactState
from finiteblock_bem.io import environment_record, fingerprint, save_dynamic, write_json
from finiteblock_bem.nodal import (
    NodalInterfaceEquilibrium,
    initialize_nodal_restart,
    load_nodal_model,
)

TOOLS = Path(__file__).resolve().parent
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))
HISTORY = runpy.run_path(str(TOOLS / "compact_radau_history.py"))
CONTACT = runpy.run_path(str(TOOLS / "radau_contact_experiment.py"))
REPLAY = runpy.run_path(str(TOOLS / "replay_resolvent.py"))


def simulate_stages(
    reference,
    state,
    first,
    second,
    wrap,
    impedance,
    elastic,
    stage_time,
    ramp,
    *,
    stop_coverage=1.0,
    diagnostic=False,
    segment_substeps=(1, 2),
    cache_mb=0,
    fft_rows=4,
    recover=False,
    kinetic_seed=False,
    failure_callback=None,
):
    """Solve coupled stages, then commit endpoint state and macrostep stop rule.

    Old state is never committed from a trial first stage. The first-stage
    path state is recomputed inside every residual and tangent evaluation.
    """
    begun = perf_counter()
    con, law = reference.constraints, reference.case.contact
    n, d, steps = len(con.contact_weights), con.size, len(stage_time) - 1
    if (
        d != first.plan.dofs
        or impedance.shape != (2 * d, 2 * d)
        or elastic.shape != (steps + 1, 2, d)
        or ramp.shape != (steps + 1, 2)
        or stage_time.shape != ramp.shape
        or not 0 < stop_coverage <= 1
        or not all(np.all(np.isfinite(a)) for a in [impedance, elastic, ramp, stage_time])
    ):
        raise ValueError("Invalid compact stage model, startup or loading")
    expected_time = np.zeros_like(stage_time)
    expected_time[1:] = (np.arange(steps)[:, None] + RADAU["C"]) * first.plan.dt
    np.testing.assert_allclose(stage_time, expected_time, rtol=1e-12, atol=1e-15)
    weights = con.contact_weights
    cdiag = np.r_[weights * law.normal_penalty, weights * law.tangential_penalty, 0]
    spring_matrix = np.zeros((d, d))
    spring_matrix[-1, -1] = reference.spring
    matrix = impedance + np.kron(np.eye(2), spring_matrix)
    a0 = matrix + np.diag(np.tile(cdiag, 2))
    instantaneous = RADAU["expand_compact_weights"](first.weights[:1], second.weights[:1], wrap)[0]
    composition = float(np.linalg.norm(instantaneous @ a0 - np.eye(2 * d), ord=np.inf))
    if not np.isfinite(composition) or composition > 1e-6:
        raise ValueError(f"Compact Radau inverse composition failed: {composition}")
    system = CONTACT["RadauContactEquilibrium"](
        reference, matrix, segment_substeps=segment_substeps
    )
    initial = law.evaluate(reference.base_dn, reference.base_ds, state)
    if np.any(initial.state.plastic_slip != state.plastic_slip) or np.any(
        initial.state.cumulative_slip != state.cumulative_slip
    ):
        raise ValueError("The initial lift must be admissible without a plastic update")
    fc0 = np.r_[weights * initial.pressure, weights * initial.shear, 0]
    q = np.zeros((steps + 1, 2, d))
    values = np.zeros((steps + 1, n, 2, 7))
    start = np.column_stack(
        (
            initial.pressure,
            initial.shear,
            state.plastic_slip,
            state.cumulative_slip,
            initial.friction_coefficient,
            np.zeros(n),
            np.zeros(n),
        )
    )
    values[0] = np.repeat(start[:, None, :], 2, axis=1)
    actuator, residual, iterations = (
        ramp.copy(),
        np.zeros(steps + 1),
        np.zeros(steps + 1, dtype=int),
    )
    memory = HISTORY["CompactCorrectionHistory"](
        first, second, wrap, cache_mb=cache_mb, fft_rows=fft_rows
    )
    memory.append(0, np.zeros(2 * d), active=False)
    old, held, hold_time = state, None, None
    recoveries = []
    for k in range(1, steps + 1):
        if held is not None:
            actuator[k] = held
        prediction = elastic[k].ravel() + memory.past(k)
        hold_force = np.zeros((2, d))
        hold_force[:, -1] = reference.spring * (actuator[k] - ramp[k])
        forcing = -np.tile(fc0, 2) - a0 @ prediction - hold_force.ravel()
        answer = system.solve(
            old,
            forcing,
            np.tile(q[k - 1, 1], 2),
            q[k - 1, 1],
            recover=recover,
            kinetic_seed=kinetic_seed,
        )
        if not answer.converged:
            if failure_callback is not None:
                failure_callback(
                    k,
                    dict(
                        matrix=matrix,
                        forcing=forcing,
                        failed_displacement=answer.displacement,
                        previous_displacement=q[k - 1, 1],
                        plastic_slip=old.plastic_slip,
                        cumulative_slip=old.cumulative_slip,
                        parameters=np.asarray(system.parameters),
                        base_dn=reference.base_dn,
                        base_ds=reference.base_ds,
                        contact_weights=weights,
                        reduced_weights=con.reduced_weights,
                        residual_mpa=answer.residual_mpa,
                        stage_time=stage_time[k],
                        accepted_displacement=q[:k],
                        accepted_response=values[:k],
                    ),
                )
            raise ValueError(f"Radau equilibrium failed at step {k}: {answer.residual_mpa}")
        q[k], values[k] = answer.displacement.reshape(2, d), answer.response
        residual[k], iterations[k] = answer.residual_mpa, answer.iterations
        if answer.recovery_attempts:
            recoveries.append(
                dict(step=k, attempts=answer.recovery_attempts, residual_mpa=answer.residual_mpa)
            )
        force = np.stack(
            [np.r_[weights * values[k, :, a, 0], weights * values[k, :, a, 1], 0] for a in range(2)]
        )
        correction = np.tile(fc0, (2, 1)) + cdiag[None, :] * q[k] - force + hold_force
        active = bool(
            np.any(hold_force != 0)
            or np.any(values[k, :, :, 0] <= 0)
            or np.any(values[k, :, :, 2] != old.plastic_slip[:, None])
            or np.any(values[k, :, :, 3] != old.cumulative_slip[:, None])
        )
        memory.append(k, correction.ravel(), active=active)
        old = ContactState(values[k, :, 1, 2], values[k, :, 1, 3])
        if (
            not diagnostic
            and held is None
            and np.average(old.cumulative_slip >= law.characteristic_slip, weights=weights)
            >= stop_coverage
        ):
            held, hold_time = actuator[k, 1], stage_time[k, 1]
    endpoint = values[:, :, 1]
    work = float(np.sum(values[1:, :, :, 5] * weights[None, :, None]))
    integrated = float(np.sum(values[1:, :, :, 6] * weights[None, :, None]))
    diagnostics = dict(
        status="completed_requested_horizon",
        time_method="radau2",
        history_representation="compact_radau_elastic_resolvent",
        dt_s=first.plan.dt,
        requested_steps=steps,
        completed_steps=steps,
        duration_s=float(stage_time[-1, 1]),
        history_dofs=[d],
        stage_dofs=2 * d,
        cq_fft_size=first.plan.fft_size,
        all_history_retained=True,
        nonlinear_backend="jax",
        contact_nodes=n,
        max_cumulative_slip_mm=float(endpoint[:, :, 3].max()),
        max_boundary_contact_residual_mpa=float(residual.max()),
        actuator_hold_time_s=hold_time,
        first_correction_step=memory.start,
        correction_history_steps=first.plan.steps,
        inverse_cq_composition_inf_error=composition,
        segment_substeps=list(segment_substeps),
        constitutive_substeps=sum(segment_substeps),
        constitutive_path="Two linear segments through the dt/3 stage and endpoint; one committed endpoint state",
        loading_stop_rule="Coverage checked at accepted macrostep endpoints; hold starts next interval",
        micro_friction_work_mpa_mm2=work,
        micro_fixed_pressure_integrated_strength_work_mpa_mm2=integrated,
        micro_fixed_pressure_work_gap_mpa_mm2=work - integrated,
        contact_root_recovery_events=recoveries,
        maximum_inactive_source_force=memory.maximum_inactive_source_force,
        time_stepping_seconds=perf_counter() - begun,
        full_trajectory_substep_ad_validated=False,
        full_original_event_reproduced=False,
        failure=None,
        scope="Complete causal compact Radau history and endpoint-committed stage contact; no FEM equivalence, gradient convergence or speed advantage asserted",
    )
    if memory.online is not None:
        diagnostics["online_fft_cache"] = memory.online.statistics()
    result = DynamicResult(
        stage_time[:, 1],
        None,
        None,
        endpoint[:, :, 0],
        endpoint[:, :, 1],
        endpoint[:, :, 4],
        endpoint[:, :, 2],
        endpoint[:, :, 3],
        actuator[:, 1],
        reference.base_platen + q[:, 1, -1],
        np.full(steps + 1, reference.case.normal_pressure),
        residual,
        iterations,
        diagnostics,
        q[:, 1, : 2 * n] + np.r_[reference.base_dn, reference.base_ds],
    )
    stages = dict(displacement=q, response=values, stage_time=stage_time, actuator=actuator)
    return result, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--steps", type=int, help="Optional original-grid prefix; no time resampling"
    )
    parser.add_argument("--segment-substeps", type=int, nargs=2, default=[1, 2])
    parser.add_argument("--contact-root-recovery", action="store_true")
    parser.add_argument("--kinetic-root-seed", action="store_true")
    parser.add_argument("--restart-seed", type=Path)
    parser.add_argument("--fft-cache-mb", type=float, default=8)
    parser.add_argument("--fft-rows", type=int, default=4)
    parser.add_argument("--max-memory-mb", type=float, default=8192)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("The original Tatva directory must remain read-only")
    if output.exists():
        raise ValueError("Choose a new simulation directory")
    if (
        any(v < 1 for v in args.segment_substeps)
        or args.fft_rows < 1
        or not np.isfinite(args.fft_cache_mb)
        or args.fft_cache_mb < 0
        or not np.isfinite(args.max_memory_mb)
        or args.max_memory_mb <= 0
        or (args.kinetic_root_seed and not args.contact_root_recovery)
    ):
        raise ValueError("Invalid substep/resource options or kinetic seed without root recovery")
    begun = perf_counter()
    metadata_path = args.kernel / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("status") != "completed_compact_radau_resolvent_kernel_only"
        or metadata.get("time_method") != "radau2"
    ):
        raise ValueError("Require a completed compact Radau kernel")
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    keys = ["source_sha256", "case_sha256", "restart_data_sha256"]
    if any(environment[k] != metadata["environment"][k] for k in keys):
        raise ValueError("Prepared Radau kernel differs from the current package or physical case")
    steps = metadata["full_steps"] if args.steps is None else args.steps
    if not 1 <= steps <= metadata["full_steps"]:
        raise ValueError("Requested steps must lie within the prepared full horizon")
    plan = CQPlan(**metadata["plan"])
    plan = replace(plan, steps=min(plan.steps, steps))
    estimate = 2 * HISTORY["BOUNDED"]["estimate_online_arrays"](
        plan, steps, args.fft_cache_mb, fft_rows=args.fft_rows
    )
    estimate += (steps + 1) * plan.dofs * 24 * 8
    if estimate > args.max_memory_mb * 1024**2:
        raise ValueError(f"Estimated stage solver arrays need {estimate / 1024**2:.1f} MiB")
    frozen_paths = [
        Path(__file__),
        metadata_path,
        *[
            TOOLS / name
            for name in [
                "compact_radau_history.py",
                "radau_contact_experiment.py",
                "radau_experiment.py",
                "substepped_contact_experiment.py",
                "robust_contact_experiment.py",
                "bounded_history.py",
                "replay_resolvent.py",
            ]
        ],
    ]
    seed_path = TOOLS / "restart_seed_experiment.py"
    if args.restart_seed is not None:
        frozen_paths.extend([seed_path, args.restart_seed])
    frozen = {str(p.resolve()): fingerprint(p) for p in frozen_paths}
    for filename, key in [
        ("shift_wrap.npy", "shift_wrap_sha256"),
        ("initial_impedance.npy", "initial_impedance_sha256"),
        ("elastic.npz", "elastic_sha256"),
    ]:
        if REPLAY["digest_file"](args.kernel / filename) != metadata[key]:
            raise ValueError(f"Prepared Radau input changed: {filename}")
    maps = []
    for record in metadata["family_weights"]:
        path = Path(record["path"])
        if REPLAY["digest_file"](path) != record["sha256"]:
            raise ValueError("Prepared compact weights changed")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            array.shape != (metadata["correction_steps"] + 1, plan.dofs, plan.dofs)
            or array.dtype != np.float64
        ):
            raise ValueError("Invalid compact time-weight shape or dtype")
        maps.append(array)
    if len(maps) != 2:
        raise ValueError("A compact Radau kernel requires two weight families")
    wrap = np.load(args.kernel / "shift_wrap.npy", allow_pickle=False)
    impedance = np.load(args.kernel / "initial_impedance.npy", allow_pickle=False)
    with np.load(args.kernel / "elastic.npz", allow_pickle=False) as data:
        stage_time = data["stage_time"][: steps + 1]
        elastic = data["elastic"][: steps + 1]
        ramp = data["uninterrupted_actuator"][: steps + 1]
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    seed_record = None
    if args.restart_seed is not None:
        model, seed_record = runpy.run_path(str(seed_path))["apply_restart_seed"](
            model, args.restart_seed
        )
    diagnostic = options.get("diagnostic_actuator_increment")
    rate = (
        model.reference.actuator_rate
        if diagnostic is None
        else diagnostic / (metadata["full_steps"] * plan.dt)
    )
    np.testing.assert_allclose(
        ramp, model.reference.restart_actuator + rate * stage_time, rtol=0, atol=1e-12
    )
    lift = initialize_nodal_restart(model, quadrature=options["quadrature"])
    reference = NodalInterfaceEquilibrium(model, lift, np.zeros((plan.dofs, plan.dofs)))
    initial = reference.evaluate(
        np.zeros(plan.dofs),
        np.zeros(plan.dofs),
        model.reference.normal_pressure,
        ramp[0, 1],
        lift.state,
        jacobian=False,
    )[0]
    if np.max(abs(initial)) >= 1e-7:
        raise ValueError("Initial static lift exceeds the equilibrium tolerance")
    initialization_seconds = perf_counter() - begun
    output.mkdir(parents=True)

    def save_failure(index, arrays):
        np.savez(output / "failed_equilibrium.npz", **arrays)
        write_json(
            output / "failure.json",
            dict(
                status="failed_radau_nonlinear_equilibrium",
                failed_step=index,
                residual_mpa=float(arrays["residual_mpa"]),
                kernel=str(args.kernel.resolve()),
                segment_substeps=args.segment_substeps,
                completed_requested_horizon=False,
                failed_equilibrium_sha256=REPLAY["digest_file"](output / "failed_equilibrium.npz"),
                scope="Captured local coupled stage equation and accepted prefix; not a resumable history checkpoint",
            ),
        )

    try:
        result, stages = simulate_stages(
            reference,
            lift.state,
            *[BoundaryHistory(plan, a[: plan.steps + 1]) for a in maps],
            wrap,
            impedance,
            elastic,
            stage_time,
            ramp,
            stop_coverage=model.reference.payload["loading"]["stop_coverage"],
            diagnostic=diagnostic is not None,
            segment_substeps=tuple(args.segment_substeps),
            cache_mb=args.fft_cache_mb,
            fft_rows=args.fft_rows,
            recover=args.contact_root_recovery,
            kinetic_seed=args.kinetic_root_seed,
            failure_callback=save_failure,
        )
    finally:
        for array in maps:
            array._mmap.close()
    if any(fingerprint(path) != digest for path, digest in frozen.items()):
        raise ValueError("Compact Radau simulation source or frozen input changed")
    current = environment_record(options["case"])
    if any(current[k] != environment[k] for k in keys):
        raise ValueError("Physical case or numerical package changed")
    result.diagnostics.update(
        prepared_kernel=str(args.kernel.resolve()),
        kernel_summary_sha256=frozen[str(metadata_path.resolve())],
        frozen_inputs_sha256=frozen,
        initialization=lift.diagnostics,
        quadrature=options["quadrature"],
        panels=[model.reference.moving.count, model.reference.stationary.count],
        restart_seed_override=seed_record,
        estimated_solver_array_peak_mib=estimate / 1024**2,
        fft_cache_mb=args.fft_cache_mb,
        timing_seconds=dict(
            static_lift_and_kernel_load=initialization_seconds,
            time_stepping=result.diagnostics["time_stepping_seconds"],
            kernel_build=metadata["timing_seconds"]["total_before_reporting"],
            total_excluding_output=perf_counter() - begun,
        ),
    )
    output_start = perf_counter()
    summary = save_dynamic(
        result,
        model.reference,
        options["case"],
        output,
        contact_y=model.reference.moving.starts[model.moving_contact, 1],
        start_environment=environment,
    )
    np.savez(output / "stages.npz", **stages)
    summary["timing_seconds"].update(
        save_outputs=perf_counter() - output_start, total_including_output=perf_counter() - begun
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary["peak_process_rss_mb"] = rss / (1024**2 if sys.platform == "darwin" else 1024)
    write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ["environment", "case", "frozen_inputs_sha256"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

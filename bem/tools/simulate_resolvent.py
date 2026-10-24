"""Run a full nonlinear event from a prepared elastic-resolvent kernel.

The kernel may use a finer time grid than its geometry/reference run. Its
instantaneous impedance, full causal elastic startup and complete correction
history are used; no original-grid Z weights or trajectory replay are needed.
This remains an experimental solver, with the original previous-state law.
"""

import argparse
import ast
import hashlib
import json
import resource
import runpy
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.cq import BoundaryHistory, CQPlan
from finiteblock_bem.dynamics import DynamicResult
from finiteblock_bem.io import environment_record, fingerprint, save_dynamic, write_json
from finiteblock_bem.nodal import (
    NodalInterfaceEquilibrium,
    initialize_nodal_restart,
    load_nodal_model,
)

HELPER_PATH = Path(__file__).with_name("replay_resolvent.py")
HELPER = runpy.run_path(str(HELPER_PATH))


def simulate_prepared(
    model, reference, state, history, elastic, time, ramp, *, diagnostic=False, history_factory=None
):
    """Preserve original Newton roots, loading stop rule, and every nonzero lag."""
    from finiteblock_bem.jax_backend import JaxContactEquilibrium

    started = perf_counter()
    steps, d, n = len(time) - 1, history.plan.dofs, len(model.moving_contact)
    if elastic.shape != (steps + 1, d) or ramp.shape != (steps + 1,):
        raise ValueError("Elastic startup/loading dimensions do not match the time grid")
    if not np.allclose(np.diff(time), history.plan.dt, rtol=1e-12, atol=1e-15):
        raise ValueError("The prepared kernel requires a uniform matching time grid")
    weights = model.constraints.contact_weights
    law = model.reference.contact
    cdiag = np.r_[weights * law.normal_penalty, weights * law.tangential_penalty, 0]
    a0 = reference.matrix + np.diag(cdiag)
    composition = float(np.linalg.norm(history.instantaneous @ a0 - np.eye(d), ord=np.inf))
    if not np.isfinite(composition) or composition > 1e-6:
        raise ValueError(f"Instantaneous CQ composition failed: {composition}")
    system = JaxContactEquilibrium(reference)
    initial, _, contact = reference.evaluate(
        np.zeros(d), np.zeros(d), model.reference.normal_pressure, ramp[0], state, jacobian=False
    )
    if np.max(abs(initial)) >= 1e-7:
        raise ValueError("The initial static lift exceeds the equilibrium tolerance")
    fc0 = np.r_[contact.pressure * weights, contact.shear * weights, 0]
    q = np.zeros((steps + 1, d))
    p, tau, mu, z, c = [np.zeros((steps + 1, n)) for _ in range(5)]
    p[0], tau[0], mu[0], z[0], c[0] = (
        contact.pressure,
        contact.shear,
        contact.friction_coefficient,
        state.plastic_slip,
        state.cumulative_slip,
    )
    residual, iterations = np.zeros(steps + 1), np.zeros(steps + 1, dtype=int)
    residual[0] = np.max(abs(initial))
    memory = (HELPER["CorrectionHistory"] if history_factory is None else history_factory)(history)
    actuator = ramp.copy()
    force = reference.spring * (ramp - ramp[0])
    old, held, hold_time = state, None, None
    for index in range(1, steps + 1):
        if held is not None:
            actuator[index] = held
        past = -a0 @ (elastic[index] + memory.past(index))
        past[-1] += force[index]
        answer = system.solve(
            model.reference.normal_pressure, actuator[index], old, past=past, guess=q[index - 1]
        )
        if not answer.converged:
            raise ValueError(f"Resolvent equilibrium failed at step {index}: {answer.residual_mpa}")
        q[index], contact = answer.reduced_displacement, answer.contact
        p[index], tau[index], mu[index], z[index], c[index] = (
            contact.pressure,
            contact.shear,
            contact.friction_coefficient,
            contact.state.plastic_slip,
            contact.state.cumulative_slip,
        )
        residual[index], iterations[index] = answer.residual_mpa, answer.iterations
        correction = fc0 + cdiag * q[index] - np.r_[p[index] * weights, tau[index] * weights, 0]
        hold_force = reference.spring * (actuator[index] - ramp[index])
        correction[-1] += hold_force
        trial = law.tangential_penalty * (
            reference.base_ds + q[index, n : 2 * n] - old.plastic_slip
        )
        active = bool(
            hold_force != 0
            or np.any(p[index] <= 0)
            or np.any(abs(trial) > law.coefficient(old.cumulative_slip) * p[index])
            or np.any(contact.state.plastic_slip != old.plastic_slip)
        )
        memory.append(index, correction, active=active)
        old = contact.state
        if not diagnostic and held is None:
            fraction = np.average(c[index] >= law.characteristic_slip, weights=weights)
            if fraction >= model.reference.payload["loading"]["stop_coverage"]:
                held, hold_time = actuator[index], time[index]
    ref = model.reference
    diagnostics = dict(
        status="completed_requested_horizon",
        boundary_formulation="P1/P0 symmetric Galerkin; nodal-force interface condensation",
        history_representation="elastic_resolvent",
        schedule="resolvent_accelerated_diagnostic"
        if diagnostic
        else "event16_original_actuator_rate",
        dt_s=history.plan.dt,
        requested_steps=steps,
        completed_steps=steps,
        duration_s=float(time[-1]),
        history_dofs=[d],
        cq_fft_size=history.plan.fft_size,
        all_history_retained=True,
        full_boundary_fields_saved=False,
        nonlinear_backend="jax",
        panels=[ref.moving.count, ref.stationary.count],
        contact_nodes=n,
        max_cumulative_slip_mm=float(c.max()),
        max_boundary_contact_residual_mpa=float(residual.max()),
        actuator_hold_time_s=None if hold_time is None else float(hold_time),
        first_correction_step=memory.start,
        correction_history_steps=history.plan.steps,
        inverse_cq_composition_inf_error=composition,
        failure=None,
        time_stepping_seconds=perf_counter() - started,
        full_original_event_reproduced=False,
        scope="Full causal elastic startup and untruncated nonlinear correction; no FEM equivalence or speed advantage asserted",
    )
    if memory.online is not None and hasattr(memory.online, "statistics"):
        diagnostics["online_fft_cache"] = memory.online.statistics()
    return DynamicResult(
        time,
        None,
        None,
        p,
        tau,
        mu,
        z,
        c,
        actuator,
        reference.base_platen + q[:, -1],
        np.full(steps + 1, ref.normal_pressure),
        residual,
        iterations,
        diagnostics,
        q[:, : 2 * n] + np.r_[reference.base_dn, reference.base_ds],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-memory-mb", type=float, default=12288)
    parser.add_argument(
        "--fft-cache-mb", type=float, help="Cap retained kernel FFTs; zero streams all"
    )
    parser.add_argument("--fft-rows", type=int, default=4)
    args = parser.parse_args()
    started = perf_counter()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new empty output directory")
    metadata = json.loads((args.kernel / "summary.json").read_text())
    if metadata["status"] != "completed_resolvent_kernel_only":
        raise ValueError("A completed prepared resolvent kernel is required")
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    for name in ["source_sha256", "case_sha256", "restart_data_sha256"]:
        if environment[name] != metadata["environment"][name]:
            raise ValueError(f"Prepared kernel differs in {name}")
    math_node = next(
        node
        for node in ast.parse(HELPER_PATH.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "causal_elastic_baseline"
    )
    if (
        metadata["helper_math_sha256"]
        != hashlib.sha256(ast.dump(math_node, include_attributes=False).encode()).hexdigest()
    ):
        raise ValueError("Prepared kernel uses different elastic CQ mathematics")
    for filename, key in [("weights.npy", "weights_sha256"), ("elastic.npz", "elastic_sha256")]:
        if HELPER["digest_file"](args.kernel / filename) != metadata[key]:
            raise ValueError(f"Prepared data digest differs: {filename}")
    plan = CQPlan(**metadata["plan"])
    bounded, bounded_sha, history_factory = None, None, None
    bounded_path = Path(__file__).with_name("bounded_history.py")
    if args.fft_cache_mb is None:
        estimate = plan.estimated_peak_bytes + 24 * (metadata["full_steps"] + 1) * plan.dofs * 8
    else:
        bounded_sha = fingerprint(bounded_path)
        bounded = runpy.run_path(str(bounded_path))
        estimate = bounded["estimate_online_arrays"](
            plan, metadata["full_steps"], args.fft_cache_mb, fft_rows=args.fft_rows
        )
        history_factory = bounded["bounded_correction_class"](
            HELPER["CorrectionHistory"], cache_mb=args.fft_cache_mb, fft_rows=args.fft_rows
        )
    if estimate > args.max_memory_mb * 1024**2:
        raise ValueError(f"Estimated solver arrays need {estimate / 1024**2:.1f} MiB")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    if plan.dofs != 2 * len(model.moving_contact) + 1:
        raise ValueError("Prepared kernel dimensions differ from the contact model")
    weights = np.load(args.kernel / "weights.npy", mmap_mode="r", allow_pickle=False)
    if weights.shape != (plan.steps + 1, plan.dofs, plan.dofs) or weights.dtype != np.float64:
        raise ValueError("Invalid physical correction weights")
    impedance = np.load(args.kernel / "initial_impedance.npy", allow_pickle=False)
    if impedance.shape != (plan.dofs, plan.dofs) or not np.all(np.isfinite(impedance)):
        raise ValueError("Invalid instantaneous impedance")
    with np.load(args.kernel / "elastic.npz", allow_pickle=False) as data:
        elastic, time, ramp = data["elastic"], data["time"], data["uninterrupted_actuator"]
    np.testing.assert_array_equal(time, np.arange(metadata["full_steps"] + 1) * plan.dt)
    rate = model.reference.actuator_rate
    diagnostic = options.get("diagnostic_actuator_increment")
    if diagnostic is not None:
        rate = diagnostic / time[-1]
    np.testing.assert_allclose(
        ramp, model.reference.restart_actuator + rate * time, rtol=0, atol=1e-12
    )
    lift = initialize_nodal_restart(model, quadrature=options["quadrature"])
    reference = NodalInterfaceEquilibrium(model, lift, impedance)
    solver_sha, helper_sha = fingerprint(Path(__file__)), fingerprint(HELPER_PATH)
    initialization_seconds = perf_counter() - started
    result = simulate_prepared(
        model,
        reference,
        lift.state,
        BoundaryHistory(plan, weights),
        elastic,
        time,
        ramp,
        diagnostic=diagnostic is not None,
        history_factory=history_factory,
    )
    if solver_sha != fingerprint(Path(__file__)) or helper_sha != fingerprint(HELPER_PATH):
        raise ValueError("Experimental solver source changed during simulation")
    if bounded_sha is not None and bounded_sha != fingerprint(bounded_path):
        raise ValueError("Bounded history implementation changed during simulation")
    result.diagnostics.update(
        prepared_kernel=str(args.kernel.resolve()),
        kernel_summary_sha256=fingerprint(args.kernel / "summary.json"),
        initial_impedance_sha256=fingerprint(args.kernel / "initial_impedance.npy"),
        solver_sha256=solver_sha,
        helper_sha256=helper_sha,
        bounded_history_sha256=bounded_sha,
        fft_cache_mb=args.fft_cache_mb,
        estimated_solver_array_peak_mib=estimate / 1024**2,
        initialization=lift.diagnostics,
        quadrature=options["quadrature"],
        spectrum_storage=metadata["spectrum_storage"],
        frequency_sampling=metadata["frequency_sampling"],
        timing_seconds=dict(
            static_lift_and_kernel_load=initialization_seconds,
            time_stepping=result.diagnostics["time_stepping_seconds"],
            kernel_build=metadata["total_seconds"],
            total_excluding_output=perf_counter() - started,
        ),
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
    )
    answer = save_dynamic(
        args.output,
        model.contact_case(),
        result,
        options["case"],
        start_environment=environment,
        contact_y=model.reference.moving.starts[model.moving_contact, 1],
        contact_weights=model.constraints.contact_weights,
    )
    write_json(
        args.output / "run_manifest.json",
        dict(
            arguments={
                **options,
                "dt": plan.dt,
                "steps": metadata["full_steps"],
                "duration": float(time[-1]),
                "name": args.output.name,
                "output": str(args.output.resolve()),
            },
            prepared_kernel=str(args.kernel.resolve()),
            source_run=metadata["source_run"],
        ),
    )
    print(
        json.dumps(
            {key: value for key, value in answer.items() if key not in ["environment", "case"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

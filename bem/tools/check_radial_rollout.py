"""Check a full nonlinear CQ rollout and adjoint with the isolated radial assembler.

This uses fresh dense experimental weights and the unchanged production
time/contact solver. Only a private copy of that solver's history-builder
binding is replaced. Original files, module globals and baseline caches stay
untouched. Short diagnostic agreement is not FEM or full-event convergence.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter
from types import FunctionType, SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.adjoint import trajectory_adjoint
from finiteblock_bem.cq import BoundaryHistory, convolution_weights
from finiteblock_bem.dynamics import LoadingSchedule
from finiteblock_bem.io import environment_record, fingerprint, save_dynamic, write_json
from finiteblock_bem.nodal import (
    NodalInterfaceEquilibrium,
    NodalInterfaceTransfer,
    initialize_nodal_restart,
    load_nodal_model,
)
from finiteblock_bem.nodal_dynamics import plan_nodal_history, simulate_nodal

EXPERIMENT = Path(__file__).with_name("radial_traction_experiment.py")


def simulate_with_fixed_history(model, history, *, lift, schedule, max_memory_mb):
    """Use an explicit, checked history without changing any package binding."""

    def history_builder(candidate, dt, steps, **kwargs):
        if candidate is not model or (dt, steps) != (history.plan.dt, history.plan.steps):
            raise ValueError("Unexpected geometry/time grid requested")
        kwargs["diagnostics"].update(method="dense_isolated_radial_traction")
        return history

    clone = FunctionType(
        simulate_nodal.__code__,
        {**simulate_nodal.__globals__, "build_nodal_history": history_builder},
        simulate_nodal.__name__,
        simulate_nodal.__defaults__,
    )
    clone.__kwdefaults__ = dict(simulate_nodal.__kwdefaults__)
    return clone(
        model,
        history.plan.dt,
        history.plan.steps,
        lift=lift,
        schedule=schedule,
        backend="jax",
        quadrature=lift.diagnostics["quadrature"],
        max_memory_mb=max_memory_mb,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-memory-mb", type=float, default=2048)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new empty output directory")
    original = json.loads((args.original / "summary.json").read_text())
    options = json.loads((args.original / "run_manifest.json").read_text())["arguments"]
    if original["status"] != "completed_requested_horizon" or original["source_changed_during_run"]:
        raise ValueError("A complete source-frozen reference is required")
    environment = environment_record(options["case"])
    for name in ["source_sha256", "case_sha256", "restart_data_sha256"]:
        if original["environment"][name] != environment[name]:
            raise ValueError(f"Reference differs in {name}")
    script_sha, experiment_sha = fingerprint(Path(__file__)), fingerprint(EXPERIMENT)
    helper = runpy.run_path(str(EXPERIMENT))
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    plan, peak = plan_nodal_history(
        model,
        original["dt_s"],
        original["completed_steps"],
        workers=args.workers,
        quadrature=options["quadrature"],
        max_memory_mb=args.max_memory_mb,
    )
    increment = options.get("diagnostic_actuator_increment")
    schedule = (
        None
        if increment is None
        else LoadingSchedule(
            lambda _: model.reference.normal_pressure,
            lambda t: model.reference.restart_actuator + increment * t / (plan.dt * plan.steps),
            "isolated_radial_accelerated_diagnostic",
        )
    )
    with threadpool_limits(limits=1, user_api="blas"):
        lift = initialize_nodal_restart(model, quadrature=options["quadrature"])
        transfer = NodalInterfaceTransfer(model, quadrature=options["quadrature"], prepared=False)
        transfer.prepared = [
            helper["RadialTractionPrepared"](
                mesh, model.reference.material, quadrature=options["quadrature"], series="horner"
            )
            for mesh in [model.reference.moving, model.reference.stationary]
        ]
        tick = perf_counter()
        weights = convolution_weights(transfer.evaluate, plan, workers=args.workers)
        build_seconds = perf_counter() - tick
        del transfer
        history = BoundaryHistory(plan, weights)
        result = simulate_with_fixed_history(
            model, history, lift=lift, schedule=schedule, max_memory_mb=args.max_memory_mb
        )
        reference = NodalInterfaceEquilibrium(model, lift, history.instantaneous)
        gradient = trajectory_adjoint(
            model.contact_case(), lift, history, result, equilibrium=reference
        )
    fields = [
        "pressure",
        "shear",
        "plastic_slip",
        "cumulative_slip",
        "actuator",
        "platen",
        "contact_jumps",
    ]
    with np.load(args.original / "trajectory.npz", allow_pickle=False) as saved:
        differences = {key: float(np.max(abs(getattr(result, key) - saved[key]))) for key in fields}
        base = SimpleNamespace(
            **{key: saved[key] for key in ["contact_jumps", "plastic_slip", "cumulative_slip"]}
        )
    branch_function = runpy.run_path(str(Path(__file__).with_name("run_adjoint.py")))[
        "exact_contact_branches"
    ]
    base_branches, actual_branches = (
        branch_function(model.reference, base),
        branch_function(model.reference, result),
    )
    branch_changes = {
        key: int(np.count_nonzero(actual_branches[key] != base_branches[key]))
        for key in base_branches
    }
    comparisons = {}
    checks = []
    with np.load(args.original / "adjoint.npz", allow_pickle=False) as saved:
        for name in [
            "gradient",
            "actuator_gradient",
            "initial_plastic_slip_gradient",
            "initial_cumulative_slip_gradient",
        ]:
            actual, target = np.asarray(gradient[name]), saved[name]
            comparisons[name + "_max_abs_difference"] = float(np.max(abs(actual - target)))
            if name == "gradient":
                comparisons["parameter_relative_differences"] = (
                    abs(actual - target) / np.maximum(abs(target), 1e-15)
                ).tolist()
            checks.append((actual, target.copy()))
    if script_sha != fingerprint(Path(__file__)) or experiment_sha != fingerprint(EXPERIMENT):
        raise ValueError("Experimental code changed during rollout")
    result.diagnostics.update(
        experimental_operator="exact radial traction and Horner evaluation of the existing nine-term series",
        experimental_operator_sha256=experiment_sha,
        validator_sha256=script_sha,
        fresh_dense_kernel_build_seconds=build_seconds,
        estimated_array_peak_mib=peak / 1024**2,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    save_dynamic(
        args.output,
        model.contact_case(),
        result,
        options["case"],
        start_environment=environment,
        contact_y=model.reference.moving.starts[model.moving_contact, 1],
        contact_weights=model.constraints.contact_weights,
    )
    np.save(args.output / "weights.npy", weights, allow_pickle=False)
    np.savez(
        args.output / "adjoint.npz",
        **{
            key: np.asarray(gradient[key])
            for key in [
                "gradient",
                "actuator_gradient",
                "initial_plastic_slip_gradient",
                "initial_cumulative_slip_gradient",
            ]
        },
    )
    answer = dict(
        maximum_absolute_field_difference=differences,
        gradient_comparisons=comparisons,
        exact_branch_changes=branch_changes,
        source_run=str(args.original.resolve()),
        source_trajectory_sha256=fingerprint(args.original / "trajectory.npz"),
        source_adjoint_sha256=fingerprint(args.original / "adjoint.npz"),
        experimental_operator_sha256=experiment_sha,
        validator_sha256=script_sha,
        environment=environment,
        scope="Full requested reference horizon, fresh experimental dense CQ weights, original local solver and conditional adjoint; a short diagnostic is not the original 45 ms event",
    )
    write_json(args.output / "equivalence.json", answer)
    print(json.dumps({k: v for k, v in answer.items() if k != "environment"}, indent=2))
    if max(differences.values()) > 1e-5 or any(branch_changes.values()):
        raise ValueError("Experimental rollout changed fields or an exact contact branch")
    for actual, target in checks:
        np.testing.assert_allclose(actual, target, rtol=1e-5, atol=1e-12)


if __name__ == "__main__":
    main()

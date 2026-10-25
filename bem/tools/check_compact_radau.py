"""Reassemble original-Z RK-CQ for a small saved compact Radau diagnostic.

The reference uses no elastic baseline, compliance families or zero-prefix
rebasing. The unchanged coupled local contact law is shared; its separate
NumPy path and finite-difference tests are not replaced by this bulk check.
"""

import argparse
import json
import runpy
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.cq import BoundaryHistory, CQPlan
from finiteblock_bem.friction import ContactState
from finiteblock_bem.galerkin_fast import prepared_geometry_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import (
    NodalInterfaceEquilibrium,
    NodalInterfaceTransfer,
    initialize_nodal_restart,
    load_nodal_model,
)

TOOLS = Path(__file__).resolve().parent
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))
CONTACT = runpy.run_path(str(TOOLS / "radau_contact_experiment.py"))


def reference_stages(
    reference,
    history,
    stage_actuator,
    initial_state,
    segment_substeps,
    *,
    recover=False,
    kinetic_seed=False,
):
    """Direct original-impedance convolution under the recorded stage loading."""
    n, d = len(reference.constraints.contact_weights), reference.constraints.size
    wc = reference.constraints.contact_weights
    initial = reference.case.contact.evaluate(reference.base_dn, reference.base_ds, initial_state)
    fc0 = np.r_[wc * initial.pressure, wc * initial.shear, 0]
    springs = np.zeros((2 * d, 2 * d))
    springs[[d - 1, 2 * d - 1], [d - 1, 2 * d - 1]] = reference.spring
    system = CONTACT["RadauContactEquilibrium"](
        reference, history.instantaneous + springs, segment_substeps=segment_substeps
    )
    steps = len(stage_actuator) - 1
    q = np.zeros((steps + 1, 2 * d))
    values = np.zeros((steps + 1, n, 2, 7))
    initial_values = np.column_stack(
        (
            initial.pressure,
            initial.shear,
            initial_state.plastic_slip,
            initial_state.cumulative_slip,
            initial.friction_coefficient,
            np.zeros(n),
            np.zeros(n),
        )
    )
    values[0] = np.repeat(initial_values[:, None, :], 2, axis=1)
    residual = np.zeros(steps + 1)
    old = initial_state
    for k in range(1, steps + 1):
        forcing = history.past_traction(q, k) - np.tile(fc0, 2)
        forcing[[d - 1, 2 * d - 1]] -= reference.spring * (stage_actuator[k] - stage_actuator[0])
        answer = system.solve(
            old,
            forcing,
            np.tile(q[k - 1, -d:], 2),
            q[k - 1, -d:],
            recover=recover,
            kinetic_seed=kinetic_seed,
        )
        if not answer.converged:
            raise ValueError(
                f"Original-Z Radau reference failed at step {k}: {answer.residual_mpa}"
            )
        q[k], values[k], residual[k] = answer.displacement, answer.response, answer.residual_mpa
        old = ContactState(values[k, :, 1, 2], values[k, :, 1, 3])
    return q.reshape(steps + 1, 2, d), values, residual


def original_equation_residuals(reference, history, stage_actuator, q, values, initial_state):
    """Evaluate the fresh original equation on every saved compact stage."""
    d = reference.constraints.size
    wc = reference.constraints.contact_weights
    wr = np.tile(reference.constraints.reduced_weights, 2)
    initial = reference.case.contact.evaluate(reference.base_dn, reference.base_ds, initial_state)
    fc0 = np.tile(np.r_[wc * initial.pressure, wc * initial.shear, 0], 2)
    flat = q.reshape(len(q), 2 * d)
    residual = np.zeros(len(q))
    for k in range(1, len(q)):
        contact = np.concatenate(
            [np.r_[wc * values[k, :, a, 0], wc * values[k, :, a, 1], 0] for a in range(2)]
        )
        lhs = history.instantaneous @ flat[k] + history.past_traction(flat, k) + contact - fc0
        lhs[[d - 1, 2 * d - 1]] += reference.spring * (
            q[k, :, -1] - stage_actuator[k] + stage_actuator[0]
        )
        residual[k] = np.max(abs(lhs / wr))
    return residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-memory-mb", type=float, default=4096)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(
        Path("/Volumes/Gauss-T7/tatva").resolve()
    ):
        raise ValueError("Choose a new check directory outside the protected original Tatva")
    if args.workers < 1 or not np.isfinite(args.max_memory_mb) or args.max_memory_mb <= 0:
        raise ValueError("Positive finite reference resources are required")
    begun = perf_counter()
    summary_path = args.run / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("status") != "completed_requested_horizon"
        or summary.get("time_method") != "radau2"
        or summary.get("source_changed_during_run")
    ):
        raise ValueError("Require a complete frozen compact Radau diagnostic")
    kernel = Path(summary["prepared_kernel"])
    metadata_path = kernel / "summary.json"
    if fingerprint(metadata_path) != summary["kernel_summary_sha256"]:
        raise ValueError("Prepared kernel metadata changed")
    metadata = json.loads(metadata_path.read_text())
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    keys = ["source_sha256", "case_sha256", "restart_data_sha256"]
    if any(environment[k] != summary["environment"][k] for k in keys):
        raise ValueError("Saved stage run differs from current package or case")
    plan = replace(
        CQPlan(**metadata["plan"]),
        steps=summary["completed_steps"],
        dofs=2 * metadata["history_dofs"],
    )
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    paths = [
        Path(__file__),
        summary_path,
        metadata_path,
        args.run / "stages.npz",
        *[
            TOOLS / name
            for name in [
                "radau_experiment.py",
                "radau_contact_experiment.py",
                "substepped_contact_experiment.py",
                "robust_contact_experiment.py",
            ]
        ],
    ]
    seed = summary.get("restart_seed_override")
    if seed is not None:
        seed_tool = TOOLS / "restart_seed_experiment.py"
        model, current_seed = runpy.run_path(str(seed_tool))["apply_restart_seed"](
            model, Path(seed["path"])
        )
        if current_seed["sha256"] != seed["sha256"]:
            raise ValueError("Recorded restart seed changed")
        paths.extend([seed_tool, Path(seed["path"])])
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    estimate = (
        (plan.fft_size // 2 + 1) * plan.dofs**2 * 16
        + (plan.steps + 1) * plan.dofs**2 * 8
        + 512 * 1024**2
    )
    estimate += sum(
        prepared_geometry_bytes(mesh, options["quadrature"])
        for mesh in [model.reference.moving, model.reference.stationary]
    )
    if estimate > args.max_memory_mb * 1024**2:
        raise ValueError(
            f"Original-Z stage reference needs about {estimate / 1024**2:.1f} MiB; use a small diagnostic"
        )
    with np.load(args.run / "stages.npz", allow_pickle=False) as source:
        actual = {k: source[k] for k in ["displacement", "response", "stage_time", "actuator"]}
    lift = initialize_nodal_restart(model, quadrature=options["quadrature"])
    reference = NodalInterfaceEquilibrium(
        model, lift, np.zeros((metadata["history_dofs"], metadata["history_dofs"]))
    )
    transfer = NodalInterfaceTransfer(model, quadrature=options["quadrature"])
    weights = RADAU["radau_weights"](transfer.evaluate, plan, workers=args.workers)
    del transfer
    history = BoundaryHistory(plan, weights)
    q, response, residual = reference_stages(
        reference,
        history,
        actual["actuator"],
        lift.state,
        tuple(summary["segment_substeps"]),
        recover=True,
        kinetic_seed=True,
    )
    original_residual = original_equation_residuals(
        reference,
        history,
        actual["actuator"],
        actual["displacement"],
        actual["response"],
        lift.state,
    )
    differences = dict(displacement=float(np.max(abs(q - actual["displacement"]))))
    for field, name in enumerate(
        [
            "pressure",
            "shear",
            "plastic_slip",
            "cumulative_slip",
            "friction_coefficient",
            "micro_work",
            "integrated_strength_work",
        ]
    ):
        differences[name] = float(
            np.max(abs(response[:, :, :, field] - actual["response"][:, :, :, field]))
        )
    passed = bool(max(differences.values()) <= 1e-5 and np.max(original_residual) < 1e-7)
    if any(fingerprint(path) != digest for path, digest in frozen.items()):
        raise ValueError("Saved run or reference source changed during verification")
    if any(environment_record(options["case"])[k] != environment[k] for k in keys):
        raise ValueError("Package or case changed during verification")
    args.output.mkdir(parents=True)
    np.savez(
        args.output / "original_z_reference.npz",
        displacement=q,
        response=response,
        residual=residual,
        original_residual_on_compact=original_residual,
    )
    record = dict(
        status="passed_original_z_stage_reference"
        if passed
        else "failed_original_z_stage_reference",
        source_run=str(args.run.resolve()),
        steps=plan.steps,
        stage_dofs=plan.dofs,
        maximum_absolute_field_differences=differences,
        field_absolute_tolerance=1e-5,
        maximum_original_equation_residual_mpa=float(np.max(original_residual)),
        original_equation_tolerance_mpa=1e-7,
        maximum_reference_root_residual_mpa=float(residual.max()),
        reference_loading="Recorded stage actuators, including any recorded endpoint hold",
        requested_workers=args.workers,
        estimated_reference_array_peak_mib=estimate / 1024**2,
        frozen_input_sha256=frozen,
        environment=environment,
        total_seconds=perf_counter() - begun,
        full_original_event_reproduced=False,
        scope="Small real-BEM original-Z RK-CQ field and residual check; no full event, full adjoint or FEM speed advantage",
    )
    write_json(args.output / "summary.json", record)
    print(
        json.dumps(
            {k: v for k, v in record.items() if k not in ["environment", "frozen_input_sha256"]},
            indent=2,
        )
    )
    if not passed:
        raise ValueError(
            "Compact Radau diagnostic failed the unchanged field or original-equation guard"
        )


if __name__ == "__main__":
    main()

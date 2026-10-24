"""Audit fixed-jump Schur elimination on the same finite-block body matrices.

Very low frequencies diagnose the free-body inverse; the actual CQ contour
endpoint is also sampled. This is not a trajectory or nonlinear stability
certificate. No timing comparison is controlled when other jobs are running.
"""

import argparse
import json
import runpy
import warnings
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import bdf2_symbol, plan_cq
from finiteblock_bem.galerkin_fast import prepared_geometry_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--max-memory-mb", type=float, default=4096)
    parser.add_argument("--formulation", choices=["joint", "trace"], default="joint")
    parser.add_argument(
        "--frequencies", type=complex, nargs="+", default=[0, 1e-3, 1, 100, 200 + 2000j, 1e6 + 3e5j]
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new empty audit directory")
    if not np.isfinite(args.max_memory_mb) or args.max_memory_mb <= 0:
        raise ValueError("Positive finite array budget required")
    if any(not np.isfinite(s) or (s != 0 and s.real <= 0) for s in args.frequencies):
        raise ValueError("Use zero for static or positive-real Laplace frequencies")
    tools = Path(__file__).resolve().parent
    helper = tools / (
        "schur_condensation_experiment.py"
        if args.formulation == "joint"
        else "trace_schur_experiment.py"
    )
    paths = [Path(__file__), helper, tools / "audit_passivity.py"]
    if args.formulation == "trace":
        paths.append(tools / "schur_condensation_experiment.py")
    paths += [args.source / name for name in ["summary.json", "run_manifest.json"]]
    frozen = {str(path): fingerprint(path) for path in paths}
    source = json.loads((args.source / "summary.json").read_text())
    options = json.loads((args.source / "run_manifest.json").read_text())["arguments"]
    environment = environment_record(options["case"])
    keys = ["source_sha256", "case_sha256", "restart_data_sha256"]
    if (
        source["status"] != "completed_requested_horizon"
        or source["source_changed_during_run"]
        or any(environment[name] != source["environment"][name] for name in keys)
    ):
        raise ValueError("A completed source-frozen run is required")
    dt = source["dt_s"] if args.dt is None else args.dt
    duration = source["duration_s"] if args.duration is None else args.duration
    if not all(np.isfinite(value) and value > 0 for value in [dt, duration]):
        raise ValueError("Positive finite dt and duration required")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    d, m = 2 * len(model.moving_contact) + 1, model.constraints.size
    bodies = [model.reference.moving, model.reference.stationary]
    order = options["quadrature"]
    peak = sum(prepared_geometry_bytes(mesh, order) for mesh in bodies)
    peak += 24 * max(mesh.dofs**2 for mesh in bodies) * 16 + 20 * m**2 * 16 + 128 * 1024**2
    if peak > args.max_memory_mb * 1024**2:
        raise ValueError(f"Schur audit array estimate needs {peak / 1024**2:.1f} MiB")
    plan = plan_cq(d, dt, round(duration / dt), disk_spectrum=True, max_memory_mb=1e9)
    endpoint = complex(bdf2_symbol(plan.radius) / dt)
    frequencies = list(dict.fromkeys([*args.frequencies, endpoint]))
    line = np.r_[
        model.constraints.contact_weights,
        model.constraints.contact_weights,
        model.constraints.platen_width,
    ]
    metrics = runpy.run_path(str(tools / "audit_passivity.py"))["metrics"]
    records, schur, compliance = [], [], []
    args.output.mkdir(parents=True, exist_ok=True)
    begin = perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        name = "ConstrainedSchurTransfer" if args.formulation == "joint" else "TraceSchurTransfer"
        operator = runpy.run_path(str(helper))[name](model, quadrature=order)
        for s in frequencies:
            tick = perf_counter()
            k, parts = operator.assemble_reduced(None if s == 0 else s)
            z, u, eliminated = operator.condense(k)
            if not np.all(np.isfinite(z)) or not np.all(np.isfinite(u)):
                raise ValueError("Nonfinite constrained Schur response")
            failure = None
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    original = operator.original_compliance(parts)
                    if not np.all(np.isfinite(original)):
                        raise ValueError("Nonfinite original compliance response")
                except (ValueError, np.linalg.LinAlgError) as error:
                    original = np.full_like(z, np.nan)
                    failure = str(error)
            force = k @ u
            residual = np.linalg.norm(force - operator.jump.T @ z) / max(np.linalg.norm(force), 1)
            static = metrics(z, 1, line) if s == 0 else None
            row = dict(
                s_real=s.real,
                s_imag=s.imag,
                actual_cq_endpoint=s == endpoint,
                supported_stiffness_condition=float(np.linalg.cond(k)),
                body_stiffness_conditions=[float(np.linalg.cond(block)) for _, block in parts],
                eliminated_stiffness_condition=float(np.linalg.cond(eliminated)),
                all_elimination_block_conditions=[
                    float(np.linalg.cond(block))
                    for block in (
                        operator.elimination_blocks(k, eliminated)
                        if hasattr(operator, "elimination_blocks")
                        else [eliminated]
                    )
                ],
                full_equilibrium_relative_residual=float(residual),
                retained_jump_inf_error=float(
                    np.linalg.norm(operator.jump @ u - np.eye(d), ord=np.inf)
                ),
                schur_reciprocity_relative=float(np.linalg.norm(z - z.T) / np.linalg.norm(z)),
                original_compliance_relative_difference=None
                if failure
                else float(
                    np.linalg.norm((z - original) / line[:, None])
                    / np.linalg.norm(z / line[:, None])
                ),
                original_compliance_failure=failure,
                original_compliance_warnings=[str(warning.message) for warning in caught],
                static_stiffness_metrics=static,
                velocity_impedance_metrics=None if s == 0 else metrics(z, s, line),
                seconds=perf_counter() - tick,
            )
            records.append(row)
            schur.append(z)
            compliance.append(original)
            print(json.dumps(row), flush=True)
    current = environment_record(options["case"])
    if any(current[name] != environment[name] for name in keys) or any(
        fingerprint(path) != value for path, value in frozen.items()
    ):
        raise ValueError("Audit implementation, package or case changed")
    np.savez(
        args.output / "matrices.npz", frequencies=frequencies, schur=schur, compliance=compliance
    )
    write_json(
        args.output / "summary.json",
        dict(
            status="completed_sampled_condensation_audit",
            source_run=str(args.source.resolve()),
            source_mesh_arguments=options,
            contour_plan=asdict(plan),
            retained_dofs=d,
            formulation=args.formulation,
            full_reduced_dofs=m,
            nullspace_dofs=m - d,
            records=records,
            environment=environment,
            frozen_inputs_sha256=frozen,
            matrices_sha256=fingerprint(args.output / "matrices.npz"),
            estimated_array_peak_mib=peak / 1024**2,
            total_seconds=perf_counter() - begin,
            scope="Same body matrices, fixed-jump Schur condensation; finite samples only, no trajectory/FEM equivalence or nonlinear stability certificate",
            timing_scope="May overlap other jobs; not a controlled speed comparison",
        ),
    )


if __name__ == "__main__":
    main()

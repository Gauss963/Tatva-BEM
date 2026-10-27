"""Pair joint and two-stage trace elimination on identical finite-body matrices.

Raw Galerkin assembly is shared. Timed condensation calls alternate in order;
full reconstruction and supported equilibrium are checked outside those timers.
This samples linear transfers, not a complete CQ kernel or friction trajectory.
"""

import argparse
import hashlib
import runpy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import bdf2_symbol, plan_cq
from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators, prepared_geometry_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--h", type=float, default=10)
    parser.add_argument("--contact-h", type=float, default=2)
    parser.add_argument("--contact-end-h", type=float, default=0.015625)
    parser.add_argument("--corner-h", type=float, default=0.015625)
    parser.add_argument("--quadrature", type=int, default=12)
    parser.add_argument("--dt", type=float, default=2e-6)
    parser.add_argument("--duration", type=float, default=0.045)
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 0.1, np.pi])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-memory-mb", type=float, default=10240)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a fresh benchmark outside protected Tatva")
    if (
        args.repeats < 2
        or args.quadrature < 2
        or not np.isfinite(args.max_memory_mb)
        or args.max_memory_mb <= 0
        or not np.isfinite(args.dt)
        or args.dt <= 0
        or not np.isfinite(args.duration)
        or args.duration <= 0
        or any(not np.isfinite(a) or not 0 <= a <= np.pi for a in args.angles)
    ):
        raise ValueError("Require valid angles, discretization, budget and at least two repeats")
    paths = [
        Path(__file__),
        TOOLS / "schur_condensation_experiment.py",
        TOOLS / "trace_schur_experiment.py",
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    environment = environment_record(args.case)
    began = perf_counter()
    model = load_nodal_model(
        args.case,
        max_panel_length=args.h,
        contact_panel_length=args.contact_h,
        contact_end_length=args.contact_end_h,
        corner_panel_length=args.corner_h,
    )
    bodies = [model.reference.moving, model.reference.stationary]
    m, d = model.constraints.size, 2 * len(model.moving_contact) + 1
    estimate = sum(prepared_geometry_bytes(mesh, args.quadrature) for mesh in bodies)
    estimate += 24 * max(mesh.dofs**2 for mesh in bodies) * 16 + 20 * m**2 * 16 + 128 * 1024**2
    if estimate > args.max_memory_mb * 1024**2:
        raise ValueError(f"Paired Schur arrays need about {estimate / 1024**2:.1f} MiB")
    plan = plan_cq(
        d, args.dt, round(args.duration / args.dt), disk_spectrum=True, max_memory_mb=1e9
    )
    factories = {
        "joint": runpy.run_path(str(paths[1]))["ConstrainedSchurTransfer"],
        "trace": runpy.run_path(str(paths[2]))["TraceSchurTransfer"],
    }
    output.mkdir(parents=True)
    write_json(
        output / "run_manifest.json",
        dict(
            status="prepared_not_completed",
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            estimated_arrays_mib=estimate / 1024**2,
            environment=environment,
            frozen_input_sha256=frozen,
            original_tatva_readonly=True,
        ),
    )
    operators, preparation = {}, {}
    records, matrices = [], []
    weights = np.r_[
        model.constraints.contact_weights,
        model.constraints.contact_weights,
        model.constraints.platen_width,
    ]
    with threadpool_limits(limits=1, user_api="blas"):
        for name, factory in factories.items():
            tick = perf_counter()
            operators[name] = factory(model, quadrature=args.quadrature, prepared=False)
            preparation[name + "_basis_seconds"] = perf_counter() - tick
        tick = perf_counter()
        prepared = [
            PreparedGalerkinOperators(mesh, model.reference.material, quadrature=args.quadrature)
            for mesh in bodies
        ]
        preparation["shared_raw_geometry_seconds"] = perf_counter() - tick
        for angle in args.angles:
            s = complex(bdf2_symbol(plan.radius * np.exp(-1j * angle)) / plan.dt)
            print(f"Building shared raw operators at angle {angle:.12g}", flush=True)
            tick = perf_counter()
            raw = [p.assemble(s) for p in prepared]
            raw_seconds = perf_counter() - tick
            tick = perf_counter()
            k, _ = operators["joint"].assemble_reduced(s, operators=raw)
            assembly_seconds = perf_counter() - tick
            k.flags.writeable = False
            samples, result = {key: [] for key in factories}, {}
            orders = []
            for repeat in range(args.repeats):
                order = ["joint", "trace"] if repeat % 2 == 0 else ["trace", "joint"]
                orders.append(order)
                for name in order:
                    tick = perf_counter()
                    result[name] = operators[name].condense(k)
                    samples[name].append(perf_counter() - tick)
            z, u, _ = result["joint"]
            zz, uu, _ = result["trace"]
            if not all(np.all(np.isfinite(a)) for a in [z, u, zz, uu]):
                raise ValueError("Nonfinite paired Schur transfer")
            checks = {}
            for name in factories:
                impedance, response, _ = result[name]
                force = k @ response
                checks[name] = dict(
                    full_equilibrium_relative_residual=float(
                        np.linalg.norm(force - operators[name].jump.T @ impedance)
                        / max(np.linalg.norm(force), 1)
                    ),
                    jump_inf_error=float(
                        np.linalg.norm(operators[name].jump @ response - np.eye(d), ord=np.inf)
                    ),
                )
            weighted = float(
                np.linalg.norm((zz - z) / weights[:, None]) / np.linalg.norm(z / weights[:, None])
            )
            reconstruction = float(np.linalg.norm(uu - u) / np.linalg.norm(u))
            passed = (
                weighted <= 1e-9
                and reconstruction <= 1e-9
                and all(
                    c["full_equilibrium_relative_residual"] <= 1e-9 and c["jump_inf_error"] <= 1e-10
                    for c in checks.values()
                )
            )
            medians = {name: float(np.median(v)) for name, v in samples.items()}
            row = dict(
                angle=angle,
                s_real=s.real,
                s_imag=s.imag,
                raw_assembly_seconds=raw_seconds,
                shared_supported_stiffness_seconds=assembly_seconds,
                supported_stiffness_sha256=hashlib.sha256(k.tobytes()).hexdigest(),
                alternating_orders=orders,
                condensation_samples_seconds=samples,
                median_seconds=medians,
                joint_over_trace_median_ratio=medians["joint"] / medians["trace"],
                work_weighted_impedance_relative_difference=weighted,
                full_reconstruction_relative_difference=reconstruction,
                checks=checks,
                passed=passed,
            )
            records.append(row)
            matrices.append([z, zz])
            write_json(
                output / "progress.json", dict(completed_frequencies=len(records), records=records)
            )
            print(
                f"Angle {angle:.12g}: weighted Z difference {weighted:.3g}; joint/trace {row['joint_over_trace_median_ratio']:.3g}; passed={passed}",
                flush=True,
            )
            del raw, k, result, z, u, zz, uu
    current = environment_record(args.case)
    if any(
        current[k] != environment[k]
        for k in ["source_sha256", "case_sha256", "restart_data_sha256"]
    ) or any(fingerprint(p) != value for p, value in frozen.items()):
        raise ValueError("Paired Schur source or physical input changed")
    np.savez(output / "matrices.npz", matrices=matrices, angles=args.angles)
    passed = all(row["passed"] for row in records)
    write_json(
        output / "summary.json",
        dict(
            status="passed_paired_sampled_schur_condensation"
            if passed
            else "failed_paired_sampled_schur_condensation",
            records=records,
            basis_preparation=preparation,
            contour_plan=asdict(plan),
            full_reduced_dofs=m,
            retained_dofs=d,
            trace_dofs=len(operators["trace"].trace_indices),
            body_interiors=[len(ids) for ids in operators["trace"].body_interiors],
            common_trace_elimination_dofs=operators["trace"].trace_nullspace.shape[1],
            environment=environment,
            frozen_input_sha256=frozen,
            estimated_arrays_mib=estimate / 1024**2,
            matrices_sha256=fingerprint(output / "matrices.npz"),
            total_seconds=perf_counter() - began,
            timing_scope="Paired alternating one-BLAS-thread condensation only; shared assembly and preparation reported separately; external concurrency must be recorded",
            scope="Finite sampled linear transfers and reconstruction; not complete CQ, event or nonlinear stability validation",
            original_tatva_readonly=True,
            same_accuracy_fem_advantage_verified=False,
        ),
    )
    if not passed:
        raise ValueError("Paired Schur comparisons failed their original recorded tolerances")
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

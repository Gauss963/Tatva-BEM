"""Test a derived high-real-frequency limit of the fixed P1/P0 bulk operator.

For each flat face C=rho*(cp*n*n.T+cs*t*t.T). The proposed limits are
s*V -> L*C^-1/2, H -> M01/2, D/s -> M11(C)/2. Therefore the symmetric
mixed discrete DtN limit is half consistent P1 mass plus half its P0
projection, not the consistent P1 mass alone. These are hypotheses tested
against newly assembled BEM operators, not an assumed production damping.
"""

import argparse
import gc
import json
import os
import resource
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import block_diag, solve

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators, prepared_geometry_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model


def body_limits(mesh, material):
    n, d = mesh.count, mesh.dofs
    v, h, consistent = (np.zeros((d, d)) for _ in range(3))
    for i, (length, normal, tangent) in enumerate(
        zip(mesh.lengths, mesh.normals, mesh.tangents, strict=True)
    ):
        c = material.density * (
            material.cp * np.outer(normal, normal) + material.cs * np.outer(tangent, tangent)
        )
        row = slice(2 * i, 2 * i + 2)
        v[row, row] = length / 2 * np.linalg.inv(c)
        ids = np.r_[
            np.arange(2 * i, 2 * i + 2), np.arange(2 * ((i + 1) % n), 2 * ((i + 1) % n) + 2)
        ]
        h[row, ids] += np.tile(length / 4 * np.eye(2), (1, 2))
        consistent[np.ix_(ids, ids)] += np.kron(length / 6 * np.array([[2.0, 1.0], [1.0, 2.0]]), c)
    derivative = consistent / 2
    mixed = h.T @ solve(v, h, assume_a="pos") + derivative
    return dict(
        single_layer=v,
        trace_double_layer=h,
        hypersingular=derivative,
        mixed=mixed,
        consistent=consistent,
    )


def condense(model, blocks):
    p = model.constraints.prolongation
    jump = np.vstack(
        (model.constraints.normal_jump, model.constraints.tangential_jump, np.eye(p.shape[1])[-1])
    )
    stiffness = p.T @ block_diag(*blocks) @ p
    compliance = jump @ solve(stiffness, jump.T, check_finite=False)
    return solve(compliance, np.eye(len(jump)), check_finite=False)


def assemble_sample(prepared, limit, s):
    op = prepared.assemble(s)
    error = {
        "s_per_s": s,
        "sV_relative_frobenius": float(
            np.linalg.norm(s * op.single_layer - limit["single_layer"])
            / np.linalg.norm(limit["single_layer"])
        ),
        "H_relative_frobenius": float(
            np.linalg.norm(op.trace_double_layer - limit["trace_double_layer"])
            / np.linalg.norm(limit["trace_double_layer"])
        ),
        "D_over_s_relative_frobenius": float(
            np.linalg.norm(op.hypersingular / s - limit["hypersingular"])
            / np.linalg.norm(limit["hypersingular"])
        ),
    }
    k = op.force_stiffness()
    if not np.all(np.isfinite(k)):
        raise ValueError("Nonfinite high-frequency BEM matrix")
    return k, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--real-s", type=float, nargs="+", default=[1e4, 1e5, 1e6, 3e6])
    parser.add_argument("--quadratures", type=int, nargs="+", default=[12, 24])
    parser.add_argument("--workers", type=int, default=int(os.environ.get("FDBEM_CORES", "1")))
    parser.add_argument("--batch-pairs", type=int, default=8)
    parser.add_argument(
        "--uniform-panel-length",
        type=float,
        help="Explicit principal-limit diagnostic mesh override; clears end/corner/seed grading",
    )
    parser.add_argument("--max-memory-mb", type=float, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new radiation-limit diagnostic directory")
    if (
        any(not np.isfinite(s) or s <= 0 for s in args.real_s)
        or any(q < 4 or q % 2 for q in args.quadratures)
        or len(set(args.quadratures)) != len(args.quadratures)
        or not np.isfinite(args.max_memory_mb)
        or args.max_memory_mb <= 0
        or args.workers < 1
        or args.batch_pairs < 1
        or (
            args.uniform_panel_length is not None
            and (not np.isfinite(args.uniform_panel_length) or args.uniform_panel_length <= 0)
        )
    ):
        raise ValueError(
            "Require positive real frequencies, distinct even quadratures and a finite budget"
        )
    begun = perf_counter()
    metadata_path = args.source / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    keys = ["source_sha256", "case_sha256", "restart_data_sha256"]
    if any(environment[k] != metadata["environment"][k] for k in keys):
        raise ValueError("Prepared kernel package or physical case differs")
    frozen = {str(p.resolve()): fingerprint(p) for p in [metadata_path, Path(__file__)]}
    mesh_arguments = {
        out: options.get(key)
        for out, key in [
            ("max_panel_length", "h"),
            ("contact_panel_length", "contact_h"),
            ("contact_end_length", "contact_end_h"),
            ("corner_panel_length", "corner_h"),
            ("restart_knot_window", "restart_knot_window"),
            ("restart_knot_subdivisions", "restart_knot_subdivisions"),
        ]
    }
    if args.uniform_panel_length is not None:
        mesh_arguments.update(
            max_panel_length=args.uniform_panel_length,
            contact_panel_length=args.uniform_panel_length,
            contact_end_length=None,
            corner_panel_length=None,
            restart_knot_window=None,
            restart_knot_subdivisions=1,
        )
    model = load_nodal_model(options["case"], **mesh_arguments)
    meshes = [model.reference.moving, model.reference.stationary]
    estimates = {
        q: max(
            prepared_geometry_bytes(m, q)
            + args.workers * (m.dofs**2 * 256 + 128 * 1024**2 + 2 * args.batch_pairs * q**2 * 1024)
            + len(args.real_s) * m.dofs**2 * 16
            for m in meshes
        )
        for q in args.quadratures
    }
    if max(estimates.values()) > args.max_memory_mb * 1024**2:
        raise ValueError("Prepared-body array estimate exceeds the requested budget")
    limits = [body_limits(m, model.reference.material) for m in meshes]
    target = condense(model, [v["mixed"] for v in limits])
    naive = condense(model, [v["consistent"] for v in limits])
    eig, vectors = np.linalg.eigh(target)
    if eig[0] <= 0:
        raise ValueError("Derived condensed radiation matrix is not positive definite")
    invroot = (vectors / np.sqrt(eig)) @ vectors.T
    args.output.mkdir(parents=True)
    records, impedance_arrays = [], {}
    for quadrature in args.quadratures:
        body_operators = [[] for _ in args.real_s]
        body_errors = []
        for mesh, limit in zip(meshes, limits, strict=True):
            prepared = PreparedGalerkinOperators(
                mesh, model.reference.material, quadrature=quadrature, batch_pairs=args.batch_pairs
            )
            errors = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                samples = pool.map(partial(assemble_sample, prepared, limit), args.real_s)
                for i, (k, error) in enumerate(samples):
                    body_operators[i].append(k)
                    errors.append(error)
            body_errors.append(errors)
            del prepared
            gc.collect()
        point_records = []
        impedances = []
        for s, blocks in zip(args.real_s, body_operators, strict=True):
            z = condense(model, blocks) / s
            impedances.append(z)
            point_records.append(
                dict(
                    s_per_s=s,
                    mixed_limit_relative_frobenius=float(
                        np.linalg.norm(z - target) / np.linalg.norm(target)
                    ),
                    mixed_limit_relative_energy_operator_norm=float(
                        np.linalg.norm(invroot @ (z - target) @ invroot, 2)
                    ),
                    consistent_mass_relative_frobenius=float(
                        np.linalg.norm(z - naive) / np.linalg.norm(naive)
                    ),
                )
            )
        records.append(
            dict(quadrature=quadrature, points=point_records, body_operator_errors=body_errors)
        )
        impedance_arrays[f"q{quadrature}"] = np.array(impedances)
        write_json(args.output / "progress.json", dict(records=records))
    reference = impedance_arrays[f"q{max(args.quadratures)}"]
    for record in records:
        actual = impedance_arrays[f"q{record['quadrature']}"]
        record["relative_difference_from_largest_quadrature"] = [
            float(np.linalg.norm(a - b) / np.linalg.norm(b))
            for a, b in zip(actual, reference, strict=True)
        ]
    current = environment_record(options["case"])
    if any(current[k] != environment[k] for k in keys) or any(
        fingerprint(p) != v for p, v in frozen.items()
    ):
        raise ValueError("Radiation diagnostic source or case changed")
    np.savez(
        args.output / "operators.npz",
        mixed_limit=target,
        consistent_mass_limit=naive,
        real_s=args.real_s,
        **impedance_arrays,
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**2 if sys.platform == "darwin" else 1024
    )
    result = dict(
        status="completed_radiation_limit_diagnostic_not_a_production_split",
        records=records,
        history_dofs=len(target),
        mesh_arguments=mesh_arguments,
        boundary_panels=[m.count for m in meshes],
        frequency_workers=args.workers,
        batch_pairs=args.batch_pairs,
        minimum_derived_radiation_eigenvalue=float(eig[0]),
        estimated_prepared_body_array_mib={str(k): v / 1024**2 for k, v in estimates.items()},
        array_estimate_scope="Prepared geometry, concurrent frequency matrices and batch scratch; not an OS RSS limit",
        peak_process_rss_mib=rss,
        total_seconds_before_reporting=perf_counter() - begun,
        environment=environment,
        frozen_input_sha256=frozen,
        operators_sha256=fingerprint(args.output / "operators.npz"),
        scope="Positive-real-frequency, fixed-space principal-limit hypothesis and quadrature diagnostic only; no history discarded, time integrator changed or nonlinear event/stability/AD/speed advantage established",
        original_tatva_readonly=True,
        same_accuracy_fem_advantage_verified=False,
    )
    write_json(args.output / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "environment"}, indent=2))


if __name__ == "__main__":
    main()

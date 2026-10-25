"""Separate auxiliary-density resolution from the retained P1 trace mesh.

Only a static experiment: subdivide panels near geometric corners for P0
density, while restricting all refined P1 traces to the original P1 space.
This does not add contact/history coordinates or change the friction law.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import block_diag, solve
from threadpoolctl import threadpool_limits

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators, prepared_geometry_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.mesh import BoundaryMesh
from finiteblock_bem.nodal import load_nodal_model

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = Path(__file__).with_name("galerkin_audit.py")
static_record = runpy.run_path(str(AUDIT_PATH))["static_record"]


def refine_density_mesh(mesh, factor, radius=16):
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
        raise ValueError("Density subdivision factor must be a positive integer")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Corner refinement radius must be finite and positive")
    corners = mesh.starts[
        np.linalg.norm(mesh.tangents - np.roll(mesh.tangents, 1, axis=0), axis=1) > 1e-10
    ]
    starts, labels, rows = [], [], []
    for index, (a, b) in enumerate(zip(mesh.starts, mesh.ends, strict=True)):
        near = np.min(np.linalg.norm((a + b)[None, :] / 2 - corners, axis=1)) < radius
        pieces = factor if near else 1
        for fraction in np.arange(pieces) / pieces:
            starts.append(a + fraction * (b - a))
            labels.append(mesh.labels[index])
            row = np.zeros(mesh.count)
            row[index], row[(index + 1) % mesh.count] = 1 - fraction, fraction
            rows.append(row)
    refined = BoundaryMesh(starts, np.roll(starts, -1, axis=0), labels)
    prolongation = np.kron(np.array(rows), np.eye(2))
    np.testing.assert_allclose(
        (prolongation @ mesh.starts.ravel()).reshape(-1, 2), refined.starts, atol=1e-12
    )
    return refined, prolongation


def restricted_density_stiffness(
    mesh,
    material,
    factor,
    *,
    radius=16,
    quadrature=12,
    reference_derivative=None,
    max_memory_mb=4096,
):
    """Refine P0 density and restrict the P1 weak forms to the original trace."""
    if not np.isfinite(max_memory_mb) or max_memory_mb <= 0:
        raise ValueError("Require a finite positive density-assembly array budget")
    if not isinstance(quadrature, int) or quadrature < 4 or quadrature % 2:
        raise ValueError("Quadrature must be even and at least four")
    refined, p = refine_density_mesh(mesh, factor, radius)
    nr, nt = refined.dofs, mesh.dofs
    estimate = prepared_geometry_bytes(refined, quadrature)
    estimate += (12 * nr**2 + 8 * nr * nt + 8 * nt**2) * 8 + 128 * 1024**2
    if estimate > max_memory_mb * 1024**2:
        raise ValueError(f"Density assembly array estimate is {estimate / 1024**2:.1f} MiB")
    prepared = PreparedGalerkinOperators(refined, material, quadrature=quadrature)
    op = prepared.assemble()
    h, d = op.trace_double_layer @ p, p.T @ op.hypersingular @ p
    error = (
        0.0
        if reference_derivative is None
        else float(np.linalg.norm(d - reference_derivative) / np.linalg.norm(reference_derivative))
    )
    if not np.isfinite(error) or error > 1e-7:
        raise ValueError(f"Restricted hypersingular form changed: relative difference {error}")
    stiffness = h.T @ solve(op.single_layer, h, check_finite=False) + d
    return (
        stiffness,
        d,
        dict(
            density_panels=refined.count,
            trace_panels=mesh.count,
            relative_hypersingular_restriction_error=error,
            estimated_body_array_peak_mib=estimate / 1024**2,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", type=Path, default=ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    )
    parser.add_argument("--h", type=float, default=20)
    parser.add_argument("--contact-h", type=float, default=5)
    parser.add_argument("--contact-end-h", type=float, default=2)
    parser.add_argument("--corner-h", type=float, default=2)
    parser.add_argument("--restart-knot-window", type=float)
    parser.add_argument("--restart-knot-subdivisions", type=int, default=1)
    parser.add_argument("--quadrature", type=int, default=12)
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--radius", type=float, default=16)
    parser.add_argument("--max-memory-mb", type=float, default=4096)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/galerkin-density")
    args = parser.parse_args()
    started = perf_counter()
    if args.factors != sorted(set(args.factors)) or args.factors[0] != 1:
        raise ValueError(
            "Use increasing positive density factors starting with the reference factor 1"
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new empty density-study directory")
    frozen = {str(p.resolve()): fingerprint(p) for p in [Path(__file__), AUDIT_PATH, args.case]}
    environment = environment_record(args.case)
    model = load_nodal_model(
        args.case,
        max_panel_length=args.h,
        contact_panel_length=args.contact_h,
        contact_end_length=args.contact_end_h,
        corner_panel_length=args.corner_h,
        restart_knot_window=args.restart_knot_window,
        restart_knot_subdivisions=args.restart_knot_subdivisions,
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    records, reference_derivative = [], []
    with threadpool_limits(limits=1):
        for factor in args.factors:
            factor_started = perf_counter()
            stiffness, body_records = [], []
            for body, mesh in enumerate([model.reference.moving, model.reference.stationary]):
                k, d, detail = restricted_density_stiffness(
                    mesh,
                    model.reference.material,
                    factor,
                    radius=args.radius,
                    quadrature=args.quadrature,
                    max_memory_mb=args.max_memory_mb,
                    reference_derivative=None if factor == 1 else reference_derivative[body],
                )
                stiffness.append(k)
                body_records.append(detail)
                if factor == 1:
                    reference_derivative.append(d)
            assembly_seconds = perf_counter() - factor_started
            record = dict(
                density_factor=factor,
                density_panels=[r["density_panels"] for r in body_records],
                trace_panels=[model.reference.moving.count, model.reference.stationary.count],
                relative_hypersingular_restriction_error=[
                    r["relative_hypersingular_restriction_error"] for r in body_records
                ],
                body_records=body_records,
            )
            record.update(
                static_record(
                    model,
                    args.quadrature,
                    output / f"factor{factor}.csv",
                    stiffness=block_diag(*stiffness),
                )
            )
            record.update(
                density_assembly_seconds=assembly_seconds,
                construction_and_static_seconds=perf_counter() - factor_started,
                seconds_scope="The existing seconds field times the static solve only; construction_and_static_seconds includes density assembly",
            )
            records.append(record)
            print(json.dumps(record), flush=True)
    current = environment_record(args.case)
    keys = ["source_sha256", "case_sha256", "restart_data_sha256"]
    if any(current[key] != environment[key] for key in keys) or any(
        fingerprint(p) != digest for p, digest in frozen.items()
    ):
        raise ValueError("Density-study implementation, physical case or seed changed")
    write_json(
        output / "summary.json",
        dict(
            records=records,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            environment=environment,
            frozen_input_sha256=frozen,
            retained_history_dofs=2 * len(model.moving_contact) + 1,
            full_original_event_reproduced=False,
            total_seconds_before_reporting=perf_counter() - started,
            scope="Static density-space refinement at fixed trace/contact mesh, not dynamic rupture",
        ),
    )


if __name__ == "__main__":
    main()

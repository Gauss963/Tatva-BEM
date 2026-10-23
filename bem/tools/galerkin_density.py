"""Separate auxiliary-density resolution from the retained P1 trace mesh.

Only a static experiment: subdivide panels near geometric corners for P0
density, while restricting all refined P1 traces to the original P1 space.
This does not add contact/history coordinates or change the friction law.
"""

import json
from pathlib import Path

import numpy as np
from galerkin_audit import static_record
from scipy.linalg import block_diag, solve
from threadpoolctl import threadpool_limits

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.mesh import BoundaryMesh
from finiteblock_bem.nodal import load_nodal_model

ROOT = Path(__file__).resolve().parents[1]


def refine_density_mesh(mesh, factor, radius=16):
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


def main():
    model = load_nodal_model(
        ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml",
        max_panel_length=20,
        contact_panel_length=5,
        contact_end_length=2,
        corner_panel_length=2,
    )
    output = ROOT / "outputs/galerkin-density"
    output.mkdir(parents=True, exist_ok=True)
    records, reference_derivative = [], []
    with threadpool_limits(limits=1):
        for factor in [1, 2, 4]:
            stiffness, counts, derivative_errors = [], [], []
            for body, mesh in enumerate([model.reference.moving, model.reference.stationary]):
                refined, p = refine_density_mesh(mesh, factor)
                prepared = PreparedGalerkinOperators(
                    refined, model.reference.material, quadrature=12
                )
                op = prepared.assemble()
                h, d = op.trace_double_layer @ p, p.T @ op.hypersingular @ p
                stiffness.append(h.T @ solve(op.single_layer, h, check_finite=False) + d)
                counts.append(refined.count)
                if factor == 1:
                    reference_derivative.append(d)
                derivative_errors.append(
                    float(
                        np.linalg.norm(d - reference_derivative[body])
                        / np.linalg.norm(reference_derivative[body])
                    )
                )
                # Restriction preserves the trace weak form; only V/H density
                # approximation is refined. Reject a bad interpolation/quadrature.
                if derivative_errors[-1] > 1e-7:
                    raise ValueError("Restricted hypersingular form changed unexpectedly")
                del prepared, op
            record = dict(
                density_factor=factor,
                density_panels=counts,
                trace_panels=[model.reference.moving.count, model.reference.stationary.count],
                relative_hypersingular_restriction_error=derivative_errors,
            )
            record.update(
                static_record(
                    model, 12, output / f"factor{factor}.csv", stiffness=block_diag(*stiffness)
                )
            )
            records.append(record)
            print(json.dumps(record), flush=True)
    write_json(
        output / "summary.json",
        dict(
            records=records,
            environment=environment_record(),
            scope="Static density-space refinement at fixed trace/contact mesh, not dynamic rupture",
        ),
    )


if __name__ == "__main__":
    main()

"""Sample reciprocity and positive-real velocity impedance of the BULK only.

For physical Laplace-domain force=Z(s)*displacement, the velocity impedance is
Z(s)/s. Its Hermitian part should be positive semidefinite in work-conjugate
coordinates for Re(s)>0. A negative eigenvalue disproves sampled passivity;
positive samples do not prove it everywhere or certify nonlinear stability.
No symmetrization or eigenvalue clipping is applied to the simulation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.case import load_case
from finiteblock_bem.condensation import InterfaceTransfer
from finiteblock_bem.io import environment_record, write_json

ROOT = Path(__file__).resolve().parents[1]


def metrics(impedance, s, row_weights):
    normalizer = 1 / np.sqrt(row_weights)
    velocity = impedance / s * normalizer[:, None] * normalizer[None, :]
    hermitian = (velocity + velocity.conj().T) / 2
    eigenvalues = np.linalg.eigvalsh(hermitian)
    spectral = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    return {
        "min_hermitian_eigenvalue": float(eigenvalues[0]),
        "max_hermitian_eigenvalue": float(eigenvalues[-1]),
        "min_over_spectral_radius": float(eigenvalues[0] / spectral),
        "reciprocity_relative_frobenius": float(
            np.linalg.norm(velocity - velocity.T)
            / max(np.linalg.norm(velocity), np.finfo(float).tiny)
        ),
        "negative_directions_below_relative_1e_8": int(
            np.count_nonzero(eigenvalues < -1e-8 * spectral)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrature-check", action="store_true")
    args = parser.parse_args()
    # Independent analytic checks of the audit's sign and displacement/velocity convention.
    assert metrics(np.eye(2) * (2 + 3j), 2 + 3j, np.ones(2))["min_hermitian_eigenvalue"] == 1
    assert (
        metrics(-np.eye(2) * (2 + 3j), 2 + 3j, np.ones(2))[
            "negative_directions_below_relative_1e_8"
        ]
        == 2
    )
    case_path = ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    output = ROOT / "outputs/passivity-audit"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    if args.quadrature_check:
        case = load_case(
            case_path,
            max_panel_length=20,
            contact_panel_length=5,
            contact_end_length=2,
            corner_panel_length=2,
        )
        wc = case.moving.lengths[case.moving_contact]
        s = 200 + 2j * np.pi * 5000
        from scipy.linalg import block_diag

        from finiteblock_bem.operators import PreparedBoundaryOperators

        with threadpool_limits(limits=1):
            for order in [12, 24, 48]:
                transfer = InterfaceTransfer(case, quadrature=order)
                weights = np.r_[wc, wc, transfer.constraints.platen_width]
                impedance, reconstruct = transfer.evaluate(s, reconstruction=True)
                d = 1 / np.sqrt(weights)
                velocity = impedance / s * d[:, None] * d[None, :]
                eigenvalues, directions = np.linalg.eigh((velocity + velocity.conj().T) / 2)
                v = d * directions[:, 0]
                u = reconstruct @ (v / s)
                full = block_diag(
                    *(
                        PreparedBoundaryOperators(m, case.material, quadrature=order)
                        .evaluate(s)
                        .impedance()
                        for m in [case.moving, case.stationary]
                    )
                )
                full_weights = np.repeat(np.r_[case.moving.lengths, case.stationary.lengths], 2)
                power = float(np.vdot(s * u, full_weights * (full @ u)).real)
                record = dict(
                    quadrature=order,
                    full_boundary_power=power,
                    eigenvector_power=float(eigenvalues[0]),
                    **metrics(impedance, s, weights),
                )
                np.testing.assert_allclose(power, eigenvalues[0], rtol=1e-7, atol=1e-10)
                records.append(record)
                print(json.dumps(record), flush=True)
        write_json(
            output / "quadrature.json",
            {"records": records, "environment": environment_record(case_path)},
        )
        return
    with threadpool_limits(limits=1):
        for label, controls in [
            ("uniform20", dict(max_panel_length=20)),
            (
                "corners2-contact20",
                dict(
                    max_panel_length=40,
                    contact_panel_length=20,
                    contact_end_length=2,
                    corner_panel_length=2,
                ),
            ),
            (
                "corners2-contact5",
                dict(
                    max_panel_length=20,
                    contact_panel_length=5,
                    contact_end_length=2,
                    corner_panel_length=2,
                ),
            ),
        ]:
            case = load_case(case_path, **controls)
            transfer = InterfaceTransfer(case, quadrature=12)
            wc = case.moving.lengths[case.moving_contact]
            weights = np.r_[wc, wc, transfer.constraints.platen_width]
            for sigma in [200.0, 2000.0]:
                for frequency in [0.0, 1000.0, 5000.0, 10000.0, 30000.0, 60000.0, 120000.0]:
                    s = sigma + 2j * np.pi * frequency
                    record = dict(
                        mesh=label,
                        sigma_per_s=sigma,
                        frequency_hz=frequency,
                        **metrics(transfer.evaluate(s), s, weights),
                    )
                    records.append(record)
                    print(json.dumps(record), flush=True)
    write_json(
        output / "summary.json",
        {
            "scope": "Sampled linear bulk velocity impedance; not a nonlinear contact stability certificate",
            "records": records,
            "environment": environment_record(case_path),
        },
    )


if __name__ == "__main__":
    main()

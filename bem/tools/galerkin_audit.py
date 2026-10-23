"""Audit the new Galerkin bulk and nodal static lift; no rupture claim."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.dynamics import RestartProjectionError
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.nodal import (
    NodalContactEquilibrium,
    NodalInterfaceTransfer,
    initialize_nodal_restart,
    load_nodal_model,
    nodal_body_stiffness,
)

ROOT = Path(__file__).resolve().parents[1]


def static_record(model, quadrature, output):
    start = perf_counter()
    ref, con = model.reference, model.constraints
    stiffness = nodal_body_stiffness(model, quadrature=quadrature)
    lift = initialize_nodal_restart(model, quadrature=quadrature, stiffness=stiffness)
    system = NodalContactEquilibrium(model, stiffness)
    guess = con.prolongation.T @ lift.displacement / np.sum(con.prolongation**2, axis=0)
    zeros = np.zeros(len(lift.displacement))
    base = system.evaluate(
        guess, zeros, ref.normal_pressure, ref.restart_actuator, lift.state, frozen=True
    )[2]
    delta = 0.01
    other = system.solve(
        ref.normal_pressure, ref.restart_actuator + delta, lift.state, guess=guess, frozen=True
    )
    if not other.converged:
        raise RuntimeError("Frozen-plastic tangent solve failed")
    p, tau = base.pressure, base.shear
    dp, dtau = (other.contact.pressure - p) / delta, (other.contact.shear - tau) / delta
    mu = ref.contact.coefficient(lift.state.cumulative_slip)
    onset = np.full(len(p), np.inf)
    for gap, slope in [(mu * p - tau, dtau - mu * dp), (mu * p + tau, -dtau - mu * dp)]:
        trial = np.full(len(p), np.inf)
        np.divide(gap, slope, out=trial, where=(slope > 0) & (gap >= 0))
        trial[(p + dp * trial <= 0) | (trial <= 0)] = np.inf
        onset = np.minimum(onset, trial)
    opening = np.full(len(p), np.inf)
    np.divide(-p, dp, out=opening, where=dp < 0)
    index = int(np.argmin(onset))
    y = ref.moving.starts[model.moving_contact, 1]
    reference = np.genfromtxt(
        ROOT / "src/finiteblock_bem/resources/data/ev16_restart_interface.csv",
        delimiter=",",
        names=True,
    )
    errors = {}
    for key, values in [("p", p), ("tau", tau)]:
        target = np.interp(y, reference["y"], reference[key])
        errors[key] = float(
            np.sqrt(
                np.dot(con.contact_weights, (values - target) ** 2)
                / np.dot(con.contact_weights, target**2)
            )
        )
    table = np.column_stack((y, p, tau, lift.state.plastic_slip, onset / ref.actuator_rate))
    np.savetxt(
        output,
        table,
        delimiter=",",
        comments="",
        header="y_mm,p_mpa,tau_mpa,plastic_slip_mm,elastic_yield_time_s",
    )
    return {
        "initialization": lift.diagnostics,
        "predicted_first_y_mm": float(y[index]),
        "predicted_shear_time_s": float(onset[index] / ref.actuator_rate),
        "prediction_valid": bool(np.isfinite(onset[index]) and np.min(opening) > onset[index]),
        "initial_profile_relative_l2": errors,
        "seconds": perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--refine-corners", action="store_true")
    args = parser.parse_args()
    case_path = ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    folder = "galerkin-corner-refinement" if args.refine_corners else "galerkin-audit"
    if args.quick:
        folder += "-quick"
    output = ROOT / "outputs" / folder
    output.mkdir(parents=True, exist_ok=True)
    controls = [
        ("uniform80", dict(max_panel_length=80), 12),
        ("uniform40", dict(max_panel_length=40), 12),
        ("uniform20", dict(max_panel_length=20), 12),
        (
            "corners2-contact20",
            dict(
                max_panel_length=40,
                contact_panel_length=20,
                contact_end_length=2,
                corner_panel_length=2,
            ),
            12,
        ),
        (
            "corners2-contact20-q24",
            dict(
                max_panel_length=40,
                contact_panel_length=20,
                contact_end_length=2,
                corner_panel_length=2,
            ),
            24,
        ),
        (
            "corners2-contact5",
            dict(
                max_panel_length=20,
                contact_panel_length=5,
                contact_end_length=2,
                corner_panel_length=2,
            ),
            12,
        ),
    ]
    if args.refine_corners:
        controls = [
            (
                f"corner{end}-contact5",
                dict(
                    max_panel_length=20,
                    contact_panel_length=5,
                    contact_end_length=end,
                    corner_panel_length=end,
                ),
                16,
            )
            for end in [2, 1, 0.5]
        ]
    records = []
    with threadpool_limits(limits=1):
        for name, mesh_options, quadrature in controls[:2] if args.quick else controls:
            model = load_nodal_model(case_path, **mesh_options)
            record = {
                "name": name,
                "mesh_options": mesh_options,
                "quadrature": quadrature,
                "panels": [model.reference.moving.count, model.reference.stationary.count],
                "contact_nodes": len(model.moving_contact),
            }
            try:
                record.update(static_record(model, quadrature, output / f"{name}.csv"))
            except RestartProjectionError as error:
                record.update(initialization=error.diagnostics, prediction_valid=False)
            if not args.static_only and not name.endswith("q24"):
                transfer = NodalInterfaceTransfer(model, quadrature=quadrature)
                weights = np.r_[
                    model.constraints.contact_weights,
                    model.constraints.contact_weights,
                    model.constraints.platen_width,
                ]
                records_s = []
                for frequency in [1000.0, 5000.0, 10000.0]:
                    s = 200 + 2j * np.pi * frequency
                    z = transfer.evaluate(s) / np.sqrt(weights[:, None] * weights[None, :])
                    velocity = z / s
                    spectrum = np.linalg.eigvalsh((velocity + velocity.conj().T) / 2)
                    records_s.append(
                        {
                            "frequency_hz": frequency,
                            "sigma_per_s": 200.0,
                            "min_velocity_hermitian_eigenvalue": float(spectrum[0]),
                            "min_over_spectral_radius": float(
                                spectrum[0] / np.max(np.abs(spectrum))
                            ),
                            "relative_reciprocity_defect": float(
                                np.linalg.norm(z - z.T) / np.linalg.norm(z)
                            ),
                        }
                    )
                record["passivity_samples"] = records_s
            records.append(record)
            print(json.dumps(record), flush=True)
            write_json(
                output / "summary.json",
                {
                    "scope": "Symmetric Galerkin bulk and frozen-plastic static lift; no dynamic event simulated",
                    "records": records,
                    "environment": environment_record(case_path),
                },
            )


if __name__ == "__main__":
    main()

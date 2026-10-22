"""Predict first yield from the frozen-plastic static tangent, before any weakening.

This isolates spatial/initial-field effects; it is not a dynamic rupture result.
The prediction is invalidated if any contact opens before the predicted yield.
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

from finiteblock_bem.case import load_case
from finiteblock_bem.constraints import build_constraints
from finiteblock_bem.dynamics import RestartProjectionError, initialize_restart
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.nonlinear import ContactEquilibrium
from finiteblock_bem.statics import assemble_static_operators

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "outputs/elastic-onset"
    output.mkdir(parents=True, exist_ok=True)
    path = ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    meshes = [
        (40, None, None, None, 20),
        (20, None, None, None, 20),
        (10, None, None, None, 20),
        (5, None, None, None, 20),
        (2, None, None, None, 20),
        (40, 20, 2, None, 20),
        (40, 20, 2, 2, 12),
        (40, 20, 2, 2, 20),
        (20, 10, 2, 2, 20),
    ]
    records = []
    for outer, contact_h, end, corner, quadrature in meshes:
        case = load_case(
            path,
            max_panel_length=outer,
            contact_panel_length=contact_h,
            contact_end_length=end,
            corner_panel_length=corner,
        )
        try:
            lift = initialize_restart(case, quadrature=quadrature)
        except RestartProjectionError as exc:
            record = {
                "outer_h_mm": outer,
                "contact_h_mm": contact_h,
                "end_h_mm": end,
                "corner_h_mm": corner,
                "quadrature": quadrature,
                "prediction_valid": False,
                "initialization": exc.diagnostics,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            continue
        con = build_constraints(case)
        w = block_diag(
            *(
                assemble_static_operators(m, case.material, quadrature=quadrature).impedance()
                for m in (case.moving, case.stationary)
            )
        )
        system = ContactEquilibrium(case, con, w)
        guess = con.prolongation.T @ lift.displacement / np.sum(con.prolongation**2, axis=0)
        delta = 0.01
        base = system.evaluate(
            guess,
            np.zeros(len(w)),
            case.normal_pressure,
            case.restart_actuator,
            lift.state,
            frozen=True,
        )[2]
        other = system.solve(
            case.normal_pressure,
            case.restart_actuator + delta,
            lift.state,
            guess=guess,
            frozen=True,
        )
        if not other.converged:
            raise RuntimeError("Frozen-plastic tangent solve failed")
        p, tau = base.pressure, base.shear
        dp, dtau = ((other.contact.pressure - p) / delta, (other.contact.shear - tau) / delta)
        mu = case.contact.coefficient(lift.state.cumulative_slip)
        candidates = []
        for gap, slope in [(mu * p - tau, dtau - mu * dp), (mu * p + tau, -dtau - mu * dp)]:
            value = np.full(len(p), np.inf)
            np.divide(gap, slope, out=value, where=(slope > 0) & (gap >= 0))
            value[(p + dp * value <= 0) | (value <= 0)] = np.inf
            candidates.append(value)
        onset = np.minimum(*candidates)
        index = int(np.argmin(onset))
        opening = np.full(len(p), np.inf)
        np.divide(-p, dp, out=opening, where=dp < 0)
        record = {
            "outer_h_mm": outer,
            "contact_h_mm": contact_h,
            "end_h_mm": end,
            "corner_h_mm": corner,
            "quadrature": quadrature,
            "contact_panels": len(p),
            "predicted_first_y_mm": float(case.moving.centers[case.moving_contact, 1][index]),
            "predicted_actuator_increment_mm": float(onset[index]),
            "predicted_shear_time_s": float(onset[index] / case.actuator_rate),
            "no_prior_opening": bool(np.min(opening) > onset[index]),
            "prediction_valid": bool(np.isfinite(onset[index]) and np.min(opening) > onset[index]),
            "initialization": lift.diagnostics,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    write_json(
        output / "summary.json",
        {
            "predictions": records,
            "environment": environment_record(path),
            "scope": "Static, pre-yield estimate; no dynamic weakening simulated",
        },
    )


if __name__ == "__main__":
    main()

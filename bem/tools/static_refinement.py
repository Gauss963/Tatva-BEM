#!/usr/bin/env python3
"""Quantify static restart spatial error against the archived FEM interface."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.case import load_case
from finiteblock_bem.comparison import panel_average, relative_l2
from finiteblock_bem.dynamics import initialize_restart
from finiteblock_bem.io import environment_record, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "outputs/static-refinement"
    output.mkdir(parents=True, exist_ok=True)
    reference = np.genfromtxt(
        ROOT.parent / "validation/static_bem2d/reference/frame_00039.csv", delimiter=",", names=True
    )
    path = ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    results = []
    for h in [80, 40, 20, 10, 5, 2]:
        case = load_case(path, max_panel_length=h)
        started = perf_counter()
        lift = initialize_restart(case)
        i, j = case.moving_contact, case.stationary_contact
        offset = case.moving.dofs
        dn = lift.displacement[2 * i] - lift.displacement[offset + 2 * j]
        ds = lift.displacement[2 * i + 1] - lift.displacement[offset + 2 * j + 1]
        response = case.contact.evaluate(dn, ds, lift.state, frozen=True)
        yy, lengths = case.moving.centers[i, 1], case.moving.lengths[i]
        pressure = panel_average(reference["y"], reference["p"], yy, lengths)
        shear = panel_average(reference["y"], reference["tau"], yy, lengths)
        record = dict(
            h_mm=h,
            panels=[case.moving.count, case.stationary.count],
            wall_seconds=perf_counter() - started,
            **lift.diagnostics,
            pressure_relative_l2=relative_l2(response.pressure, pressure, lengths),
            shear_relative_l2=relative_l2(response.shear, shear, lengths),
        )
        np.savetxt(
            output / f"h{h}.csv",
            np.column_stack((yy, lengths, response.pressure, pressure, response.shear, shear)),
            delimiter=",",
            comments="",
            header="y_mm,panel_length_mm,bem_pressure_mpa,fem_panel_pressure_mpa,bem_shear_mpa,fem_panel_shear_mpa",
        )
        results.append(record)
        print(json.dumps(record), flush=True)
    write_json(
        output / "summary.json", {"refinements": results, "environment": environment_record(path)}
    )


if __name__ == "__main__":
    main()

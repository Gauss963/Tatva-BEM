"""Small, portable artifacts with explicit method and input provenance."""

import hashlib
import json
import platform
import sys
import tomllib
from pathlib import Path

import numpy as np
import scipy

from . import __version__


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_record(case_path=None):
    package = Path(__file__).parent
    record = {
        "package_version": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "source_sha256": {p.name: fingerprint(p) for p in sorted(package.glob("*.py"))},
    }
    if case_path is not None:
        record["case_sha256"] = fingerprint(case_path)
        with Path(case_path).open("rb") as stream:
            source = tomllib.load(stream)["source"]["restart_interface"]
        restart = Path(source)
        if not restart.is_absolute():
            restart = Path(case_path).parent / restart
        record["restart_data_sha256"] = fingerprint(restart)
    return record


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def save_bar(output, result, parameters):
    output.mkdir(parents=True, exist_ok=True)
    data = np.column_stack(
        (
            result.time,
            result.loading_displacement,
            result.loading_exact,
            result.probe_displacement,
            result.probe_exact,
        )
    )
    np.savetxt(
        output / "wave_reflection.csv",
        data,
        delimiter=",",
        comments="",
        header="time,loading_displacement,loading_exact,probe_displacement,probe_exact",
    )
    summary = {
        "status": "completed_analytic_wave_benchmark",
        "parameters": parameters,
        "loading_relative_l2": result.loading_relative_l2,
        "probe_relative_l2": result.probe_relative_l2,
        "precursor_relative_max": result.precursor_relative_max,
        "boundary_residual": result.boundary_residual,
        "probe_x": result.probe_x,
        "panels": result.panels,
        "full_original_event_reproduced": False,
        "environment": environment_record(),
    }
    write_json(output / "summary.json", summary)
    return summary


def save_dynamic(output, case, result, case_path):
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "trajectory.npz",
        time=result.time,
        displacement=result.displacement,
        traction=result.traction,
        pressure=result.pressure,
        shear=result.shear,
        friction_coefficient=result.friction_coefficient,
        plastic_slip=result.plastic_slip,
        cumulative_slip=result.cumulative_slip,
        actuator=result.actuator,
        platen=result.platen,
        normal_load=result.normal_load,
        moving_starts=case.moving.starts,
        moving_ends=case.moving.ends,
        stationary_starts=case.stationary.starts,
        stationary_ends=case.stationary.ends,
        contact_y=case.moving.centers[case.moving_contact, 1],
        moving_contact=case.moving_contact,
        stationary_contact=case.stationary_contact,
    )
    weights = case.moving.lengths[case.moving_contact]
    columns = np.column_stack(
        (
            result.time,
            result.normal_load,
            np.average(result.pressure, axis=1, weights=weights),
            np.average(result.shear, axis=1, weights=weights),
            np.max(result.cumulative_slip, axis=1),
            result.actuator,
            result.platen,
            case.spring_stiffness * (result.actuator - result.platen),
            result.residual_mpa,
            result.newton_iterations,
        )
    )
    np.savetxt(
        output / "history.csv",
        columns,
        delimiter=",",
        comments="",
        header="time_s,normal_load_mpa,mean_contact_pressure_mpa,mean_contact_shear_mpa,max_cumulative_slip_mm,actuator_mm,platen_mm,applied_shear_mpa,residual_mpa,newton_iterations",
    )
    summary = dict(result.diagnostics)
    summary["environment"] = environment_record(case_path)
    summary["case"] = case.payload
    write_json(output / "summary.json", summary)
    return summary

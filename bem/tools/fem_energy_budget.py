"""Saved-sample mechanical energy budget of the verified original FEM rerun.

The remainder contains friction and numerical/integration error. It is not
direct friction-work output, a time-convergence bound or a BEM stability proof.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from finiteblock_bem.io import fingerprint, write_json

READER_PATH = Path(__file__).with_name("fem_reference_io.py")
READER = runpy.run_path(str(READER_PATH))


def mechanical_budget(history, spring_stiffness, *, stride=1):
    """Integrate external work against control coordinates, retaining endpoints."""
    if not np.isfinite(spring_stiffness) or spring_stiffness <= 0:
        raise ValueError("Require a finite positive total platen spring stiffness")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("Sampling stride must be a positive integer")
    fields = [
        "time",
        "applied_shear_displacement",
        "loading_face_displacement",
        "normal_external_force",
        "normal_loading_coordinate",
        "elastic_energy",
        "interface_energy",
        "kinetic_energy",
    ]
    values = {k: np.asarray(history[k], dtype=float) for k in fields}
    count = len(values["time"])
    if count < 2 or any(v.shape != (count,) or not np.all(np.isfinite(v)) for v in values.values()):
        raise ValueError("Energy budget requires aligned finite scalar histories")
    if np.any(np.diff(values["time"]) <= 0):
        raise ValueError("Saved times must increase strictly")
    ids = np.unique(np.r_[np.arange(0, count, stride), count - 1])
    data = {k: v[ids] for k, v in values.items()}
    gap = np.maximum(data["applied_shear_displacement"] - data["loading_face_displacement"], 0)
    force = spring_stiffness * gap
    actuator_work = float(
        np.dot(0.5 * (force[1:] + force[:-1]), np.diff(data["applied_shear_displacement"]))
    )
    normal_work = float(
        np.dot(
            0.5 * (data["normal_external_force"][1:] + data["normal_external_force"][:-1]),
            np.diff(data["normal_loading_coordinate"]),
        )
    )
    changes = {
        k: float(data[k][-1] - data[k][0])
        for k in ["elastic_energy", "interface_energy", "kinetic_energy"]
    }
    spring_change = float(0.5 * spring_stiffness * (gap[-1] ** 2 - gap[0] ** 2))
    return dict(
        stride=stride,
        samples=len(ids),
        first_time_s=float(data["time"][0]),
        last_time_s=float(data["time"][-1]),
        actuator_work=actuator_work,
        normal_external_work=normal_work,
        energy_changes=changes,
        spring_energy_change=spring_change,
        friction_and_numerical_remainder=actuator_work
        + normal_work
        - spring_change
        - sum(changes.values()),
        minimum_spring_gap_mm=float(
            np.min(data["applied_shear_displacement"] - data["loading_face_displacement"])
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rerun", type=Path)
    parser.add_argument("--field-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new energy-budget output directory")
    begun, cpu = perf_counter(), process_time()
    source = args.rerun / "summary.json"
    frozen = {
        str(p.resolve()): fingerprint(p)
        for p in [source, args.field_comparison, Path(__file__), READER_PATH]
    }
    summary = json.loads(source.read_text())
    comparison = json.loads(args.field_comparison.read_text())
    hdf5 = (args.rerun / "simulation.h5").resolve()
    if not (
        summary["status"] == "completed"
        and summary["computational_dtype"] == "float64"
        and summary["source_and_protected_input_unchanged"]
        and summary["full_original_horizon"]
        and comparison["original_fem_reference_verified"]
        and Path(comparison["rerun"]["path"]).resolve() == hdf5
        and comparison["frozen_input_sha256"][str(source.resolve())] == fingerprint(source)
    ):
        raise ValueError("Require the completed original-precision, field-verified FEM rerun")
    model = summary["summary"]
    if (
        model["normal_loading_mode"] != "stress"
        or model["shear_loading_mode"] != "spring-displacement"
    ):
        raise ValueError("This energy budget requires stress normal loading and the platen spring")
    if not model["shear_spring_rigid_face"]:
        raise ValueError("A single platen coordinate requires the verified rigid loading face")
    width = float(model["legacy"]["geometry"]["moving"]["dimensions"][0])
    stiffness = width * float(model["shear_loading_stiffness"])
    trace = READER["read_fem_trace"](hdf5)
    if any(
        trace["provenance"][k] != comparison["rerun"][k]
        for k in ["bytes", "mtime_ns", "saved_frames"]
    ):
        raise ValueError("FEM trace changed after its original-field comparison")
    shear = trace["phase_id"] == 2
    history = {k: v[shear] for k, v in trace["history"].items()}
    if np.count_nonzero(shear) != comparison["shear_rows_compared"]:
        raise ValueError("Energy-budget sample count differs from the verified shear trace")
    gap = history["applied_shear_displacement"] - history["loading_face_displacement"]
    spring_traction = float(model["shear_loading_stiffness"]) * np.maximum(gap, 0)
    difference = abs(spring_traction - history["applied_shear"])
    eps = np.finfo(np.dtype(summary["stored_history_dtype"])).eps
    rounding_bound = (
        2
        * eps
        * (
            float(model["shear_loading_stiffness"])
            * (
                abs(history["applied_shear_displacement"])
                + abs(history["loading_face_displacement"])
            )
            + abs(history["applied_shear"])
        )
    )
    if np.any(difference > rounding_bound):
        raise ValueError("Saved shear traction disagrees with the spring beyond storage roundoff")
    records = [mechanical_budget(history, stiffness, stride=stride) for stride in [1, 2, 4, 8]]
    if any(fingerprint(p) != digest for p, digest in frozen.items()):
        raise ValueError("Verified FEM metadata or energy-budget implementation changed")
    args.output.mkdir(parents=True)
    record = dict(
        units="MPa mm^2 per unit thickness",
        records=records,
        platen_width_mm=width,
        total_spring_stiffness=stiffness,
        source_provenance=trace["provenance"],
        frozen_input_sha256=frozen,
        stored_history_dtype=summary["stored_history_dtype"],
        maximum_spring_traction_reconstruction_error_mpa=float(np.max(difference)),
        maximum_storage_roundoff_bound_mpa=float(np.max(rounding_bound)),
        maximum_coarsened_budget_difference=float(
            max(
                abs(
                    r["friction_and_numerical_remainder"]
                    - records[0]["friction_and_numerical_remainder"]
                )
                for r in records
            )
        ),
        total_seconds_before_reporting=perf_counter() - begun,
        process_cpu_seconds=process_time() - cpu,
        full_fem_equivalent_bem_verified=False,
        scope="Saved FEM work minus bulk, kinetic, interface and unilateral spring storage changes; remainder contains friction and numerical/integration error",
        cadence_scope="Coarsening checks saved-output quadrature sensitivity only; not time-step convergence or a certified bound",
    )
    write_json(args.output / "summary.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

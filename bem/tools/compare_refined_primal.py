"""Compare all saved physical fields for the same refined-contact problem.

Different prepared kernels are allowed; physical case, bulk/time/contact
grids, original restart, initialization and local law must be identical.
This is a solver equivalence gate, not a BEM/FEM accuracy claim.
"""

import argparse
import json
import runpy
from pathlib import Path

import numpy as np

from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
TOLERANCES = {
    "pressure": 1e-3,
    "shear": 1e-3,
    "plastic_slip": 1e-6,
    "cumulative_slip": 1e-6,
    "actuator": 1e-6,
    "platen": 1e-6,
    "contact_jumps": 1e-6,
    "retained_displacement": 1e-6,
    "initial_retained_displacement": 1e-9,
    "initial_retained_bulk_force": 1e-7,
}


def validate_records(a, b):
    for record in (a, b):
        if (
            record["status"] != "completed_requested_horizon"
            or not record["all_history_retained"]
            or record["correction_start"] != 1
            or record["max_boundary_contact_residual_mpa"] >= 1e-7
            or record["history_representation"] != "elastic_resolvent_refined_contact"
        ):
            raise ValueError("Require complete converged refined-contact primals")
    for key in [
        "duration_s",
        "dt_s",
        "schedule",
        "panels",
        "constitutive_substeps",
        "case",
        "subdivisions",
        "seed_interpolation",
        "static_initialization",
        "bulk_contact_nodes",
        "contact_nodes",
        "completed_steps",
    ]:
        if a[key] != b[key]:
            raise ValueError(f"Same-grid primal differs in {key}")
    if a["restart_seed_override"]["sha256"] != b["restart_seed_override"]["sha256"]:
        raise ValueError("Original restart input differs")
    for key in ["source_sha256", "case_sha256", "restart_data_sha256"]:
        if a["environment"][key] != b["environment"][key]:
            raise ValueError("Bulk package or physical input differs")


def compare_fields(old, new):
    for key in [
        "time",
        "contact_y",
        "contact_weights",
        "contact_interpolation",
        "retained_contact_y",
    ]:
        np.testing.assert_array_equal(old[key], new[key])
    fields = {}
    for key, tolerance in TOLERANCES.items():
        expected, actual = old[key], new[key]
        if expected.shape != actual.shape or not all(
            np.all(np.isfinite(v)) for v in [expected, actual]
        ):
            raise ValueError(f"Invalid aligned complete physical field: {key}")
        error = float(np.max(abs(actual - expected)))
        fields[key] = dict(
            maximum_absolute_difference=error,
            absolute_tolerance=tolerance,
            passed=error <= tolerance,
        )
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a new equivalence output outside protected Tatva")
    paths = [
        args.reference / "summary.json",
        args.candidate / "summary.json",
        Path(__file__),
        TOOLS / "replay_resolvent.py",
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    a, b = [json.loads(p.read_text()) for p in paths[:2]]
    validate_records(a, b)
    digest = runpy.run_path(str(TOOLS / "replay_resolvent.py"))["digest_file"]
    files = [p / "trajectory.npz" for p in [args.reference, args.candidate]]
    before = {str(p.resolve()): digest(p) for p in files}
    with np.load(files[0], allow_pickle=False) as old, np.load(files[1], allow_pickle=False) as new:
        fields = compare_fields(old, new)
    if any(fingerprint(p) != v for p, v in frozen.items()) or any(
        digest(p) != v for p, v in before.items()
    ):
        raise ValueError("Frozen same-grid comparison inputs changed")
    passed = all(v["passed"] for v in fields.values())
    output.mkdir(parents=True)
    write_json(
        output / "summary.json",
        dict(
            status="passed_complete_same_grid_primal_check"
            if passed
            else "failed_same_grid_primal_check",
            reference_run=str(args.reference.resolve()),
            candidate_run=str(args.candidate.resolve()),
            frozen_input_sha256=frozen,
            trajectory_sha256=before,
            fields=fields,
            scope="Complete same-grid solver equivalence; not continuum convergence or BEM/FEM accuracy",
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
        ),
    )
    print(output / "summary.json", flush=True)
    if not passed:
        raise ValueError("Complete same-grid fields differ at original tolerances")


if __name__ == "__main__":
    main()

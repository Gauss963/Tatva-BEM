"""Gate fine-grid attribution on recovery of the existing one-grid event.

This is an implementation identity check, not a BEM/FEM accuracy criterion.
The complete time grids, original inputs and all seven saved physical fields
are compared. A failed gate remains saved and stops an automated study.
"""

import argparse
import json
import runpy
from pathlib import Path

import numpy as np

from finiteblock_bem.io import fingerprint, write_json

TOLERANCES = {
    "pressure": 1e-3,
    "shear": 1e-3,
    "plastic_slip": 1e-6,
    "cumulative_slip": 1e-6,
    "actuator": 1e-6,
    "platen": 1e-6,
    "contact_jumps": 1e-6,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("identity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a new comparison output outside protected Tatva")
    paths = [args.original / "summary.json", args.identity / "summary.json", Path(__file__)]
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    a, b = [json.loads(p.read_text()) for p in paths[:2]]
    for key in (
        "status",
        "duration_s",
        "dt_s",
        "schedule",
        "panels",
        "constitutive_substeps",
        "case",
    ):
        if a[key] != b[key]:
            raise ValueError(f"Identity control differs in {key}")
    if (
        a["status"] != "completed_requested_horizon"
        or a["history_representation"] != "elastic_resolvent"
        or b["history_representation"] != "elastic_resolvent_refined_contact"
        or b["subdivisions"] != 1
        or b["seed_interpolation"] != "coarse-p1"
        or a["restart_seed_override"]["sha256"] != b["restart_seed_override"]["sha256"]
        or not b["all_history_retained"]
        or b["correction_start"] != 1
    ):
        raise ValueError("Require the same complete original event and an identity contact grid")
    for key in ("source_sha256", "case_sha256", "restart_data_sha256"):
        if a["environment"][key] != b["environment"][key]:
            raise ValueError("Bulk package or physical inputs differ")
    digest = runpy.run_path(str(Path(__file__).with_name("replay_resolvent.py")))["digest_file"]
    files = [folder / "trajectory.npz" for folder in (args.original, args.identity)]
    before = {str(p.resolve()): digest(p) for p in files}
    fields = {}
    with np.load(files[0], allow_pickle=False) as old, np.load(files[1], allow_pickle=False) as new:
        np.testing.assert_array_equal(old["time"], new["time"])
        np.testing.assert_array_equal(old["contact_y"], new["contact_y"])
        np.testing.assert_array_equal(old["contact_y"], new["retained_contact_y"])
        for key, tolerance in TOLERANCES.items():
            expected, actual = old[key], new[key]
            if expected.shape != actual.shape or not all(
                np.all(np.isfinite(v)) for v in (expected, actual)
            ):
                raise ValueError(f"Invalid aligned physical field: {key}")
            maximum = float(np.max(abs(actual - expected)))
            fields[key] = dict(
                maximum_absolute_difference=maximum,
                absolute_tolerance=tolerance,
                passed=maximum <= tolerance,
            )
    if any(fingerprint(p) != value for p, value in frozen.items()) or any(
        digest(p) != value for p, value in before.items()
    ):
        raise ValueError("Frozen identity inputs changed")
    passed = all(field["passed"] for field in fields.values())
    output.mkdir(parents=True)
    write_json(
        output / "summary.json",
        dict(
            status="passed_complete_identity_grid_check"
            if passed
            else "failed_identity_grid_check",
            original_run=str(args.original.resolve()),
            identity_run=str(args.identity.resolve()),
            frozen_input_sha256=frozen,
            trajectory_sha256=before,
            fields=fields,
            scope="Implementation identity at fixed bulk/time/local grid; not physical convergence or agreement with FEM",
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
        ),
    )
    print(output / "summary.json", flush=True)
    if not passed:
        raise ValueError(
            "Identity trajectory differs; do not attribute changes to contact refinement"
        )


if __name__ == "__main__":
    main()

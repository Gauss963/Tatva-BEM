"""Compare all saved physical fields for the same refined-contact problem.

Different prepared kernels are allowed; physical case, bulk/time/contact
grids, original restart, initialization and local law must be identical.
The default is a solver equivalence gate. The explicit constitutive refinement
mode compares a finer local-time reference with a coarser candidate using all
saved P1 fields. Neither mode certifies continuum or BEM/FEM accuracy.
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


def validate_records(a, b, *, mode="solver"):
    if mode not in ("solver", "constitutive-refinement"):
        raise ValueError("Unknown same-grid comparison mode")
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
    if mode == "solver" and a["constitutive_substeps"] != b["constitutive_substeps"]:
        raise ValueError("Same-grid primal differs in constitutive_substeps")
    if mode == "constitutive-refinement" and not (
        isinstance(a["constitutive_substeps"], int)
        and isinstance(b["constitutive_substeps"], int)
        and 1 <= b["constitutive_substeps"] < a["constitutive_substeps"]
    ):
        raise ValueError("The constitutive reference must have more local substeps")
    if mode == "constitutive-refinement" and (
        a["duration_s"] != 0.045 or a["kernel_summary_sha256"] != b["kernel_summary_sha256"]
    ):
        raise ValueError("Constitutive refinement requires the same complete 45 ms kernel")
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


def validate_constitutive_arguments(manifests, records):
    required = {
        "kernel",
        "restart_seed",
        "subdivisions",
        "constitutive_substeps",
        "seed_interpolation",
        "static_initialization",
        "kinetic_root_seed",
    }
    for manifest, record in zip(manifests, records, strict=True):
        if not required <= manifest["arguments"].keys():
            raise ValueError("Constitutive refinement requires the original invocation arguments")
        if manifest["arguments"]["constitutive_substeps"] != record["constitutive_substeps"]:
            raise ValueError("Constitutive invocation differs from its completed result")
    a, b = [manifest["arguments"] for manifest in manifests]
    for key in set(a) | set(b):
        if key not in {"output", "constitutive_substeps"} and a.get(key) != b.get(key):
            raise ValueError(f"Constitutive control changes another invocation argument: {key}")
    if manifests[0]["frozen_input_sha256"] != manifests[1]["frozen_input_sha256"]:
        raise ValueError("Constitutive invocations used different numerical source or input files")


def compare_constitutive_refinement(reference, candidate, *, tolerance=0.01):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Require a positive finite constitutive screen tolerance")
    for key in [
        "time",
        "contact_y",
        "contact_weights",
        "contact_interpolation",
        "retained_contact_y",
    ]:
        np.testing.assert_array_equal(reference[key], candidate[key])
    initialization = {}
    for key in ["initial_retained_displacement", "initial_retained_bulk_force"]:
        if not all(np.all(np.isfinite(data[key])) for data in [reference, candidate]):
            raise ValueError("Initial constitutive-control traces must be finite")
        np.testing.assert_allclose(reference[key], candidate[key], rtol=0, atol=TOLERANCES[key])
        initialization[key] = float(np.max(abs(candidate[key] - reference[key])))
    p1_compare = runpy.run_path(str(TOOLS / "compare_refined_fem_history.py"))[
        "compare_saved_p1_fields"
    ]
    fields = {}
    keys = [
        "pressure",
        "shear",
        "plastic_slip",
        "cumulative_slip",
        "friction_coefficient",
        "friction_strength",
    ]
    for key in keys:
        values = [
            data["pressure"] * data["friction_coefficient"]
            if key == "friction_strength"
            else data[key]
            for data in [reference, candidate]
        ]
        if any(value.dtype != np.float64 for value in values):
            raise ValueError("Constitutive comparison requires true float64 fields")
        fields[key] = p1_compare(
            (reference["time"], reference["contact_y"], values[0]),
            (candidate["time"], candidate["contact_y"], values[1]),
        )
    scalars = {}
    for key in [
        "max_cumulative_slip_mm",
        "platen_mm",
        "actuator_increment_mm",
        "spring_extension_mm",
    ]:
        rows = []
        for data in [reference, candidate]:
            if key == "max_cumulative_slip_mm":
                value = data["cumulative_slip"].max(axis=1)
            elif key == "platen_mm":
                value = data["platen"]
            elif key == "actuator_increment_mm":
                value = data["actuator"] - data["actuator"][0]
            else:
                value = data["actuator"] - data["platen"]
            if value.dtype != np.float64:
                raise ValueError("Constitutive comparison requires true float64 histories")
            rows.append(np.repeat(value[:, None], 2, axis=1))
        scalars[key] = p1_compare(
            (reference["time"], [0, 1], rows[0]),
            (candidate["time"], [0, 1], rows[1]),
        )
    critical = ["pressure", "shear", "cumulative_slip"]
    checks = {}
    for key in critical:
        result = fields[key]
        checks[key] = all(
            result[name] <= tolerance
            if result[name] is not None
            else result["maximum_saved_interpolant_absolute_difference"] == 0
            for name in ["relative_space_time_p1_l2", "final_shared_time_relative_p1_l2"]
        )
    return dict(
        fields=fields,
        scalar_errors=scalars,
        initial_trace_maximum_absolute_differences=initialization,
        critical_fields=critical,
        relative_screen_tolerance=tolerance,
        checks=checks,
        all_critical_fields_passed=all(checks.values()),
        scope="Whole saved space-time and final P1 fields on a fixed bulk/time/contact grid; local constitutive refinement only",
        physical_bulk_time_convergence_verified=False,
        same_accuracy_fem_advantage_verified=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["solver", "constitutive-refinement"], default="solver")
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
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
    if args.mode == "constitutive-refinement":
        paths.extend(
            TOOLS / name
            for name in [
                "compare_refined_fem_history.py",
                "compare_fem_mesh.py",
                "fem_reference_io.py",
            ]
        )
        paths.extend(run / "run_manifest.json" for run in [args.reference, args.candidate])
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    a, b = [json.loads(p.read_text()) for p in paths[:2]]
    validate_records(a, b, mode=args.mode)
    if args.mode == "constitutive-refinement":
        manifests = [
            json.loads((run / "run_manifest.json").read_text())
            for run in [args.reference, args.candidate]
        ]
        validate_constitutive_arguments(manifests, [a, b])
    digest = runpy.run_path(str(TOOLS / "replay_resolvent.py"))["digest_file"]
    files = [p / "trajectory.npz" for p in [args.reference, args.candidate]]
    before = {str(p.resolve()): digest(p) for p in files}
    with np.load(files[0], allow_pickle=False) as old, np.load(files[1], allow_pickle=False) as new:
        if args.mode == "solver":
            fields = compare_fields(old, new)
            refinement = None
            passed = all(v["passed"] for v in fields.values())
        else:
            np.testing.assert_array_equal(
                old["time"], np.arange(a["completed_steps"] + 1) * a["dt_s"]
            )
            refinement = compare_constitutive_refinement(
                old, new, tolerance=args.relative_tolerance
            )
            fields = refinement["fields"]
            passed = refinement["all_critical_fields_passed"]
    if any(fingerprint(p) != v for p, v in frozen.items()) or any(
        digest(p) != v for p, v in before.items()
    ):
        raise ValueError("Frozen same-grid comparison inputs changed")
    output.mkdir(parents=True)
    write_json(
        output / "summary.json",
        dict(
            status=(
                "passed_complete_same_grid_primal_check"
                if passed
                else "failed_same_grid_primal_check"
            )
            if args.mode == "solver"
            else (
                "passed_complete_constitutive_screen" if passed else "failed_constitutive_screen"
            ),
            comparison_mode=args.mode,
            reference_run=str(args.reference.resolve()),
            candidate_run=str(args.candidate.resolve()),
            frozen_input_sha256=frozen,
            trajectory_sha256=before,
            fields=fields,
            constitutive_refinement=refinement,
            reference_constitutive_substeps=a["constitutive_substeps"],
            candidate_constitutive_substeps=b["constitutive_substeps"],
            scope="Complete same-grid solver equivalence"
            if args.mode == "solver"
            else "Fixed bulk/time/contact grids; finer constitutive reference",
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
        ),
    )
    print(output / "summary.json", flush=True)
    if not passed:
        raise ValueError("Complete saved field comparison failed its recorded tolerances")


if __name__ == "__main__":
    main()

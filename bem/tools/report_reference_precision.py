"""Record the excluded FEM precision variant and corrected initialization check."""

import json
import re
from pathlib import Path

import h5py
import numpy as np

from finiteblock_bem.io import fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path("/Volumes/Gauss-T7/tatva/runs/local-mus08-muk06-gap0-v2-ev16")


def read_record(path):
    return dict(path=str(path), sha256=fingerprint(path), data=json.loads(path.read_text()))


def main():
    wrong = ROOT / "outputs/fem-event16-original-h2-4c-interface"
    short = ROOT / "outputs/fem-authored-f64-shortcheck"
    source_h5, short_h5 = REFERENCE / "data/simulation.h5", short / "simulation.h5"
    before = source_h5.stat()
    differences = {}
    with h5py.File(source_h5, "r") as original, h5py.File(short_h5, "r") as corrected:
        for body in ["moving", "stationary"]:
            for field in ["coords", "displacement"]:
                a, b = original[f"{body}/{field}"], corrected[f"{body}/{field}"]
                a, b = (a[:], b[:]) if field == "coords" else (a[0], b[0])
                differences[f"{body}/{field}"] = dict(
                    bitwise_equal=bool(np.array_equal(a, b)),
                    maximum_absolute_difference=float(np.max(abs(a - b))),
                )
        for field in ["plastic_slip", "cumulative_slip", "friction_coefficient"]:
            a, b = original[f"interface/{field}"][0], corrected[f"interface/{field}"][0]
            differences[f"initial_interface/{field}"] = dict(
                bitwise_equal=bool(np.array_equal(a, b)),
                maximum_absolute_difference=float(np.max(abs(a - b))),
            )
        static = {
            k: dict(
                original=float(v),
                corrected=float(corrected.attrs[k]),
                equal=bool(v == corrected.attrs[k]),
            )
            for k, v in original.attrs.items()
            if k.startswith("static_normal_")
        }
    after = source_h5.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Original HDF5 changed during read-only comparison")
    paths = {
        "authored_resolved_input": REFERENCE / "input/resolved.json",
        "historical_statistics": REFERENCE / "stats/summary.json",
        "excluded_float32_configuration": wrong / "resolved_run_config.json",
        "excluded_float32_run": wrong / "summary.json",
        "corrected_short_configuration": short / "resolved_run_config.json",
        "corrected_short_run": short / "summary.json",
        "contact1_identity": ROOT / "outputs/galerkin-fine2-q12-contact1/identity_comparison.json",
        "postfem_workflow": ROOT / "outputs/fine2-postfem-20261024.json",
    }
    records = {k: read_record(p) for k, p in paths.items()}
    for count in [4, 16]:
        path = ROOT / f"outputs/galerkin-fine2-q12-contact{count}-recovered/summary.json"
        if path.is_file():
            records[f"recovered_contact{count}"] = read_record(path)
    excluded = dict(
        status="excluded_from_original_configuration_reference",
        reason="Old wrapper treated stats dtype (stored scalar-history dtype) as the solver model dtype",
        expected_model_dtype="float64",
        actual_model_dtype="float32",
        measured_variant_wall_seconds=records["excluded_float32_run"]["data"]["wall_seconds"],
        same_accuracy_performance_baseline=False,
    )
    write_json(wrong / "reference_fidelity.json", excluded)
    log = ROOT / "outputs/fine2-postfem-contact4.log"
    error = re.search(r"failed at step (\d+): ([0-9.eE+-]+)", log.read_text())
    failure = dict(
        log=str(log),
        log_sha256=fingerprint(log),
        completed_requested_horizon=False,
        failed_step=None if error is None else int(error[1]),
        residual_mpa=None if error is None else float(error[2]),
    )
    record = dict(
        goal_achieved=False,
        original_h2_authored_precision_full_reference_verified=False,
        interpretation="Statistics dtype describes HDF5 history storage, not the model precision requested by input/resolved.json",
        excluded_float32_reference=excluded,
        corrected_initial_output_comparison=differences,
        corrected_static_attributes=static,
        corrected_initial_fields_bitwise_equal=all(
            v["bitwise_equal"] for v in differences.values()
        ),
        original_hdf5_size_and_mtime_unchanged=True,
        contact4_failure=failure,
        contact16_complete_result_available="recovered_contact16" in records,
        records=records,
        scope="Initialization/source fidelity and an excluded float32 control; no full corrected FEM match, BEM convergence or same-accuracy advantage is asserted",
    )
    write_json(ROOT / "docs/fem_precision_audit_results.json", record)
    print(ROOT / "docs/fem_precision_audit_results.json")


if __name__ == "__main__":
    main()

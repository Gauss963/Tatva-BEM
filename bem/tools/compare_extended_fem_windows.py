"""Compare dt/4, dt/8 and dt/16 over the same conditional physical window.

All original contact endpoints and shared saved times are retained. Scalars
use every actual fine step. A passed local diagnostic is not full-event,
spatial, energy-stability or PDE convergence certification.
"""

import argparse
import json
import runpy
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
BASE = runpy.run_path(str(TOOLS / "compare_fem_timestep_window.py"))
DIGEST, WRITE = BASE["DIGEST"], BASE["WRITE"]


def load_branch(folder, record, first, last, expected_y=None, expected_steps=None):
    factor = record["factor"]
    with np.load(folder / f"factor{factor}.npz", allow_pickle=False) as a:
        steps = a["step_id"]
        use = (steps > first) & (steps <= last)
        selected = steps[use]
        y = a["contact_y"]
        count = (last - first) * factor
        scalar = a["all_scalar_history"][:count]
        if (
            scalar.shape != (count, 17)
            or scalar.dtype != np.float64
            or len(selected) < 2
            or np.any(np.diff(selected) <= 0)
            or np.any(np.diff(y) <= 0)
            or not np.all(np.isfinite(scalar))
            or not np.all(np.isfinite(y))
        ):
            raise ValueError("Require complete ordered fine scalar and contact observations")
        if expected_steps is not None and not np.array_equal(selected, expected_steps):
            raise ValueError("Shared physical observation steps differ")
        if expected_y is not None and not np.array_equal(y, expected_y):
            raise ValueError("Original complete contact grid differs")
        time = selected * (record["dt_s"] * factor)
        if not np.array_equal(a["time"][use], time):
            raise ValueError("Contact observation times differ from integer physical steps")
        fields = {k: a[k][use] for k in BASE["BASE"]["FIELDS"]}
    if any(v.shape != (len(time), len(y)) or not np.all(np.isfinite(v)) for v in fields.values()):
        raise ValueError("Incomplete or nonfinite shared contact fields")
    return dict(time=time, step_id=selected, y=y, scalar=scalar, fields=fields)


def verified_extension(folder):
    summary = json.loads((folder / "summary.json").read_text())
    manifest = json.loads((folder / "extension_manifest.json").read_text())
    if (
        summary["status"] != "completed_conditional_fem_window_extension"
        or [v["factor"] for v in summary["records"]] != [8, 16]
        or not summary["every_fine_scalar_and_accepted_state_retained"]
        or not summary["all_source_inputs_unchanged"]
        or summary["full_event_or_spatial_convergence_verified"]
        or [summary[k] for k in ["first_shear_step", "last_shear_step"]]
        != [manifest[k] for k in ["first_step", "last_step"]]
        or manifest["factors"] != [8, 16]
        or not summary["replay_control"]["every_float64_scalar_row_bitwise"]
        or not all(summary["replay_control"]["contact_fields_bitwise"].values())
        or summary["replay_control"]["shared_observations"] < 2
        or any(DIGEST(p) != v for p, v in manifest["frozen_inputs_sha256"].items())
    ):
        raise ValueError("Require both completed fine factors and unchanged numerical inputs")
    files = [folder / "summary.json", folder / "extension_manifest.json"]
    for record in summary["records"]:
        factor = record["factor"]
        final = folder / f"factor{factor}.npz"
        receipt_path = folder / f"factor{factor}-receipt.json"
        receipt = json.loads(receipt_path.read_text())
        chunks = sorted((folder / f"factor{factor}-chunks").glob("chunk-*.npz"))
        if (
            DIGEST(final) != record["output_sha256"]
            or receipt["output_sha256"] != record["output_sha256"]
            or len(chunks) != record["chunks"]
            or [DIGEST(p) for p in chunks] != receipt["chunk_sha256"]
        ):
            raise ValueError("Completed fine export or accepted-state chunks changed")
        files.extend([final, receipt_path, *chunks])
    return summary, manifest, files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent", type=Path)
    p.add_argument("extension", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    parent, extension, output = (v.resolve() for v in [args.parent, args.extension, args.output])
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh report outside protected Tatva")
    summary, manifest, files = verified_extension(extension)
    original = json.loads((parent / "summary.json").read_text())
    if (
        original["status"] != "completed_local_fem_timestep_window"
        or not original["all_source_inputs_unchanged"]
        or not all(original["records"][0]["original_contact_bitwise_checks"].values())
        or DIGEST(parent / "summary.json")
        != manifest["frozen_inputs_sha256"][str(parent / "summary.json")]
        or DIGEST(parent / "factor4.npz")
        != manifest["frozen_inputs_sha256"][str(parent / "factor4.npz")]
        or original["first_shear_step"] != summary["first_shear_step"]
        or original["last_shear_step"] < summary["last_shear_step"]
    ):
        raise ValueError("Parent dt/4 does not represent the identical original starting state")
    files.extend(
        [
            Path(__file__),
            TOOLS / "compare_fem_timestep_window.py",
            TOOLS / "compare_float64_contact_history.py",
            TOOLS / "compare_refined_fem_history.py",
            TOOLS / "event_observables.py",
            parent / "summary.json",
            parent / "factor4.npz",
        ]
    )
    frozen = {str(p): DIGEST(p) for p in files}
    records = {r["factor"]: r for r in [original["records"][-1], *summary["records"]]}
    dt = records[4]["dt_s"] * 4
    if not np.isfinite(dt) or dt <= 0 or any(r["dt_s"] * f != dt for f, r in records.items()):
        raise ValueError("Refinement factors do not share the original physical time step")
    first, last = summary["first_shear_step"], summary["last_shear_step"]
    reference = load_branch(extension, records[16], first, last)
    data = {16: reference}
    for factor, folder in [(4, parent), (8, extension)]:
        data[factor] = load_branch(
            folder, records[factor], first, last, reference["y"], reference["step_id"]
        )
    fields, scalars, holds, crossings = {}, {}, {}, {}
    start = summary["physical_shear_interval_s"][0]
    fine_clock = start + np.arange(1, len(reference["scalar"]) + 1) * records[16]["dt_s"]
    for factor, a in data.items():
        clock = start + np.arange(1, len(a["scalar"]) + 1) * records[factor]["dt_s"]
        ids = np.flatnonzero(a["scalar"][:, 11] > 0.5)
        holds[str(factor)] = None if not len(ids) else float(clock[ids[0]])
        crossings[str(factor)] = {
            name: BASE["EVENT"]["crossing_brackets"](clock, a["scalar"][:, 5, None], threshold)[0]
            for name, threshold in {"Dc10": 0.00125959, "Dc": 0.0125959}.items()
        }
        if factor == 16:
            continue
        fields[str(factor)] = {
            k: BASE["BASE"]["COMPARE"]["compare_saved_p1_fields"](
                (reference["time"], reference["y"], reference["fields"][k]),
                (a["time"], a["y"], a["fields"][k]),
            )
            for k in reference["fields"]
        }
        scalars[str(factor)] = {
            k: BASE["EVENT"]["waveform_error"](
                fine_clock, reference["scalar"][:, index], clock, a["scalar"][:, index]
            )
            for k, index in BASE["SCALARS"].items()
        }
    if any(DIGEST(p) != v for p, v in frozen.items()):
        raise ValueError("Comparison inputs changed")
    WRITE(
        output,
        dict(
            status="completed_conditional_fem_extended_time_comparison",
            contact_errors_against_dt_over_16=fields,
            every_fine_scalar_errors_against_dt_over_16=scalars,
            loading_hold_first_fine_endpoint_s=holds,
            global_max_cumulative_first_passage_brackets=crossings,
            physical_shear_interval_s=summary["physical_shear_interval_s"],
            contact_observation_interval_s=[
                float(reference["time"][0]),
                float(reference["time"][-1]),
            ],
            contact_observation_rows=len(reference["time"]),
            finer_local_one_percent_diagnostic=all(
                v[k] is not None and v[k] <= 0.01
                for v in fields["8"].values()
                for k in ["relative_space_time_p1_l2", "final_shared_time_relative_p1_l2"]
            ),
            frozen_inputs_sha256=frozen,
            full_event_or_spatial_convergence_verified=False,
            fitted_time_shift_s=0,
            all_contact_endpoints_retained=True,
        ),
    )


if __name__ == "__main__":
    main()

"""Compare saved local FEM step-refinement fields and full-rate scalar clocks.

These are conditional window diagnostics, never a full-event convergence or
stability certificate. All spatial endpoints and saved contact rows are used.
"""

import argparse
import json
import runpy
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
BASE = runpy.run_path(str(TOOLS / "compare_float64_contact_history.py"))
EVENT = runpy.run_path(str(TOOLS / "event_observables.py"))
DIGEST = BASE["DIGEST"]
WRITE = runpy.run_path(str(TOOLS / "run_checkpointed_fem_reference.py"))["write_json"]
SCALARS = {
    "max_cumulative_slip": 5,
    "mean_pressure": 3,
    "mean_signed_shear": 2,
    "kinetic_energy": 9,
}


def scalar_clock(data, summary, record):
    count = (summary["last_shear_step"] - summary["first_shear_step"]) * record["factor"]
    rows = data["all_scalar_history"]
    if rows.shape != (count, 17) or not np.all(np.isfinite(rows)):
        raise ValueError("Require every actual fine scalar step, without downsampling")
    start = summary["physical_shear_interval_s"][0]
    return start + np.arange(1, count + 1) * record["dt_s"], rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("window", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(
        Path("/Volumes/Gauss-T7/tatva")
    ):
        raise ValueError("Choose a fresh report outside protected Tatva")
    folder = args.window.resolve()
    summary = json.loads((folder / "summary.json").read_text())
    if (
        summary["status"] != "completed_local_fem_timestep_window"
        or not summary["all_source_inputs_unchanged"]
        or not summary["all_scalar_steps_within_window_retained"]
        or summary["full_event_time_convergence_verified"]
        or [r["factor"] for r in summary["records"]] != [1, 2, 4]
        or not all(summary["records"][0]["original_contact_bitwise_checks"].values())
    ):
        raise ValueError("Require all completed branches and original contact identity")
    files = [
        Path(__file__),
        TOOLS / "event_observables.py",
        TOOLS / "compare_float64_contact_history.py",
        TOOLS / "compare_refined_fem_history.py",
        folder / "manifest.json",
        folder / "summary.json",
        *[folder / f"factor{i}.npz" for i in [1, 2, 4]],
    ]
    frozen = {str(a.resolve()): DIGEST(a) for a in files}
    datasets = {}
    for record in summary["records"]:
        factor = record["factor"]
        with np.load(folder / f"factor{factor}.npz", allow_pickle=False) as a:
            datasets[factor] = {
                k: a[k] for k in ["time", "contact_y", "all_scalar_history", *BASE["FIELDS"]]
            }
    reference = datasets[4]
    errors, scalars, crossings = {}, {}, {}
    for factor, data in datasets.items():
        np.testing.assert_array_equal(data["time"], reference["time"])
        np.testing.assert_array_equal(data["contact_y"], reference["contact_y"])
        record = next(r for r in summary["records"] if r["factor"] == factor)
        clock, values = scalar_clock(data, summary, record)
        crossings[str(factor)] = {}
        for name, threshold in {"Dc10": 0.00125959, "Dc": 0.0125959}.items():
            crossings[str(factor)][name] = EVENT["crossing_brackets"](
                clock, values[:, 5, None], threshold
            )[0]
        if factor == 4:
            continue
        errors[str(factor)] = {
            k: BASE["COMPARE"]["compare_saved_p1_fields"](
                (reference["time"], reference["contact_y"], reference[k]),
                (data["time"], data["contact_y"], data[k]),
            )
            for k in BASE["FIELDS"]
        }
        fine_clock, fine_values = scalar_clock(reference, summary, summary["records"][-1])
        scalars[str(factor)] = {
            k: EVENT["waveform_error"](fine_clock, fine_values[:, index], clock, values[:, index])
            for k, index in SCALARS.items()
        }
    if any(DIGEST(a) != v for a, v in frozen.items()):
        raise ValueError("Window input or comparator changed")
    WRITE(
        args.output,
        dict(
            status="completed_conditional_local_fem_timestep_comparison",
            contact_field_errors_against_dt_over_4=errors,
            scalar_errors_against_dt_over_4=scalars,
            global_max_cumulative_first_passage_brackets=crossings,
            physical_shear_interval_s=summary["physical_shear_interval_s"],
            contact_observation_interval_s=[
                float(reference["time"][0]),
                float(reference["time"][-1]),
            ],
            contact_observation_rows=len(reference["time"]),
            frozen_inputs_sha256=frozen,
            local_finer_field_one_percent_diagnostic=all(
                r[k] is not None and r[k] <= 0.01
                for r in errors["2"].values()
                for k in ["relative_space_time_p1_l2", "final_shared_time_relative_p1_l2"]
            ),
            full_event_or_spatial_convergence_verified=False,
            scope="Same original accepted starting state; no fitted time shift, endpoint clipping, or extrapolation; saved contact P1 interpolants and every fine scalar step",
        ),
    )


if __name__ == "__main__":
    main()

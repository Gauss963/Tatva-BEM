"""Analyze committed native FEM fields independently of their expensive replay.

Saved-field interpolation remains meaningful if a local-law replay fails.
In that case the constitutive projection is explicitly rejected and its
differences are retained, rather than changing tolerances or losing data.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.io import write_json

TOOLS = Path(__file__).resolve().parent
DIGEST = runpy.run_path(str(TOOLS / "fem_refinement_chunks.py"))["digest"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder, output = args.observations.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Require a fresh analysis outside protected Tatva")
    begun = perf_counter()
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (
        summary["status"] != "completed_bitwise_fem_native_contact_observations"
        or not summary["source_and_inputs_unchanged"]
        or not summary["all_original_contact_endpoints_retained"]
        or not summary["every_observed_step_has_original_float64_scalars"]
        or any(DIGEST(path) != value for path, value in summary["frozen_inputs_sha256"].items())
    ):
        raise ValueError("Require complete source-unchanged native observations")
    sources = [
        Path(__file__),
        TOOLS / "fem_contact_path_observations.py",
        TOOLS / "compare_refined_fem_history.py",
        summary_path,
    ]
    frozen = {str(p.resolve()): DIGEST(p) for p in sources}
    path_model = runpy.run_path(str(TOOLS / "fem_contact_path_observations.py"))
    output.mkdir(parents=True)
    results = []
    for record in summary["results"]:
        path = Path(record["output"])
        if (
            not record["every_native_scalar_row_bitwise"]
            or not all(record["all_final_carry_arrays_bitwise"])
            or not all(record["canonical_endpoint_fields_bitwise"].values())
            or DIGEST(path) != record["output_sha256"]
        ):
            raise ValueError("Committed native data or original replay proof changed")
        frozen[str(path)] = record["output_sha256"]
        write_json(output / "progress.json", dict(index=record["index"], phase="path_projection"))
        with np.load(path, allow_pickle=False) as data:
            time, y, fields, parameters = (
                data[k] for k in ["time_s", "contact_y", "fields", "parameters"]
            )
        if fields.dtype != np.float64 or fields.shape[0] != record["native_rows"] + 1:
            raise ValueError("Missing native observation rows or non-f64 fields")
        report = path_model["projection_report"](time, y, fields, parameters)
        results.append(
            dict(
                index=record["index"],
                actual_shear_interval_s=record["actual_shear_interval_s"],
                input=str(path),
                input_sha256=record["output_sha256"],
                **report,
            )
        )
        write_json(output / "completed_analyses.json", dict(results=results))
        del fields
    if any(DIGEST(path) != value for path, value in frozen.items()):
        raise ValueError("Source or native observations changed during analysis")
    write_json(
        output / "summary.json",
        dict(
            status="completed_saved_field_projection_and_local_law_gate_report",
            results=results,
            all_constitutive_projection_gates_passed=all(
                r["constitutive_projection_accepted"] for r in results
            ),
            frozen_inputs_sha256=frozen,
            original_full_event_gates_unchanged=True,
            actual_bem_solution_error_or_bound=False,
            seconds=perf_counter() - begun,
        ),
    )
    print(
        json.dumps(
            {
                str(r["index"]): dict(
                    production_native_gate=r["production_bem_native_path_replay"],
                    source_order_native_gate=r["source_order_native_path_replay"],
                    projected_fields={
                        h: {
                            k: v["relative_space_time_p1_l2"]
                            for k, v in p["saved_field_projection_errors"].items()
                        }
                        for h, p in r["projections"].items()
                    },
                )
                for r in results
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Distinguish net slip, saved-sample variation and accumulated plastic slip.

Accumulated slip is advanced at solver steps, so it retains reversals between
saved frames. The difference from absolute net slip measures all reversal
variation, whereas sample variation alone can miss it. Storage and arithmetic
roundoff prevent interpreting a tiny difference as an exact zero.
"""

import argparse
import json
import runpy
from pathlib import Path

import numpy as np

from finiteblock_bem.io import fingerprint, write_json

READER_PATH = Path(__file__).with_name("fem_reference_io.py")


def variation_profiles(y, plastic, cumulative):
    y, plastic, cumulative = (np.asarray(a, dtype=float) for a in (y, plastic, cumulative))
    if (
        y.ndim != 1
        or len(y) < 2
        or plastic.ndim != 2
        or plastic.shape[0] < 2
        or plastic.shape[1] != len(y)
        or cumulative.shape != plastic.shape
        or not all(np.all(np.isfinite(a)) for a in (y, plastic, cumulative))
        or np.any(np.diff(y) <= 0)
    ):
        raise ValueError("Require finite aligned histories on increasing interface nodes")
    weights = np.r_[np.diff(y), 0] / 2 + np.r_[0, np.diff(y)] / 2
    net = abs(plastic[-1] - plastic[0])
    sampled = np.sum(abs(np.diff(plastic, axis=0)), axis=0)
    accumulated = cumulative[-1] - cumulative[0]
    profiles = dict(
        net_plastic_slip_mm=net,
        saved_sample_variation_mm=sampled,
        accumulated_slip_increment_mm=accumulated,
        accumulated_minus_net_mm=accumulated - net,
        accumulated_minus_sampled_variation_mm=accumulated - sampled,
    )
    metrics = {
        key: dict(
            minimum=float(np.min(value)),
            maximum=float(np.max(value)),
            p1_weighted_mean=float(np.average(value, weights=weights)),
        )
        for key, value in profiles.items()
    }
    metrics["minimum_saved_cumulative_increment_mm"] = float(np.min(np.diff(cumulative, axis=0)))
    return profiles, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rerun", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--field-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new variation-audit directory")
    source = args.rerun / "summary.json"
    frozen_paths = [source, args.field_comparison, Path(__file__), READER_PATH]
    for run in args.runs:
        frozen_paths.extend([run / "summary.json", run / "trajectory.npz"])
    frozen = {str(p.resolve()): fingerprint(p) for p in frozen_paths}
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
    trace = runpy.run_path(str(READER_PATH))["read_fem_trace"](
        hdf5, fields=("plastic_slip", "cumulative_slip")
    )
    if any(
        trace["provenance"][k] != comparison["rerun"][k]
        for k in ["bytes", "mtime_ns", "saved_frames"]
    ):
        raise ValueError("FEM trace changed after its original-field comparison")
    shear = trace["phase_id"] == 2
    if np.count_nonzero(shear) != comparison["shear_rows_compared"]:
        raise ValueError("Shear sample count differs from the verified trace")
    arrays = trace["interface_fields"]
    z, c = (arrays[key][shear] for key in ("plastic_slip", "cumulative_slip"))
    if np.any(c[0] != 0):
        raise ValueError("Audit requires the original zero-slip shear start to be stored")
    profiles, metrics = variation_profiles(trace["y"], z, c)
    normal_time = summary["summary"]["pressure_time"]
    horizon = trace["history"]["time"][shear][-1] - normal_time
    records = [dict(source="verified_fem", samples=len(c), metrics=metrics)]
    export = [(trace["y"], profiles)]
    for run in args.runs:
        record = json.loads((run / "summary.json").read_text())
        if record["status"] != "completed_requested_horizon":
            raise ValueError("Require completed BEM runs")
        with np.load(run / "trajectory.npz", allow_pickle=False) as data:
            if abs(data["time"][-1] - horizon) > 2e-8 or np.any(data["cumulative_slip"][0] != 0):
                raise ValueError("BEM run must cover the same zero-slip shear horizon")
            yy = data["contact_y"]
            pp, mm = variation_profiles(yy, data["plastic_slip"], data["cumulative_slip"])
            export.append((yy, pp))
            records.append(dict(source=str(run.resolve()), samples=len(data["time"]), metrics=mm))
    if any(fingerprint(p) != digest for p, digest in frozen.items()):
        raise ValueError("Audit code or input artifacts changed while reading")
    args.output.mkdir(parents=True)
    for index, (yy, pp) in enumerate(export):
        np.savetxt(
            args.output / f"profiles-{index}.csv",
            np.column_stack([yy, *pp.values()]),
            delimiter=",",
            header=",".join(["y_mm", *pp]),
            comments="",
        )
    result = dict(
        records=records,
        fem_provenance=trace["provenance"],
        fem_computational_dtype=summary["computational_dtype"],
        fem_stored_interface_dtype=summary["stored_history_dtype"],
        frozen_input_sha256=frozen,
        scope="Accumulated minus absolute net plastic slip measures reversal variation, including between saved frames, up to computation and storage error; not a bound on bulk time-discretization error",
        caveat="Small signed deficits are retained; neither clipping nor a certified floating-point error bound is applied",
        same_accuracy_fem_advantage_verified=False,
    )
    write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

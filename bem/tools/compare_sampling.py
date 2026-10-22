"""Compare complete trajectories and optional weight arrays, without hiding switches."""

import argparse
import json
from pathlib import Path

import numpy as np

from finiteblock_bem.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference-weights", type=Path)
    parser.add_argument("--candidate-weights", type=Path)
    args = parser.parse_args()
    summaries = [
        json.loads((p / "summary.json").read_text()) for p in [args.reference, args.candidate]
    ]
    for s in summaries:
        if s["status"] != "completed_requested_horizon" or s["source_changed_during_run"]:
            raise ValueError("Require complete frozen-source trajectories")
    for name in ["dt_s", "requested_steps", "boundary_formulation"]:
        if summaries[0][name] != summaries[1][name]:
            raise ValueError(f"Different simulation controls: {name}")
    for name in ["case_sha256", "restart_data_sha256"]:
        if summaries[0]["environment"][name] != summaries[1]["environment"][name]:
            raise ValueError(f"Different physical inputs: {name}")
    controls = [
        json.loads((p / "run_manifest.json").read_text())["arguments"]
        for p in [args.reference, args.candidate]
    ]
    if controls[0]["quadrature"] != controls[1]["quadrature"]:
        raise ValueError("Different spatial quadrature")
    # These kernels, contact equations and static initialization must coincide.
    for name in [
        "operators.py",
        "kernels.py",
        "friction.py",
        "jax_backend.py",
        "nonlinear.py",
        "material.py",
        "mesh.py",
        "constraints.py",
        "statics.py",
        "case.py",
    ]:
        if (
            summaries[0]["environment"]["source_sha256"][name]
            != summaries[1]["environment"]["source_sha256"][name]
        ):
            raise ValueError(f"Reference physics/discretization differs: {name}")
    record = {"reference": args.reference.name, "candidate": args.candidate.name, "errors": {}}
    with (
        np.load(args.reference / "trajectory.npz", allow_pickle=False) as a,
        np.load(args.candidate / "trajectory.npz", allow_pickle=False) as b,
    ):
        for name in [
            "time",
            "moving_starts",
            "moving_ends",
            "stationary_starts",
            "stationary_ends",
        ]:
            np.testing.assert_array_equal(a[name], b[name])
        for name in ["pressure", "shear", "cumulative_slip", "plastic_slip", "platen", "actuator"]:
            diff = b[name] - a[name]
            record["errors"][name] = {
                "max_absolute": float(np.max(np.abs(diff))),
                "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(a[name]), 1e-30)),
            }
        record["changed_sliding_entries"] = int(
            np.count_nonzero(
                (np.abs(np.diff(a["plastic_slip"], axis=0)) > 1e-10)
                != (np.abs(np.diff(b["plastic_slip"], axis=0)) > 1e-10)
            )
        )
    if (args.reference_weights is None) != (args.candidate_weights is None):
        raise ValueError("Provide both weight arrays or neither")
    if args.reference_weights is not None:
        a, b = [
            np.load(p, mmap_mode="r", allow_pickle=False)
            for p in [args.reference_weights, args.candidate_weights]
        ]
        if a.shape != b.shape:
            raise ValueError("Different weight shapes")
        error, reference, maximum = 0.0, 0.0, 0.0
        for start in range(0, len(a), 64):
            x, y = a[start : start + 64], b[start : start + 64]
            diff = x - y
            error += float(np.sum(diff * diff))
            reference += float(np.sum(x * x))
            maximum = max(maximum, float(np.max(np.abs(diff))))
        record["weight_errors"] = {
            "relative_frobenius": (error / reference) ** 0.5,
            "max_absolute": maximum,
            "instantaneous_relative_frobenius": float(
                np.linalg.norm(a[0] - b[0]) / np.linalg.norm(a[0])
            ),
        }
    record["candidate_summary"] = summaries[1]
    write_json(args.candidate / "sampling_comparison.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

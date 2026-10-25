"""Verify a completed original-precision FEM rerun against its archived fields.

Comparison uses identical shear step IDs, never interpolation of a failed or
different trajectory. This establishes FEM reference fidelity only, not BEM
agreement or a same-accuracy performance advantage.
"""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import h5py
import numpy as np

from finiteblock_bem.io import write_json

ROOT = Path(__file__).resolve().parents[1]
READER = runpy.run_path(str(Path(__file__).with_name("fem_reference_io.py")))


def differences(expected, actual, *, rtol, atol):
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape != actual.shape or not (
        np.all(np.isfinite(expected)) and np.all(np.isfinite(actual))
    ):
        raise ValueError("FEM arrays must have matching shapes and finite values")
    delta, norm = actual - expected, float(np.linalg.norm(expected))
    return dict(
        bitwise_equal=bool(np.array_equal(expected, actual)),
        maximum_absolute_difference=float(np.max(abs(delta), initial=0)),
        relative_l2=None if norm == 0 else float(np.linalg.norm(delta) / norm),
        within_tolerance=bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
    )


def aligned_shear_rows(reference, actual):
    rows = [np.flatnonzero(data["phase_id"] == 2) for data in [reference, actual]]
    steps = [data["step_id"][r] for data, r in zip([reference, actual], rows, strict=True)]
    if not len(steps[0]) or any(np.any(np.diff(s) <= 0) for s in steps):
        raise ValueError("Shear samples must have nonempty, strictly increasing step IDs")
    if not np.array_equal(*steps):
        raise ValueError("Full shear output must have exactly matching step IDs; no interpolation")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rerun", type=Path)
    parser.add_argument(
        "--reference-run",
        type=Path,
        default=Path("/Volumes/Gauss-T7/tatva/runs/local-mus08-muk06-gap0-v2-ev16"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists():
        raise ValueError("Choose a new comparison folder in this checkout's bem/outputs")
    if any(not np.isfinite(v) or v < 0 for v in [args.rtol, args.atol]):
        raise ValueError("Comparison tolerances must be finite and nonnegative")
    summary_path = args.rerun / "summary.json"
    config_path = args.rerun / "resolved_run_config.json"
    manifest_path = args.rerun / "reference_manifest.json"
    summary, config, manifest = [
        json.loads(p.read_text()) for p in [summary_path, config_path, manifest_path]
    ]
    paths = [
        Path(__file__),
        Path(__file__).with_name("fem_reference_io.py"),
        summary_path,
        config_path,
        manifest_path,
        args.reference_run / "input/case.toml",
        args.reference_run / "input/resolved.json",
    ]

    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    frozen = {str(p.resolve()): digest(p) for p in paths}
    authored = json.loads((args.reference_run / "input/resolved.json").read_text())
    if not (
        summary["status"] == "completed"
        and summary["source_and_protected_input_unchanged"]
        and summary["full_original_horizon"]
        and not summary["restart_projected_to_coarse_mesh"]
        and summary.get("computational_dtype") == authored["solver"]["dtype"] == "float64"
        and config["dtype"] == "float64"
        and summary.get("jax_enable_x64")
        and config["mesh_size"] == authored["solver"]["mesh_size"] == 2.0
        and summary["source_revision"] == authored["git_revision"]
    ):
        raise ValueError(
            "Require a completed original h2/float64 FEM reference with verified inputs"
        )
    for path in paths[-2:]:
        if manifest["protected_input_sha256"].get(str(path)) != digest(path):
            raise ValueError("Rerun did not use the supplied archived FEM input")
    fields = ["plastic_slip", "cumulative_slip", "friction_coefficient", "friction_strength"]
    files = [args.reference_run / "data/simulation.h5", args.rerun / "simulation.h5"]
    before = [p.stat() for p in files]
    reference, actual = [READER["read_fem_trace"](p, fields=fields) for p in files]
    ia, ib = aligned_shear_rows(reference, actual)

    def compare(a, b):
        return differences(a, b, rtol=args.rtol, atol=args.atol)

    if set(reference["history"]) != set(actual["history"]):
        raise ValueError("FEM scalar history columns differ")
    history = {k: compare(v[ia], actual["history"][k][ib]) for k, v in reference["history"].items()}
    interface = {
        k: compare(reference["interface_fields"][k][ia], actual["interface_fields"][k][ib])
        for k in fields
    }
    other = {"interface_y": compare(reference["y"], actual["y"])}
    with h5py.File(files[0], "r") as source, h5py.File(files[1], "r") as target:
        for body in ["moving", "stationary"]:
            for field in ["coords", "displacement"]:
                a, b = source[f"{body}/{field}"], target[f"{body}/{field}"]
                a, b = (a[:], b[:]) if field == "coords" else (a[0], b[0])
                other[f"initial/{body}/{field}"] = compare(a, b)
        for field in ["u", "v_half", "plastic_slip", "cumulative_slip"]:
            other[f"final_state/{field}"] = compare(
                source[f"final_state/{field}"][:], target[f"final_state/{field}"][:]
            )
        for field in fields:
            a, b = reference["interface_fields"][field][0], actual["interface_fields"][field][0]
            other[f"initial_interface/{field}"] = compare(a, b)
    after = [p.stat() for p in files]
    if any(
        (a.st_size, a.st_mtime_ns) != (b.st_size, b.st_mtime_ns)
        for a, b in zip(before, after, strict=True)
    ) or any(digest(p) != checksum for p, checksum in frozen.items()):
        raise ValueError("FEM comparison source changed during read-only verification")
    records = [*history.values(), *interface.values(), *other.values()]
    passed = all(r["within_tolerance"] for r in records)
    output.mkdir()
    write_json(
        output / "summary.json",
        dict(
            status="passed_original_fem_reference" if passed else "failed_fem_field_comparison",
            original_fem_reference_verified=passed,
            full_original_event_reproduced_by_bem=False,
            all_compared_arrays_bitwise_equal=all(r["bitwise_equal"] for r in records),
            shear_rows_compared=len(ia),
            rtol=args.rtol,
            atol=args.atol,
            scalar_history=history,
            interface_fields=interface,
            other_fields=other,
            original=reference["provenance"],
            rerun=actual["provenance"],
            frozen_input_sha256=frozen,
            timing_seconds=summary["timing_seconds"],
            scope="Original FEM field fidelity only; BEM accuracy and fair same-accuracy performance remain separate gates",
        ),
    )
    print(output / "summary.json", flush=True)
    if not passed:
        raise ValueError("FEM fields differ beyond the declared original-reference tolerances")


if __name__ == "__main__":
    main()

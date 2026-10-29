"""Compare all frequency/weight bytes and startup arrays of matched Radau builds."""

import argparse
import json
import runpy
from pathlib import Path

import numpy as np

from finiteblock_bem.io import write_json

TOOLS = Path(__file__).resolve().parent
DIGEST = runpy.run_path(str(TOOLS / "radau_sampling_checkpoint.py"))["digest_range"]


def compare(reference, candidate):
    reference, candidate = Path(reference).resolve(), Path(candidate).resolve()
    a, b = [json.loads((p / "summary.json").read_text()) for p in [reference, candidate]]
    for r in [a, b]:
        if (
            r["status"] != "completed_compact_radau_resolvent_kernel_only"
            or r["time_method"] != "radau2"
        ):
            raise ValueError("Require complete Radau kernels")
    for key in [
        "plan",
        "source_mesh_arguments",
        "family_spectra_coordinates",
        "time_weights_layout",
    ]:
        if a[key] != b[key]:
            raise ValueError("Kernel discretization differs: " + key)
    for key in ["source_sha256", "case_sha256", "restart_data_sha256", "numpy", "scipy"]:
        if a["environment"][key] != b["environment"][key]:
            raise ValueError("Kernel numerical environment differs")

    def raw(r):
        return [
            (x["branch"], x["pre_combination_sha256"]) for x in r["pre_combination_branch_digests"]
        ]

    if raw(a) != raw(b):
        raise ValueError("Complete original raw branch receipts differ")
    verified = {}
    for key in ["family_spectra", "family_weights"]:
        if len(a[key]) != 2 or len(b[key]) != 2:
            raise ValueError("Both complete families are required")
        for old, new in zip(a[key], b[key], strict=True):
            for r in [old, new]:
                p = Path(r["path"])
                actual = DIGEST(p)
                if actual != r["sha256"] or p.stat().st_size != r["bytes"]:
                    raise ValueError("Kernel file changed: " + str(p))
                verified[str(p)] = actual
            if old["sha256"] != new["sha256"] or old["bytes"] != new["bytes"]:
                raise ValueError("Complete kernel family file bytes differ")
    for name, key in [
        ("initial_impedance.npy", "initial_impedance_sha256"),
        ("shift_wrap.npy", "shift_wrap_sha256"),
    ]:
        for p, r in [(reference, a), (candidate, b)]:
            if DIGEST(p / name) != r[key]:
                raise ValueError("Kernel initial data changed")
        if a[key] != b[key]:
            raise ValueError("Initial impedance/wrap file bytes differ")
    for p, r in [(reference, a), (candidate, b)]:
        if DIGEST(p / "elastic.npz") != r["elastic_sha256"]:
            raise ValueError("Complete causal startup changed")
    with (
        np.load(reference / "elastic.npz", allow_pickle=False) as x,
        np.load(candidate / "elastic.npz", allow_pickle=False) as y,
    ):
        if set(x.files) != set(y.files):
            raise ValueError("Causal startup fields differ")
        for name in x.files:
            left, right = x[name], y[name]
            if (
                left.dtype != right.dtype
                or left.shape != right.shape
                or left.tobytes() != right.tobytes()
            ):
                raise ValueError("Complete causal startup coefficient bits differ")
        startup_fields = sorted(x.files)
    return dict(
        status="passed_complete_matched_radau_kernel_identity",
        reference=str(reference),
        candidate=str(candidate),
        plan=a["plan"],
        complete_raw_branch_receipts_equal=True,
        complete_compact_file_bytes_equal=True,
        complete_physical_weight_file_bytes_equal=True,
        initial_impedance_and_wrap_file_bytes_equal=True,
        complete_causal_startup_fields_bitwise_equal=startup_fields,
        verified_family_file_sha256=verified,
        candidate_attempt_timings_accounted=b.get("all_attempt_timings_accounted"),
        full_original_event_reproduced=False,
        same_accuracy_fem_advantage_verified=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh comparison output outside protected Tatva")
    write_json(output, compare(args.reference, args.candidate))
    print(output, flush=True)


if __name__ == "__main__":
    main()

"""Check owned in-place compaction and direct IFFT against a complete BEM kernel.

Copy only retained raw branches into NEW staging, consume them, compare all
compact bytes, then run the complete direct-weight/wrap/causal-startup checker.
The source kernel is read-only. Default validation disk limit is 2 GiB.
"""

import argparse
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.cq import CQPlan
from finiteblock_bem.io import environment_record, write_json

TOOLS = Path(__file__).resolve().parent
INPLACE = runpy.run_path(str(TOOLS / "inplace_radau_compaction.py"))
DIGEST = INPLACE["CHECKPOINT"]["digest_range"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-workspace-mb", type=float, default=512)
    parser.add_argument("--max-validation-disk-mb", type=float, default=2048)
    args = parser.parse_args()
    begun = perf_counter()
    kernel, output = args.kernel.resolve(), args.output.resolve()
    if (
        output.exists()
        or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve())
        or not np.isfinite(args.max_validation_disk_mb)
        or args.max_validation_disk_mb <= 0
        or not np.isfinite(args.max_workspace_mb)
        or args.max_workspace_mb <= 0
    ):
        raise ValueError("Require fresh output outside protected Tatva and positive budgets")
    metadata = json.loads((kernel / "summary.json").read_text())
    if (
        metadata["status"] != "completed_compact_radau_resolvent_kernel_only"
        or metadata["time_method"] != "radau2"
        or metadata["time_weights_layout"] != "fortran"
        or metadata["family_spectra_coordinates"] != "diag(sqrt(w)) * L * diag(sqrt(w))"
    ):
        raise ValueError("Require a complete Fortran Radau kernel retaining raw branches")
    environment = environment_record(metadata["source_mesh_arguments"]["case"])
    for key in ["source_sha256", "case_sha256", "restart_data_sha256", "numpy", "scipy"]:
        if environment[key] != metadata["environment"][key]:
            raise ValueError("Kernel source or numerical runtime changed")
    plan = CQPlan(**metadata["plan"])
    branches = metadata["pre_combination_branch_digests"]
    if len(branches) != 2 or any(
        r["branch"] != i or r["retained_separately_after_combination"] is not True
        for i, r in enumerate(branches)
    ):
        raise ValueError("Require both retained complete raw branch receipts")
    raw_paths = [kernel / n for n in INPLACE["NAMES"]]
    for path in raw_paths:
        INPLACE["CHECKPOINT"]["header"](path, plan)
    needed = (
        sum(p.stat().st_size for p in raw_paths)
        + 2 * plan.weight_bytes
        + INPLACE["compaction_resources"](plan, 3)["maximum_extra_disk_bytes"]
    )
    if needed > args.max_validation_disk_mb * 1024**2:
        raise ValueError("Complete validation copy exceeds its explicit disk budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    if needed > shutil.disk_usage(output.parent).free:
        raise ValueError("Insufficient disk for the complete validation copy")
    protected = [
        kernel / "summary.json",
        *raw_paths,
        *[Path(r["path"]) for r in metadata["family_spectra"]],
        *[kernel / n for n in ["elastic.npz", "initial_impedance.npy", "shift_wrap.npy"]],
        Path(__file__),
        TOOLS / "inplace_radau_compaction.py",
        TOOLS / "direct_fortran_radau.py",
        TOOLS / "check_direct_fortran_radau.py",
        TOOLS / "radau_raw_sampling.py",
        TOOLS / "radau_sampling_checkpoint.py",
    ]
    frozen = {str(p.resolve()): DIGEST(p) for p in protected}
    for i, path in enumerate(raw_paths):
        if frozen[str(path)] != branches[i]["pre_combination_sha256"]:
            raise ValueError("Retained original raw branch bytes changed")
    stage = output / "compact-stage"
    stage.mkdir(parents=True)
    records, inodes = [], []
    for i, path in enumerate(raw_paths):
        target = stage / path.name
        shutil.copy2(path, target)
        records.append(
            dict(
                path=str(target),
                shape=[plan.fft_size // 2 + 1, plan.dofs, plan.dofs],
                bytes=target.stat().st_size,
                sha256=branches[i]["pre_combination_sha256"],
                all_frequency_matrices_written=True,
                role=INPLACE["ROLE"],
                mode_branch=i,
            )
        )
        inodes.append(target.stat().st_ino)
    compact, state = INPLACE["compact_inplace"](
        records,
        plan,
        stage,
        batch_matrices=3,
        max_workspace_mb=args.max_workspace_mb,
    )
    for i, (new, old) in enumerate(zip(compact, metadata["family_spectra"], strict=True)):
        if new["sha256"] != old["sha256"] or new["bytes"] != old["bytes"]:
            raise ValueError("Complete compact spectrum file bytes differ")
        if Path(new["path"]).stat().st_ino != inodes[i]:
            raise ValueError("Compaction replaced an owned file instead of reusing it")
    view = output / "kernel-view"
    view.mkdir()
    for name in ["elastic.npz", "initial_impedance.npy", "shift_wrap.npy"]:
        shutil.copy2(kernel / name, view / name)
    write_json(view / "summary.json", {**metadata, "family_spectra": compact})
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(TOOLS / "check_direct_fortran_radau.py"),
            str(view),
            "--output",
            str(output / "direct-weights"),
            "--max-workspace-mb",
            str(args.max_workspace_mb),
        ],
        check=True,
    )
    direct = json.loads((output / "direct-weights/summary.json").read_text())
    if any(DIGEST(p) != value for p, value in frozen.items()):
        raise ValueError("Original kernel or validation sources changed")
    write_json(
        output / "summary.json",
        dict(
            status="passed_complete_bem_inplace_compaction_and_direct_weights_identity",
            source_kernel=str(kernel),
            source_plan=metadata["plan"],
            frozen_input_sha256=frozen,
            complete_compact_files_byte_equal=[True, True],
            owned_source_inodes_preserved=True,
            raw_paths_consumed=True,
            original_kernel_unmodified=True,
            compaction_resources=state["resources"],
            compaction_attempts=state["attempts"],
            all_compaction_attempt_timings_accounted=state["all_attempt_timings_accounted"],
            complete_weight_files_byte_equal=direct["complete_weight_files_byte_equal"],
            wrap_coefficients_bitwise_equal=direct["wrap_coefficients_bitwise_equal"],
            complete_causal_startup_bitwise_equal=direct["complete_causal_startup_bitwise_equal"],
            initial_composition_inf_error=direct["initial_composition_inf_error"],
            total_validation_seconds_including_hashes_and_copies=perf_counter() - begun,
            producer_ownership_transfer_implemented=False,
            integrated_into_production_pipeline=False,
            full_original_event_reproduced=False,
            same_accuracy_fem_advantage_verified=False,
            scope="Complete small BEM arithmetic/storage identity, not a nonlinear or FEM accuracy/speed proof",
        ),
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

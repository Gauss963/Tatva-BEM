"""Verify direct inversion against a complete Fortran-weight Radau kernel.

Compare every physical weight byte, wrap coefficient and causal actuator
startup value. No new BEM frequency assembly and no fitted loading/time shift.
Use a fresh output; this validation may itself require a complete weight copy.
"""

import argparse
import json
import runpy
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.cq import CQPlan
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
DIRECT = runpy.run_path(str(TOOLS / "direct_fortran_radau.py"))
STORAGE = runpy.run_path(str(TOOLS / "compact_radau_storage.py"))
RK = runpy.run_path(str(TOOLS / "radau_experiment.py"))
DIGEST = DIRECT["JOURNAL"]["digest_range"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-workspace-mb", type=float, default=10240)
    args = parser.parse_args()
    begun = perf_counter()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh output outside protected Tatva")
    metadata = json.loads((args.kernel / "summary.json").read_text())
    if (
        metadata["status"] != "completed_compact_radau_resolvent_kernel_only"
        or metadata["time_method"] != "radau2"
        or metadata["time_weights_layout"] != "fortran"
        or metadata["family_spectra_coordinates"] != "diag(sqrt(w)) * L * diag(sqrt(w))"
    ):
        raise ValueError("Require a complete work-coordinate compact Radau kernel")
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    for key in ["source_sha256", "case_sha256", "restart_data_sha256", "numpy", "scipy"]:
        if environment[key] != metadata["environment"][key]:
            raise ValueError("Kernel source or runtime differs from this validation")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    con, ref, law = model.constraints, model.reference, model.reference.contact
    line = np.r_[con.contact_weights, con.contact_weights, con.platen_width]
    plan = CQPlan(**metadata["plan"])
    if len(line) != plan.dofs:
        raise ValueError("Retained trace differs from recorded kernel")
    files = [
        Path(__file__),
        TOOLS / "direct_fortran_radau.py",
        TOOLS / "compact_radau_storage.py",
        TOOLS / "radau_experiment.py",
        TOOLS / "radau_sampling_checkpoint.py",
        args.kernel / "summary.json",
    ]
    frozen = {str(p.resolve()): DIGEST(p) for p in files}
    rows = metadata["inversion"]["fft_rows"]
    paths, wrap, state = DIRECT["invert_physical_fortran"](
        metadata["family_spectra"],
        plan,
        line,
        output,
        fft_rows=rows,
        max_workspace_mb=args.max_workspace_mb,
    )
    byte_equal = []
    for old, new in zip(metadata["family_weights"], state["files"][:2], strict=True):
        if (
            DIGEST(old["path"]) != old["sha256"]
            or old["sha256"] != new["sha256"]
            or old["bytes"] != new["bytes"]
        ):
            raise ValueError("Complete physical Fortran weight files differ")
        byte_equal.append(True)
    old_wrap = np.load(args.kernel / "shift_wrap.npy", allow_pickle=False)
    if DIGEST(args.kernel / "shift_wrap.npy") != metadata[
        "shift_wrap_sha256"
    ] or not np.array_equal(wrap.view(np.uint64), old_wrap.view(np.uint64)):
        raise ValueError("Physical finite-DFT shift-wrap coefficients differ")
    if DIGEST(args.kernel / "elastic.npz") != metadata["elastic_sha256"]:
        raise ValueError("Original causal startup data changed")
    with np.load(args.kernel / "elastic.npz", allow_pickle=False) as saved:
        stage_time = saved["stage_time"]
        expected = saved["elastic"]
    increment = options.get("diagnostic_actuator_increment")
    rate = ref.actuator_rate if increment is None else increment / metadata["arguments"]["duration"]
    spring = ref.spring_stiffness * con.platen_width
    forcing = spring * rate * stage_time
    actual = np.empty_like(expected)
    spectra = [
        np.load(r["path"], mmap_mode="r", allow_pickle=False) for r in metadata["family_spectra"]
    ]
    try:
        for first in range(0, plan.dofs, metadata["arguments"]["startup_rows"]):
            last = min(first + metadata["arguments"]["startup_rows"], plan.dofs)
            columns = [s[:, first:last, -1] / np.sqrt(line[first:last] * line[-1]) for s in spectra]
            actual[:, :, first:last] = STORAGE["causal_compact_startup"](
                *columns, wrap[first:last, -1], replace(plan, dofs=last - first), forcing
            )
    finally:
        for s in spectra:
            s._mmap.close()
    if not np.array_equal(actual.view(np.uint64), expected.view(np.uint64)):
        raise ValueError("Complete causal actuator startup differs")
    np.save(output / "elastic-startup.npy", actual, allow_pickle=False)
    maps = [np.load(path, mmap_mode="r", allow_pickle=False) for path in paths]
    try:
        instantaneous = RK["expand_compact_weights"](maps[0][:1], maps[1][:1], wrap)[0]
    finally:
        for m in maps:
            m._mmap.close()
    if DIGEST(args.kernel / "initial_impedance.npy") != metadata["initial_impedance_sha256"]:
        raise ValueError("Original initial impedance changed")
    impedance = np.load(args.kernel / "initial_impedance.npy", allow_pickle=False)
    tangent = np.diag(
        np.r_[
            con.contact_weights * law.normal_penalty,
            con.contact_weights * law.tangential_penalty,
            spring,
        ]
    )
    composition = float(
        np.linalg.norm(
            instantaneous @ (impedance + np.kron(np.eye(2), tangent)) - np.eye(2 * plan.dofs),
            ord=np.inf,
        )
    )
    if not np.isfinite(composition) or composition > 1e-6:
        raise ValueError("Direct Fortran lag-zero inverse composition failed")
    if any(DIGEST(p) != value for p, value in frozen.items()):
        raise ValueError("Validation sources changed")
    write_json(
        output / "summary.json",
        dict(
            status="passed_complete_direct_fortran_weights_wrap_and_causal_startup",
            source_kernel=str(args.kernel.resolve()),
            source_plan=metadata["plan"],
            frozen_input_sha256=frozen,
            complete_weight_files_byte_equal=byte_equal,
            wrap_coefficients_bitwise_equal=True,
            complete_causal_startup_bitwise_equal=True,
            initial_composition_inf_error=composition,
            resources=state["resources"],
            inversion_attempts=state["attempts"],
            all_attempt_timings_accounted=state["all_attempt_timings_accounted"],
            no_extra_c_order_time_file=True,
            source_frequency_or_weights_modified=False,
            expanded_full_stage_time_weights_stored=False,
            total_validation_seconds_including_hashes=perf_counter() - begun,
            full_original_event_reproduced=False,
            same_accuracy_fem_advantage_verified=False,
            scope="Storage/inversion identity against a complete BEM kernel; no nonlinear/FEM convergence or same-accuracy speed proof",
        ),
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

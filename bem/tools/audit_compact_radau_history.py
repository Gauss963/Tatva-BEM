"""Audit all saved stages against the complete causal original RK operator."""

import argparse
import gc
import json
import os
import resource
import runpy
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.fft import set_workers

from finiteblock_bem.cq import CQPlan
from finiteblock_bem.cq_storage import DiskSpectrum, spectrum_bytes
from finiteblock_bem.galerkin_fast import prepared_geometry_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOLS = Path(__file__).resolve().parent
READ = runpy.run_path(str(TOOLS / "replay_resolvent.py"))
AUDIT = runpy.run_path(str(TOOLS / "radau_original_history_audit.py"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fft-rows", type=int, default=4)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("FDBEM_CORES", "1")))
    parser.add_argument("--max-workspace-mb", type=float, default=8192)
    parser.add_argument("--max-memory-mb", type=float, default=12288)
    parser.add_argument("--residual-tolerance-mpa", type=float, default=1e-4)
    parser.add_argument("--frequency-relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--allow-prefix", action="store_true", help="Explicit diagnostic only")
    parser.add_argument("--remove-reconstructed-families", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh audit output outside protected Tatva")
    if (
        args.workers < 1
        or args.fft_rows < 1
        or len(set(p.resolve() for p in args.runs)) != len(args.runs)
        or not all(
            np.isfinite(v) and v > 0
            for v in [
                args.max_workspace_mb,
                args.max_memory_mb,
                args.residual_tolerance_mpa,
                args.frequency_relative_tolerance,
            ]
        )
    ):
        raise ValueError("Require distinct runs and positive finite audit budgets/tolerances")
    begun = perf_counter()
    records = [json.loads((p / "summary.json").read_text()) for p in args.runs]
    kernel = Path(records[0]["prepared_kernel"]).resolve()
    metadata_path = kernel / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    full_plan = CQPlan(**metadata["plan"])
    steps = records[0]["completed_steps"]
    if (
        metadata["status"] != "completed_compact_radau_resolvent_kernel_only"
        or metadata["time_method"] != "radau2"
        or full_plan.steps != metadata["full_steps"]
        or not 1 <= steps <= full_plan.steps
        or (steps != full_plan.steps and not args.allow_prefix)
    ):
        raise ValueError("Require a complete Radau kernel/horizon, or explicitly allow a prefix")
    plan = replace(full_plan, steps=steps)
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    keys = ["source_sha256", "case_sha256", "restart_data_sha256"]
    for record in [metadata, *records]:
        if any(record["environment"][k] != environment[k] for k in keys):
            raise ValueError("Prepared Radau package or physical inputs changed")
    for record in records:
        if (
            record["status"] != "completed_requested_horizon"
            or record["time_method"] != "radau2"
            or Path(record["prepared_kernel"]).resolve() != kernel
            or record["completed_steps"] != steps
            or record["dt_s"] != plan.dt
            or not record["all_history_retained"]
            or record["kernel_summary_sha256"] != fingerprint(metadata_path)
        ):
            raise ValueError("Require aligned complete saved Radau trajectories")
    storage = metadata["family_spectra"]
    if (
        len(storage) != 2
        or not all(r["all_frequency_matrices_written"] for r in storage)
        or metadata["family_spectra_coordinates"] != "diag(sqrt(w)) * L * diag(sqrt(w))"
    ):
        raise ValueError("Require both completed work-coordinate RK families")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    trajectories_bytes = (steps + 1) * (2 * plan.dofs * len(records) * 8 + 16 * plan.dofs * 8)
    workspace = (
        AUDIT["AUDIT"]["estimate_action_arrays"](plan, steps, 2 * len(records), args.fft_rows)
        + 6 * (steps + 1) * 2 * plan.dofs * len(records) * 8
    )
    geometry = sum(
        prepared_geometry_bytes(mesh, options["quadrature"])
        for mesh in [model.reference.moving, model.reference.stationary]
    )
    direct_solver = (
        32
        * (
            model.constraints.size**2
            + model.reference.moving.dofs**2
            + model.reference.stationary.dofs**2
        )
        * 16
    )
    peak = max(workspace + trajectories_bytes + 1024**3, geometry + direct_solver + 1024**3)
    if workspace > args.max_workspace_mb * 1024**2 or peak > args.max_memory_mb * 1024**2:
        raise ValueError("Complete original RK action exceeds its explicit array budget")
    selected = [
        Path(__file__),
        TOOLS / "radau_original_history_audit.py",
        TOOLS / "original_history_audit.py",
        TOOLS / "radau_raw_sampling.py",
        TOOLS / "radau_experiment.py",
        TOOLS / "replay_resolvent.py",
        metadata_path,
        *[p / "summary.json" for p in args.runs],
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in selected}

    def verify():
        current = environment_record(options["case"])
        if any(fingerprint(p) != value for p, value in frozen.items()) or any(
            current[k] != environment[k] for k in keys
        ):
            raise ValueError("Frozen original RK audit inputs changed")

    required = 2 * spectrum_bytes(plan) + trajectories_bytes + 1024**2
    output.parent.mkdir(parents=True, exist_ok=True)
    if required > shutil.disk_usage(output.parent).free:
        raise ValueError("Insufficient disk for both original RK families and residual evidence")
    output.mkdir()
    write_json(
        output / "preflight.json",
        dict(
            status="preflight_only_not_an_original_equation_result",
            plan=asdict(plan),
            complete_kernel_steps=full_plan.steps,
            prefix_explicitly_allowed=args.allow_prefix,
            estimated_action_array_mib=workspace / 1024**2,
            estimated_total_array_mib=peak / 1024**2,
            required_disk_bytes=required,
            frozen_inputs_sha256=frozen,
            original_tatva_readonly=True,
        ),
    )
    if args.preflight_only:
        print(output / "preflight.json", flush=True)
        return
    source_stats, sources = [], []
    for item in storage:
        path = Path(item["path"])
        if READ["digest_file"](path) != item["sha256"]:
            raise ValueError("Completed source RK spectrum changed")
        source_stats.append((path, path.stat().st_size, path.stat().st_mtime_ns))
        sources.append(READ["SavedDiskSpectrum"](path, plan))
    primal_digests = {
        str((p / "stages.npz").resolve()): READ["digest_file"](p / "stages.npz") for p in args.runs
    }
    write_json(
        output / "run_manifest.json",
        dict(
            status="prepared_not_completed",
            verified_family_spectra=storage,
            primal_stage_sha256=primal_digests,
            frozen_inputs_sha256=frozen,
            workers=args.workers,
            automatic_restart=False,
        ),
    )
    target_paths = [output / f"work-original-family{i}.npy" for i in range(2)]
    targets = [DiskSpectrum(p, plan) for p in target_paths]
    inversion = AUDIT["invert_compact_spectra"](*sources, targets, plan, workers=args.workers)
    target_storage = [target.record() for target in targets]
    write_json(
        output / "completed_operator.json", dict(storage=target_storage, inversion=inversion)
    )
    wc = model.constraints.contact_weights
    line = np.r_[wc, wc, model.constraints.platen_width]
    scale = np.sqrt(line)
    law = model.reference.contact
    diagonal = np.r_[wc * law.normal_penalty, wc * law.tangential_penalty, 0]
    spring = model.reference.spring_stiffness * model.constraints.platen_width
    reference_matrix = np.diag(diagonal)
    reference_matrix[-1, -1] += spring
    transfer = NodalInterfaceTransfer(model, quadrature=options["quadrature"])
    indices = sorted(
        {0, plan.fft_size // 2}
        | {min(round(f * plan.fft_size * plan.dt), plan.fft_size // 2) for f in [1000, 10000]}
    )
    checks = []
    check_start = perf_counter()
    for index in indices:
        verify()
        zeta = plan.radius * np.exp(-2j * np.pi * index / plan.fft_size)
        modes = AUDIT["RAW"]["lower_half_radau_modes"](zeta, 1.0)
        for branch, w in enumerate(modes):
            fresh = (transfer.evaluate(w / plan.dt) + reference_matrix) / (
                scale[:, None] * scale[None, :]
            )
            reconstructed = targets[0][index] + w * targets[1][index]
            sampled_l = sources[0][index] + w * sources[1][index]
            error = float(np.linalg.norm(reconstructed - fresh) / np.linalg.norm(fresh))
            composition = float(np.linalg.norm(fresh @ sampled_l - np.eye(plan.dofs), ord=np.inf))
            checks.append(
                dict(
                    index=index,
                    branch=branch,
                    relative_work_frobenius=error,
                    original_solve_composition_inf=composition,
                )
            )
            if max(error, composition) > args.frequency_relative_tolerance:
                raise ValueError("Fresh original RK frequency check failed")
    fresh_seconds = perf_counter() - check_start
    del transfer, sources
    q = np.empty((steps + 1, 2, plan.dofs, len(records)))
    for column, folder in enumerate(args.runs):
        with np.load(folder / "stages.npz", allow_pickle=False) as data:
            q[..., column] = data["displacement"]
            np.testing.assert_array_equal(q[0, ..., column], 0)
            expected_time = np.zeros((steps + 1, 2))
            expected_time[1:] = (np.arange(steps)[:, None] + AUDIT["RADAU"]["C"]) * plan.dt
            np.testing.assert_allclose(data["stage_time"], expected_time, rtol=1e-12, atol=1e-15)
    gc.collect()
    with set_workers(args.workers):
        action, action_diagnostics = AUDIT["causal_compact_original_action"](
            *targets, plan, q, line, fft_rows=args.fft_rows, max_workspace_mb=args.max_workspace_mb
        )
    results = []
    for column, folder in enumerate(args.runs):
        with np.load(folder / "stages.npz", allow_pickle=False) as data:
            response = data["response"]
            force = np.zeros((steps + 1, 2, plan.dofs))
            force[:, :, : len(wc)] = response[:, :, :, 0].transpose(0, 2, 1) * wc
            force[:, :, len(wc) : 2 * len(wc)] = response[:, :, :, 1].transpose(0, 2, 1) * wc
            residual = action[..., column] - diagonal * q[..., column] + force
            residual += data["initial_residual_force"] - data["initial_contact_force"]
            residual[:, :, -1] -= spring * (data["actuator"] - data["actuator"][0])
            residual /= line
        if not np.all(np.isfinite(residual)):
            raise ValueError("Nonfinite original RK equation residual")
        location = np.unravel_index(np.argmax(abs(residual)), residual.shape)
        np.savez(output / f"residual-{column}.npz", residual_mpa=residual)
        results.append(
            dict(
                run=str(folder.resolve()),
                maximum_original_equation_residual_mpa=float(np.max(abs(residual))),
                worst_step=int(location[0]),
                worst_stage=int(location[1]),
                worst_dof=int(location[2]),
                all_saved_stages_checked=True,
            )
        )
    verify()
    for path, size, modified in source_stats:
        if path.stat().st_size != size or path.stat().st_mtime_ns != modified:
            raise ValueError("Source RK spectrum changed during the audit")
    if any(READ["digest_file"](p) != value for p, value in primal_digests.items()):
        raise ValueError("Saved primal RK stages changed during the audit")
    passed = all(
        r["maximum_original_equation_residual_mpa"] <= args.residual_tolerance_mpa for r in results
    )
    write_json(
        output / "summary.json",
        dict(
            status="passed_original_causal_radau_equation_audit"
            if passed
            else "failed_original_causal_radau_equation_audit",
            plan=asdict(plan),
            complete_prepared_horizon_checked=steps == full_plan.steps,
            kernel=str(kernel),
            results=results,
            original_equation_tolerance_mpa=args.residual_tolerance_mpa,
            fresh_original_mode_checks=checks,
            frequency_relative_tolerance=args.frequency_relative_tolerance,
            inversion=inversion,
            action=action_diagnostics,
            reconstructed_operator_storage=target_storage,
            reconstructed_families_removed_after_check=args.remove_reconstructed_families,
            original_initial_residual_force_retained=True,
            environment=environment,
            frozen_inputs_sha256=frozen,
            timing_seconds=dict(fresh_original_samples=fresh_seconds, total=perf_counter() - begun),
            peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2 if sys.platform == "darwin" else 1024),
            full_original_event_reproduced=False,
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
            scope="All saved stage equations and fresh sampled original modes; interpolation/physical convergence and initialized AD remain separate",
        ),
    )
    if args.remove_reconstructed_families:
        del targets
        gc.collect()
        for path in target_paths:
            path.unlink()
    print(output / "summary.json", flush=True)
    if not passed:
        raise ValueError("Complete causal original RK equilibrium did not pass")


if __name__ == "__main__":
    main()

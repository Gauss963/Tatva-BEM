"""Audit full refined trajectories in the original causal BEM equations.

Reconstruct A=Z+Dref+spring from the completed work-coordinate resolvent,
retain every required coefficient, and independently apply the causal
operator to all selected trajectories. Fresh BEM frequency evaluations check
the reconstruction at several contour points. All outputs use new paths.
"""

import argparse
import gc
import json
import os
import resource
import runpy
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.fft import set_workers

from finiteblock_bem.cq import CQPlan, bdf2_symbol
from finiteblock_bem.cq_storage import DiskSpectrum, spectrum_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOLS = Path(__file__).resolve().parent
READ = runpy.run_path(str(TOOLS / "replay_resolvent.py"))
AUDIT = runpy.run_path(str(TOOLS / "original_history_audit.py"))
REFINED = runpy.run_path(str(TOOLS / "refined_contact.py"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fft-rows", type=int, default=4)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("FDBEM_CORES", "1")))
    parser.add_argument("--max-workspace-mb", type=float, default=4096)
    parser.add_argument("--max-memory-mb", type=float, default=8192)
    parser.add_argument("--residual-tolerance-mpa", type=float, default=1e-4)
    parser.add_argument("--frequency-relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a new audit output outside protected Tatva")
    if (
        args.fft_rows < 1
        or args.workers < 1
        or len(set(p.resolve() for p in args.runs)) != len(args.runs)
        or not all(
            np.isfinite(v) and v > 0
            for v in (
                args.max_workspace_mb,
                args.max_memory_mb,
                args.residual_tolerance_mpa,
                args.frequency_relative_tolerance,
            )
        )
    ):
        raise ValueError("Require distinct runs and positive finite audit budgets/tolerances")
    begun = perf_counter()
    records = [json.loads((p / "summary.json").read_text()) for p in args.runs]
    kernel = Path(records[0]["prepared_kernel"])
    metadata_path = kernel / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    plan = CQPlan(**metadata["plan"])
    if (
        metadata["status"] != "completed_resolvent_kernel_only"
        or plan.steps != metadata["full_steps"]
        or plan.steps * plan.dt != 0.045
    ):
        raise ValueError("Require the complete original 45 ms history")
    for record in records:
        if (
            record["status"] != "completed_requested_horizon"
            or record["history_representation"] != "elastic_resolvent_refined_contact"
            or record["completed_steps"] != plan.steps
            or not record["all_history_retained"]
            or record["correction_start"] != 1
            or Path(record["prepared_kernel"]).resolve() != kernel.resolve()
            or record["kernel_summary_sha256"] != fingerprint(metadata_path)
        ):
            raise ValueError("Require complete fine-contact primals using the same complete kernel")
        if any(fingerprint(path) != value for path, value in record["frozen_input_sha256"].items()):
            raise ValueError("A frozen primal implementation or lifting input changed")
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    for key in ("source_sha256", "case_sha256", "restart_data_sha256"):
        if environment[key] != metadata["environment"][key] or any(
            environment[key] != record["environment"][key] for record in records
        ):
            raise ValueError("Prepared bulk package or original physical inputs changed")
    storage = metadata["spectrum_storage"]
    if (
        storage["coordinates"] != "diag(sqrt(w)) * L * diag(sqrt(w))"
        or not storage["all_frequency_matrices_written"]
    ):
        raise ValueError("Require a completed work-coordinate resolvent spectrum")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    nodes = model.reference.moving.starts[model.moving_contact, 1]
    spaces = [REFINED["ContactQuadrature"].build(nodes, r["subdivisions"]) for r in records]
    workspace = AUDIT["estimate_action_arrays"](plan, plan.steps, len(records), args.fft_rows)
    arrays = workspace + 2 * (plan.steps + 1) * max(s.points for s in spaces) * 8 + 1024**3
    if workspace > args.max_workspace_mb * 1024**2 or arrays > args.max_memory_mb * 1024**2:
        raise ValueError("Complete independent operator action exceeds the explicit array budget")
    selected = [
        Path(__file__),
        TOOLS / "original_history_audit.py",
        TOOLS / "refined_contact.py",
        TOOLS / "replay_resolvent.py",
        metadata_path,
        *[p / "summary.json" for p in args.runs],
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in selected}
    required_disk = spectrum_bytes(plan) + (plan.steps + 1) * plan.dofs * len(records) * 8 + 65536
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    if required_disk > free:
        raise ValueError("Insufficient disk for the reconstructed operator and residual evidence")
    output.mkdir()
    write_json(
        output / "preflight.json",
        dict(
            status="preflight_only_not_an_original_equation_result",
            plan=asdict(plan),
            runs=list(map(str, args.runs)),
            required_disk_bytes=required_disk,
            available_disk_bytes=free,
            estimated_action_array_mib=workspace / 1024**2,
            estimated_total_array_mib=arrays / 1024**2,
            no_new_time_grid_or_physical_parameters=True,
            frozen_input_sha256=frozen,
            expected_source_resolvent_sha256=storage["sha256"],
            original_tatva_readonly=True,
        ),
    )
    if args.preflight_only:
        print(output / "preflight.json", flush=True)
        return
    source_path = Path(storage["path"])
    source_stat = source_path.stat()
    if READ["digest_file"](source_path) != storage["sha256"]:
        raise ValueError("Completed source resolvent changed")
    primal_digests = {
        str(p.resolve()): READ["digest_file"](p / "trajectory.npz") for p in args.runs
    }
    write_json(
        output / "run_manifest.json",
        dict(
            status="prepared_not_completed",
            frozen_input_sha256=frozen,
            primal_trajectory_sha256=primal_digests,
            verified_source_resolvent_sha256=storage["sha256"],
            workers=args.workers,
            blas_threads_per_worker=1,
            automatic_restart=False,
        ),
    )
    source = READ["SavedDiskSpectrum"](source_path, plan)
    inverse = DiskSpectrum(output / "work-original-operator.npy", plan)
    inversion = AUDIT["invert_work_spectrum"](source, inverse, workers=args.workers)
    inverse_storage = inverse.record()
    inverse_storage["coordinates"] = "inv(diag(sqrt(w)) * L * diag(sqrt(w)))"
    write_json(
        output / "completed_operator.json", dict(storage=inverse_storage, inversion=inversion)
    )
    wc = model.constraints.contact_weights
    line = np.r_[wc, wc, model.constraints.platen_width]
    scale = np.sqrt(line)
    law = model.reference.contact
    diagonal = np.r_[wc * law.normal_penalty, wc * law.tangential_penalty, 0]
    spring = model.reference.spring_stiffness * model.constraints.platen_width
    reference_matrix = np.diag(diagonal)
    reference_matrix[-1, -1] += spring
    # These freshly assembled matrices are independent of saved L and its
    # adaptive interpolation. They are spot checks, not an all-contour bound.
    transfer = NodalInterfaceTransfer(model, quadrature=options["quadrature"])
    indices = sorted(
        {0, plan.fft_size // 2}
        | {
            min(round(f * plan.fft_size * plan.dt), plan.fft_size // 2)
            for f in (1000, 10000, 50000)
        }
    )
    checks = []
    check_start = perf_counter()
    for index in indices:
        zeta = plan.radius * np.exp(-2j * np.pi * index / plan.fft_size)
        s = bdf2_symbol(zeta) / plan.dt
        fresh = (transfer.evaluate(s) + reference_matrix) / (scale[:, None] * scale[None, :])
        reconstructed = inverse[index]
        error = float(np.linalg.norm(reconstructed - fresh) / np.linalg.norm(fresh))
        checks.append(
            dict(index=index, s_real=s.real, s_imag=s.imag, relative_work_frobenius=error)
        )
    frequency_seconds = perf_counter() - check_start
    del transfer, source
    q = np.empty((plan.steps + 1, plan.dofs, len(records)))
    for column, (folder, space) in enumerate(zip(args.runs, spaces, strict=True)):
        with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["time"], np.arange(plan.steps + 1) * plan.dt)
            np.testing.assert_array_equal(data["retained_contact_y"], nodes)
            np.testing.assert_array_equal(data["contact_y"], space.y)
            np.testing.assert_array_equal(data["contact_weights"], space.weights)
            q[..., column] = data["retained_displacement"]
            np.testing.assert_array_equal(q[0, :, column], 0)
    gc.collect()
    with set_workers(args.workers):
        action, action_diagnostics = AUDIT["causal_original_action"](
            inverse,
            plan,
            q,
            line,
            fft_rows=args.fft_rows,
            max_workspace_mb=args.max_workspace_mb,
        )
    results = []
    for column, (folder, space) in enumerate(zip(args.runs, spaces, strict=True)):
        with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
            contact = space.pull_force(data["pressure"], data["shear"])
            residual = (
                AUDIT["physical_residual"](
                    action[..., column],
                    q[..., column],
                    contact,
                    data["initial_retained_bulk_force"],
                    diagonal,
                    spring,
                    data["initial_retained_displacement"][-1],
                    data["actuator"],
                )
                / np.r_[wc, wc, model.constraints.platen_width]
            )
        if not np.all(np.isfinite(residual)):
            raise ValueError("Nonfinite original-equation residual")
        location = np.unravel_index(np.argmax(abs(residual)), residual.shape)
        maximum = float(np.max(abs(residual)))
        results.append(
            dict(
                run=str(folder.resolve()),
                subdivisions=space.subdivisions,
                maximum_original_residual_mpa=maximum,
                largest_residual_time_s=location[0] * plan.dt,
                largest_residual_retained_row=int(location[1]),
                original_residual_divided_by_normal_load=maximum / model.reference.normal_pressure,
                requested_absolute_tolerance_mpa=args.residual_tolerance_mpa,
                original_equilibrium_passed=maximum <= args.residual_tolerance_mpa,
                replay_root_residual_mpa=records[column]["max_boundary_contact_residual_mpa"],
            )
        )
        np.savez(output / f"original_residual_{column}.npz", residual_mpa=residual)
        del residual, contact
    frequency_passed = (
        max(c["relative_work_frobenius"] for c in checks) <= args.frequency_relative_tolerance
    )
    passed = frequency_passed and all(r["original_equilibrium_passed"] for r in results)
    if any(fingerprint(p) != value for p, value in frozen.items()) or any(
        READ["digest_file"](Path(p) / "trajectory.npz") != value
        for p, value in primal_digests.items()
    ):
        raise ValueError("Audit implementation or frozen primal changed")
    current = source_path.stat()
    if (current.st_size, current.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise ValueError("Read-only source spectrum changed during audit")
    final_environment = environment_record(options["case"])
    if any(
        environment[k] != final_environment[k]
        for k in ("source_sha256", "case_sha256", "restart_data_sha256")
    ):
        raise ValueError("Bulk package or physical input changed during audit")
    write_json(
        output / "summary.json",
        dict(
            status="passed_original_causal_equation_audit"
            if passed
            else "failed_original_equation_audit",
            environment=environment,
            frozen_input_sha256=frozen,
            primal_trajectory_sha256=primal_digests,
            source_resolvent_sha256=storage["sha256"],
            inverse_operator_storage=inverse_storage,
            inversion=inversion,
            fresh_frequency_checks=checks,
            fresh_frequency_relative_tolerance=args.frequency_relative_tolerance,
            fresh_frequency_check_seconds=frequency_seconds,
            complete_causal_action=action_diagnostics,
            runs=results,
            total_including_hashes_and_output_seconds=perf_counter() - begun,
            peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2 if sys.platform == "darwin" else 1024),
            scope="Complete original-equation causal algebra plus fresh frequency spot checks; not an all-frequency interpolation certificate or physical convergence proof",
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
        ),
    )
    print(output / "summary.json", flush=True)
    if not passed:
        raise ValueError("Original-equation audit failed; residuals and all evidence preserved")


if __name__ == "__main__":
    main()

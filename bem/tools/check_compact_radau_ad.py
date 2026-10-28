"""Reproduce complete RK adjoint/JVP checks with a measured healed preload.

The original causal equation audit is required. The default differentiates
the selected static active set; fixed initialization is an explicit control.
The recorded stage actuator is independent of the contact parameters, so
the endogenous stop-time derivative is outside this check.
"""

import argparse
import gc
import json
import resource
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.cq import BoundaryHistory, CQPlan
from finiteblock_bem.galerkin_fast import prepared_geometry_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model, nodal_body_stiffness

TOOLS = Path(__file__).resolve().parent
READ = runpy.run_path(str(TOOLS / "replay_resolvent.py"))


def estimate_ad_arrays(plan, contact_points, *, batch_size=3, cache_mb=8, fft_rows=2):
    """Conservative explicit-array model for the existing batched RK engines.

    Counts both pending/input arrays of both families, padded batched source
    FFTs, streamed kernel rows, saved stages and dense root workspaces. The
    reserve covers JAX/runtime allocations heuristically; mapped weight pages
    and the OS page cache are excluded. This is not a hard process RSS cap.
    """
    if (
        not isinstance(batch_size, int)
        or not 1 <= batch_size <= 7
        or not isinstance(fft_rows, int)
        or fft_rows < 1
        or not isinstance(contact_points, int)
        or contact_points < 1
        or plan.steps < 1
        or plan.dofs != 2 * contact_points + 1
        or not np.isfinite(cache_mb)
        or cache_mb < 0
    ):
        raise ValueError("Require an aligned identity contact grid and positive AD budgets")
    d, count, channels = plan.dofs, plan.steps + 1, 2 * batch_size
    largest = 1 << (int(plan.steps).bit_length() - 1)
    rows = min(fft_rows, d)
    histories = 4 * count * d * channels * 8
    source_fft = 4 * (largest + 1) * d * channels * 16
    row_fft = 4 * (largest + 1) * rows * d * 16
    product_fft = 4 * (largest + 1) * rows * channels * 16
    saved_stages = count * (2 * d + 14 * contact_points + 32) * 8
    dense_root = 64 * (2 * d) ** 2 * 8
    runtime_reserve = 2 * 1024**3
    cached = int(cache_mb * 1024**2)
    return dict(
        peak=histories
        + source_fft
        + row_fft
        + product_fft
        + saved_stages
        + dense_root
        + runtime_reserve
        + cached,
        histories=histories,
        source_fft=source_fft,
        streamed_kernel_fft=row_fft,
        product_fft=product_fft,
        saved_stages=saved_stages,
        dense_root=dense_root,
        runtime_reserve=runtime_reserve,
        configured_fft_cache=cached,
        direction_batch_size=batch_size,
        largest_dyadic_block=largest,
    )


def restart_path(record, override):
    if record is None:
        if override is not None:
            raise ValueError("An AD control cannot change the primal's stock restart seed")
        return None
    path = Path(record["path"]) if override is None else Path(override)
    if not path.is_file() and override is None:
        path = TOOLS.parent / "src/finiteblock_bem/resources/data/ev16_restart_float64.npz"
    if not path.is_file() or READ["digest_file"](path) != record["sha256"]:
        raise ValueError("Require the exact restart values used by the saved RK trajectory")
    return path.resolve()


def verification_directions(steps):
    parameters = np.c_[np.eye(5), np.zeros((5, 2))]
    actuator = np.zeros((steps + 1, 2, 7))
    actuator[1:, :, 5] = np.sin(np.arange(2 * steps).reshape(-1, 2) * 0.41)
    actuator[1:, :, 6] = np.random.default_rng(98).normal(size=(steps, 2))
    return parameters, actuator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--operator-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initialization-gradient", choices=["active-set", "fixed"], default="active-set"
    )
    parser.add_argument("--restart-seed", type=Path)
    parser.add_argument("--jvp-batch-size", type=int, choices=range(1, 8), default=3)
    parser.add_argument("--fft-cache-mb", type=float, default=8)
    parser.add_argument("--fft-rows", type=int, default=2)
    parser.add_argument("--max-memory-mb", type=float, default=12288)
    parser.add_argument("--gradient-rtol", type=float, default=1e-7)
    parser.add_argument("--gradient-atol", type=float, default=1e-18)
    parser.add_argument("--allow-prefix", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh AD output outside protected Tatva")
    if (
        args.fft_rows < 1
        or not np.isfinite(args.fft_cache_mb)
        or args.fft_cache_mb < 0
        or not all(
            np.isfinite(v) and v > 0
            for v in [args.max_memory_mb, args.gradient_rtol, args.gradient_atol]
        )
    ):
        raise ValueError("Require positive finite resource and gradient-check options")
    begun = perf_counter()
    run_path = args.run / "summary.json"
    run = json.loads(run_path.read_text())
    kernel = Path(run["prepared_kernel"])
    metadata_path = kernel / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    audit_path = args.operator_audit / "summary.json"
    audit = json.loads(audit_path.read_text())
    manifest_path = args.operator_audit / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    full_plan = CQPlan(**metadata["plan"])
    steps = run["completed_steps"]
    plan = replace(full_plan, steps=steps)
    if (
        run["status"] != "completed_requested_horizon"
        or run["time_method"] != "radau2"
        or not run["all_history_retained"]
        or run["source_changed_during_run"]
        or run["static_initialization"] != "fem-body-active-set"
        or run["kernel_summary_sha256"] != fingerprint(metadata_path)
        or metadata["status"] != "completed_compact_radau_resolvent_kernel_only"
        or metadata["time_method"] != "radau2"
        or not 1 <= steps <= full_plan.steps
        or run["dt_s"] != plan.dt
        or audit["plan"] != dict(metadata["plan"], steps=steps)
        or bool(audit["complete_prepared_horizon_checked"]) != (steps == full_plan.steps)
        or (steps != full_plan.steps and not args.allow_prefix)
        or audit["status"] != "passed_original_causal_radau_equation_audit"
        or Path(audit["kernel"]).resolve() != kernel.resolve()
    ):
        raise ValueError(
            "Require matching complete measured-body RK stages and a passed original audit"
        )
    nonlinear_error = run["max_boundary_contact_residual_mpa"]
    if not np.isfinite(nonlinear_error) or not 0 <= nonlinear_error <= 1e-7:
        raise ValueError("Implicit RK derivatives require the original nonlinear root tolerance")
    matches = [r for r in audit["results"] if Path(r["run"]).resolve() == args.run.resolve()]
    if len(matches) != 1 or not matches[0]["all_saved_stages_checked"]:
        raise ValueError("Original audit must identify every stage of this trajectory")
    residual = matches[0]["maximum_original_equation_residual_mpa"]
    if not np.isfinite(residual) or not 0 <= residual <= audit["original_equation_tolerance_mpa"]:
        raise ValueError("Original RK equation bound did not pass")
    for p in [run_path, metadata_path]:
        if audit["frozen_inputs_sha256"][str(p.resolve())] != fingerprint(p):
            raise ValueError("Saved primal differs from its original causal audit")
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    for item in [run, metadata, audit]:
        if any(
            item["environment"][k] != environment[k]
            for k in ["source_sha256", "case_sha256", "restart_data_sha256"]
        ):
            raise ValueError("Numerical package or physical case changed")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    m = len(model.moving_contact)
    arrays = estimate_ad_arrays(
        plan, m, batch_size=args.jvp_batch_size, cache_mb=args.fft_cache_mb, fft_rows=args.fft_rows
    )
    geometry = sum(
        prepared_geometry_bytes(mesh, options["quadrature"])
        for mesh in [model.reference.moving, model.reference.stationary]
    )
    static_peak = (
        geometry
        + 32
        * (
            model.constraints.size**2
            + model.reference.moving.dofs**2
            + model.reference.stationary.dofs**2
        )
        * 16
        + arrays["runtime_reserve"]
    )
    total_peak = max(arrays["peak"], static_peak)
    if total_peak > args.max_memory_mb * 1024**2:
        raise ValueError(
            f"RK AD array estimate exceeds memory budget: {total_peak / 1024**2:.1f} MiB; reduce the JVP batch size or FFT rows"
        )
    seed = restart_path(run.get("restart_seed_override"), args.restart_seed)
    names = [
        "initialized_radau_adjoint.py",
        "radau_contact_adjoint.py",
        "radau_contact_experiment.py",
        "compact_radau_history.py",
        "radau_experiment.py",
        "bounded_history.py",
        "radau_initialization.py",
        "refined_contact.py",
        "uncondensed_static_contact.py",
        "uncondensed_static_contact_ad.py",
        "fem_static_contact.py",
        "fem_static_contact_ad.py",
        "restart_seed_experiment.py",
        "replay_resolvent.py",
        "substepped_contact_experiment.py",
        "robust_contact_experiment.py",
    ]
    lift_path = args.run / "initial_lift.npz"
    if fingerprint(lift_path) != run["initial_lift_record"]["sha256"]:
        raise ValueError("The measured primal static lift changed")
    selected = [
        Path(__file__),
        run_path,
        metadata_path,
        audit_path,
        manifest_path,
        lift_path,
        *[TOOLS / n for n in names],
    ]
    if seed is not None:
        selected.append(seed)
        if seed.with_suffix(".json").is_file():
            selected.append(seed.with_suffix(".json"))
    frozen = {str(p.resolve()): fingerprint(p) for p in selected}

    def verify():
        current = environment_record(options["case"])
        if any(fingerprint(p) != v for p, v in frozen.items()) or any(
            current[k] != environment[k]
            for k in ["source_sha256", "case_sha256", "restart_data_sha256"]
        ):
            raise ValueError("Frozen RK AD inputs changed")

    output.mkdir(parents=True)
    write_json(
        output / "preflight.json",
        dict(
            status="preflight_only_not_a_gradient_result",
            plan=metadata["plan"],
            checked_steps=steps,
            complete_prepared_horizon_checked=steps == full_plan.steps,
            prefix_explicitly_allowed=args.allow_prefix,
            initialization_gradient=args.initialization_gradient,
            estimated_ad_array_mib={
                k: v / 1024**2
                for k, v in arrays.items()
                if k not in ["direction_batch_size", "largest_dyadic_block"]
            },
            jvp_batch_size=args.jvp_batch_size,
            largest_dyadic_block=arrays["largest_dyadic_block"],
            estimated_static_array_peak_mib=static_peak / 1024**2,
            estimated_total_array_peak_mib=total_peak / 1024**2,
            frozen_input_sha256=frozen,
            original_tatva_readonly=True,
        ),
    )
    if args.preflight_only:
        print(output / "preflight.json", flush=True)
        return

    def progress(phase, **extra):
        write_json(
            output / "progress.json",
            dict(
                phase=phase,
                elapsed_seconds=perf_counter() - begun,
                complete_gradient_check=False,
                **extra,
            ),
        )

    progress("checking_complete_primal_and_time_weights")
    io_start = perf_counter()
    stages_path = args.run / "stages.npz"
    stage_sha = READ["digest_file"](stages_path)
    if manifest["primal_stage_sha256"][str(stages_path.resolve())] != stage_sha:
        raise ValueError("Primal stages differ from the original causal operator audit")
    weight_stats = []
    histories = []
    for record in metadata["family_weights"]:
        path = Path(record["path"])
        if READ["digest_file"](path) != record["sha256"]:
            raise ValueError("Completed RK weights changed")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            array.shape != (metadata["correction_steps"] + 1, plan.dofs, plan.dofs)
            or array.dtype != np.float64
        ):
            raise ValueError("Require complete original float64 RK time weights")
        histories.append(BoundaryHistory(plan, array[: steps + 1]))
        weight_stats.append((path, path.stat().st_size, path.stat().st_mtime_ns))
    if len(histories) != 2:
        raise ValueError("Both complete RK history families are required")
    for filename, key in [
        ("shift_wrap.npy", "shift_wrap_sha256"),
        ("initial_impedance.npy", "initial_impedance_sha256"),
    ]:
        if READ["digest_file"](kernel / filename) != metadata[key]:
            raise ValueError(f"Prepared RK {filename} changed")
    wrap = np.load(kernel / "shift_wrap.npy", allow_pickle=False)
    impedance = np.load(kernel / "initial_impedance.npy", allow_pickle=False)
    io_seconds = perf_counter() - io_start
    static_start = perf_counter()
    progress("recomputing_measured_static_body")
    if seed is not None:
        model, _ = runpy.run_path(str(TOOLS / "restart_seed_experiment.py"))["apply_restart_seed"](
            model, seed
        )
    refined = runpy.run_path(str(TOOLS / "refined_contact.py"))
    body = runpy.run_path(str(TOOLS / "uncondensed_static_contact.py"))
    space = refined["ContactQuadrature"].build(
        model.reference.moving.starts[model.moving_contact, 1], 1
    )
    reduction = refined["reduce_static_trace"](
        model, nodal_body_stiffness(model, quadrature=options["quadrature"])
    )
    lift = body["initialize_uncondensed_contact"](model, space, reduction, model.restart)
    reference = refined["dynamic_refined_reference"](
        model, space, lift, np.zeros((plan.dofs, plan.dofs))
    )
    line = reference.constraints.reduced_weights
    with np.load(lift_path, allow_pickle=False) as original:
        np.testing.assert_allclose(
            original["retained_displacement"], lift.retained_displacement, rtol=0, atol=1e-9
        )
        if np.max(abs(original["retained_bulk_force"] - lift.retained_bulk_force) / line) > 1e-7:
            raise ValueError("Measured initial bulk force differs in physical stress coordinates")
    sensitivity = None
    if args.initialization_gradient == "active-set":
        sensitivity = runpy.run_path(str(TOOLS / "uncondensed_static_contact_ad.py"))[
            "uncondensed_static_parameter_jacobian"
        ](model, space, reduction, model.restart, lift)
    del reduction
    gc.collect()
    static_seconds = perf_counter() - static_start
    with np.load(stages_path, allow_pickle=False) as data:
        stages = {k: data[k] for k in data.files}
    if stages["displacement"].shape != (steps + 1, 2, plan.dofs):
        raise ValueError("Incomplete saved RK stages")
    np.testing.assert_allclose(
        stages["response"][0, :, 1, 2], lift.state.plastic_slip, rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(stages["response"][0, :, 1, 3], lift.state.cumulative_slip)
    np.testing.assert_array_equal(stages["actuator"][0], model.reference.restart_actuator)
    initial = reference.evaluate(
        np.zeros(plan.dofs),
        np.zeros(plan.dofs),
        model.reference.normal_pressure,
        model.reference.restart_actuator,
        lift.state,
        jacobian=False,
    )[0]
    if (
        np.max(abs(initial)) >= 1e-7
        or np.max(abs(stages["initial_residual_force"] / line - initial)) > 1e-7
    ):
        raise ValueError("Saved measured initial residual differs from the static body")
    response = model.reference.contact.evaluate(reference.base_dn, reference.base_ds, lift.state)
    fc0 = np.r_[space.weights * response.pressure, space.weights * response.shear, 0]
    if np.max(abs(stages["initial_contact_force"] - fc0) / line) > 1e-7:
        raise ValueError("Saved initial contact force differs from the original healed state")
    import jax

    base = runpy.run_path(str(TOOLS / "radau_contact_adjoint.py"))
    initialized = runpy.run_path(str(TOOLS / "initialized_radau_adjoint.py"))
    arguments = (reference, *histories, wrap, impedance, stages)
    keywords = dict(
        segment_substeps=tuple(run["segment_substeps"]),
        cache_mb=args.fft_cache_mb,
        fft_rows=args.fft_rows,
    )

    def execute(method, *extra):
        verify()
        if sensitivity is None:
            answer = base[method](*arguments, *extra, **keywords)
        else:
            answer = initialized["initialized_gradient"](
                method, sensitivity, *arguments, *extra, **keywords
            )
        verify()
        gc.collect()
        jax.clear_caches()
        return answer

    reverse = execute("reverse_gradient")
    progress("completed_reverse_pending_independent_jvps")
    print("Complete RK reverse pass finished; checking independent JVP batches", flush=True)
    dp, da = verification_directions(steps)
    jvp = np.empty(7)
    batches = []
    for first in range(0, 7, args.jvp_batch_size):
        last = min(first + args.jvp_batch_size, 7)
        answer = execute("forward_directions", dp[:, first:last], da[:, :, first:last])
        jvp[first:last] = answer["objective_directional_derivative"]
        batches.append(
            dict(
                directions=list(range(first, last)),
                seconds_including_jit=answer["seconds_including_jit"],
                history=answer["history"],
            )
        )
        progress("checking_independent_jvp_batches", completed_directions=last, total_directions=7)
        print(f"Complete RK JVP directions {first}:{last} finished", flush=True)
    dual = np.asarray(reverse["gradient"]) @ dp + np.einsum(
        "ts,tsp->p", reverse["actuator_gradient"], da
    )
    difference = abs(jvp - dual)
    scale = np.maximum(abs(jvp), abs(dual))
    limit = args.gradient_atol + args.gradient_rtol * scale
    passed = bool(np.all(np.isfinite(difference)) and np.all(difference <= limit))
    verify()
    if READ["digest_file"](stages_path) != stage_sha:
        raise ValueError("Primal RK stages changed during AD")
    if any(
        p.stat().st_size != size or p.stat().st_mtime_ns != modified
        for p, size, modified in weight_stats
    ):
        raise ValueError("Read-only RK weights changed during AD")
    arrays_out = {
        k: reverse[k]
        for k in [
            "gradient",
            "actuator_gradient",
            "initial_state_cotangent",
            "initial_path_cotangent",
        ]
    }
    arrays_out.update(
        jvp=jvp, vjp_dot_products=dual, absolute_difference=difference, absolute_tolerance=limit
    )
    if sensitivity is not None:
        arrays_out.update(
            initial_displacement_jacobian=sensitivity["displacement_jacobian"],
            initial_plastic_jacobian=sensitivity["plastic_jacobian"],
            initial_measured_force_jacobian=sensitivity["retained_bulk_force_jacobian"],
        )
    np.savez(output / "gradients.npz", **arrays_out)
    write_json(
        output / "summary.json",
        dict(
            status="passed_complete_radau_adjoint_jvp_control"
            if passed
            else "failed_complete_radau_adjoint_jvp_control",
            run=str(args.run.resolve()),
            kernel=str(kernel.resolve()),
            original_operator_audit=str(args.operator_audit.resolve()),
            completed_steps=steps,
            dt_s=plan.dt,
            complete_prepared_horizon_checked=steps == full_plan.steps,
            initialization_gradient=args.initialization_gradient,
            full_initialized_gradient=sensitivity is not None,
            parameter_order=base["PARAMETERS"],
            gradient=np.asarray(reverse["gradient"]).tolist(),
            objective=reverse["objective"],
            value_mm=reverse["value_mm"],
            jvp=jvp.tolist(),
            vjp_dot_products=dual.tolist(),
            maximum_absolute_difference=float(np.max(difference)),
            maximum_relative_nonzero_difference=float(
                np.max(difference / np.maximum(scale, 1e-30))
            ),
            gradient_rtol=args.gradient_rtol,
            gradient_atol=args.gradient_atol,
            jvp_batches=batches,
            reverse_history=reverse["history"],
            every_history_source_included=True,
            timing_seconds=dict(
                input_hashes=io_seconds,
                static_reinitialization_and_jacobian=static_seconds,
                reverse_including_jit=reverse["seconds_including_jit"],
                forward_including_jit=sum(b["seconds_including_jit"] for b in batches),
                total_before_reporting=perf_counter() - begun,
            ),
            jit_caches_cleared_between_passes=True,
            estimated_total_array_peak_mib=total_peak / 1024**2,
            peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2 if sys.platform == "darwin" else 1024),
            primal_stage_sha256=stage_sha,
            gradient_file_sha256=READ["digest_file"](output / "gradients.npz"),
            frozen_input_sha256=frozen,
            environment=environment_record(options["case"]),
            full_original_event_reproduced=False,
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
            event_time_derivative_included=False,
            scope="Complete saved RK history, selected healed-static/transient branches, fixed bulk/restart and recorded stage loading; no material/geometry/stop-time derivative or gradient convergence certificate",
        ),
    )
    print(output / "summary.json", flush=True)
    write_json(
        output / "progress.json",
        dict(
            phase="completed_gradient_check" if passed else "failed_gradient_check",
            elapsed_seconds=perf_counter() - begun,
            complete_gradient_check=passed,
        ),
    )
    if not passed:
        raise ValueError("Complete Radau adjoint/JVP dot products did not pass")


if __name__ == "__main__":
    main()

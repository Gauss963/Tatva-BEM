"""Complete short two-body sliding/initialized-AD control for paired kernels.

Both backends assemble every physical Radau mode on the same complete small
contour. The shared healed preload uses the original static body operator.
The accelerated actuator is an explicit diagnostic, not event-16 loading.
This checks backend integration; it is not full-event accuracy or timing.
"""

import argparse
import json
import runpy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import BoundaryHistory, plan_cq
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model, nodal_body_stiffness

TOOLS = Path(__file__).resolve().parent
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
SEED = CASE.parent.parent / "data/ev16_restart_float64.npz"


def relative_error(actual, exact):
    denominator = np.linalg.norm(exact)
    error = np.linalg.norm(actual - exact)
    return float(error / denominator) if denominator else (0.0 if error == 0 else None)


def mixed_error(actual, exact, *, rtol=1e-7, atol=1e-18):
    return float(np.max(abs(actual - exact) / (atol + rtol * np.maximum(abs(actual), abs(exact)))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Require fresh owned diagnostic output outside original Tatva")
    output.mkdir(parents=True)
    begun = perf_counter()
    import jax

    jax.config.update("jax_enable_x64", True)
    frozen = {str(path.resolve()): fingerprint(path) for path in [*TOOLS.glob("*.py"), CASE, SEED]}
    environment = environment_record(CASE)

    def verify():
        if any(fingerprint(path) != value for path, value in frozen.items()) or (
            environment_record(CASE) != environment
        ):
            raise ValueError("Original or candidate numerical inputs changed")

    try:
        with threadpool_limits(limits=1, user_api="blas"):
            result = experiment(output, frozen, environment, begun)
        verify()
        result.update(
            total_seconds_including_all_controls=perf_counter() - begun,
            frozen_inputs_sha256=frozen,
            environment=environment,
            original_tatva_readonly=True,
            full_original_event_reproduced=False,
            same_accuracy_fem_advantage_verified=False,
            independent_full_trajectory_finite_difference_verified=False,
            scope="Complete 128-step coarse accelerated sliding diagnostic, fixed bulk and recorded loading; selected healed-static/transient branches. Not original-rate event 16, fine/full event convergence, stopping-time/material/geometry AD, or a uniform kernel certificate.",
        )
        write_json(output / "summary.json", result)
        print(json.dumps({k: v for k, v in result.items() if k.endswith("errors")}), flush=True)
    except BaseException as error:
        write_json(
            output / "failure.json",
            dict(error=repr(error), seconds=perf_counter() - begun, completed=False),
        )
        raise


def experiment(output, frozen, environment, begun):
    ray = runpy.run_path(str(TOOLS / "paired_ray_galerkin.py"))
    rk = runpy.run_path(str(TOOLS / "radau_experiment.py"))
    modes = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))["lower_half_radau_modes"]
    compact = runpy.run_path(str(TOOLS / "compact_radau_storage.py"))
    simulate = runpy.run_path(str(TOOLS / "simulate_compact_radau.py"))["simulate_stages"]
    refined = runpy.run_path(str(TOOLS / "refined_contact.py"))
    body = runpy.run_path(str(TOOLS / "uncondensed_static_contact.py"))
    ad = runpy.run_path(str(TOOLS / "initialized_radau_adjoint.py"))["initialized_gradient"]
    model = load_nodal_model(
        CASE,
        max_panel_length=160,
        contact_panel_length=160,
        contact_end_length=160,
        corner_panel_length=160,
    )
    model, seed_record = runpy.run_path(str(TOOLS / "restart_seed_experiment.py"))[
        "apply_restart_seed"
    ](model, SEED)
    transfer = runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"](
        model, quadrature=12
    )
    plan = plan_cq(transfer.dofs, 2e-6, 128, max_memory_mb=512)
    law, con = model.reference.contact, model.constraints
    spring = model.reference.spring_stiffness * con.platen_width
    diagonal = np.diag(
        np.r_[
            con.contact_weights * law.normal_penalty,
            con.contact_weights * law.tangential_penalty,
            spring,
        ]
    )
    providers = {
        "original": transfer.prepared,
        "paired": [ray["PairedRayGalerkin"](a) for a in transfer.prepared],
    }
    write_json(
        output / "preflight.json",
        dict(
            plan=asdict(plan),
            frozen_inputs_sha256=frozen,
            environment=environment,
            seed=seed_record,
            mesh=dict(h=160, contact_h=160, contact_end_h=160, corner_h=160, quadrature=12),
            diagnostic_actuator_rate_mm_per_s=7812.5,
            original_actuator_rate_mm_per_s=model.reference.actuator_rate,
            shared_original_static_preload=True,
        ),
    )
    raw_max, schur_max = 0.0, 0.0

    def impedances(s):
        nonlocal raw_max, schur_max
        operators = {name: [a.assemble(s) for a in group] for name, group in providers.items()}
        for actual, exact in zip(operators["paired"], operators["original"], strict=True):
            for field in ["single_layer", "trace_double_layer", "hypersingular"]:
                x, y = getattr(actual, field), getattr(exact, field)
                value = float(np.linalg.norm(x - y) / (1e-14 + 1e-11 * np.linalg.norm(y)))
                raw_max = max(raw_max, value)
                if not np.isfinite(value) or value > 1:
                    raise ValueError("Complete small-contour raw gate failed")
        maps = {
            name: transfer.evaluate_operators(op, reconstruction=True)
            for name, op in operators.items()
        }
        for x, y in zip(maps["paired"], maps["original"], strict=True):
            for order in [None, np.inf]:
                value = float(np.linalg.norm(x - y, ord=order) / np.linalg.norm(y, ord=order))
                schur_max = max(schur_max, value)
                if not np.isfinite(value) or value > 1e-8:
                    raise ValueError("Complete small-contour Schur gate failed")
        return {name: value[0] for name, value in maps.items()}

    spectra = {
        name: [np.empty((plan.fft_size // 2 + 1, plan.dofs, plan.dofs), complex) for _ in range(2)]
        for name in providers
    }
    for index in range(plan.fft_size // 2 + 1):
        zeta = plan.radius * np.exp(-2j * np.pi * index / plan.fft_size)
        a, b = modes(zeta, plan.dt)
        za, zb = impedances(a), impedances(b)
        for name, families in spectra.items():
            la, lb = np.linalg.inv(za[name] + diagonal), np.linalg.inv(zb[name] + diagonal)
            families[0][index] = (b * la - a * lb) / (b - a)
            families[1][index] = (lb - la) / ((b - a) * plan.dt)
        write_json(
            output / "progress.json",
            dict(phase="complete_contour", index=index, total=plan.fft_size // 2 + 1),
        )
    contour_seconds = perf_counter() - begun
    # Use the actual coupled RK symbol, independently of contour interpolation.
    impedances0 = {}
    zero_cache = {}

    def at_zero(s, name):
        if s not in zero_cache:
            zero_cache[s] = impedances(s)
        return zero_cache[s][name]

    for name in providers:
        impedances0[name] = rk["matrix_function"](
            lambda s, name=name: at_zero(s, name), rk["radau_symbol"](0) / plan.dt
        ).real
    y = model.reference.moving.starts[model.moving_contact, 1]
    space = refined["ContactQuadrature"].build(y, 1)
    reduction = refined["reduce_static_trace"](model, nodal_body_stiffness(model, quadrature=12))
    lift = body["initialize_uncondensed_contact"](model, space, reduction, model.restart)
    reference = refined["dynamic_refined_reference"](
        model, space, lift, np.zeros((plan.dofs, plan.dofs))
    )
    sensitivity = runpy.run_path(str(TOOLS / "uncondensed_static_contact_ad.py"))[
        "uncondensed_static_parameter_jacobian"
    ](model, space, reduction, model.restart, lift)
    initial_residual = reference.evaluate(
        np.zeros(plan.dofs),
        np.zeros(plan.dofs),
        model.reference.normal_pressure,
        model.reference.restart_actuator,
        lift.state,
        jacobian=False,
    )[0]
    initial_force = initial_residual * reference.constraints.reduced_weights
    if np.max(abs(initial_residual)) > 1e-7:
        raise ValueError("Original measured static root failed")
    clock = np.zeros((plan.steps + 1, 2))
    clock[1:] = (np.arange(plan.steps)[:, None] + rk["C"]) * plan.dt
    ramp = model.reference.restart_actuator + 7812.5 * clock
    dp, da = runpy.run_path(str(TOOLS / "check_compact_radau_ad.py"))["verification_directions"](
        plan.steps
    )
    outputs, duality, complete_weights = {}, {}, {}
    for name, families in spectra.items():
        write_json(output / "progress.json", dict(phase="primal_and_initialized_ad", backend=name))
        w0, w1, wrap = rk["invert_compact_spectrum"](*families, plan)
        complete_weights[name] = rk["expand_compact_weights"](w0, w1, wrap)
        histories = BoundaryHistory(plan, w0), BoundaryHistory(plan, w1)
        elastic = compact["causal_compact_startup"](
            families[0][:, :, -1], families[1][:, :, -1], wrap[:, -1], plan, spring * 7812.5 * clock
        )
        result, stages = simulate(
            reference,
            lift.state,
            *histories,
            wrap,
            impedances0[name],
            elastic,
            clock,
            ramp,
            diagnostic=True,
            segment_substeps=(8, 16),
            recover=True,
            kinetic_seed=True,
            initial_residual_force=initial_force,
        )
        if np.max(stages["response"][-1, :, 1, 3]) <= law.characteristic_slip:
            raise ValueError("Control did not reach nontrivial slip weakening")
        arguments = (reference, *histories, wrap, impedances0[name], stages)
        reverse = ad("reverse_gradient", sensitivity, *arguments, segment_substeps=(8, 16))
        forward = ad(
            "forward_directions", sensitivity, *arguments, dp, da, segment_substeps=(8, 16)
        )
        dual = reverse["gradient"] @ dp + np.einsum("ts,tsp->p", reverse["actuator_gradient"], da)
        jvp = np.asarray(forward["objective_directional_derivative"])
        duality[name] = mixed_error(jvp, dual)
        if not np.isfinite(duality[name]) or duality[name] > 1:
            raise ValueError("Initialized VJP/JVP gate failed")
        outputs[name] = dict(
            displacement=stages["displacement"],
            response=stages["response"],
            gradient=reverse["gradient"],
            actuator_gradient=reverse["actuator_gradient"],
            jvp=jvp,
        )
        np.savez(output / f"{name}-arrays.npz", **outputs[name])
        write_json(output / f"{name}-diagnostics.json", result.diagnostics)
    errors = {
        k: relative_error(outputs["paired"][k], outputs["original"][k]) for k in outputs["paired"]
    }
    gradient_errors = {
        k: mixed_error(outputs["paired"][k], outputs["original"][k])
        for k in ["gradient", "actuator_gradient", "jvp"]
    }
    for k in ["displacement", "response"]:
        if errors[k] is None or not np.isfinite(errors[k]) or errors[k] > 1e-8:
            raise ValueError("Complete saved-stage primal gate failed")
    if any(not np.isfinite(v) or v > 1 for v in gradient_errors.values()):
        raise ValueError("Complete initialized backend-gradient gate failed")
    weight_error = relative_error(complete_weights["paired"], complete_weights["original"])
    if weight_error is None or weight_error > 1e-9:
        raise ValueError("Complete causal weight gate failed")
    return dict(
        status="passed_complete_coarse_sliding_and_selected_initialized_ad_backend_control",
        plan=asdict(plan),
        all_physical_contour_modes_checked=True,
        physical_contour_mode_evaluations=2 * (plan.fft_size // 2 + 1),
        additional_unique_lag_zero_modes=len(zero_cache),
        maximum_raw_mixed_normalized_error=raw_max,
        maximum_schur_relative_error=schur_max,
        complete_causal_weight_relative_error=weight_error,
        saved_stage_relative_errors=errors,
        initialized_gradient_mixed_normalized_errors=gradient_errors,
        initialized_jvp_vjp_mixed_normalized_errors=duality,
        final_max_cumulative_slip_mm={
            name: float(np.max(value["response"][-1, :, 1, 3])) for name, value in outputs.items()
        },
        contour_seconds_including_initial_preparation=contour_seconds,
        all_history_retained=True,
        fitted_time_shift_s=0,
    )


if __name__ == "__main__":
    main()

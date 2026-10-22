"""Fully elastodynamic two-domain CQ history plus nonlinear contact.

This is a dense reference engine with a deliberately bounded horizon. It
retains all past boundary increments and never substitutes a static matrix
or local damping for the elastodynamic Dirichlet-to-Neumann map.
"""

import resource
import sys
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter, process_time

import numpy as np
from scipy.linalg import block_diag

from .benchmarks import smooth_step
from .case import TwoBlockCase
from .constraints import build_constraints
from .cq import OnlineConvolution, build_history, plan_domains
from .friction import ContactState
from .nonlinear import ContactEquilibrium
from .operators import assemble_operators


class RestartProjectionError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class StaticLift:
    displacement: np.ndarray
    traction: np.ndarray
    state: ContactState
    diagnostics: dict


def initialize_restart(case: TwoBlockCase, *, quadrature=20, allow_relaxation=False):
    """Map measured plastic slip to a static BEM equilibrium with zero velocity.

    A static lift is legitimate because perturbations then have homogeneous
    initial conditions. It is not an exact import of the FEM interior field
    or a recovery of the event-15 wave history. Inadmissible transfers fail.
    """
    con = build_constraints(case)
    matrices = [
        assemble_operators(mesh, case.material, quadrature=quadrature).impedance()
        for mesh in (case.moving, case.stationary)
    ]
    system = ContactEquilibrium(case, con, block_diag(*matrices))
    guess = system.closed_elastic_guess(case.normal_pressure, case.restart_actuator, case.restart)
    frozen = system.solve(
        case.normal_pressure, case.restart_actuator, case.restart, guess=guess, frozen=True
    )
    if not frozen.converged:
        raise RestartProjectionError(
            "Frozen-plastic static restart did not converge",
            {"status": "static_equilibrium_failed", "residual_mpa": frozen.residual_mpa},
        )
    strength = frozen.contact.pressure * case.contact.coefficient(case.restart.cumulative_slip)
    ratio = float(np.max(np.abs(frozen.contact.shear) / np.maximum(strength, 1e-20)))
    diagnostics = {
        "status": "admissible_static_lift",
        "initial_max_strength_ratio": ratio,
        "static_residual_mpa": frozen.residual_mpa,
        "relaxation_allowed": bool(allow_relaxation),
        "imported_fem_interior_field": False,
        "zero_initial_perturbation_velocity": True,
    }
    result = frozen
    if ratio > 1 + 1e-6:
        if not allow_relaxation:
            diagnostics["status"] = "incompatible_restart_projection"
            raise RestartProjectionError(
                "Transferred FEM plastic slip violates the BEM friction bound. "
                "No event-16 dynamic trajectory was generated.",
                diagnostics,
            )
        result = system.solve(
            case.normal_pressure,
            case.restart_actuator,
            case.restart,
            guess=frozen.reduced_displacement,
            max_iterations=100,
        )
        if not result.converged:
            diagnostics.update(status="restart_relaxation_failed", residual_mpa=result.residual_mpa)
            raise RestartProjectionError("Nonlinear static restart relaxation failed", diagnostics)
        diagnostics["status"] = "relaxed_static_lift_changes_initial_state"
    u = con.prolongation @ result.reduced_displacement
    traction = system.impedance @ u
    diagnostics["introduced_cumulative_slip_mm"] = float(
        np.max(result.contact.state.cumulative_slip - case.restart.cumulative_slip)
    )
    diagnostics["max_plastic_slip_change_mm"] = float(
        np.max(np.abs(result.contact.state.plastic_slip - case.restart.plastic_slip))
    )
    diagnostics["platen_displacement_mm"] = float(result.reduced_displacement[-1])
    diagnostics["mean_pressure_mpa"] = float(
        np.average(result.contact.pressure, weights=con.contact_weights)
    )
    diagnostics["mean_shear_mpa"] = float(
        np.average(result.contact.shear, weights=con.contact_weights)
    )
    return StaticLift(u, traction, result.contact.state, diagnostics)


@dataclass(frozen=True)
class LoadingSchedule:
    pressure: Callable[[float], float]
    actuator: Callable[[float], float]
    name: str


def cold_smoke_schedule(case):
    d = case.payload["diagnostic"]
    return LoadingSchedule(
        lambda t: float(case.normal_pressure * smooth_step(t, d["normal_rise"])),
        lambda t: float(d["shear_amplitude"] * smooth_step(t - d["shear_start"], d["shear_rise"])),
        "cold_start_accelerated_loading_diagnostic",
    )


def event16_schedule(case):
    return LoadingSchedule(
        lambda t: case.normal_pressure,
        lambda t: case.restart_actuator + case.actuator_rate * t,
        "event16_static_lift_then_original_actuator_rate",
    )


@dataclass(frozen=True)
class DynamicResult:
    time: np.ndarray
    displacement: np.ndarray
    traction: np.ndarray
    pressure: np.ndarray
    shear: np.ndarray
    friction_coefficient: np.ndarray
    plastic_slip: np.ndarray
    cumulative_slip: np.ndarray
    actuator: np.ndarray
    platen: np.ndarray
    normal_load: np.ndarray
    residual_mpa: np.ndarray
    newton_iterations: np.ndarray
    diagnostics: dict


def simulate(
    case: TwoBlockCase,
    dt,
    steps,
    *,
    schedule: LoadingSchedule,
    lift: StaticLift | None = None,
    quadrature=20,
    max_memory_mb=512,
    progress=None,
    workers=1,
    history_method="fft",
    backend="numpy",
    cache_directory=None,
):
    started, cpu_started = perf_counter(), process_time()
    if history_method not in ("direct", "fft") or backend not in ("numpy", "jax"):
        raise ValueError("Unknown history method or nonlinear backend")
    con = build_constraints(case)
    meshes = (case.moving, case.stationary)
    plans, peak = plan_domains(
        [mesh.dofs for mesh in meshes], dt, steps, max_memory_mb=max_memory_mb
    )
    histories = []
    for domain, mesh in enumerate(meshes):
        callback = None if progress is None else lambda i, n, d=domain: progress("kernel", d, i, n)
        histories.append(
            build_history(
                mesh,
                case.material,
                dt,
                steps,
                quadrature=quadrature,
                max_memory_mb=max_memory_mb,
                progress=callback,
                workers=workers,
                cache_directory=cache_directory,
            )
        )
    w0 = block_diag(*(history.instantaneous for history in histories))
    kernel_seconds = perf_counter() - started
    size = w0.shape[0]
    base_u = np.zeros(size) if lift is None else lift.displacement
    base_t = np.zeros(size) if lift is None else lift.traction
    old = (
        ContactState(np.zeros(len(case.moving_contact)), np.zeros(len(case.moving_contact)))
        if lift is None
        else lift.state
    )
    system = ContactEquilibrium(case, con, w0, base_displacement=base_u, base_traction=base_t)
    if backend == "jax":
        try:
            from .jax_backend import JaxContactEquilibrium
        except ModuleNotFoundError as exc:
            if exc.name == "jax":
                raise ValueError("JAX backend requires installation with the [jax] extra") from exc
            raise

        system = JaxContactEquilibrium(system)
    online = [OnlineConvolution(h) for h in histories] if history_method == "fft" else None
    if online is not None:
        for h in online:
            h.append(np.zeros(h.history.plan.dofs))
    perturbation = np.zeros((steps + 1, size))
    u = np.zeros_like(perturbation)
    u[0] = base_u
    traction = np.zeros_like(u)
    traction[0] = base_t
    count = len(case.moving_contact)
    pressure = np.zeros((steps + 1, count))
    shear = np.zeros_like(pressure)
    coefficient = np.zeros_like(pressure)
    z = np.zeros_like(pressure)
    cum = np.zeros_like(pressure)
    time = np.arange(steps + 1) * dt
    actuator = np.array([schedule.actuator(float(t)) for t in time])
    normal = np.array([schedule.pressure(float(t)) for t in time])
    if not np.all(np.isfinite(actuator)) or not np.all(np.isfinite(normal)) or np.any(normal < 0):
        raise ValueError("Nonfinite loading or negative prescribed compression")
    platen = np.zeros(steps + 1)
    residual = np.zeros(steps + 1)
    iterations = np.zeros(steps + 1, dtype=int)
    reduced = np.zeros(con.size)
    r, _, initial = system.evaluate(
        reduced, np.zeros(size), normal[0], actuator[0], old, jacobian=False
    )
    if np.max(abs(r)) > 1e-6:
        raise ValueError(
            "Initial boundary/contact state is not in equilibrium; use a valid static lift"
        )
    pressure[0] = initial.pressure
    shear[0] = initial.shear
    coefficient[0] = initial.friction_coefficient
    z[0] = old.plastic_slip
    cum[0] = old.cumulative_slip
    platen[0] = system.base_platen
    failure = None
    valid = steps + 1
    held_actuator = None
    stop_time = None
    offset = case.moving.dofs
    history_seconds = 0.0
    newton_seconds = 0.0
    step_started = perf_counter()
    for step in range(1, steps + 1):
        if held_actuator is not None:
            actuator[step] = held_actuator
        tick = perf_counter()
        past = (
            np.concatenate([h.past(step) for h in online])
            if online is not None
            else np.concatenate(
                (
                    histories[0].past_traction(perturbation[:, :offset], step),
                    histories[1].past_traction(perturbation[:, offset:], step),
                )
            )
        )
        history_seconds += perf_counter() - tick
        tick = perf_counter()
        try:
            result = system.solve(normal[step], actuator[step], old, past=past, guess=reduced)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            valid = step
            failure = {"step": step, "time_s": float(time[step]), "reason": str(exc)}
            break
        newton_seconds += perf_counter() - tick
        if not result.converged:
            valid = step
            failure = {
                "step": step,
                "time_s": float(time[step]),
                "residual_mpa": result.residual_mpa,
            }
            break
        reduced = result.reduced_displacement
        perturbation[step] = con.prolongation @ reduced
        tick = perf_counter()
        if online is not None:
            online[0].append(perturbation[step, :offset])
            online[1].append(perturbation[step, offset:])
        history_seconds += perf_counter() - tick
        u[step] = base_u + perturbation[step]
        traction[step] = base_t + w0 @ perturbation[step] + past
        response = result.contact
        pressure[step] = response.pressure
        shear[step] = response.shear
        coefficient[step] = response.friction_coefficient
        z[step] = response.state.plastic_slip
        cum[step] = response.state.cumulative_slip
        platen[step] = system.base_platen + reduced[-1]
        residual[step] = result.residual_mpa
        iterations[step] = result.iterations
        old = response.state
        # Match the original load-hold criterion only in the warm event mode.
        if schedule.name.startswith("event16") and held_actuator is None:
            coverage = float(np.mean(cum[step] >= case.contact.characteristic_slip))
            if coverage >= case.payload["loading"]["stop_coverage"]:
                held_actuator = float(actuator[step])
                stop_time = float(time[step])
        if progress is not None:
            progress("step", 0, step, steps)
    strength = pressure[:valid] * coefficient[:valid]
    ratio = np.abs(shear[:valid]) / np.maximum(strength, 1e-20)
    slip_increment = np.diff(z[:valid], axis=0)
    friction_work = np.sum(shear[1:valid] * slip_increment * con.contact_weights)
    data = [
        time,
        u,
        traction,
        pressure,
        shear,
        coefficient,
        z,
        cum,
        actuator,
        platen,
        normal,
        residual,
        iterations,
    ]
    diagnostics = {
        "status": "completed_requested_horizon" if failure is None else "nonlinear_step_failed",
        "formulation": "2D plane-strain direct elastodynamic BEM, BDF2 convolution quadrature",
        "schedule": schedule.name,
        "requested_steps": steps,
        "completed_steps": valid - 1,
        "dt_s": dt,
        "duration_s": float(time[valid - 1]),
        "wave_speeds": {"cp": case.material.cp, "cs": case.material.cs},
        "panels": [mesh.count for mesh in meshes],
        "cq_fft_size": plans[0].fft_size,
        "all_history_retained": True,
        "history_method": history_method,
        "nonlinear_backend": backend,
        "kernel_workers": workers,
        "timing_seconds": {
            "kernel_build_or_cache_load": kernel_seconds,
            "time_stepping": perf_counter() - step_started,
            "history_updates": history_seconds,
            "nonlinear_solves_including_jit": newton_seconds,
            "total_excluding_static_lift_and_output": perf_counter() - started,
            "process_cpu": process_time() - cpu_started,
        },
        "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        "dense_peak_estimate_mb": peak / 1024**2,
        "weakening_evaluation": case.contact.weakening_evaluation,
        "max_contact_strength_ratio": float(np.max(ratio)),
        "max_boundary_contact_residual_mpa": float(np.max(residual[:valid])),
        "max_cumulative_slip_mm": float(np.max(cum[:valid])),
        "friction_work_per_unit_thickness_mpa_mm2": float(friction_work),
        "actuator_hold_time_s": stop_time,
        "failure": failure,
        "initialization": {"kind": "zero_initial_fields"} if lift is None else lift.diagnostics,
        "full_original_event_reproduced": False,
        "limitations": [
            "P0 collocation with dense spatial kernels; spatial refinement still required",
            "No certified energy/passivity estimate for the nonlinear collocation scheme",
            "Static lift differs from the FEM interior initial field",
        ],
    }
    return DynamicResult(*(array[:valid] for array in data), diagnostics)

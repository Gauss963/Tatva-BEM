"""Fully dynamic retained-coordinate CQ with the Galerkin nodal-force backend.

The P0 reference remains available. This path uses continuous boundary traces,
the same local LSW/Newton/JAX law, and complete online convolution histories.
It currently requires a fixed static lift and constant normal pressure.
"""

import hashlib
import json
import os
import resource
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .cq import (
    BoundaryHistory,
    OnlineConvolution,
    ResourceBudgetError,
    convolution_weights,
    plan_cq,
)
from .dynamics import DynamicResult, event16_schedule
from .galerkin_fast import prepared_geometry_bytes
from .nodal import NodalInterfaceEquilibrium, NodalInterfaceTransfer, initialize_nodal_restart
from .sampling import parsimonious_weights


def plan_nodal_history(
    model, dt, steps, *, workers=1, quadrature=12, sampling=None, max_memory_mb=1024
):
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    plan = plan_cq(2 * len(model.moving_contact) + 1, dt, steps, max_memory_mb=max_memory_mb)
    sizes = [model.reference.moving.dofs, model.reference.stationary.dofs]
    # Read-only quadrature geometry is shared by frequency workers. Count it
    # once, then bounded batch temporaries and dense factorizations per worker.
    geometry = sum(
        prepared_geometry_bytes(mesh, quadrature)
        for mesh in (model.reference.moving, model.reference.stationary)
    )
    workspace = geometry + workers * (
        14 * max(s * s for s in sizes) * 16 + 32 * quadrature**2 * 1024
    )
    if sampling is not None:
        workspace += (4 * sampling.degree + 2 * sampling.check_points + 16) * plan.dofs**2 * 16
    peak = plan.estimated_peak_bytes + workspace
    if peak > max_memory_mb * 1024**2:
        raise ResourceBudgetError(
            f"Galerkin CQ matrix/workspace estimate {peak / 1024**2:.1f} MiB exceeds budget"
        )
    return plan, peak


def build_nodal_history(
    model,
    dt,
    steps,
    *,
    workers=1,
    quadrature=12,
    sampling=None,
    max_memory_mb=1024,
    cache_directory=None,
    progress=None,
    diagnostics=None,
):
    plan, _ = plan_nodal_history(
        model,
        dt,
        steps,
        workers=workers,
        quadrature=quadrature,
        sampling=sampling,
        max_memory_mb=max_memory_mb,
    )
    cached = None
    if cache_directory is not None:
        digest = hashlib.sha256(
            repr((plan, model.reference.material, quadrature, sampling)).encode()
        )
        for mesh in [model.reference.moving, model.reference.stationary]:
            for data in [mesh.starts, mesh.ends, mesh.labels]:
                digest.update(data.tobytes())
        digest.update(model.constraints.prolongation.tobytes())
        for name in [
            "galerkin.py",
            "galerkin_fast.py",
            "nodal.py",
            "cq.py",
            "sampling.py",
            "kernels.py",
            "material.py",
            "mesh.py",
        ]:
            digest.update(Path(__file__).with_name(name).read_bytes())
        directory = Path(cache_directory)
        directory.mkdir(parents=True, exist_ok=True)
        cached = directory / f"galerkin-{digest.hexdigest()}.npy"
        metadata_path = cached.with_suffix(".json")
        if cached.exists() and metadata_path.exists():
            weights = np.load(cached, mmap_mode="r", allow_pickle=False)
            if weights.shape != (steps + 1, plan.dofs, plan.dofs) or weights.dtype != np.float64:
                raise ValueError("Invalid Galerkin history cache")
            if diagnostics is not None:
                diagnostics.update(json.loads(metadata_path.read_text()), cache_hit=True)
            return BoundaryHistory(plan, weights)
    transfer = NodalInterfaceTransfer(model, quadrature=quadrature)
    if sampling is None:
        weights = convolution_weights(transfer.evaluate, plan, workers=workers, progress=progress)
        metadata = {"method": "dense", "transfer_evaluations": plan.fft_size // 2 + 1}
    else:
        wc = model.constraints.contact_weights
        scale = 1 / np.r_[wc, wc, model.constraints.platen_width]
        weights, metadata = parsimonious_weights(
            transfer.evaluate,
            plan,
            options=sampling,
            row_scale=scale,
            workers=workers,
            progress=progress,
        )
    if cached is not None:
        temporary = cached.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, weights, allow_pickle=False)
        temporary.replace(cached)
        temporary = metadata_path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        temporary.replace(metadata_path)
    if diagnostics is not None:
        diagnostics.update(metadata, cache_hit=False)
    return BoundaryHistory(plan, weights)


def simulate_nodal(
    model,
    dt,
    steps,
    *,
    schedule=None,
    lift=None,
    backend="numpy",
    workers=1,
    quadrature=12,
    sampling=None,
    max_memory_mb=1024,
    cache_directory=None,
    progress=None,
):
    start = perf_counter()
    ref = model.reference
    if backend not in ["numpy", "jax"]:
        raise ValueError("Unknown nonlinear backend")
    plan, peak = plan_nodal_history(
        model,
        dt,
        steps,
        workers=workers,
        quadrature=quadrature,
        sampling=sampling,
        max_memory_mb=max_memory_mb,
    )
    schedule = event16_schedule(ref) if schedule is None else schedule
    time = np.arange(steps + 1) * dt
    pressure_load = np.array([schedule.pressure(float(t)) for t in time])
    actuator = np.array([schedule.actuator(float(t)) for t in time])
    if not np.all(pressure_load == ref.normal_pressure):
        raise ValueError("Condensed Galerkin mode needs constant normal pressure")
    if not np.all(np.isfinite(actuator)):
        raise ValueError("Nonfinite actuator loading")
    if lift is None:
        lift = initialize_nodal_restart(model, quadrature=quadrature)
    static_seconds = perf_counter() - start
    tick = perf_counter()
    sampling_diagnostics = {}
    history = build_nodal_history(
        model,
        dt,
        steps,
        workers=workers,
        quadrature=quadrature,
        sampling=sampling,
        max_memory_mb=max_memory_mb,
        cache_directory=cache_directory,
        progress=progress,
        diagnostics=sampling_diagnostics,
    )
    kernel_seconds = perf_counter() - tick
    reference = NodalInterfaceEquilibrium(model, lift, history.instantaneous)
    system = reference
    if backend == "jax":
        from .jax_backend import JaxContactEquilibrium

        system = JaxContactEquilibrium(reference)
    n = len(model.moving_contact)
    q = np.zeros((steps + 1, plan.dofs))
    p, tau, mu, z, c = [np.zeros((steps + 1, n)) for _ in range(5)]
    platen = np.zeros(steps + 1)
    residual = np.zeros(steps + 1)
    iterations = np.zeros(steps + 1, dtype=int)
    initial, _, contact = system.evaluate(
        q[0], np.zeros(plan.dofs), pressure_load[0], actuator[0], lift.state, jacobian=False
    )
    if np.max(np.abs(initial)) > 1e-6:
        raise ValueError("Initial nodal boundary/contact state is not in equilibrium")
    p[0], tau[0], mu[0], z[0], c[0] = (
        contact.pressure,
        contact.shear,
        contact.friction_coefficient,
        lift.state.plastic_slip,
        lift.state.cumulative_slip,
    )
    platen[0] = reference.base_platen
    residual[0] = np.max(np.abs(initial))
    online = OnlineConvolution(history)
    online.append(q[0])
    old = lift.state
    held, hold_time, failure, valid = None, None, None, steps + 1
    tick = perf_counter()
    for index in range(1, steps + 1):
        if held is not None:
            actuator[index] = held
        try:
            result = system.solve(
                pressure_load[index],
                actuator[index],
                old,
                past=online.past(index),
                guess=q[index - 1],
            )
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            failure = {"step": index, "reason": str(error)}
            valid = index
            break
        if not result.converged:
            failure = {"step": index, "residual_mpa": result.residual_mpa}
            valid = index
            break
        q[index] = result.reduced_displacement
        online.append(q[index])
        contact = result.contact
        p[index], tau[index], mu[index] = (
            contact.pressure,
            contact.shear,
            contact.friction_coefficient,
        )
        z[index], c[index] = contact.state.plastic_slip, contact.state.cumulative_slip
        platen[index] = reference.base_platen + q[index, -1]
        residual[index], iterations[index] = result.residual_mpa, result.iterations
        old = contact.state
        if schedule.name.startswith("event16") and held is None:
            fraction = np.average(
                c[index] >= ref.contact.characteristic_slip,
                weights=model.constraints.contact_weights,
            )
            if fraction >= ref.payload["loading"]["stop_coverage"]:
                held, hold_time = actuator[index], time[index]
        if progress is not None and index % max(1, steps // 10) == 0:
            progress(index, steps)
    step_seconds = perf_counter() - tick
    contact_jumps = q[:valid, : 2 * n] + np.r_[reference.base_dn, reference.base_ds]
    diagnostic = {
        "status": "completed_requested_horizon" if failure is None else "nonlinear_step_failed",
        "boundary_formulation": "P1/P0 symmetric Galerkin; nodal-force interface condensation",
        "schedule": schedule.name,
        "dt_s": dt,
        "requested_steps": steps,
        "completed_steps": valid - 1,
        "duration_s": float(time[valid - 1]),
        "history_dofs": [plan.dofs],
        "cq_fft_size": plan.fft_size,
        "all_history_retained": True,
        "full_boundary_fields_saved": False,
        "nonlinear_backend": backend,
        "panels": [ref.moving.count, ref.stationary.count],
        "contact_nodes": n,
        "quadrature": quadrature,
        "kernel_workers": workers,
        "frequency_sampling": sampling_diagnostics,
        "max_cumulative_slip_mm": float(np.max(c[:valid])),
        "max_boundary_contact_residual_mpa": float(np.max(residual[:valid])),
        "actuator_hold_time_s": None if hold_time is None else float(hold_time),
        "failure": failure,
        "initialization": lift.diagnostics,
        "estimated_array_peak_mb": peak / 1024**2,
        "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        "timing_seconds": {
            "static_lift": static_seconds,
            "kernel_build_or_cache_load": kernel_seconds,
            "time_stepping": step_seconds,
            "total_excluding_output": perf_counter() - start,
        },
        "wave_speeds": {"cp": ref.material.cp, "cs": ref.material.cs},
        "sampling_options": None if sampling is None else asdict(sampling),
        "full_original_event_reproduced": False,
    }
    return DynamicResult(
        time[:valid],
        None,
        None,
        p[:valid],
        tau[:valid],
        mu[:valid],
        z[:valid],
        c[:valid],
        actuator[:valid],
        platen[:valid],
        pressure_load[:valid],
        residual[:valid],
        iterations[:valid],
        diagnostic,
        contact_jumps,
    )

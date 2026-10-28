"""Signed incremental bulk-port work with the actual RK stage quadrature.

This is a trajectory diagnostic, not a reconstruction of interior energy or
a theorem for slip weakening. Stage velocity uses the causal RK derivative
of the displacement trace, including the previous accepted endpoint.
"""

import runpy
from pathlib import Path

import numpy as np

RADAU = runpy.run_path(str(Path(__file__).with_name("radau_experiment.py")))


def stage_derivative(values, dt):
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 3
        or values.shape[1] != 2
        or len(values) < 2
        or not np.all(np.isfinite(values))
        or not np.isfinite(dt)
        or dt <= 0
        or np.any(values[0, 0] != values[0, 1])
    ):
        raise ValueError("Require finite aligned RK stages, initial endpoint and positive dt")
    answer = np.zeros_like(values)
    increment = values[1:] - values[:-1, 1, None, :]
    answer[1:] = np.einsum("ab,tbi->tai", np.linalg.inv(RADAU["A"]), increment) / dt
    return answer


def equilibrium_bulk_force(
    displacement, response, actuator, initial_force, initial_contact, line, spring
):
    """Recover Z*q from physical balance, leaving its audited residual as error.

    R = Z*q + spring*q_platen + fc - fc0 + r0 - spring*(a-a0).
    Return the R=0 reconstruction. The work caller bounds the effect of the
    actual R using the independently audited maximum residual in MPa.
    """
    q, response, actuator = map(np.asarray, [displacement, response, actuator])
    line, initial_force, initial_contact = map(np.asarray, [line, initial_force, initial_contact])
    if q.ndim != 3 or len(q) < 2 or q.shape[1] != 2 or q.shape[2] % 2 != 1:
        raise ValueError("Require stage traces on the identity nodal contact grid")
    n, d = (q.shape[2] - 1) // 2, q.shape[2]
    if (
        response.shape != (len(q), n, 2, 7)
        or actuator.shape != (len(q), 2)
        or line.shape != (d,)
        or initial_force.shape != (d,)
        or initial_contact.shape != (d,)
        or not all(
            np.all(np.isfinite(v))
            for v in [q, response, actuator, line, initial_force, initial_contact]
        )
        or np.any(line <= 0)
        or not np.array_equal(line[:n], line[n : 2 * n])
        or not np.isfinite(spring)
        or spring <= 0
    ):
        raise ValueError(
            "Require finite aligned physical contact/force fields and positive weights"
        )
    answer = np.zeros_like(q)
    answer[:, :, :n] = -response[:, :, :, 0].transpose(0, 2, 1) * line[:n]
    answer[:, :, n : 2 * n] = -response[:, :, :, 1].transpose(0, 2, 1) * line[n : 2 * n]
    answer += initial_contact - initial_force
    answer[:, :, -1] += spring * (actuator - actuator[0] - q[:, :, -1])
    return answer


def bulk_port_work(force, displacement, dt, line, *, residual_bound_mpa=0.0, radius=1.0):
    """Signed b-weighted work and a bound from every saved force residual.

    If |R_i| <= epsilon*line_i, then the absolute cumulative work error is
    bounded by epsilon*dt*sum(b*line*|v|). No force/power/work is clipped.
    The optional radius weights are a CQ diagnostic, not physical damping.
    """
    q, force, line = map(np.asarray, [displacement, force, line])
    if (
        q.ndim != 3
        or len(q) < 2
        or force.shape != q.shape
        or line.shape != (q.shape[-1],)
        or not np.all(np.isfinite(force))
        or not np.all(np.isfinite(line))
        or np.any(line <= 0)
        or np.any(q[0] != 0)
        or not np.isfinite(residual_bound_mpa)
        or residual_bound_mpa < 0
        or not np.isfinite(radius)
        or not 0 < radius <= 1
    ):
        raise ValueError(
            "Require a zero-initial perturbation, finite nodal forces and valid bounds"
        )
    velocity = stage_derivative(q, dt)
    power = np.einsum("tai,tai->ta", force, velocity)
    step_work = dt * (power @ RADAU["B"])
    step_bound = (
        residual_bound_mpa * dt * (np.einsum("tai,i->ta", abs(velocity), line) @ RADAU["B"])
    )
    cumulative = np.cumsum(step_work)
    error = np.cumsum(step_bound)
    damping = radius ** (2 * np.arange(len(q)))
    weighted = np.cumsum(step_work * damping)
    weighted_error = np.cumsum(step_bound * damping)
    if not all(
        np.all(np.isfinite(a))
        for a in [velocity, power, cumulative, error, weighted, weighted_error]
    ):
        raise ValueError("Nonfinite incremental RK port work")
    return dict(
        stage_power=power,
        step_work=step_work,
        cumulative_work=cumulative,
        cumulative_residual_error_bound=error,
        radius_weighted_cumulative_work=weighted,
        radius_weighted_residual_error_bound=weighted_error,
    ), dict(
        final_work_mpa_mm2=float(cumulative[-1]),
        final_residual_work_error_bound_mpa_mm2=float(error[-1]),
        minimum_prefix_work_mpa_mm2=float(np.min(cumulative)),
        minimum_prefix_lower_bound_mpa_mm2=float(np.min(cumulative - error)),
        minimum_prefix_upper_bound_mpa_mm2=float(np.min(cumulative + error)),
        final_radius_weighted_work_mpa_mm2=float(weighted[-1]),
        final_radius_weighted_residual_error_bound_mpa_mm2=float(weighted_error[-1]),
        minimum_radius_weighted_prefix_upper_bound_mpa_mm2=float(np.min(weighted + weighted_error)),
        sum_absolute_step_work_mpa_mm2=float(np.sum(abs(step_work))),
        audited_force_residual_bound_mpa=residual_bound_mpa,
        stage_weights=RADAU["B"].tolist(),
        cq_diagnostic_radius=radius,
        previous_endpoint_in_velocity=True,
        signed_work_not_clipped=True,
        scope="Selected incremental trace only; no interior energy reconstruction, global passivity or slip-weakening stability theorem",
    )

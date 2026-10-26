"""Positive refined contact quadrature on an unchanged P1 bulk trace.

Fine internal states and traction are distinct from the retained bulk jumps.
Interpolation and force pullback are virtual-work conjugates. No bulk modes,
friction parameters or history terms are removed by this adapter.
"""

from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter

import numpy as np
from scipy.linalg import solve

from finiteblock_bem.constraints import BoundaryConstraints
from finiteblock_bem.friction import ContactState
from finiteblock_bem.nonlinear import ContactEquilibrium


def line_weights(nodes):
    widths = np.diff(nodes)
    return (np.r_[widths, 0] + np.r_[0, widths]) / 2


@dataclass(frozen=True)
class ContactQuadrature:
    bulk_y: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    interpolation: np.ndarray
    subdivisions: int

    @classmethod
    def build(cls, nodes, subdivisions=1):
        nodes = np.asarray(nodes, dtype=float)
        if (
            nodes.ndim != 1
            or len(nodes) < 2
            or not np.all(np.isfinite(nodes))
            or np.any(np.diff(nodes) <= 0)
            or not isinstance(subdivisions, int)
            or isinstance(subdivisions, bool)
            or subdivisions < 1
        ):
            raise ValueError(
                "Require increasing finite bulk nodes and positive integer subdivisions"
            )
        fine = np.unique(
            np.concatenate([np.linspace(a, b, subdivisions + 1) for a, b in pairwise(nodes)])
        )
        left = np.clip(np.searchsorted(nodes, fine, side="right") - 1, 0, len(nodes) - 2)
        fraction = (fine - nodes[left]) / (nodes[left + 1] - nodes[left])
        interpolation = np.zeros((len(fine), len(nodes)))
        interpolation[np.arange(len(fine)), left] = 1 - fraction
        interpolation[np.arange(len(fine)), left + 1] = fraction
        weights = line_weights(fine)
        if np.any(weights <= 0) or not np.allclose(
            interpolation.T @ weights, line_weights(nodes), rtol=1e-13, atol=1e-13
        ):
            raise ValueError("Fine quadrature does not preserve positive P1 line weights")
        values = [nodes.copy(), fine, weights, interpolation]
        for value in values:
            value.setflags(write=False)
        return cls(*values, subdivisions)

    @property
    def bulk_nodes(self):
        return len(self.bulk_y)

    @property
    def points(self):
        return len(self.y)

    @property
    def dofs(self):
        return 2 * self.bulk_nodes + 1

    def interpolate(self, values):
        values = np.asarray(values)
        if values.shape[-1] != self.bulk_nodes:
            raise ValueError("P1 trace values must end in the retained contact nodes")
        return values @ self.interpolation.T

    def pull_force(self, pressure, shear):
        pressure, shear = (np.asarray(v) for v in (pressure, shear))
        if pressure.shape != shear.shape or pressure.shape[-1] != self.points:
            raise ValueError("Fine pressure and shear must align with the quadrature points")
        matrix = self.weights[:, None] * self.interpolation
        return np.concatenate(
            (pressure @ matrix, shear @ matrix, np.zeros((*pressure.shape[:-1], 1))), axis=-1
        )

    def seed_from_bulk_p1(self, state):
        return ContactState(
            self.interpolate(state.plastic_slip), self.interpolate(state.cumulative_slip)
        )

    def constraints(self, platen_width):
        n, d = self.bulk_nodes, self.dofs
        eye = np.eye(d)
        normal = np.pad(self.interpolation, ((0, 0), (0, n + 1)))
        tangent = np.pad(self.interpolation, ((0, 0), (n, 1)))
        coarse = line_weights(self.bulk_y)
        return BoundaryConstraints(
            eye,
            np.ones(d),
            np.r_[coarse, coarse, platen_width],
            normal,
            tangent,
            self.weights,
            platen_width,
            np.array([], dtype=int),
        )


@dataclass(frozen=True)
class StaticTraceReduction:
    impedance: np.ndarray
    effective_load: np.ndarray
    displacement_response: np.ndarray
    loaded_displacement: np.ndarray
    reduced_stiffness: np.ndarray
    reduced_load: np.ndarray
    jump: np.ndarray
    diagnostics: dict


def reduce_static_trace(model, stiffness):
    """Constrained static energy/load reduction without a free-body inverse.

    [Kr B.T; B 0] [u;lambda] = [F; q]. Thus Z0=-lambda_q and the
    external retained load is lambda_F. Rigid modes are fixed by prescribed
    contact jumps/platen, with no artificial stiffness or regularization.
    """
    con = model.constraints
    p = con.prolongation
    stiffness = np.asarray(stiffness, dtype=float)
    if stiffness.shape != (len(p), len(p)) or not np.all(np.isfinite(stiffness)):
        raise ValueError("Require a finite full static Galerkin stiffness")
    jump = np.vstack((con.normal_jump, con.tangential_jump, np.eye(con.size)[-1]))
    k = p.T @ stiffness @ p
    load = p.T @ (model.reference.normal_pressure * model.normal_force_shape)
    d, size = len(jump), len(k)
    saddle = np.block([[k, jump.T], [jump, np.zeros((d, d))]])
    rhs = np.zeros((size + d, d + 1))
    rhs[:size, 0] = load
    rhs[size:, 1:] = np.eye(d)
    response = solve(saddle, rhs, assume_a="sym", check_finite=False)
    if not np.all(np.isfinite(response)):
        raise ValueError("Nonfinite static trace reduction")
    residual = saddle @ response - rhs
    scaled = np.max(abs(residual) / (1 + abs(rhs)))
    z = -response[size:, 1:]
    diagnostic = dict(
        saddle_scaled_residual=float(scaled),
        relative_impedance_symmetry=float(np.linalg.norm(z - z.T) / max(np.linalg.norm(z), 1)),
        no_free_body_inverse_or_regularization=True,
    )
    if scaled > 1e-7:
        raise ValueError("Static trace saddle balance exceeds tolerance")
    return StaticTraceReduction(
        z, response[size:, 0], response[:size, 1:], response[:size, 0], k, load, jump, diagnostic
    )


class RefinedContactEquilibrium(ContactEquilibrium):
    """Original NumPy/JAX root APIs with fine contact maps and fixed bulk DOFs."""

    def __init__(
        self,
        model,
        quadrature,
        matrix,
        *,
        base_normal=None,
        base_tangent=None,
        base_platen=0.0,
        base_force=None,
    ):
        self.model = model
        self.case = model.contact_case()
        self.quadrature = quadrature
        if not np.array_equal(
            quadrature.bulk_y, model.reference.moving.starts[model.moving_contact, 1]
        ):
            raise ValueError("Quadrature bulk nodes differ from the unchanged boundary model")
        self.constraints = quadrature.constraints(model.constraints.platen_width)
        d, m = quadrature.dofs, quadrature.points
        self.impedance = np.asarray(matrix, dtype=float)
        if self.impedance.shape != (d, d) or not np.all(np.isfinite(self.impedance)):
            raise ValueError("Refined contact requires the unchanged retained bulk impedance")
        self.spring = model.reference.spring_stiffness * model.constraints.platen_width
        self.matrix = self.impedance.copy()
        self.matrix[-1, -1] += self.spring
        self.base_u = np.zeros(d)
        self.base_dn = np.zeros(m) if base_normal is None else np.asarray(base_normal, dtype=float)
        self.base_ds = (
            np.zeros(m) if base_tangent is None else np.asarray(base_tangent, dtype=float)
        )
        self.base_platen = float(base_platen)
        self.base_t = np.zeros(d) if base_force is None else np.asarray(base_force, dtype=float)
        if self.base_dn.shape != (m,) or self.base_ds.shape != (m,) or self.base_t.shape != (d,):
            raise ValueError("Lifted fine jumps and coarse nodal forces do not align")

    def external_traction(self, normal_pressure):
        if normal_pressure != self.model.reference.normal_pressure:
            raise ValueError("This condensed lifting requires the original constant normal load")
        return np.zeros(self.quadrature.dofs)


@dataclass(frozen=True)
class RefinedStaticLift:
    retained_displacement: np.ndarray
    full_displacement: np.ndarray
    retained_bulk_force: np.ndarray
    state: ContactState
    pressure: np.ndarray
    shear: np.ndarray
    diagnostics: dict


def initialize_refined_contact(model, quadrature, reduction, state):
    if len(state.plastic_slip) != quadrature.points:
        raise ValueError("Static contact state must align with the fine quadrature")
    system = RefinedContactEquilibrium(
        model, quadrature, reduction.impedance, base_force=-reduction.effective_load
    )
    normal, actuator = model.reference.normal_pressure, model.reference.restart_actuator
    guess = system.closed_elastic_guess(normal, actuator, state)
    answer = system.solve(normal, actuator, state, guess=guess, frozen=True)
    if not answer.converged:
        raise ValueError("Refined static lift did not converge without changing the seed")
    q, response = answer.reduced_displacement, answer.contact
    strength = model.reference.contact.coefficient(state.cumulative_slip) * response.pressure
    ratio = float(np.max(abs(response.shear) / np.maximum(strength, 1e-20)))
    coarse_force = quadrature.pull_force(response.pressure, response.shear)
    coarse_force[-1] += system.spring * (q[-1] - actuator)
    reduced = reduction.loaded_displacement + reduction.displacement_response @ q
    uncondensed = (
        reduction.reduced_stiffness @ reduced
        - reduction.reduced_load
        + reduction.jump.T @ coarse_force
    )
    physical_error = float(np.max(abs(uncondensed) / model.constraints.reduced_weights))
    trace_error = float(np.max(abs(reduction.jump @ reduced - q)))
    moving_test = np.sum(model.constraints.prolongation[: model.reference.moving.dofs : 2], axis=0)
    bulk_normal_force = float(moving_test @ (reduction.reduced_stiffness @ reduced))
    applied_normal_force = float(moving_test @ reduction.reduced_load)
    contact_normal_force = float(np.dot(quadrature.weights, response.pressure))
    if ratio > 1 + 1e-6 or physical_error >= 1e-7 or trace_error >= 1e-9:
        raise ValueError("Refined static lift violates friction or original full-body balance")
    if not np.array_equal(response.state.plastic_slip, state.plastic_slip) or not np.array_equal(
        response.state.cumulative_slip, state.cumulative_slip
    ):
        raise ValueError("Frozen initialization changed the prescribed contact seed")
    diagnostic = dict(
        status="admissible_refined_contact_static_lift",
        bulk_dofs=quadrature.dofs,
        contact_state_points=quadrature.points,
        subdivisions=quadrature.subdivisions,
        residual_mpa=answer.residual_mpa,
        original_full_balance_mpa=physical_error,
        trace_constraint_error_mm=trace_error,
        maximum_strength_ratio=ratio,
        mean_pressure_mpa=float(np.average(response.pressure, weights=quadrature.weights)),
        moving_bulk_normal_force_mpa_mm=bulk_normal_force,
        applied_normal_force_mpa_mm=applied_normal_force,
        contact_normal_force_mpa_mm=contact_normal_force,
        moving_normal_balance_defect_mpa_mm=bulk_normal_force
        + contact_normal_force
        - applied_normal_force,
        force_scope="Original discrete Galerkin balance; its finite-quadrature rigid-translation defect is retained, not clipped",
        mean_shear_mpa=float(np.average(response.shear, weights=quadrature.weights)),
        platen_mm=float(q[-1]),
        seed_unchanged=True,
        reduction=reduction.diagnostics,
    )
    return RefinedStaticLift(
        q,
        model.constraints.prolongation @ reduced,
        reduction.impedance @ q - reduction.effective_load,
        state,
        response.pressure,
        response.shear,
        diagnostic,
    )


def dynamic_refined_reference(model, quadrature, lift, instantaneous_impedance):
    n = quadrature.bulk_nodes
    q0 = lift.retained_displacement
    return RefinedContactEquilibrium(
        model,
        quadrature,
        instantaneous_impedance,
        base_normal=quadrature.interpolate(q0[:n]),
        base_tangent=quadrature.interpolate(q0[n : 2 * n]),
        base_platen=q0[-1],
        base_force=lift.retained_bulk_force,
    )


@dataclass(frozen=True)
class RefinedTrajectory:
    time: np.ndarray
    bulk_displacement: np.ndarray
    pressure: np.ndarray
    shear: np.ndarray
    friction_coefficient: np.ndarray
    plastic_slip: np.ndarray
    cumulative_slip: np.ndarray
    actuator: np.ndarray
    platen: np.ndarray
    residual: np.ndarray
    iterations: np.ndarray
    diagnostics: dict


def affine_initial_tail(history, residual_force, steps, *, block_lags=64):
    """Strictly past L*r0 for a fixed initial lift residual starting at step 1.

    The instantaneous root already contains r0, so lag zero must NOT be
    included again. This preserves the original body force rather than
    silently balancing it with a small artificial initial contact force.
    """
    value = np.asarray(residual_force, dtype=float)
    if value.shape != (history.plan.dofs,) or history.plan.steps < steps:
        raise ValueError("Initial-residual convolution needs complete matching time weights")
    answer = np.zeros((steps + 1, history.plan.dofs))
    running = np.zeros(history.plan.dofs)
    for first in range(1, steps, block_lags):
        last = min(first + block_lags, steps)
        block = np.asarray(history.weights[first:last]) @ value
        cumulative = np.cumsum(block, axis=0) + running
        answer[first + 1 : last + 1] = cumulative
        running = cumulative[-1]
    return answer


def simulate_refined_prepared(
    model,
    quadrature,
    reference,
    state,
    history,
    elastic,
    time,
    ramp,
    *,
    equilibrium_factory=None,
    history_factory=None,
    failure_callback=None,
    schedule_name="event16_original_actuator_rate",
):
    """Complete original-grid CQ with work-conjugate refined contact forces.

    Dref remains the original coarse diagonal elastic reference. Changing
    contact integration can activate its correction immediately, so require
    every lag from step one, even when the nominal prefix remains sticking.
    """
    from finiteblock_bem.jax_backend import JaxContactEquilibrium

    steps, d, m = len(time) - 1, quadrature.dofs, quadrature.points
    if (
        history.plan.dofs != d
        or history.plan.steps < steps
        or elastic.shape != (steps + 1, d)
        or ramp.shape != (steps + 1,)
        or len(state.plastic_slip) != m
        or not np.array_equal(time, np.arange(steps + 1) * history.plan.dt)
        or not np.all(np.isfinite(elastic))
        or not np.all(np.isfinite(ramp))
    ):
        raise ValueError("Refined contact requires complete same-grid history from the first step")
    if history_factory is None:
        import runpy
        from pathlib import Path

        history_factory = runpy.run_path(str(Path(__file__).with_name("replay_resolvent.py")))[
            "CorrectionHistory"
        ]
    begun = perf_counter()
    law = model.reference.contact
    wc = model.constraints.contact_weights
    diagonal = np.r_[wc * law.normal_penalty, wc * law.tangential_penalty, 0]
    a0 = reference.matrix + np.diag(diagonal)
    composition = float(np.linalg.norm(history.instantaneous @ a0 - np.eye(d), ord=np.inf))
    if composition > 1e-6 or not np.isfinite(composition):
        raise ValueError("Original coarse-reference lag-zero inverse composition failed")
    initial, _, contact = reference.evaluate(
        np.zeros(d), np.zeros(d), model.reference.normal_pressure, ramp[0], state, jacobian=False
    )
    if np.max(abs(initial)) >= 1e-7:
        raise ValueError("Refined static initialization is not in the original equilibrium")
    initial_force = initial * reference.constraints.reduced_weights
    affine = affine_initial_tail(history, initial_force, steps)
    fc0 = quadrature.pull_force(contact.pressure, contact.shear)
    q = np.zeros((steps + 1, d))
    pressure, shear, mu, plastic, cumulative = [np.zeros((steps + 1, m)) for _ in range(5)]
    pressure[0], shear[0], mu[0], plastic[0], cumulative[0] = (
        contact.pressure,
        contact.shear,
        contact.friction_coefficient,
        state.plastic_slip,
        state.cumulative_slip,
    )
    residual, iterations = np.zeros(steps + 1), np.zeros(steps + 1, dtype=int)
    residual[0] = np.max(abs(initial))
    factory = JaxContactEquilibrium if equilibrium_factory is None else equilibrium_factory
    system, memory = factory(reference), history_factory(history)
    actuator = ramp.copy()
    force = reference.spring * (ramp - ramp[0])
    old, held, hold_time, work = state, None, None, 0.0
    for index in range(1, steps + 1):
        if held is not None:
            actuator[index] = held
        past = -a0 @ (elastic[index] - affine[index] + memory.past(index))
        past[-1] += force[index]
        answer = system.solve(
            model.reference.normal_pressure, actuator[index], old, past=past, guess=q[index - 1]
        )
        if not answer.converged:
            if failure_callback is not None:
                failure_callback(
                    index,
                    dict(
                        failed_displacement=answer.reduced_displacement,
                        previous_displacement=q[index - 1],
                        plastic_slip=old.plastic_slip,
                        cumulative_slip=old.cumulative_slip,
                        past=past,
                        time=time[index],
                        residual_mpa=answer.residual_mpa,
                    ),
                )
            raise ValueError(
                f"Refined contact root failed at {time[index]} s: {answer.residual_mpa}"
            )
        q[index], contact = answer.reduced_displacement, answer.contact
        pressure[index], shear[index], mu[index], plastic[index], cumulative[index] = (
            contact.pressure,
            contact.shear,
            contact.friction_coefficient,
            contact.state.plastic_slip,
            contact.state.cumulative_slip,
        )
        residual[index], iterations[index] = answer.residual_mpa, answer.iterations
        correction = (
            fc0 + diagonal * q[index] - quadrature.pull_force(pressure[index], shear[index])
        )
        correction[-1] += reference.spring * (actuator[index] - ramp[index])
        memory.append(index, correction, active=True)
        micro = (
            system.accepted_micro_work[-1]
            if hasattr(system, "accepted_micro_work")
            else shear[index] * (plastic[index] - old.plastic_slip)
        )
        work += float(np.dot(quadrature.weights, micro))
        old = contact.state
        if (
            held is None
            and np.average(cumulative[index] >= law.characteristic_slip, weights=quadrature.weights)
            >= model.reference.payload["loading"]["stop_coverage"]
        ):
            held, hold_time = actuator[index], time[index]
    diagnostics = dict(
        status="completed_requested_horizon",
        duration_s=float(time[-1]),
        dt_s=history.plan.dt,
        boundary_formulation="P1/P0 symmetric Galerkin; coarse P1 jumps and refined contact quadrature",
        history_representation="elastic_resolvent_refined_contact",
        schedule=schedule_name,
        bulk_contact_nodes=quadrature.bulk_nodes,
        contact_nodes=m,
        subdivisions=quadrature.subdivisions,
        history_dofs=[d],
        panels=[model.reference.moving.count, model.reference.stationary.count],
        completed_steps=steps,
        requested_steps=steps,
        correction_start=memory.start,
        all_history_retained=True,
        correction_history_steps=history.plan.steps,
        inverse_cq_composition_inf_error=composition,
        initial_residual_mpa=float(np.max(abs(initial))),
        original_lift_force_preserved=True,
        strict_past_initial_residual_response_included=True,
        actuator_hold_time_s=hold_time,
        micro_friction_work_mpa_mm2=work,
        max_cumulative_slip_mm=float(np.max(cumulative)),
        max_boundary_contact_residual_mpa=float(np.max(residual)),
        time_stepping_seconds=perf_counter() - begun,
        full_original_event_reproduced=False,
        full_trajectory_substep_ad_validated=False,
        scope="Complete original CQ history; refined state quadrature is not bulk spatial or temporal convergence",
    )
    if hasattr(system, "substeps"):
        diagnostics["constitutive_substeps"] = system.substeps
    if hasattr(system, "recovery_events"):
        diagnostics["contact_root_recovery_events"] = system.recovery_events
    if memory.online is not None and hasattr(memory.online, "statistics"):
        diagnostics["online_fft_cache"] = memory.online.statistics()
    return RefinedTrajectory(
        time,
        q,
        pressure,
        shear,
        mu,
        plastic,
        cumulative,
        actuator,
        reference.base_platen + q[:, -1],
        residual,
        iterations,
        diagnostics,
    )

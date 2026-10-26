"""Positive refined contact quadrature on an unchanged P1 bulk trace.

Fine internal states and traction are distinct from the retained bulk jumps.
Interpolation and force pullback are virtual-work conjugates. No bulk modes,
friction parameters or history terms are removed by this adapter.
"""

from dataclasses import dataclass
from itertools import pairwise

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
        mean_shear_mpa=float(np.average(response.shear, weights=quadrature.weights)),
        platen_mm=float(q[-1]),
        seed_unchanged=True,
        reduction=reduction.diagnostics,
    )
    return RefinedStaticLift(
        q,
        model.constraints.prolongation @ reduced,
        state,
        response.pressure,
        response.shear,
        diagnostic,
    )


def dynamic_refined_reference(model, quadrature, lift, instantaneous_impedance):
    n = quadrature.bulk_nodes
    q0 = lift.retained_displacement
    force0 = quadrature.pull_force(lift.pressure, lift.shear)
    spring = model.reference.spring_stiffness * model.constraints.platen_width
    base_force = -force0
    base_force[-1] -= spring * (q0[-1] - model.reference.restart_actuator)
    return RefinedContactEquilibrium(
        model,
        quadrature,
        instantaneous_impedance,
        base_normal=quadrature.interpolate(q0[:n]),
        base_tangent=quadrature.interpolate(q0[n : 2 * n]),
        base_platen=q0[-1],
        base_force=base_force,
    )

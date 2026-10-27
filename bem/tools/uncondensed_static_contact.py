"""Resolve the healed preload in all constrained boundary displacement DOFs.

Avoid cancellation in loaded_u + response_u @ trace when the endpoint panels
are small. The original finite penalties, active set and acceptance tolerances
are retained. This is an explicit static backend, not a fallback or a new
dynamic/AD solver.
"""

import runpy
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq

from finiteblock_bem.constraints import BoundaryConstraints
from finiteblock_bem.nonlinear import ContactEquilibrium

TOOLS = Path(__file__).resolve().parent


class BodyStaticEquilibrium(ContactEquilibrium):
    """Original root API in constrained body coordinates, before trace elimination."""

    def __init__(self, model, quadrature, reduction):
        con = model.constraints
        y = model.reference.moving.starts[model.moving_contact, 1]
        if not np.array_equal(y, quadrature.bulk_y):
            raise ValueError("Contact quadrature and body trace differ")
        size = con.size
        self.model, self.case = model, model.contact_case()
        self.constraints = BoundaryConstraints(
            np.eye(size),
            np.ones(size),
            con.reduced_weights,
            quadrature.interpolation @ con.normal_jump,
            quadrature.interpolation @ con.tangential_jump,
            quadrature.weights,
            con.platen_width,
            np.array([], dtype=int),
        )
        self.impedance = reduction.reduced_stiffness
        self.matrix = self.impedance.copy()
        self.spring = model.reference.spring_stiffness * con.platen_width
        self.matrix[-1, -1] += self.spring
        self.base_u = np.zeros(size)
        self.base_t = -reduction.reduced_load
        self.base_dn = np.zeros(quadrature.points)
        self.base_ds = np.zeros(quadrature.points)
        self.base_platen = 0.0

    def external_traction(self, normal_pressure):
        if normal_pressure != self.model.reference.normal_pressure:
            raise ValueError("Static reduction contains the original constant normal load")
        return np.zeros(self.constraints.size)


def initialize_uncondensed_contact(model, quadrature, reduction, state):
    """Return a checked lift with bulk force measured from the body equations."""
    if len(state.plastic_slip) != quadrature.points:
        raise ValueError("Static state must align with contact quadrature")
    static = runpy.run_path(str(TOOLS / "fem_static_contact.py"))
    refined = runpy.run_path(str(TOOLS / "refined_contact.py"))
    system = BodyStaticEquilibrium(model, quadrature, reduction)
    normal, actuator = model.reference.normal_pressure, model.reference.restart_actuator
    u, corrected, active = static["solve_healed_static_contact"](system, normal, actuator, state)
    # Check the generic frozen-state root with the same force/strength criteria.
    checked = system.solve(normal, actuator, corrected, guess=u, frozen=True)
    if not checked.converged or np.max(abs(checked.reduced_displacement - u)) > 1e-8:
        raise ValueError("Healed and frozen uncondensed roots differ")
    u, response = checked.reduced_displacement, checked.contact
    q = reduction.jump @ u
    force = quadrature.pull_force(response.pressure, response.shear)
    force[-1] += system.spring * (q[-1] - actuator)
    body_force = reduction.reduced_stiffness @ u - reduction.reduced_load
    uncondensed = body_force + reduction.jump.T @ force
    physical_error = float(np.max(abs(uncondensed) / model.constraints.reduced_weights))
    # Measure the actual body port force. Do not set it to -contact_force.
    bulk = lstsq(reduction.jump.T, body_force, lapack_driver="gelsy")[0]
    nonport = float(
        np.max(abs(body_force - reduction.jump.T @ bulk) / model.constraints.reduced_weights)
    )
    line = np.r_[
        refined["line_weights"](quadrature.bulk_y),
        refined["line_weights"](quadrature.bulk_y),
        model.constraints.platen_width,
    ]
    port_error = float(np.max(abs(bulk + force) / line))
    strength = model.reference.contact.coefficient(corrected.cumulative_slip) * response.pressure
    ratio = float(np.max(abs(response.shear) / np.maximum(strength, 1e-20)))
    n = quadrature.bulk_nodes
    trace_error = float(
        max(
            np.max(abs(system.constraints.normal_jump @ u - quadrature.interpolate(q[:n]))),
            np.max(abs(system.constraints.tangential_jump @ u - quadrature.interpolate(q[n:-1]))),
        )
    )
    if (
        not np.all(np.isfinite([physical_error, nonport, port_error, ratio, trace_error]))
        or max(physical_error, nonport, port_error) >= 1e-7
        or trace_error >= 1e-9
        or ratio > 1 + 1e-6
    ):
        raise ValueError(
            "Uncondensed static lift violates original force/trace/strength tolerances"
        )
    if not np.array_equal(
        response.state.plastic_slip, corrected.plastic_slip
    ) or not np.array_equal(response.state.cumulative_slip, state.cumulative_slip):
        raise ValueError("Frozen recheck changed the prescribed healed state")
    moving_test = np.sum(model.constraints.prolongation[: model.reference.moving.dofs : 2], axis=0)
    bulk_normal = float(moving_test @ (reduction.reduced_stiffness @ u))
    applied_normal = float(moving_test @ reduction.reduced_load)
    contact_normal = float(quadrature.weights @ response.pressure)
    diagnostics = dict(
        status="admissible_uncondensed_static_lift",
        static_backend="uncondensed_body_active_set",
        body_displacement_dofs=len(u),
        bulk_dofs=len(q),
        contact_state_points=quadrature.points,
        subdivisions=quadrature.subdivisions,
        residual_mpa=checked.residual_mpa,
        original_full_balance_mpa=physical_error,
        nonport_force_projection_residual_mpa=nonport,
        actual_port_equilibrium_residual_mpa=port_error,
        trace_constraint_error_mm=trace_error,
        maximum_strength_ratio=ratio,
        retained_schur_on_direct_trace_residual_mpa=float(
            np.max(abs(reduction.impedance @ q - reduction.effective_load + force) / line)
        ),
        bulk_force_source="Measured Kr*u-F, projected through the original work-conjugate trace",
        mean_pressure_mpa=float(np.average(response.pressure, weights=quadrature.weights)),
        mean_shear_mpa=float(np.average(response.shear, weights=quadrature.weights)),
        moving_bulk_normal_force_mpa_mm=bulk_normal,
        applied_normal_force_mpa_mm=applied_normal,
        contact_normal_force_mpa_mm=contact_normal,
        moving_normal_balance_defect_mpa_mm=bulk_normal + contact_normal - applied_normal,
        force_scope="Original Galerkin balance, including its finite-quadrature rigid-mode defect",
        platen_mm=float(q[-1]),
        seed_unchanged=active["input_seed_unchanged"],
        static_initialization=active,
        reduction=reduction.diagnostics,
        dynamic_or_ad_backend_changed=False,
        original_tolerance_relaxed=False,
    )
    return refined["RefinedStaticLift"](
        q,
        model.constraints.prolongation @ u,
        bulk,
        corrected,
        response.pressure,
        response.shear,
        diagnostics,
    )

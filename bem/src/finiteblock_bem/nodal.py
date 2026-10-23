"""Continuous boundary traces and work-conjugate nodal-force contact systems.

This adapter deliberately keeps Galerkin nodal forces separate from the legacy
P0 traction arrays. Contact uses positive lumped line weights, including half
weights at each contact endpoint. A vertex shared by contact and platen has
one displacement DOF, so supports/loading remain continuous at that vertex.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, solve

from .case import load_case
from .constraints import BoundaryConstraints
from .dynamics import RestartProjectionError
from .friction import ContactState
from .galerkin import assemble_galerkin_operators
from .galerkin_fast import PreparedGalerkinOperators
from .nonlinear import ContactEquilibrium


def edge_nodal_weights(mesh, label):
    """Integral of each P1 hat over the selected boundary edges."""
    result = np.zeros(mesh.count)
    ids = mesh.indices(label)
    np.add.at(result, ids, mesh.lengths[ids] / 2)
    np.add.at(result, (ids + 1) % mesh.count, mesh.lengths[ids] / 2)
    return result


def edge_nodes(mesh, label):
    return np.flatnonzero(edge_nodal_weights(mesh, label) > 0)


@dataclass(frozen=True)
class NodalModel:
    reference: object
    moving_contact: np.ndarray
    stationary_contact: np.ndarray
    restart: ContactState
    constraints: BoundaryConstraints
    normal_force_shape: np.ndarray
    case_path: Path

    def contact_case(self):
        """Internal adapter for the shared local contact/Newton implementation."""
        return replace(
            self.reference,
            moving_contact=self.moving_contact,
            stationary_contact=self.stationary_contact,
            restart=self.restart,
        )


def load_nodal_model(path, **mesh_options):
    path = Path(path).resolve()
    ref = load_case(path, **mesh_options)
    moving, stationary = ref.moving, ref.stationary
    im, js = edge_nodes(moving, "contact"), edge_nodes(stationary, "contact")
    im = im[np.argsort(moving.starts[im, 1])]
    js = js[np.argsort(stationary.starts[js, 1])]
    np.testing.assert_allclose(moving.starts[im], stationary.starts[js], rtol=0, atol=1e-10)
    wc = edge_nodal_weights(moving, "contact")[im]
    np.testing.assert_allclose(
        wc, edge_nodal_weights(stationary, "contact")[js], rtol=0, atol=1e-10
    )
    offset, size = moving.dofs, moving.dofs + stationary.dofs
    fixed = np.r_[
        offset + 2 * edge_nodes(stationary, "support_x"),
        offset + 2 * edge_nodes(stationary, "support_y") + 1,
    ]
    shared = 2 * edge_nodes(moving, "platen") + 1
    free = np.setdiff1d(np.arange(size), np.r_[fixed, shared])
    p = np.zeros((size, len(free) + 1))
    p[free, np.arange(len(free))] = 1
    p[shared, -1] = 1
    nodal_weight = np.r_[
        (moving.lengths + np.roll(moving.lengths, 1)) / 2,
        (stationary.lengths + np.roll(stationary.lengths, 1)) / 2,
    ]
    reduced_weights = p.T @ np.repeat(nodal_weight, 2)
    width = float(np.sum(moving.lengths[moving.indices("platen")]))
    reduced_weights[-1] = width
    con = BoundaryConstraints(
        p,
        np.ones(size),  # Input is already nodal force; never apply panel weights again.
        reduced_weights,
        p[2 * im] - p[offset + 2 * js],
        p[2 * im + 1] - p[offset + 2 * js + 1],
        wc,
        width,
        fixed,
    )
    seed = Path(ref.payload["source"]["restart_interface"])
    if not seed.is_absolute():
        seed = path.parent / seed
    data = np.genfromtxt(seed, delimiter=",", names=True)
    yy = moving.starts[im, 1]
    state = ContactState(
        np.interp(yy, data["y"], data["plastic_slip"]),
        np.interp(yy, data["y"], data["cumulative_slip"]),
    )
    normal = np.zeros(size)
    normal[:offset:2] = edge_nodal_weights(moving, "normal")
    return NodalModel(ref, im, js, state, con, normal, path)


def nodal_body_stiffness(model, s=None, *, quadrature=12):
    ref = model.reference
    return block_diag(
        *(
            assemble_galerkin_operators(
                mesh, ref.material, s, quadrature=quadrature
            ).force_stiffness()
            for mesh in (ref.moving, ref.stationary)
        )
    )


class NodalContactEquilibrium(ContactEquilibrium):
    """Shared Newton/LSW, with a Galerkin stiffness and NODAL FORCE loading."""

    def __init__(self, model, stiffness, *, base_displacement=None, base_force=None):
        self.model = model
        super().__init__(
            model.contact_case(),
            model.constraints,
            stiffness,
            base_displacement=base_displacement,
            base_traction=base_force,
        )

    def external_traction(self, normal_pressure):
        # The base class method name is retained internally; values here are forces.
        return normal_pressure * self.model.normal_force_shape


@dataclass(frozen=True)
class NodalStaticLift:
    displacement: np.ndarray
    boundary_force: np.ndarray
    state: ContactState
    diagnostics: dict


def initialize_nodal_restart(model, *, quadrature=12, stiffness=None):
    ref, con = model.reference, model.constraints
    if stiffness is None:
        stiffness = nodal_body_stiffness(model, quadrature=quadrature)
    if np.shape(stiffness) != (len(con.prolongation), len(con.prolongation)):
        raise ValueError("Expected a full nodal-force stiffness for this model")
    system = NodalContactEquilibrium(model, stiffness)
    guess = system.closed_elastic_guess(ref.normal_pressure, ref.restart_actuator, model.restart)
    result = system.solve(
        ref.normal_pressure, ref.restart_actuator, model.restart, guess=guess, frozen=True
    )
    strength = result.contact.pressure * ref.contact.coefficient(model.restart.cumulative_slip)
    ratio = float(np.max(np.abs(result.contact.shear) / np.maximum(strength, 1e-20)))
    diagnostics = {
        "status": "admissible_nodal_static_lift",
        "basis": "P1 displacement / P0 auxiliary density; symmetric Galerkin",
        "residual_mpa": result.residual_mpa,
        "initial_max_strength_ratio": ratio,
        "max_cumulative_slip_change_mm": 0.0,
        "platen_mm": float(result.reduced_displacement[-1]),
        "mean_pressure_mpa": float(
            np.average(result.contact.pressure, weights=con.contact_weights)
        ),
        "mean_shear_mpa": float(np.average(result.contact.shear, weights=con.contact_weights)),
        "quadrature": quadrature,
    }
    if not result.converged or ratio > 1 + 1e-6:
        diagnostics["status"] = "incompatible_nodal_static_lift"
        raise RestartProjectionError(
            "Nodal static lift does not satisfy equilibrium/friction", diagnostics
        )
    displacement = con.prolongation @ result.reduced_displacement
    return NodalStaticLift(
        displacement, stiffness @ displacement, result.contact.state, diagnostics
    )


class NodalInterfaceTransfer:
    """Eliminate Galerkin nodal forces in frequency space, before CQ."""

    def __init__(self, model, *, quadrature=12, prepared=True):
        self.model, self.quadrature = model, quadrature
        self.constraints = con = model.constraints
        self.jump = np.vstack((con.normal_jump, con.tangential_jump, np.eye(con.size)[-1]))
        self.dofs = len(self.jump)
        self.prepared = (
            [
                PreparedGalerkinOperators(mesh, model.reference.material, quadrature=quadrature)
                for mesh in (model.reference.moving, model.reference.stationary)
            ]
            if prepared
            else None
        )

    def evaluate(self, s, *, reconstruction=False):
        ref, con = self.model.reference, self.constraints
        compliance = np.zeros((self.dofs, self.dofs), complex)
        responses = []
        first = 0
        for body, mesh in enumerate((ref.moving, ref.stationary)):
            p = con.prolongation[first : first + mesh.dofs]
            columns = np.flatnonzero(np.any(p != 0, axis=0))
            p = p[:, columns]
            b = self.jump[:, columns]
            operators = (
                self.prepared[body].assemble(s)
                if self.prepared is not None
                else assemble_galerkin_operators(mesh, ref.material, s, quadrature=self.quadrature)
            )
            stiffness = operators.force_stiffness()
            response = solve(p.T @ stiffness @ p, b.T, check_finite=False)
            compliance += b @ response
            if reconstruction:
                responses.append((columns, response))
            first += mesh.dofs
        impedance = solve(compliance, np.eye(self.dofs), check_finite=False)
        if not reconstruction:
            return impedance
        reduced = np.zeros((con.size, self.dofs), complex)
        for columns, response in responses:
            reduced[columns] = response @ impedance
        return impedance, con.prolongation @ reduced


class NodalInterfaceEquilibrium(ContactEquilibrium):
    """Retained q=(nodal normal/tangent jumps, platen) with fixed static bulk force."""

    def __init__(self, model, lift, impedance):
        self.model = model
        self.case = model.contact_case()
        self.impedance = np.asarray(impedance, float)
        con, ref = model.constraints, model.reference
        n = len(model.moving_contact)
        size = 2 * n + 1
        eye = np.eye(size)
        self.constraints = BoundaryConstraints(
            eye,
            np.ones(size),
            np.r_[con.contact_weights, con.contact_weights, con.platen_width],
            eye[:n],
            eye[n : 2 * n],
            con.contact_weights,
            con.platen_width,
            np.array([], int),
        )
        im, js, offset = model.moving_contact, model.stationary_contact, ref.moving.dofs
        self.base_dn = lift.displacement[2 * im] - lift.displacement[offset + 2 * js]
        self.base_ds = lift.displacement[2 * im + 1] - lift.displacement[offset + 2 * js + 1]
        self.base_platen = float(
            np.mean(lift.displacement[2 * edge_nodes(ref.moving, "platen") + 1])
        )
        jump = np.vstack((con.normal_jump, con.tangential_jump, np.eye(con.size)[-1]))
        residual_force = con.reduce_force(
            lift.boundary_force - ref.normal_pressure * model.normal_force_shape
        )
        self.base_t = solve(jump @ jump.T, jump @ residual_force, check_finite=False)
        np.testing.assert_allclose(jump.T @ self.base_t, residual_force, rtol=1e-7, atol=1e-6)
        self.base_u = np.zeros(size)
        self.spring = ref.spring_stiffness * con.platen_width
        self.matrix = self.impedance.copy()
        self.matrix[-1, -1] += self.spring

    def external_traction(self, normal_pressure):
        if normal_pressure != self.model.reference.normal_pressure:
            raise ValueError("Condensed nodal mode requires constant normal pressure")
        return np.zeros(len(self.base_u))

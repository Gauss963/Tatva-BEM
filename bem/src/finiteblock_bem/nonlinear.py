"""Newton solve for one implicit dynamic or static contact equilibrium."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve

from .case import TwoBlockCase
from .constraints import BoundaryConstraints
from .friction import ContactResponse, ContactState


@dataclass(frozen=True)
class NewtonResult:
    reduced_displacement: np.ndarray
    contact: ContactResponse
    residual_mpa: float
    iterations: int
    converged: bool


class ContactEquilibrium:
    """Operator-specific system; u=base_u+P*z, t=base_t+W0*P*z+past.

    P.T*diag(panel length) enforces the resultant spring/platen equation.
    The panel-length weighted interface forces cancel pairwise by construction.
    """

    def __init__(
        self,
        case: TwoBlockCase,
        constraints: BoundaryConstraints,
        impedance,
        *,
        base_displacement=None,
        base_traction=None,
    ):
        self.case = case
        self.constraints = constraints
        self.impedance = np.asarray(impedance, dtype=float)
        p = constraints.prolongation
        size = len(p)
        if self.impedance.shape != (size, size):
            raise ValueError("Incorrect combined boundary impedance size")
        self.base_u = (
            np.zeros(size)
            if base_displacement is None
            else np.asarray(base_displacement, dtype=float)
        )
        self.base_t = (
            np.zeros(size) if base_traction is None else np.asarray(base_traction, dtype=float)
        )
        if self.base_u.shape != (size,) or self.base_t.shape != (size,):
            raise ValueError("Incorrect static-lift boundary field size")
        self.matrix = p.T @ (constraints.full_weights[:, None] * (self.impedance @ p))
        self.spring = case.spring_stiffness * constraints.platen_width
        self.matrix[-1, -1] += self.spring
        self.base_platen = float(np.mean(self.base_u[2 * case.moving.indices("platen") + 1]))
        offset = case.moving.dofs
        im, js = case.moving_contact, case.stationary_contact
        self.base_dn = self.base_u[2 * im] - self.base_u[offset + 2 * js]
        self.base_ds = self.base_u[2 * im + 1] - self.base_u[offset + 2 * js + 1]

    def external_traction(self, normal_pressure):
        result = np.zeros(len(self.base_u))
        result[2 * self.case.moving.indices("normal")] = normal_pressure
        return result

    def evaluate(self, z, past, normal_pressure, actuator, old, *, frozen=False, jacobian=True):
        con = self.constraints
        dn = self.base_dn + con.normal_jump @ z
        ds = self.base_ds + con.tangential_jump @ z
        contact = self.case.contact.evaluate(dn, ds, old, frozen=frozen)
        forcing = self.base_t + past - self.external_traction(normal_pressure)
        residual = self.matrix @ z + con.reduce_force(forcing)
        residual[-1] += self.spring * (self.base_platen - actuator)
        residual += con.normal_jump.T @ (con.contact_weights * contact.pressure)
        residual += con.tangential_jump.T @ (con.contact_weights * contact.shear)
        residual /= con.reduced_weights
        derivative = None
        if jacobian:
            derivative = self.matrix.copy()
            derivative += con.normal_jump.T @ (
                (con.contact_weights * contact.dp_dn)[:, None] * con.normal_jump
            )
            derivative += con.tangential_jump.T @ (
                (con.contact_weights * contact.dtau_ds)[:, None] * con.tangential_jump
            )
            derivative += con.tangential_jump.T @ (
                (con.contact_weights * contact.dtau_dn)[:, None] * con.normal_jump
            )
            derivative /= con.reduced_weights[:, None]
        return residual, derivative, contact

    def closed_elastic_guess(self, normal_pressure, actuator, old):
        """Static initialization only: avoid the open freely moving rigid modes."""
        con = self.constraints
        a = self.matrix.copy()
        a += con.normal_jump.T @ (
            (con.contact_weights * self.case.contact.normal_penalty)[:, None] * con.normal_jump
        )
        a += con.tangential_jump.T @ (
            (con.contact_weights * self.case.contact.tangential_penalty)[:, None]
            * con.tangential_jump
        )
        rhs = con.reduce_force(self.external_traction(normal_pressure) - self.base_t)
        rhs += con.tangential_jump.T @ (
            con.contact_weights * self.case.contact.tangential_penalty * old.plastic_slip
        )
        rhs[-1] += self.spring * (actuator - self.base_platen)
        return solve(a, rhs, check_finite=False)

    def solve(
        self,
        normal_pressure,
        actuator,
        old: ContactState,
        *,
        past=None,
        guess=None,
        frozen=False,
        tolerance=1e-7,
        max_iterations=50,
    ):
        con = self.constraints
        past = np.zeros(len(self.base_u)) if past is None else np.asarray(past)
        z = np.zeros(con.size) if guess is None else np.array(guess, dtype=float, copy=True)
        if z.shape != (con.size,) or past.shape != self.base_u.shape:
            raise ValueError("Incorrect displacement guess or history-traction shape")
        for iteration in range(1, max_iterations + 1):
            residual, jacobian, contact = self.evaluate(
                z, past, normal_pressure, actuator, old, frozen=frozen
            )
            norm = float(np.max(np.abs(residual)))
            if norm < tolerance:
                return NewtonResult(z, contact, norm, iteration, True)
            direction = solve(jacobian, -residual, check_finite=False)
            accepted = False
            alpha = 1.0
            for _ in range(24):
                trial = z + alpha * direction
                r, _, _ = self.evaluate(
                    trial, past, normal_pressure, actuator, old, frozen=frozen, jacobian=False
                )
                error = np.max(np.abs(r))
                if error < (1 - 1e-4 * alpha) * norm or error < tolerance:
                    z = trial
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                return NewtonResult(z, contact, norm, iteration, False)
        r, _, contact = self.evaluate(
            z, past, normal_pressure, actuator, old, frozen=frozen, jacobian=False
        )
        return NewtonResult(z, contact, float(np.max(abs(r))), max_iterations, False)

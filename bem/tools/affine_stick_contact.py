"""Exact linear sticking shortcut for the original substepped contact root.

Accept only when both endpoints of every local jump path lie strictly
inside the same closed Coulomb cone with unchanged old state. Convexity
then proves all local substeps stick. The original force residual must
also pass; every other case uses the existing robust nonlinear solver.

This prototype does not change the physical law, CQ history, time grid,
implicit AD API or selected nonlinear root recovery. Full-event speed and
field equivalence need separate validation before using it in a study.
"""

import runpy
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from finiteblock_bem.friction import ContactResponse, ContactState
from finiteblock_bem.nonlinear import NewtonResult

BASE = runpy.run_path(str(Path(__file__).with_name("robust_contact_experiment.py")))[
    "RobustSubsteppedContactEquilibrium"
]


class AffineStickContactEquilibrium(BASE):
    def __init__(self, reference, *, substeps=1, kinetic_seed_recovery=False):
        super().__init__(reference, substeps=substeps, kinetic_seed_recovery=kinetic_seed_recovery)
        con = reference.constraints
        self.bn, self.bt = np.asarray(con.normal_jump), np.asarray(con.tangential_jump)
        self.wc, self.wr = np.asarray(con.contact_weights), np.asarray(con.reduced_weights)
        self.normal_gram = self.bn.T @ (self.wc[:, None] * self.bn)
        self.tangent_gram = self.bt.T @ (self.wc[:, None] * self.bt)
        self.factorization = None
        self.factorization_penalties = None
        self.fast_stick_steps = 0
        self.nonlinear_steps = 0

    def affine_candidate(self, normal_pressure, actuator, old, past):
        ref = self.reference
        p = np.asarray(self.parameters)
        kn, kt, mus, muk, dc = p
        if not np.all(np.isfinite(p)) or kn <= 0 or kt <= 0 or dc <= 0:
            return None
        if self.factorization_penalties is None or not np.array_equal(
            p[:2], self.factorization_penalties
        ):
            matrix = np.asarray(ref.matrix) + kn * self.normal_gram + kt * self.tangent_gram
            self.factorization = lu_factor(matrix, check_finite=False)
            self.factorization_penalties = p[:2].copy()
        external = ref.constraints.reduce_force(ref.external_traction(normal_pressure))
        forcing = np.asarray(ref.base_t) - external + past
        forcing[-1] += ref.spring * (ref.base_platen - actuator)
        constant = self.bn.T @ (self.wc * kn * ref.base_dn)
        constant += self.bt.T @ (self.wc * kt * (ref.base_ds - old.plastic_slip))
        q = lu_solve(self.factorization, -forcing - constant, check_finite=False)
        if not np.all(np.isfinite(q)):
            return None
        mu = mus - (mus - muk) * np.minimum(old.cumulative_slip / dc, 1.0)
        pressure = kn * (ref.base_dn + self.bn @ q)
        shear = kt * (ref.base_ds + self.bt @ q - old.plastic_slip)
        old_pressure = kn * (ref.base_dn + self.bn @ self.previous_displacement)
        old_shear = kt * (ref.base_ds + self.bt @ self.previous_displacement - old.plastic_slip)
        # The previous endpoint must be tested using the CURRENT old state.
        # After weakening, the previous applied shear can exceed this cone.
        # Such a path may slip even if its final candidate lies inside it.
        margin = 1e-10
        if (
            not np.all(pressure > 0)
            or not np.all(old_pressure > 0)
            or not np.all(mu * pressure - abs(shear) > margin)
            or not np.all(mu * old_pressure - abs(old_shear) > margin)
        ):
            return None
        residual = (
            np.asarray(ref.matrix) @ q
            + forcing
            + self.bn.T @ (self.wc * pressure)
            + self.bt.T @ (self.wc * shear)
        ) / self.wr
        error = float(np.max(abs(residual)))
        if not np.isfinite(error) or error >= 1e-7:
            return None
        contact = ContactResponse(
            pressure,
            shear,
            np.full_like(pressure, kn),
            np.zeros_like(shear),
            np.full_like(shear, kt),
            mu,
            ContactState(old.plastic_slip.copy(), old.cumulative_slip.copy()),
        )
        return NewtonResult(q, contact, error, 1, True)

    def solve(self, normal_pressure, actuator, old, *, past=None, guess=None):
        # Keep the ordinary one-substep fallback consistent if parameters
        # are explicitly replaced between solves (JAX arrays are immutable).
        self.base.parameters = self.parameters
        past = np.zeros(self.reference.constraints.size) if past is None else np.asarray(past)
        answer = self.affine_candidate(normal_pressure, actuator, old, past)
        if answer is None:
            self.nonlinear_steps += 1
            return super().solve(normal_pressure, actuator, old, past=past, guess=guess)
        self.fast_stick_steps += 1
        self.previous_displacement = answer.reduced_displacement.copy()
        self.accepted_micro_work.append(np.zeros_like(old.plastic_slip))
        self.accepted_integrated_strength_work.append(np.zeros_like(old.plastic_slip))
        return answer

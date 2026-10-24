"""Eliminate unretained displacements without inverting free-body compliance.

For jump map B, choose a fixed lifting P and nullspace N with BP=I, BN=0.
Stationarity at fixed jumps gives U=P-N (N.T K N)^-1 N.T K P and Z=P.T K U.
This uses the same supported Galerkin stiffness as the original transfer. It
does not add springs, clip modes, regularize a singular inverse or change CQ.
All transposes are bilinear at complex Laplace frequencies.
"""

import numpy as np
from scipy.linalg import qr, solve, solve_triangular

from finiteblock_bem.nodal import NodalInterfaceTransfer


class ConstrainedSchurTransfer(NodalInterfaceTransfer):
    def __init__(self, model, *, quadrature=12):
        super().__init__(model, quadrature=quadrature)
        q, r = qr(self.jump.T, mode="full", check_finite=False)
        d = self.dofs
        if np.min(abs(np.diag(r[:d]))) <= np.finfo(float).eps * np.linalg.norm(r):
            raise ValueError("The retained jump map must have full row rank")
        self.lifting = solve_triangular(r[:d], q[:, :d].T, check_finite=False).T
        self.interior = q[:, d:]
        np.testing.assert_allclose(self.jump @ self.lifting, np.eye(d), atol=1e-12, rtol=0)
        np.testing.assert_allclose(self.jump @ self.interior, 0, atol=1e-12, rtol=0)
        self.lifting.flags.writeable = False
        self.interior.flags.writeable = False

    def assemble_reduced(self, s):
        """Supported nodal stiffness plus separate body blocks for an audit."""
        con, ref = self.constraints, self.model.reference
        stiffness = np.zeros((con.size, con.size), dtype=float if s is None else complex)
        parts = []
        first = 0
        for prepared, mesh in zip(self.prepared, [ref.moving, ref.stationary], strict=True):
            p = con.prolongation[first : first + mesh.dofs]
            columns = np.flatnonzero(np.any(p != 0, axis=0))
            p = p[:, columns]
            block = p.T @ prepared.assemble(s).force_stiffness() @ p
            stiffness[np.ix_(columns, columns)] += block
            parts.append((columns, block))
            first += mesh.dofs
        return stiffness, parts

    def condense(self, stiffness):
        """Return retained force and stationary full reduced displacement maps."""
        p, n = self.lifting, self.interior
        eliminated = n.T @ stiffness @ n
        response = p - n @ solve(eliminated, n.T @ stiffness @ p, check_finite=False)
        impedance = p.T @ stiffness @ response
        return impedance, response, eliminated

    def evaluate(self, s, *, reconstruction=False):
        stiffness, _ = self.assemble_reduced(s)
        impedance, response, _ = self.condense(stiffness)
        if not reconstruction:
            return impedance
        return impedance, self.constraints.prolongation @ response

    def original_compliance(self, parts):
        """Original algebra on the exact SAME body matrices; audit only."""
        compliance = np.zeros((self.dofs, self.dofs), dtype=parts[0][1].dtype)
        for columns, block in parts:
            jump = self.jump[:, columns]
            compliance += jump @ solve(block, jump.T, check_finite=False)
        return solve(compliance, np.eye(self.dofs), check_finite=False)

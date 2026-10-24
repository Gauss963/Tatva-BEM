"""Two-stage fixed-trace Schur elimination with the original bulk matrices.

First eliminate each body's unretained coordinates with its contact/platen
trace fixed. Then eliminate the common trace motion at fixed jumps. This
preserves rigid modes in the retained map and avoids a large joint-nullspace
factorization. No frequency-dependent switching or regularization is used.
"""

import runpy
from pathlib import Path

import numpy as np
from scipy.linalg import qr, solve, solve_triangular

from finiteblock_bem.nodal import NodalInterfaceTransfer

BASE = runpy.run_path(str(Path(__file__).with_name("schur_condensation_experiment.py")))[
    "ConstrainedSchurTransfer"
]


class TraceSchurTransfer(BASE):
    def __init__(self, model, *, quadrature=12):
        # Reuse the assembly/comparison interface without constructing the
        # larger QR basis of BASE. The two trace bases below replace it.
        NodalInterfaceTransfer.__init__(self, model, quadrature=quadrature)
        trace = np.any(self.jump != 0, axis=0)
        self.trace_indices = np.flatnonzero(trace)
        b = self.jump[:, trace]
        q, r = qr(b.T, mode="full", check_finite=False)
        d = self.dofs
        self.trace_lifting = solve_triangular(r[:d], q[:, :d].T, check_finite=False).T
        self.trace_nullspace = q[:, d:]
        np.testing.assert_allclose(b @ self.trace_lifting, np.eye(d), atol=1e-12, rtol=0)
        np.testing.assert_allclose(b @ self.trace_nullspace, 0, atol=1e-12, rtol=0)
        interior, first = [], 0
        for mesh in [model.reference.moving, model.reference.stationary]:
            p = self.constraints.prolongation[first : first + mesh.dofs]
            ids = np.flatnonzero(np.any(p != 0, axis=0) & ~trace)
            if len(ids):
                interior.append(ids)
            first += mesh.dofs
        combined = np.concatenate(interior) if interior else np.array([], int)
        np.testing.assert_array_equal(np.sort(combined), np.flatnonzero(~trace))
        self.body_interiors = tuple(interior)
        for array in [
            self.trace_indices,
            self.trace_lifting,
            self.trace_nullspace,
            *self.body_interiors,
        ]:
            array.flags.writeable = False

    def condense(self, stiffness):
        trace = self.trace_indices
        trace_matrix = stiffness[np.ix_(trace, trace)].copy()
        recover = np.zeros((self.constraints.size, len(trace)), dtype=stiffness.dtype)
        recover[trace] = np.eye(len(trace))
        for ids in self.body_interiors:
            response = solve(
                stiffness[np.ix_(ids, ids)], stiffness[np.ix_(ids, trace)], check_finite=False
            )
            trace_matrix -= stiffness[np.ix_(trace, ids)] @ response
            recover[ids] = -response
        p, n = self.trace_lifting, self.trace_nullspace
        eliminated = n.T @ trace_matrix @ n
        trace_response = p - n @ solve(eliminated, n.T @ trace_matrix @ p, check_finite=False)
        impedance = p.T @ trace_matrix @ trace_response
        return impedance, recover @ trace_response, eliminated

    def elimination_blocks(self, stiffness, final):
        return [stiffness[np.ix_(ids, ids)] for ids in self.body_interiors] + [final]

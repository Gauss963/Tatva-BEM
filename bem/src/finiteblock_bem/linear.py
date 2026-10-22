"""Mixed-boundary transient solve for a single finite elastic domain."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .cq import BoundaryHistory


@dataclass(frozen=True)
class LinearHistoryResult:
    time: np.ndarray
    displacement: np.ndarray
    traction: np.ndarray
    max_boundary_residual: float


def solve_mixed(history: BoundaryHistory, prescribed, applied_traction):
    """prescribed maps displacement DOF to a length-(steps+1) time series.

    Tractions are imposed at all remaining DOFs. Displacement and traction
    series must start from zero; nonzero initial fields require a static lift
    or separate initial-condition potentials.
    """
    size = history.plan.dofs
    times = np.arange(history.plan.steps + 1) * history.plan.dt
    applied = np.asarray(applied_traction, dtype=float)
    if applied.shape != (len(times), size) or not np.all(np.isfinite(applied)):
        raise ValueError("Expected finite applied tractions of shape (steps+1, dofs)")
    fixed = np.array(sorted(prescribed), dtype=int)
    if np.any((fixed < 0) | (fixed >= size)):
        raise ValueError("Invalid prescribed DOF")
    free = np.setdiff1d(np.arange(size), fixed)
    u = np.zeros_like(applied)
    for dof, value in prescribed.items():
        if np.shape(value) != (len(times),) or not np.all(np.isfinite(value)):
            raise ValueError("Prescribed displacement series has incorrect shape or values")
        u[:, dof] = value
    if np.max(np.abs(u[0]), initial=0) > 1e-14 or np.max(np.abs(applied[0]), initial=0) > 1e-14:
        raise ValueError(
            "This solver requires zero initial perturbation and initially zero forcing"
        )
    traction = np.zeros_like(applied)
    w0 = history.instantaneous
    factor = lu_factor(w0[np.ix_(free, free)]) if len(free) else None
    max_residual = 0.0
    for step in range(1, len(times)):
        past = history.past_traction(u, step)
        rhs = applied[step, free] - past[free]
        rhs -= w0[np.ix_(free, fixed)] @ u[step, fixed]
        if len(free):
            u[step, free] = lu_solve(factor, rhs)
        traction[step] = w0 @ u[step] + past
        max_residual = max(
            max_residual, np.max(np.abs(traction[step, free] - applied[step, free]), initial=0)
        )
    return LinearHistoryResult(times, u, traction, max_residual)

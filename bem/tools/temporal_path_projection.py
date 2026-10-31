"""Best time-P1 approximation of a saved P1 field on fixed stage knots.

The exact temporal mass matrix and RHS use the union of the two clocks.
All spatial coordinates are retained. Since the spatial mass matrix is
common and invertible, the time projection acts independently at each
spatial node and is also optimal in the tensor-product space-time norm.
This is a representation bound for the saved field, not a PDE error bound.
"""

import runpy
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

TOOLS = Path(__file__).resolve().parent
PATH = runpy.run_path(str(TOOLS / "fem_contact_path_observations.py"))


def best_p1_time_projection(time, values, knots):
    """Unconstrained orthogonal projection; coefficients are not sampled values."""
    time, values, knots = (np.asarray(a, dtype=float) for a in [time, values, knots])
    if (
        time.ndim != 1
        or len(time) < 2
        or knots.ndim != 1
        or len(knots) < 2
        or values.ndim != 2
        or values.shape[0] != len(time)
        or values.shape[1] < 1
        or not all(np.all(np.isfinite(a)) for a in [time, values, knots])
        or np.any(np.diff(time) <= 0)
        or np.any(np.diff(knots) <= 0)
        or knots[0] != time[0]
        or knots[-1] != time[-1]
    ):
        raise ValueError("Require finite P1 fields on ordered clocks with identical boundaries")
    union = np.union1d(time, knots)
    field = PATH["interpolate_time"](time, values, union)
    left = np.clip(np.searchsorted(knots, union[:-1], side="right") - 1, 0, len(knots) - 2)
    h = np.diff(knots)
    a = (union[:-1] - knots[left]) / h[left]
    b = (union[1:] - knots[left]) / h[left]
    scale = np.diff(union)[:, None] / 6
    rhs = np.zeros((len(knots), values.shape[1]))
    # Exact integral of each coarse hat times a native P1 function.
    np.add.at(
        rhs,
        left,
        scale * ((3 - 2 * a - b)[:, None] * field[:-1] + (3 - a - 2 * b)[:, None] * field[1:]),
    )
    np.add.at(
        rhs,
        left + 1,
        scale * ((2 * a + b)[:, None] * field[:-1] + (a + 2 * b)[:, None] * field[1:]),
    )
    diagonal = np.r_[h, 0] / 3 + np.r_[0, h] / 3
    band = np.zeros((3, len(knots)))
    band[1], band[0, 1:], band[2, :-1] = diagonal, h / 6, h / 6
    coefficients = solve_banded((1, 1), band, rhs, check_finite=False)
    residual = diagonal[:, None] * coefficients - rhs
    residual[:-1] += h[:, None] / 6 * coefficients[1:]
    residual[1:] += h[:, None] / 6 * coefficients[:-1]
    absolute = float(np.linalg.norm(residual))
    denominator = float(np.linalg.norm(rhs))
    orthogonality = absolute / denominator if denominator else absolute
    if not np.all(np.isfinite(coefficients)) or orthogonality > 1e-12:
        raise ValueError("Temporal projection mass solve did not meet its orthogonality check")
    return coefficients, dict(
        temporal_mass_equation_relative_residual=orthogonality,
        original_boundary_times_retained=True,
        endpoint_coefficients_unconstrained=True,
        spatial_coordinates_unchanged=True,
        scope="Exact L2 orthogonal time-P1 projection of the saved P1 field; endpoints are free coefficients. Representation lower bound in this fixed time space only, without a continuous-solution or nonlinear BEM error certificate.",
    )

"""Measured static-body initialization and its complete Radau residual history.

The static preload is solved before trace elimination, using the existing
healed active set. A small nonzero original equilibrium residual is retained
as a force, rather than silently replacing the measured body force by -fc0.
"""

import runpy
from pathlib import Path

import numpy as np

from finiteblock_bem.nodal import nodal_body_stiffness

TOOLS = Path(__file__).resolve().parent
REFINED = runpy.run_path(str(TOOLS / "refined_contact.py"))
BODY = runpy.run_path(str(TOOLS / "uncondensed_static_contact.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))


def initialize_radau_body(model, *, quadrature):
    """Identity contact quadrature; same original healed preload and tolerances."""
    y = model.reference.moving.starts[model.moving_contact, 1]
    space = REFINED["ContactQuadrature"].build(y, 1)
    stiffness = nodal_body_stiffness(model, quadrature=quadrature)
    reduction = REFINED["reduce_static_trace"](model, stiffness)
    lift = BODY["initialize_uncondensed_contact"](model, space, reduction, model.restart)
    reference = REFINED["dynamic_refined_reference"](
        model, space, lift, np.zeros((space.dofs, space.dofs))
    )
    return lift, reference


def initial_residual_past(first, second, residual_force, steps, *, rows=8):
    """Strictly past compact L*r0 for a constant force beginning in interval 1.

    The current root contains r0 itself, so lag zero is excluded. Summing the
    two compact families against a repeated stage force gives, for k>=1,
    1*sum_{l=1}^{k-1}(W0_l*r0) + Delta0*1*((W1_{k-1}-W1_0)*r0).
    The finite-DFT wrap occurs at lag zero and is already in the current root.
    Project row tiles across all time lags to support both C/Fortran storage.
    """
    value = np.asarray(residual_force)
    shape = np.shape(first)
    if (
        len(shape) != 3
        or shape != np.shape(second)
        or shape[1] != shape[2]
        or value.shape != (shape[1],)
        or not np.all(np.isfinite(value))
        or not isinstance(steps, int)
        or isinstance(steps, bool)
        or not 1 <= steps < shape[0]
        or not isinstance(rows, int)
        or isinstance(rows, bool)
        or rows < 1
    ):
        raise ValueError("Require complete matching weights, initial force and positive row budget")
    d = shape[1]
    dtype = np.result_type(first.dtype, second.dtype, value.dtype)
    answer = np.zeros((steps + 1, 2, d), dtype=dtype)
    coefficient = np.linalg.inv(RADAU["A"]) @ np.ones(2)
    for begin in range(0, d, rows):
        end = min(begin + rows, d)
        projected = np.einsum("kij,j->ki", first[1:steps, begin:end], value)
        cumulative = np.cumsum(projected, axis=0)
        delta = np.einsum("kij,j->ki", second[1:steps, begin:end], value)
        delta -= np.asarray(second[0, begin:end]) @ value
        if not np.all(np.isfinite(cumulative)) or not np.all(np.isfinite(delta)):
            raise ValueError("Nonfinite original initial-residual history")
        answer[2:, :, begin:end] = (
            cumulative[:, None, :] + coefficient[None, :, None] * delta[:, None, :]
        )
    return answer

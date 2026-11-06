"""Experimental full-length Radau startup using two physical force actions."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.fft import irfft
from scipy.linalg import solve

TOOLS = Path(__file__).resolve().parent
STORAGE = runpy.run_path(str(TOOLS / "compact_radau_storage.py"))
RAW = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))
RADAU = STORAGE["RADAU"]


def projected_resolvent(stiffness, jump, tangent, forces):
    """Apply the closed/stick trace compliance through the supported body system.

    Inputs use physical forces and bilinear transposes. The jump map sends
    supported body displacements to the physical contact and platen traces.
    """
    rhs = jump.T @ forces
    matrix = stiffness + jump.T @ (tangent[:, None] * jump)
    displacement = solve(matrix, rhs)
    error = np.linalg.norm(matrix @ displacement - rhs, axis=0)
    scale = np.linalg.norm(rhs, axis=0)
    if (
        not np.all(np.isfinite(displacement))
        or not np.all(np.isfinite(error))
        or not np.all(np.isfinite(scale))
        or np.any(error > 1e-8 * scale)
    ):
        raise ValueError("Projected body solve fails the physical force residual")
    return jump @ displacement


def projected_histories(plan, branch_actions, actuator_force, *, rows=8):
    """Build the complete ramp and strictly past affine history in row tiles.

    Each half-circle mode has two actions: the unit platen force and the
    physical initial equilibrium residual. The caller supplies both using
    the same fixed tangent as the nonlinear correction kernel.
    """
    modes = plan.fft_size // 2 + 1
    actions = np.asarray(branch_actions)
    force = np.asarray(actuator_force)
    if (
        actions.shape != (modes, 2, plan.dofs, 2)
        or force.shape != (plan.steps + 1, 2)
        or not np.all(np.isfinite(force))
        or not 1 <= plan.steps < plan.fft_size // 2
        or not isinstance(rows, int)
        or isinstance(rows, bool)
        or rows < 1
    ):
        raise ValueError("Require aligned projected modes and a complete Radau startup grid")
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(modes) / plan.fft_size)
    eigenvalues = RAW["lower_half_radau_modes"](zeta, 1.0)
    first = np.empty((len(force), 2, plan.dofs))
    past = np.zeros_like(first)
    amplification = plan.radius ** (-np.arange(1, plan.steps))[:, None]
    coefficient = np.linalg.solve(RADAU["A"], np.ones(2))
    for start in range(0, plan.dofs, rows):
        section = slice(start, start + rows)
        values = actions[:, :, section, :]
        f0, f1 = [
            family.reshape(modes, values.shape[2], 2)
            for family in STORAGE["compact_rows"](eigenvalues, values.reshape(modes, 2, -1))
        ]
        for family in (f0, f1):
            endpoints = family[[0, -1]]
            if np.any(
                np.linalg.norm(endpoints.imag, axis=1)
                > 1e-10 * np.maximum(np.linalg.norm(endpoints.real, axis=1), 1e-300)
            ):
                raise ValueError("Projected compact spectrum violates real endpoint symmetry")
            family[[0, -1]] = endpoints.real
        local_plan = replace(plan, dofs=values.shape[2])
        wrap = plan.radius * irfft(f1[:, :, 0], n=plan.fft_size, axis=0)[-1]
        first[:, :, section] = STORAGE["causal_compact_startup"](
            f0[:, :, 0], f1[:, :, 0], wrap, local_plan, force
        )
        w0 = irfft(f0[:, :, 1], n=plan.fft_size, axis=0)
        w1 = irfft(f1[:, :, 1], n=plan.fft_size, axis=0)
        affine_sum = np.cumsum(w0[1 : plan.steps] * amplification, axis=0)
        affine_delta = w1[1 : plan.steps] * amplification - w1[0]
        past[2:, :, section] = (
            affine_sum[:, None, :] + coefficient[None, :, None] * affine_delta[:, None, :]
        )
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(past)):
        raise ValueError("Nonfinite complete projected startup")
    return first, past

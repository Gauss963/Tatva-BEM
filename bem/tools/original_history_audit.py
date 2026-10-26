"""Causal original-operator action from a completed work resolvent spectrum.

This is independent of the nonlinear replay recurrence: invert each saved
frequency matrix, recover time coefficients by row IFFT, truncate only the
zero-padding beyond the requested horizon, and apply a linear convolution.
An untruncated frequency product would wrap future inputs into early outputs.

The reconstruction shares the sampled frequency operator with the primal.
It checks time-domain algebra and causality, not interpolation accuracy or
physical convergence. Fresh BEM frequency checks are a separate requirement.
"""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
from scipy.fft import irfft, rfft
from scipy.linalg import solve
from threadpoolctl import threadpool_limits


def invert_work_spectrum(source, target, *, block_matrices=16, residual_limit=1e-7, workers=1):
    """Invert complete frequency matrices with an independently checked identity."""
    if (
        len(source.shape) != 3
        or source.shape[1] != source.shape[2]
        or source.shape != target.shape
        or not isinstance(block_matrices, int)
        or block_matrices < 1
        or not isinstance(workers, int)
        or workers < 1
        or not np.isfinite(residual_limit)
        or residual_limit <= 0
    ):
        raise ValueError("Require matching square half spectra and positive inversion budgets")
    identity = np.eye(source.shape[1])
    maximum, asymmetry = 0.0, 0.0
    begun = perf_counter()

    def invert_one(a):
        value = solve(a, identity, check_finite=False)
        if not np.all(np.isfinite(value)):
            raise ValueError("Nonfinite reconstructed original operator")
        error = float(np.linalg.norm(value @ a - identity, ord=np.inf))
        relative = float(np.linalg.norm(value - value.T) / max(np.linalg.norm(value), 1e-300))
        return value, error, relative

    # Independent frequencies provide parallelism with one BLAS thread each.
    with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(workers) as pool:
        for first in range(0, source.shape[0], block_matrices):
            last = min(first + block_matrices, source.shape[0])
            block = np.asarray(source[first:last])
            if not np.all(np.isfinite(block)):
                raise ValueError("Nonfinite source resolvent")
            results = list(pool.map(invert_one, block))
            maximum = max(maximum, max(v[1] for v in results))
            asymmetry = max(asymmetry, max(v[2] for v in results))
            if maximum > residual_limit:
                raise ValueError(f"Original-operator inverse identity failed: {maximum}")
            # Preserve source arithmetic, including any small asymmetry.
            target[first:last] = np.stack([v[0] for v in results])
    if hasattr(target, "sync"):
        target.sync()
    return dict(
        complete_frequency_matrices=source.shape[0],
        maximum_inverse_composition_inf_error=maximum,
        maximum_relative_transpose_asymmetry=asymmetry,
        inversion_seconds=perf_counter() - begun,
        frequency_workers=workers,
        blas_threads_per_worker=1,
        no_symmetrization_or_regularization=True,
    )


def estimate_action_arrays(plan, steps, directions, rows):
    """Conservative explicit-array estimate; excludes OS pages and FFT runtime."""
    if (
        not isinstance(steps, int)
        or steps < 1
        or not isinstance(directions, int)
        or directions < 1
        or not isinstance(rows, int)
        or rows < 1
        or steps > plan.steps
        or 2 * (steps + 1) > plan.fft_size
    ):
        raise ValueError("Require covered causal horizon, directions and positive FFT rows")
    rows = min(rows, plan.dofs)
    return (
        4 * plan.fft_size * rows * plan.dofs * 8
        + 3 * plan.fft_size * plan.dofs * directions * 8
        + 2 * plan.fft_size * rows * directions * 8
        + 64 * plan.dofs**2 * 16
    )


def causal_original_action(
    original_work_spectrum,
    plan,
    displacement,
    line_weights,
    *,
    fft_rows=4,
    max_workspace_mb=4096,
):
    """Apply every causal coefficient to multiple bulk displacement histories.

    Lwork=sqrt(W) Lphysical sqrt(W), hence its inverse Awork satisfies
    Aphysical=sqrt(W) Awork sqrt(W). The returned force includes the fixed
    elastic-reference and spring terms in A; subtract them explicitly when
    evaluating the physical original-Z residual.
    """
    q = np.asarray(displacement, dtype=float)
    line = np.asarray(line_weights, dtype=float)
    if q.ndim == 2:
        q = q[..., None]
    if (
        q.ndim != 3
        or q.shape[1] != plan.dofs
        or not np.all(np.isfinite(q))
        or line.shape != (plan.dofs,)
        or not np.all(np.isfinite(line))
        or np.any(line <= 0)
        or original_work_spectrum.shape != (plan.fft_size // 2 + 1, plan.dofs, plan.dofs)
        or not np.isfinite(max_workspace_mb)
        or max_workspace_mb <= 0
    ):
        raise ValueError("Require finite aligned physical histories and positive work weights")
    steps, directions = len(q) - 1, q.shape[-1]
    estimate = estimate_action_arrays(plan, steps, directions, fft_rows)
    if estimate > max_workspace_mb * 1024**2:
        raise ValueError("Causal original-operator FFT action exceeds its array workspace")
    begun = perf_counter()
    damping = plan.radius ** np.arange(steps + 1)
    scale = np.sqrt(line)
    transformed = rfft(q * damping[:, None, None] * scale[None, :, None], n=plan.fft_size, axis=0)
    answer = np.empty_like(q)
    for first in range(0, plan.dofs, fft_rows):
        last = min(first + fft_rows, plan.dofs)
        sampled = original_work_spectrum[:, first:last]
        for index in (0, -1):
            if np.linalg.norm(sampled[index].imag) > 1e-10 * max(
                np.linalg.norm(sampled[index].real), 1.0
            ):
                raise ValueError("Original spectrum violates real-axis conjugate symmetry")
        coefficients = irfft(sampled, n=plan.fft_size, axis=0)
        del sampled
        # Keep every requested lag. Zero only the padding beyond the horizon.
        # This is the essential causal step; a direct spectral product omits it.
        coefficients[steps + 1 :] = 0
        kernel = rfft(coefficients, n=plan.fft_size, axis=0)
        del coefficients
        product = np.einsum("kij,kjp->kip", kernel, transformed, optimize=False)
        del kernel
        value = irfft(product, n=plan.fft_size, axis=0)[: steps + 1]
        answer[:, first:last] = value * scale[None, first:last, None] / damping[:, None, None]
    if not np.all(np.isfinite(answer)):
        raise ValueError("Nonfinite complete causal original-operator force")
    return answer, dict(
        complete_lags=steps + 1,
        directions=directions,
        estimated_action_array_mib=estimate / 1024**2,
        causal_action_seconds=perf_counter() - begun,
        future_input_wrap_excluded=True,
        reconstructed_from_same_sampled_frequency_operator=True,
        scope="Independent time-domain algebra; fresh BEM samples and physical convergence remain separate",
    )


def physical_residual(
    action, displacement, contact_force, base_force, diagonal, spring, platen0, actuator
):
    """Original finite-body equilibrium, in integrated nodal force coordinates."""
    q, action, contact = (np.asarray(v) for v in (displacement, action, contact_force))
    if (
        q.ndim != 2
        or q.shape != action.shape
        or q.shape != contact.shape
        or np.shape(base_force) != (q.shape[1],)
        or np.shape(diagonal) != (q.shape[1],)
        or np.shape(actuator) != (len(q),)
    ):
        raise ValueError("Original residual fields must share the retained time/grid coordinates")
    # A=Z+Dref+spring. Its instantaneous spring*q cancels the physical
    # spring*q, leaving the base-platen/actuator term below.
    residual = action - diagonal * q + contact + base_force
    residual[:, -1] += spring * (platen0 - actuator)
    return residual

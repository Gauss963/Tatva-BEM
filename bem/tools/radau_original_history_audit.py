"""Reconstruct and apply every causal original RK coefficient in bounded rows.

This shares the saved interpolated frequency operator with the primal. It
audits the time-domain equations independently of the elastic baseline and
nonlinear replay, but does not replace fresh BEM samples or convergence.
"""

import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.fft import irfft, rfft
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

TOOLS = Path(__file__).resolve().parent
RAW = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))
AUDIT = runpy.run_path(str(TOOLS / "original_history_audit.py"))


def invert_compact_spectra(
    first, second, targets, plan, *, workers=1, block_matrices=8, residual_limit=1e-7
):
    """Invert both actual modes and retain both original-operator families."""
    shape = (plan.fft_size // 2 + 1, plan.dofs, plan.dofs)
    if (
        first.shape != shape
        or second.shape != shape
        or len(targets) != 2
        or any(target.shape != shape for target in targets)
        or not isinstance(workers, int)
        or workers < 1
        or not isinstance(block_matrices, int)
        or block_matrices < 1
        or not np.isfinite(residual_limit)
        or residual_limit <= 0
    ):
        raise ValueError("Require two aligned complete families and positive inversion budgets")
    identity = np.eye(plan.dofs)
    maximum = 0.0
    begun = perf_counter()

    def invert_one(item):
        index, f0, f1 = item
        zeta = plan.radius * np.exp(-2j * np.pi * index / plan.fft_size)
        modes = RAW["lower_half_radau_modes"](zeta, 1.0)
        values, errors = [], []
        for w in modes:
            original = solve(f0 + w * f1, identity, check_finite=False)
            if not np.all(np.isfinite(original)):
                raise ValueError("Nonfinite reconstructed original RK operator")
            errors.append(float(np.linalg.norm(original @ (f0 + w * f1) - identity, ord=np.inf)))
            values.append(original)
        if max(errors) > residual_limit:
            raise ValueError(f"Original RK inverse identity failed: {max(errors)}")
        return RAW["compact_coefficients"](zeta, *values), max(errors)

    with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(workers) as pool:
        for start in range(0, shape[0], block_matrices):
            stop = min(start + block_matrices, shape[0])
            a, b = np.asarray(first[start:stop]), np.asarray(second[start:stop])
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                raise ValueError("Nonfinite source RK family")
            rows = list(pool.map(invert_one, zip(range(start, stop), a, b, strict=True)))
            maximum = max(maximum, max(row[1] for row in rows))
            for family, target in enumerate(targets):
                target[start:stop] = np.stack([row[0][family] for row in rows])
    for target in targets:
        if hasattr(target, "sync"):
            target.sync()
    return dict(
        complete_half_spectrum_matrices=shape[0],
        actual_modes_inverted_per_matrix=2,
        maximum_inverse_composition_inf_error=maximum,
        inversion_seconds=perf_counter() - begun,
        frequency_workers=workers,
        blas_threads_per_worker=1,
        no_symmetrization_or_regularization=True,
    )


def causal_compact_original_action(
    first, second, plan, displacement, line_weights, *, fft_rows=4, max_workspace_mb=4096
):
    """Apply I*G0 + Delta*G1, including its finite-DFT instantaneous wrap.

    q is (time, stage, dof[, direction]); forces use physical integrated nodal
    coordinates. Every coefficient through the requested horizon is retained.
    Future input is excluded by zeroing only the coefficient padding before
    multiplication, rather than multiplying untruncated frequency samples.
    """
    q = np.asarray(displacement, dtype=float)
    if q.ndim == 3:
        q = q[..., None]
    line = np.asarray(line_weights, dtype=float)
    shape = (plan.fft_size // 2 + 1, plan.dofs, plan.dofs)
    if (
        q.ndim != 4
        or q.shape[1:3] != (2, plan.dofs)
        or not np.all(np.isfinite(q))
        or line.shape != (plan.dofs,)
        or not np.all(np.isfinite(line))
        or np.any(line <= 0)
        or first.shape != shape
        or second.shape != shape
        or not isinstance(fft_rows, int)
        or fft_rows < 1
        or not np.isfinite(max_workspace_mb)
        or max_workspace_mb <= 0
    ):
        raise ValueError("Require finite aligned stage histories, families and positive budgets")
    steps, directions = len(q) - 1, q.shape[-1]
    estimate = AUDIT["estimate_action_arrays"](plan, steps, 2 * directions, fft_rows)
    estimate += 6 * q.size * 8
    if estimate > max_workspace_mb * 1024**2:
        raise ValueError("Causal RK original-operator action exceeds its array workspace")
    begun = perf_counter()
    damping = plan.radius ** np.arange(steps + 1)
    scale = np.sqrt(line)
    transformed = rfft(
        q * damping[:, None, None, None] * scale[None, None, :, None],
        n=plan.fft_size,
        axis=0,
    )
    delta0 = RADAU["radau_symbol"](0).real
    delta1 = (RADAU["radau_symbol"](1) - delta0).real
    answer = np.empty_like(q)
    maximum_wrap = 0.0
    for start in range(0, plan.dofs, fft_rows):
        stop = min(start + fft_rows, plan.dofs)
        parts = []
        for family, spectrum in enumerate((first, second)):
            sampled = np.asarray(spectrum[:, start:stop])
            if not np.all(np.isfinite(sampled)):
                raise ValueError("Nonfinite original RK family")
            for index in (0, -1):
                if np.linalg.norm(sampled[index].imag) > 1e-10 * max(
                    np.linalg.norm(sampled[index].real), 1.0
                ):
                    raise ValueError("Original RK family violates real-axis conjugate symmetry")
            coefficients = irfft(sampled, n=plan.fft_size, axis=0)
            del sampled
            if family == 1:
                wrap = plan.radius * coefficients[-1].copy()
                maximum_wrap = max(maximum_wrap, float(np.max(abs(wrap))))
            coefficients[steps + 1 :] = 0
            kernel = rfft(coefficients, n=plan.fft_size, axis=0)
            del coefficients
            product = np.einsum("kij,ksjp->ksip", kernel, transformed, optimize=False)
            del kernel
            value = irfft(product, n=plan.fft_size, axis=0)[: steps + 1]
            parts.append(value / damping[:, None, None, None])
            del product, value
        block = parts[0] + np.einsum("ab,tbip->taip", delta0, parts[1])
        block[1:] += np.einsum("ab,tbip->taip", delta1, parts[1][:-1])
        wrapped = np.einsum("ij,tsjp->tsip", wrap, q * scale[None, None, :, None])
        block += np.einsum("ab,tbip->taip", delta1, wrapped)
        answer[:, :, start:stop] = block * scale[None, None, start:stop, None]
    if not np.all(np.isfinite(answer)):
        raise ValueError("Nonfinite complete causal original RK force")
    return answer, dict(
        complete_lags=steps + 1,
        stages=2,
        directions=directions,
        estimated_action_array_mib=estimate / 1024**2,
        causal_action_seconds=perf_counter() - begun,
        maximum_absolute_work_family_shift_wrap=maximum_wrap,
        finite_dft_shift_wrap_retained=True,
        future_input_wrap_excluded=True,
        reconstructed_from_same_sampled_frequency_operator=True,
        scope="Independent time-domain algebra; fresh BEM samples and physical convergence remain separate",
    )

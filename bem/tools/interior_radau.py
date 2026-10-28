"""Small dense rectangular Radau CQ reference for interior-field validation.

Keep every lag and the finite-DFT alias terms, with a zero artificial initial
stage block. This is an offline reference, not the bounded large-event history
implementation. The explicit resource guard rejects oversized reconstructions.
"""

import runpy
from pathlib import Path

import numpy as np
from scipy.fft import irfft

from finiteblock_bem.cq import ResourceBudgetError, plan_cq

RK = runpy.run_path(str(Path(__file__).with_name("radau_experiment.py")))


def stage_matrix(transfer, zeta, dt):
    if not np.isfinite(zeta) or abs(zeta) >= 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Require a unit-disk node and positive dt")
    values, vectors = np.linalg.eig(RK["radau_symbol"](zeta))
    if np.linalg.cond(vectors) > 1e8:
        raise ValueError("Near-defective Radau symbol needs a derivative evaluator")
    inverse = np.linalg.inv(vectors)
    maps = [np.asarray(transfer(s / dt)) for s in values]
    if (
        maps[0].ndim != 2
        or min(maps[0].shape) < 1
        or any(m.shape != maps[0].shape or not np.all(np.isfinite(m)) for m in maps)
    ):
        raise ValueError("Require finite rectangular transfer matrices with matching shapes")
    m, n = maps[0].shape
    result = np.zeros((2 * m, 2 * n), complex)
    for k, value in enumerate(maps):
        result += np.kron(np.outer(vectors[:, k], inverse[k]), value)
    return result


def rectangular_weights(
    transfer, dt, steps, inputs, outputs, *, max_memory_mb=512, alias_tolerance=1e-10
):
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in [inputs, outputs]):
        raise ValueError("Require positive integer input and output channel counts")
    plan = plan_cq(2, dt, steps, max_memory_mb=max_memory_mb, alias_tolerance=alias_tolerance)
    elements = 4 * inputs * outputs
    peak = ((plan.fft_size // 2 + 1) * 16 + 2 * plan.fft_size * 8 + (steps + 1) * 8 + 64) * elements
    if peak > max_memory_mb * 1024**2:
        raise ResourceBudgetError(
            f"Rectangular CQ arrays need about {peak / 1024**2:.1f} MiB before caller state; choose a bounded history backend"
        )
    spectrum = np.empty((plan.fft_size // 2 + 1, 2 * outputs, 2 * inputs), complex)
    for k in range(len(spectrum)):
        zeta = plan.radius * np.exp(-2j * np.pi * k / plan.fft_size)
        matrix = stage_matrix(transfer, zeta, dt)
        if matrix.shape != spectrum.shape[1:]:
            raise ValueError("Declared interior input/output channels differ from the transfer")
        spectrum[k] = matrix
    for k in [0, len(spectrum) - 1]:
        if np.linalg.norm(spectrum[k].imag) > 1e-10 * max(np.linalg.norm(spectrum[k].real), 1):
            raise ValueError("Interior transfer violates real-axis conjugate symmetry")
        spectrum[k] = spectrum[k].real
    transformed = irfft(spectrum, n=plan.fft_size, axis=0)
    weights = transformed[: steps + 1].copy()
    weights *= plan.radius ** (-np.arange(steps + 1))[:, None, None]
    return weights, dict(
        dt_s=dt,
        steps=steps,
        fft_size=plan.fft_size,
        radius=plan.radius,
        array_peak_estimate_bytes=peak,
        weight_bytes=weights.nbytes,
        sampled_frequency_evaluations=2 * len(spectrum),
        every_requested_lag_retained=True,
        caller_state_and_operator_assembly_excluded_from_estimate=True,
    )


def apply_weights(weights, stages):
    weights, stages = np.asarray(weights), np.asarray(stages)
    if (
        weights.ndim != 3
        or stages.shape != (len(weights), weights.shape[-1])
        or weights.shape[1] % 2
        or stages.shape[1] % 2
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(stages))
        or np.any(stages[0] != 0)
    ):
        raise ValueError(
            "Require complete finite weights and a zero artificial initial stage block"
        )
    answer = np.zeros((len(weights), weights.shape[1]), dtype=np.result_type(weights, stages))
    for k in range(1, len(weights)):
        answer[k] = np.einsum("lij,lj->i", weights[: k + 1], stages[k::-1])
    return answer

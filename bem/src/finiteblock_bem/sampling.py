"""Experimental parsimonious BDF2-CQ frequency sampling.

Inspired by Melenk/Nick, arXiv:2410.15079, sections 3, 5 and 7. This is an
adaptive variant, not their theorem's prescribed mesh or error bound. Each
piece uses Chebyshev interpolation, coefficient-tail and independent point
checks; unresolved pieces revert to exact FFT-grid evaluations. No time tail
is discarded. Dense CQ remains the default and the validation reference.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
from scipy.fft import dct
from scipy.interpolate import BarycentricInterpolator
from threadpoolctl import threadpool_limits

from .cq import bdf2_symbol, invert_spectrum


@dataclass(frozen=True)
class SamplingOptions:
    relative_tolerance: float = 1e-12
    absolute_tolerance: float = 1e-8
    degree: int = 16
    check_points: int = 7

    def __post_init__(self):
        if not 1e-14 <= self.relative_tolerance <= 1e-3:
            raise ValueError("Sampling relative tolerance must be between 1e-14 and 1e-3")
        if not np.isfinite(self.absolute_tolerance) or self.absolute_tolerance < 0:
            raise ValueError("Sampling absolute tolerance must be finite and nonnegative")
        if not isinstance(self.degree, int) or not 8 <= self.degree <= 64:
            raise ValueError("Chebyshev degree must be an integer between 8 and 64")
        if not isinstance(self.check_points, int) or self.check_points < 3:
            raise ValueError("At least three independent check points required")


def parsimonious_weights(transfer, plan, *, options=None, row_scale=None, workers=1, progress=None):
    """Return weights and measured interpolation diagnostics.

    Point checks are empirical, not an operator-norm certificate. Validate the
    actual contact trajectory as well as weights before using a new resolution.
    row_scale maps generalized forces to traction units for the error checks.
    """
    options = SamplingOptions() if options is None else options
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    scale = np.ones(plan.dofs) if row_scale is None else np.asarray(row_scale, dtype=float)
    if scale.shape != (plan.dofs,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Positive finite row scales required")
    half = plan.fft_size // 2
    theta = 2 * np.pi * np.arange(half + 1) / plan.fft_size
    spectrum = np.empty((half + 1, plan.dofs, plan.dofs), dtype=complex)
    records = {
        "method": "adaptive_piecewise_chebyshev",
        "options": asdict(options),
        "dense_grid_evaluations": half + 1,
        "transfer_evaluations": 0,
        "interpolated_grid_points": 0,
        "direct_grid_points": 0,
        "accepted_intervals": 0,
        "rejected_intervals": 0,
        "max_accepted_normalized_check_error": 0.0,
        "max_accepted_normalized_tail": 0.0,
        "certified_error_bound": False,
    }

    def evaluate(angle):
        s = bdf2_symbol(plan.radius * np.exp(-1j * angle)) / plan.dt
        result = transfer(s)
        if np.shape(result) != (plan.dofs, plan.dofs) or not np.all(np.isfinite(result)):
            raise FloatingPointError("Nonfinite or incorrectly sized Laplace transfer matrix")
        return result

    def norm(values):
        return np.linalg.norm(values * scale[None, :, None], axis=(-2, -1))

    with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(workers) as pool:

        def samples(angles):
            result = []
            for first in range(0, len(angles), workers):
                batch = angles[first : first + workers]
                values = map(evaluate, batch) if workers == 1 else pool.map(evaluate, batch)
                result.extend(values)
                records["transfer_evaluations"] += len(batch)
            return np.asarray(result)

        # Broad initial pieces; adaptive bisection resolves low-frequency poles.
        # The near-zero grid is exact, as in the parsimonious CQ construction.
        first_angle = min(np.sqrt(abs(np.log(plan.radius))), np.pi)
        first_index = min(half, max(1, int(np.ceil(first_angle * plan.fft_size / (2 * np.pi)))))
        spectrum[:first_index] = samples(theta[:first_index])
        records["direct_grid_points"] = first_index
        stack = [(first_index, half + 1)]
        degree = options.degree
        nodes = np.cos(np.arange(degree + 1) * np.pi / degree)
        barycentric = (-1.0) ** np.arange(degree + 1)
        barycentric[[0, -1]] *= 0.5
        # Non-Chebyshev fractions, disjoint from interpolation nodes.
        check = (
            2 * (np.arange(options.check_points) + 0.3819660112501051) / options.check_points - 1
        )
        while stack:
            left, right = stack.pop()
            count = right - left
            if count <= 2 * (degree + 1 + options.check_points):
                spectrum[left:right] = samples(theta[left:right])
                records["direct_grid_points"] += count
            else:
                middle = (theta[left] + theta[right - 1]) / 2
                radius = (theta[right - 1] - theta[left]) / 2
                values = samples(middle + radius * nodes)
                poly = BarycentricInterpolator(nodes, values, wi=barycentric, axis=0)
                exact = samples(middle + radius * check)
                predicted = poly(check)
                reference_norm = max(float(np.max(norm(values))), float(np.max(norm(exact))))
                tolerance = max(
                    options.absolute_tolerance + options.relative_tolerance * reference_norm,
                    np.finfo(float).tiny,
                )
                error = float(np.max(norm(predicted - exact))) / tolerance
                coefficients = dct(values, type=1, axis=0) / degree
                coefficients[[0, -1]] *= 0.5
                tail = float(np.sum(norm(coefficients[-4:]))) / tolerance
                if error <= 1 and tail <= 0.25:
                    for first in range(left, right, 8):
                        last = min(first + 8, right)
                        spectrum[first:last] = poly((theta[first:last] - middle) / radius)
                    records["interpolated_grid_points"] += count
                    records["accepted_intervals"] += 1
                    records["max_accepted_normalized_check_error"] = max(
                        records["max_accepted_normalized_check_error"], error
                    )
                    records["max_accepted_normalized_tail"] = max(
                        records["max_accepted_normalized_tail"], tail
                    )
                else:
                    records["rejected_intervals"] += 1
                    split = (left + right) // 2
                    stack.extend([(split, right), (left, split)])
            if progress is not None:
                done = records["direct_grid_points"] + records["interpolated_grid_points"]
                progress(done, half + 1)
    records["evaluation_ratio"] = records["transfer_evaluations"] / (half + 1)
    return invert_spectrum(spectrum, plan), records

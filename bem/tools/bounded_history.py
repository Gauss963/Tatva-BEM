"""Capacity-limited, exact dyadic history for experimental resolvent runs.

Frequently reused small levels keep their kernel FFTs. Larger levels stream
output rows on every use. This changes storage and recomputation only: every
accepted source contributes to every required causal target, without fitting
the kernel or deleting a time tail. The production package stays unchanged.
"""

from types import FunctionType

import numpy as np
from scipy.fft import irfft, rfft

from finiteblock_bem.cq import OnlineConvolution


def cache_layout(plan, cache_mb, *, direct_block=16):
    if not np.isfinite(cache_mb) or cache_mb < 0:
        raise ValueError("FFT cache capacity must be finite and nonnegative")
    if not isinstance(direct_block, int) or direct_block < 1:
        raise ValueError("direct_block must be a positive integer")
    limit = int(cache_mb * 1024**2)
    levels, stored = {}, 0
    for k in range(int(plan.steps).bit_length()):
        b = 1 << k
        if b <= direct_block or 3 * b > plan.steps:
            continue
        size = (b + 1) * plan.dofs**2 * 16
        if stored + size <= limit:
            levels[b] = size
            stored += size
    return levels, stored


def estimate_online_arrays(plan, full_steps, cache_mb, *, fft_rows=4):
    """Conservative array estimate with disk weights; not a hard RSS limit.

    Counts full trajectory/adjoint vectors, retained FFTs and concurrent row
    transforms/padding. Read-only file pages and library/OS caches are excluded.
    The reserve covers smaller local arrays and runtime workspace heuristically.
    """
    if not isinstance(fft_rows, int) or fft_rows < 1 or full_steps < plan.steps:
        raise ValueError("Positive FFT rows and a covering full horizon are required")
    levels, cached = cache_layout(plan, cache_mb)
    largest = 1 << (int(plan.steps).bit_length() - 1)
    rows, d = min(fft_rows, plan.dofs), plan.dofs
    stream = 4 * (largest + 1) * rows * d * 16
    temporary = max(stream, max(levels.values(), default=0))
    vectors = (28 * (full_steps + 1) + 4 * (plan.steps + 1)) * d * 8
    return cached + temporary + vectors + 512 * 1024**2


class BoundedOnlineConvolution(OnlineConvolution):
    def __init__(self, history, *, cache_mb=256, fft_rows=4, direct_block=16):
        if fft_rows is None:
            raise ValueError("Bounded history requires a positive number of FFT rows")
        self.cache_levels, _ = cache_layout(history.plan, cache_mb, direct_block=direct_block)
        super().__init__(history, direct_block=direct_block, fft_rows=fft_rows)
        self.cache_limit_bytes = int(cache_mb * 1024**2)
        self.cache_peak_bytes, self.cached_bytes = 0, 0
        self.kernel_fft_builds, self.streamed_updates = 0, 0

    def append(self, displacement):
        n = self.next_step
        b = (n + 1) & -(n + 1)
        if b <= self.direct_block or n >= self.history.plan.steps:
            return super().append(displacement)
        value = np.asarray(displacement)
        if value.shape != (self.history.plan.dofs,) or not np.all(np.isfinite(value)):
            raise ValueError("Invalid accepted boundary displacement")
        self.inputs[n] = value
        self.next_step += 1
        start, count = n + 1 - b, min(b, len(self.pending) - n - 1)
        source = rfft(self.inputs[start : n + 1], n=2 * b, axis=0)
        if b in self.cache_levels:
            if b not in self.transforms:
                kernel = self.history.weights[: min(2 * b, len(self.history.weights))]
                self.transforms[b] = rfft(kernel, n=2 * b, axis=0)
                self.kernel_fft_builds += 1
                self.cached_bytes += self.transforms[b].nbytes
                self.cache_peak_bytes = max(self.cache_peak_bytes, self.cached_bytes)
                if self.cached_bytes > self.cache_limit_bytes:
                    raise ValueError("Kernel FFT cache exceeded its planned capacity")
            product = np.einsum("kij,kj->ki", self.transforms[b], source, optimize=False)
            self.pending[n + 1 : n + 1 + count] += irfft(product, n=2 * b, axis=0)[b : b + count]
            return
        self.streamed_levels.append(b)
        self.streamed_updates += 1
        for row in range(0, self.history.plan.dofs, self.fft_rows):
            kernel = self.history.weights[
                : min(2 * b, len(self.history.weights)), row : row + self.fft_rows
            ]
            transform = rfft(kernel, n=2 * b, axis=0)
            self.kernel_fft_builds += 1
            product = np.einsum("kij,kj->ki", transform, source, optimize=False)
            self.pending[n + 1 : n + 1 + count, row : row + self.fft_rows] += irfft(
                product, n=2 * b, axis=0
            )[b : b + count]
            del transform, product

    def statistics(self):
        return dict(
            capacity_bytes=self.cache_limit_bytes,
            peak_retained_kernel_fft_bytes=self.cache_peak_bytes,
            cached_levels=sorted(self.transforms),
            streamed_levels=sorted(set(self.streamed_levels)),
            streamed_updates=self.streamed_updates,
            kernel_fft_builds=self.kernel_fft_builds,
            fft_rows=self.fft_rows,
            all_history_terms_retained=True,
        )


def bounded_correction_class(original, *, cache_mb, fft_rows=4):
    """Keep the original zero-prefix checks; replace only its private factory."""

    def convolution(history):
        return BoundedOnlineConvolution(history, cache_mb=cache_mb, fft_rows=fft_rows)

    append = FunctionType(
        original.append.__code__,
        {**original.append.__globals__, "OnlineConvolution": convolution},
        original.append.__name__,
        original.append.__defaults__,
    )
    append.__kwdefaults__ = original.append.__kwdefaults__
    return type("BoundedCorrectionHistory", (original,), {"append": append})

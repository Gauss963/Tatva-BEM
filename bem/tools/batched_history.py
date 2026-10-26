"""Exact bounded CQ history for several derivative right-hand sides.

One kernel FFT serves every direction, including streamed large blocks.
No rank approximation, time-tail truncation or change of physical weights.
"""

import runpy
from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft

BASE = runpy.run_path(str(Path(__file__).with_name("bounded_history.py")))[
    "BoundedOnlineConvolution"
]


class BatchedBoundedConvolution(BASE):
    def __init__(self, history, *, right_hand_sides, cache_mb=256, fft_rows=4, direct_block=16):
        if (
            not isinstance(right_hand_sides, int)
            or isinstance(right_hand_sides, bool)
            or right_hand_sides < 1
        ):
            raise ValueError("Require a positive integer number of right-hand sides")
        super().__init__(history, cache_mb=cache_mb, fft_rows=fft_rows, direct_block=direct_block)
        self.right_hand_sides = right_hand_sides
        self.pending = np.zeros((history.plan.steps + 1, history.plan.dofs, right_hand_sides))
        self.inputs = np.zeros_like(self.pending)

    def append(self, displacement):
        n = self.next_step
        if n > self.history.plan.steps:
            raise ValueError("Online history is already complete")
        value = np.asarray(displacement)
        if value.shape != self.inputs.shape[1:] or not np.all(np.isfinite(value)):
            raise ValueError("Invalid accepted batched boundary displacement")
        self.inputs[n] = value
        self.next_step += 1
        b = (n + 1) & -(n + 1)
        start, count = n + 1 - b, min(b, len(self.pending) - n - 1)
        if count <= 0:
            return
        target = self.pending[n + 1 : n + 1 + count]
        if b <= self.direct_block:
            for j in range(b):
                lag = b - j
                target += np.einsum(
                    "kij,jr->kir",
                    self.history.weights[lag : lag + count],
                    self.inputs[start + j],
                    optimize=False,
                )
            return
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
            product = np.einsum("kij,kjr->kir", self.transforms[b], source, optimize=False)
            target += irfft(product, n=2 * b, axis=0)[b : b + count]
            return
        self.streamed_levels.append(b)
        self.streamed_updates += 1
        for row in range(0, self.history.plan.dofs, self.fft_rows):
            kernel = self.history.weights[
                : min(2 * b, len(self.history.weights)), row : row + self.fft_rows
            ]
            transform = rfft(kernel, n=2 * b, axis=0)
            self.kernel_fft_builds += 1
            product = np.einsum("kij,kjr->kir", transform, source, optimize=False)
            target[:, row : row + self.fft_rows] += irfft(product, n=2 * b, axis=0)[b : b + count]
            del transform, product

    def statistics(self):
        return {
            **super().statistics(),
            "right_hand_sides": self.right_hand_sides,
            "kernel_fft_shared_across_right_hand_sides": True,
        }

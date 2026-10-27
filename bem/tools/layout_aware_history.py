"""Exact bounded CQ with column streaming for transposed disk weights.

A C-order (time,row,column) file transposed in its last two axes makes
output-row streaming touch almost every file page for each small row block.
Stream INPUT columns of that view instead, accumulate the complete frequency
product and perform one inverse FFT. This preserves every causal pair and
avoids another full transposed weight file. It changes floating-point summation
grouping, so full gradient equivalence needs its own measured check.
"""

import runpy
from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft

BASE = runpy.run_path(str(Path(__file__).with_name("bounded_history.py")))[
    "BoundedOnlineConvolution"
]


class LayoutAwareBoundedConvolution(BASE):
    def __init__(self, history, *, cache_mb=256, fft_rows=4, direct_block=16):
        super().__init__(history, cache_mb=cache_mb, fft_rows=fft_rows, direct_block=direct_block)
        stride = history.weights.strides
        self.prefer_input_columns = abs(stride[1]) < abs(stride[2])
        self.column_streamed_updates = 0
        self.maximum_frequency_product_bytes = 0

    def append(self, displacement):
        n = self.next_step
        if n > self.history.plan.steps:
            raise ValueError("Online history is already complete")
        b = (n + 1) & -(n + 1)
        if (
            not self.prefer_input_columns
            or b <= self.direct_block
            or b in self.cache_levels
            or n >= self.history.plan.steps
        ):
            return super().append(displacement)
        value = np.asarray(displacement)
        d = self.history.plan.dofs
        if value.shape != (d,) or not np.all(np.isfinite(value)):
            raise ValueError("Invalid accepted boundary displacement")
        self.inputs[n] = value
        self.next_step += 1
        start, count = n + 1 - b, min(b, len(self.pending) - n - 1)
        if count <= 0:
            return
        source = rfft(self.inputs[start : n + 1], n=2 * b, axis=0)
        product = np.zeros((b + 1, d), complex)
        self.maximum_frequency_product_bytes = max(
            self.maximum_frequency_product_bytes, product.nbytes
        )
        self.streamed_levels.append(b)
        self.streamed_updates += 1
        self.column_streamed_updates += 1
        for column in range(0, d, self.fft_rows):
            stop = min(column + self.fft_rows, d)
            kernel = self.history.weights[: min(2 * b, len(self.history.weights)), :, column:stop]
            transform = rfft(kernel, n=2 * b, axis=0)
            self.kernel_fft_builds += 1
            product += np.einsum("kij,kj->ki", transform, source[:, column:stop], optimize=False)
            del transform, kernel
        self.pending[n + 1 : n + 1 + count] += irfft(product, n=2 * b, axis=0)[b : b + count]

    def statistics(self):
        return {
            **super().statistics(),
            "transposed_input_column_streaming": self.prefer_input_columns,
            "column_streamed_updates": self.column_streamed_updates,
            "maximum_frequency_product_bytes": self.maximum_frequency_product_bytes,
            "duplicate_transposed_weight_file_required": False,
            "floating_point_sum_grouping_changed": self.prefer_input_columns,
        }

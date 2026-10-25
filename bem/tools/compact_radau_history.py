"""Exact two-family Radau history with batched, capacity-limited dyadic FFTs.

Both stages share each family FFT, including streamed row transforms. No
2d-by-2d time weights are expanded. Transpose mode supplies the same causal
operator on reversed adjoint inputs; the caller controls time reversal.
"""

import runpy
from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft

from finiteblock_bem.cq import BoundaryHistory, cq_plan_signature

TOOLS = Path(__file__).resolve().parent
BOUNDED = runpy.run_path(str(TOOLS / "bounded_history.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))


class BatchedOnlineConvolution:
    """One kernel FFT for every input channel at a given dyadic block."""

    def __init__(self, history, channels, *, cache_mb=0, fft_rows=4, direct_block=16):
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
            raise ValueError("Input channel count must be a positive integer")
        if not isinstance(fft_rows, int) or isinstance(fft_rows, bool) or fft_rows < 1:
            raise ValueError("FFT rows must be a positive integer")
        plan = history.plan
        if history.weights.shape != (plan.steps + 1, plan.dofs, plan.dofs):
            raise ValueError("Time weights do not match their CQ plan")
        self.cache_levels, _ = BOUNDED["cache_layout"](plan, cache_mb, direct_block=direct_block)
        self.history, self.channels = history, channels
        self.pending = np.zeros((plan.steps + 1, plan.dofs, channels))
        self.inputs = np.zeros_like(self.pending)
        self.transforms, self.next_step = {}, 0
        self.direct_block, self.fft_rows = direct_block, fft_rows
        self.cache_limit_bytes = int(cache_mb * 1024**2)
        self.cache_peak_bytes, self.kernel_fft_builds, self.streamed_updates = 0, 0, 0

    def past(self, step):
        if step != self.next_step or step > self.history.plan.steps:
            raise ValueError("Batched history must be consumed in order")
        return self.pending[step].copy()

    def append(self, value):
        n, plan = self.next_step, self.history.plan
        if n > plan.steps:
            raise ValueError("Batched history is already complete")
        value = np.asarray(value)
        if value.shape != (plan.dofs, self.channels) or not np.all(np.isfinite(value)):
            raise ValueError("Invalid accepted batched source")
        self.inputs[n] = value
        self.next_step += 1
        b = (n + 1) & -(n + 1)
        first, count = n + 1 - b, min(b, len(self.pending) - n - 1)
        if count <= 0:
            return
        weights = self.history.weights
        if b <= self.direct_block:
            for j in range(b):
                lag = b - j
                self.pending[n + 1 : n + 1 + count] += np.einsum(
                    "kij,jp->kip", weights[lag : lag + count], self.inputs[first + j]
                )
            return
        source = rfft(self.inputs[first : n + 1], n=2 * b, axis=0)
        if b in self.cache_levels:
            if b not in self.transforms:
                self.transforms[b] = rfft(weights[: min(2 * b, len(weights))], n=2 * b, axis=0)
                self.kernel_fft_builds += 1
                retained = sum(v.nbytes for v in self.transforms.values())
                if retained > self.cache_limit_bytes:
                    raise ValueError("Batched kernel FFT cache exceeded its capacity")
                self.cache_peak_bytes = max(self.cache_peak_bytes, retained)
            product = np.einsum("kij,kjp->kip", self.transforms[b], source)
            self.pending[n + 1 : n + 1 + count] += irfft(product, n=2 * b, axis=0)[b : b + count]
            return
        self.streamed_updates += 1
        for row in range(0, plan.dofs, self.fft_rows):
            kernel = weights[: min(2 * b, len(weights)), row : row + self.fft_rows]
            transform = rfft(kernel, n=2 * b, axis=0)
            self.kernel_fft_builds += 1
            product = np.einsum("kij,kjp->kip", transform, source)
            self.pending[n + 1 : n + 1 + count, row : row + self.fft_rows] += irfft(
                product, n=2 * b, axis=0
            )[b : b + count]

    def statistics(self):
        return dict(
            channels=self.channels,
            capacity_bytes=self.cache_limit_bytes,
            peak_retained_kernel_fft_bytes=self.cache_peak_bytes,
            cached_levels=sorted(self.transforms),
            streamed_updates=self.streamed_updates,
            kernel_fft_builds=self.kernel_fft_builds,
            fft_rows=self.fft_rows,
            all_history_terms_retained=True,
        )


class CompactRadauConvolution:
    """I*F0 + (Delta0 + z*Delta1)*F1 with the finite-DFT lag-zero wrap."""

    def __init__(self, first, second, wrap, *, cache_mb=0, fft_rows=4, transpose=False):
        if cq_plan_signature(first.plan) != cq_plan_signature(second.plan):
            raise ValueError("Compact families require the same mathematical CQ plan")
        d = first.plan.dofs
        wrap = np.array(wrap, copy=True)
        if wrap.shape != (d, d) or not np.all(np.isfinite(wrap)):
            raise ValueError("Invalid compact lag-zero wrap")
        if not np.isfinite(cache_mb) or cache_mb < 0:
            raise ValueError("FFT cache capacity must be finite and nonnegative")
        delta0 = np.linalg.inv(RADAU["A"])
        delta1 = -np.outer(delta0 @ np.ones(2), [0, 1])
        if transpose:
            first = BoundaryHistory(first.plan, first.weights.transpose(0, 2, 1))
            second = BoundaryHistory(second.plan, second.weights.transpose(0, 2, 1))
            delta0, delta1, wrap = delta0.T, delta1.T, wrap.T
        self.plan, self.delta0, self.delta1 = first.plan, delta0, delta1
        self.instantaneous = np.kron(np.eye(2), first.instantaneous)
        self.instantaneous += np.kron(delta0, second.instantaneous) + np.kron(delta1, wrap)
        if not np.all(np.isfinite(self.instantaneous)):
            raise ValueError("Invalid compact instantaneous operator")
        self.streams = [
            BatchedOnlineConvolution(h, 2, cache_mb=cache_mb / 2, fft_rows=fft_rows)
            for h in [first, second]
        ]
        self.previous_f1 = np.zeros((d, 2))

    @property
    def next_step(self):
        return self.streams[0].next_step

    def past(self, step):
        p0, p1 = [stream.past(step) for stream in self.streams]
        value = p0.T + self.delta0 @ p1.T + self.delta1 @ self.previous_f1.T
        return value.ravel()

    def append(self, value):
        if self.next_step > self.plan.steps:
            raise ValueError("Compact history is already complete")
        value = np.asarray(value)
        if value.shape != (2 * self.plan.dofs,) or not np.all(np.isfinite(value)):
            raise ValueError("Invalid accepted Radau stage source")
        stages = value.reshape(2, self.plan.dofs).T
        f1 = self.streams[1]
        self.previous_f1 = f1.history.instantaneous @ stages + f1.past(self.next_step)
        for stream in self.streams:
            stream.append(stages)

    def statistics(self):
        families = [s.statistics() for s in self.streams]
        return dict(
            families=families,
            peak_retained_kernel_fft_bytes=sum(
                s["peak_retained_kernel_fft_bytes"] for s in families
            ),
            kernel_fft_builds=sum(s["kernel_fft_builds"] for s in families),
            expanded_stage_time_weights_stored=False,
            finite_dft_shift_wrap_retained=True,
            all_history_terms_retained=True,
        )

"""Bounded exact short-lag copies of immutable disk-backed CQ weights.

This optional storage adapter retains every uncached lag on its original array.
It does not change the time grid, compress coefficients, or exceed its prefix budget.
Callers must keep the source files and their arrays unchanged for its lifetime.
The cache budget is additional to any online kernel-FFT cache/workspace budget.
"""

from numbers import Integral

import numpy as np

from .cq import BoundaryHistory, cq_plan_signature


class _PrefixWeights:
    """Array slices backed by a small owned prefix or the unchanged source."""

    def __init__(self, source, prefix):
        self._source, self._prefix = source, prefix
        self.shape, self.dtype, self.ndim = source.shape, source.dtype, source.ndim
        self.nbytes = source.nbytes

    def __len__(self):
        return len(self._source)

    def __array__(self, dtype=None, copy=None):
        raise TypeError("Slice cached time weights explicitly; full materialization is disabled")

    def __getitem__(self, key):
        parts = key if isinstance(key, tuple) else (key,)
        if not parts:
            return self._source[key]
        first, count = parts[0], len(self._prefix)
        if isinstance(first, Integral) and not isinstance(first, (bool, np.bool_)):
            index = int(first)
            if index < 0:
                index += len(self)
            if 0 <= index < count:
                return self._prefix[(index, *parts[1:])]
        elif isinstance(first, slice) and first.step in (None, 1):
            start, stop, _ = first.indices(len(self))
            if start <= count and stop <= count:
                return self._prefix[(slice(start, stop), *parts[1:])]
        return self._source[key]

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        elif not axes:
            axes = tuple(reversed(range(self.ndim)))
        if len(axes) != self.ndim or axes[0] != 0 or set(axes) != set(range(self.ndim)):
            raise ValueError("Cached weight transpose must retain time as its first axis")
        return type(self)(self._source.transpose(axes), self._prefix.transpose(axes))


def cache_history_prefixes(histories, *, capacity_bytes, max_lags=32):
    """Return BoundaryHistory adapters and an explicit additional-memory record.

    All families receive the same number of complete cached time matrices.
    Prefix layout follows the source's stride order, preserving its contraction
    traversal as far as possible. Transpose views share the same allocation.
    A zero/insufficient budget returns the original histories without copying.
    This is an opt-in helper; existing production simulations do not enable it.
    """
    for value, name, minimum in [(capacity_bytes, "capacity_bytes", 0), (max_lags, "max_lags", 1)]:
        if (
            not isinstance(value, Integral)
            or isinstance(value, (bool, np.bool_))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    histories = tuple(histories)
    if not histories or any(not isinstance(h, BoundaryHistory) for h in histories):
        raise ValueError("Require at least one BoundaryHistory")
    plan = histories[0].plan
    if any(cq_plan_signature(h.plan) != cq_plan_signature(plan) for h in histories):
        raise ValueError("Cached families require the same mathematical CQ plan")
    shape = (plan.steps + 1, plan.dofs, plan.dofs)
    for history in histories:
        source = history.weights
        if (
            not isinstance(source, np.ndarray)
            or source.shape != shape
            or source.dtype != np.dtype("float64")
            or source.flags.writeable
        ):
            raise ValueError("Require aligned read-only float64 source weights")
    bytes_per_lag = sum(h.weights[0].nbytes for h in histories)
    count = min(int(max_lags), plan.steps + 1, int(capacity_bytes) // bytes_per_lag)
    adapted, layouts = [], []
    for history in histories:
        source = history.weights
        if count:
            prefix = np.array(source[:count], copy=True, order="K", subok=False)
            prefix.flags.writeable = False
            adapted.append(BoundaryHistory(history.plan, _PrefixWeights(source, prefix)))
            prefix_strides = list(prefix.strides)
        else:
            adapted.append(history)
            prefix_strides = None
        layouts.append(dict(source_strides=list(source.strides), prefix_strides=prefix_strides))
    return tuple(adapted), dict(
        policy="exact_readonly_short_lag_prefix",
        capacity_bytes=int(capacity_bytes),
        cached_lags=count,
        cached_bytes=count * bytes_per_lag,
        families=len(histories),
        layouts=layouts,
        additional_to_fft_cache_and_workspace=True,
        all_history_terms_retained=True,
        time_grid_unchanged=True,
        coefficient_approximation=False,
    )

"""Compare forecasts with coefficient requests made by the actual history code."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import BoundaryHistory, CQPlan

TOOLS = Path(__file__).resolve().parents[1] / "tools"
COST = runpy.run_path(str(TOOLS / "history_access_cost.py"))
HISTORY = runpy.run_path(str(TOOLS / "compact_radau_history.py"))
PREFIX = runpy.run_path(str(TOOLS / "history_prefix_cache.py"))


class TracedWeights(np.ndarray):
    def __new__(cls, array):
        obj = np.asarray(array).view(cls)
        obj.requests = []
        return obj

    def __array_finalize__(self, old):
        if old is not None:
            self.requests = getattr(old, "requests", [])

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.requests.append(int(getattr(value, "nbytes", self.dtype.itemsize)))
        return value


def test_forecast_matches_actual_restored_history_accesses_and_unchanged_values():
    steps, d, first, last = 4096, 5, 386, 1696
    plan = CQPlan(2e-6, steps, d, 16384, 0.999, 0, (steps + 1) * d * d * 8)
    rng = np.random.default_rng(35014)
    decay = np.exp(-np.arange(steps + 1)[:, None, None] / 20) * 1e-5
    weights = [rng.normal(size=(steps + 1, d, d)) * decay for _ in range(2)]
    sources = rng.normal(size=(last + 1, d, 2))
    reference = None
    configurations = [(32, 8, 4), (512, 8, 4), (32, 2048, 4)]
    configurations += [(512, 2048, 1), (1024, 8, 1), (256, 2048, 2)]
    for prefix, fine_cache_mb, rows in configurations:
        traced = [TracedWeights(np.array(w, order="F")) for w in weights]
        for array in traced:
            array.flags.writeable = False
        cache_mb = fine_cache_mb * (d / 531) ** 2
        histories, _ = PREFIX["cache_history_prefixes"](
            [BoundaryHistory(plan, w) for w in traced],
            capacity_bytes=prefix * 2 * d * d * 8,
            max_lags=prefix,
        )
        streams = [
            HISTORY["BatchedOnlineConvolution"](h, 2, cache_mb=cache_mb / 2, fft_rows=rows)
            for h in histories
        ]
        observed = []
        for index in range(last + 1):
            if index == first:
                for array in traced:
                    array.requests.clear()
                for stream in streams:
                    stream.transforms.clear()
                    stream.kernel_fft_builds = 0
            row = []
            for stream in streams:
                row.append(stream.past(index))
                stream.append(sources[index])
            if index >= first:
                observed.append(row)
        actual = np.asarray(observed)
        if reference is None:
            reference = actual.copy()
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)
        expected = COST["forecast_history_accesses"](
            plan, first, last, prefix_lags=prefix, fft_cache_mb=cache_mb, fft_rows=rows
        )
        assert (
            sum(sum(w.requests) for w in traced)
            == expected["logical_source_coefficient_bytes_requested"]
        )
        assert sum(s.kernel_fft_builds for s in streams) == expected["kernel_fft_builds"]
        assert sorted(streams[0].transforms) == expected["retained_levels_built"]
        assert (
            sum(sum(v.nbytes for v in s.transforms.values()) for s in streams)
            == expected["retained_fft_bytes_at_end"]
        )


def test_final_local_append_has_no_future_target():
    plan = CQPlan(1, 64, 3, 256, 0.9, 0, 65 * 3 * 3 * 8)
    result = COST["forecast_history_accesses"](plan, 64, 64)
    assert result["logical_source_coefficient_bytes_requested"] == 0
    assert result["kernel_fft_builds"] == 0


@pytest.mark.parametrize("interval", [(-1, 5), (5, 4), (0, 65)])
def test_reject_outside_original_local_horizon(interval):
    plan = CQPlan(1, 64, 3, 256, 0.9, 0, 65 * 3 * 3 * 8)
    with pytest.raises(ValueError):
        COST["forecast_history_accesses"](plan, *interval)

"""Short-lag storage copies preserve complete primal and transpose histories."""

import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import BoundaryHistory, plan_cq
from finiteblock_bem.history_cache import cache_history_prefixes

TOOLS = Path(__file__).resolve().parents[1] / "tools"
COMPACT = runpy.run_path(str(TOOLS / "compact_radau_history.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))


def saved_histories(tmp_path, order, *, steps=129):
    plan = plan_cq(3, 0.01, steps, max_memory_mb=64)
    rng = np.random.default_rng(893)
    families, paths = [], []
    for index in range(2):
        weights = rng.normal(size=(steps + 1, plan.dofs, plan.dofs))
        weights /= 1 + np.arange(steps + 1)[:, None, None]
        path = tmp_path / f"weights{index}.npy"
        np.save(path, np.array(weights, order=order), allow_pickle=False)
        families.append(BoundaryHistory(plan, np.load(path, mmap_mode="r", allow_pickle=False)))
        paths.append(path)
    return families, paths, rng


@pytest.mark.parametrize("order", ["C", "F"])
def test_capacity_readonly_slices_transpose_and_complete_disk_source(tmp_path, order):
    families, paths, _ = saved_histories(tmp_path, order)
    digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    per_lag = sum(h.weights[0].nbytes for h in families)
    for capacity in [0, per_lag - 1, per_lag, 8 * per_lag + 1, 200 * per_lag]:
        cached, record = cache_history_prefixes(families, capacity_bytes=capacity, max_lags=32)
        assert record["cached_bytes"] == record["cached_lags"] * per_lag <= capacity
        assert record["cached_lags"] == min(32, capacity // per_lag)
        for original, adapted in zip(families, cached, strict=True):
            for key in [
                0,
                -1,
                5,
                slice(0, 8),
                slice(7, 35),
                slice(None, None, -1),
                (slice(0, 4), 1),
            ]:
                np.testing.assert_array_equal(adapted.weights[key], original.weights[key])
            if not record["cached_lags"]:
                assert adapted is original
                continue
            count = record["cached_lags"]
            assert not np.shares_memory(adapted.weights[:count], original.weights)
            assert np.shares_memory(adapted.weights[count:], original.weights)
            assert not adapted.weights[:count].flags.writeable
            transposed = adapted.weights.transpose(0, 2, 1)
            np.testing.assert_array_equal(
                transposed[:count], original.weights[:count].transpose(0, 2, 1)
            )
            assert np.shares_memory(transposed[:count], adapted.weights[:count])
            with pytest.raises(ValueError, match="read-only"):
                adapted.weights[0][0, 0] = 1
            with pytest.raises(TypeError, match="full materialization"):
                np.asarray(adapted.weights)
            with pytest.raises(ValueError, match="time"):
                adapted.weights.transpose(2, 1, 0)
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == digests


@pytest.mark.parametrize("order,fft_cache_mb", [("C", 0), ("F", 0), ("F", 0.02)])
def test_complete_original_compact_history_and_transpose_with_prefix_cache(
    tmp_path, order, fft_cache_mb
):
    families, _, rng = saved_histories(tmp_path, order)
    plan = families[0].plan
    capacity = 32 * sum(h.weights[0].nbytes for h in families)
    cached, record = cache_history_prefixes(families, capacity_bytes=capacity)
    wrap = rng.normal(size=(plan.dofs, plan.dofs)) * 0.01
    dense = RADAU["expand_compact_weights"](*[h.weights for h in families], wrap)
    x, cotangent = [rng.normal(size=(plan.steps + 1, 2 * plan.dofs)) for _ in range(2)]
    outputs = []
    for transpose, inputs in [(False, x), (True, cotangent[::-1])]:
        streams = [
            COMPACT["CompactRadauConvolution"](
                *histories, wrap, cache_mb=fft_cache_mb, fft_rows=1, transpose=transpose
            )
            for histories in [families, cached]
        ]
        kernels = dense.transpose(0, 2, 1) if transpose else dense
        result = np.empty_like(inputs)
        for step, value in enumerate(inputs):
            original, adapted = [s.instantaneous @ value + s.past(step) for s in streams]
            np.testing.assert_allclose(adapted, original, rtol=2e-12, atol=2e-12)
            expected = np.einsum("kij,kj->i", kernels[: step + 1], inputs[step::-1])
            np.testing.assert_allclose(adapted, expected, rtol=2e-12, atol=2e-12)
            result[step] = adapted
            for stream in streams:
                stream.append(value)
        for stream in streams:
            assert stream.statistics()["peak_retained_kernel_fft_bytes"] <= fft_cache_mb * 1024**2
            assert all(s.next_step == plan.steps + 1 for s in stream.streams)
        outputs.append(result)
    np.testing.assert_allclose(
        np.sum(outputs[0] * cotangent), np.sum(x * outputs[1][::-1]), rtol=2e-12, atol=2e-12
    )
    assert record["cached_bytes"] == capacity
    assert record["all_history_terms_retained"]


def test_reject_ambiguous_sources_and_budgets(tmp_path):
    families, _, _ = saved_histories(tmp_path, "F")
    for capacity in [-1, 1.0, True]:
        with pytest.raises(ValueError, match="capacity_bytes"):
            cache_history_prefixes(families, capacity_bytes=capacity)
    with pytest.raises(ValueError, match="max_lags"):
        cache_history_prefixes(families, capacity_bytes=1000, max_lags=0)
    with pytest.raises(ValueError, match="at least one"):
        cache_history_prefixes([], capacity_bytes=0)
    writable = BoundaryHistory(families[0].plan, np.array(families[0].weights))
    with pytest.raises(ValueError, match="read-only"):
        cache_history_prefixes([writable], capacity_bytes=1000)
    wrong = BoundaryHistory(plan_cq(3, 0.02, 129, max_memory_mb=64), families[0].weights)
    with pytest.raises(ValueError, match="same mathematical"):
        cache_history_prefixes([families[0], wrong], capacity_bytes=1000)

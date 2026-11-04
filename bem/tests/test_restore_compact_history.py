"""Resume across dyadic/FFT boundaries without losing or double-counting lags."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import BoundaryHistory, plan_cq

RESTORE = runpy.run_path(str(Path(__file__).parents[1] / "tools/restore_compact_history.py"))
HISTORY = RESTORE["HISTORY"]


def model():
    rng = np.random.default_rng(1103)
    plan = plan_cq(3, 2e-6, 137)
    families = [
        BoundaryHistory(plan, np.asfortranarray(rng.normal(size=(138, 3, 3)))) for _ in range(2)
    ]
    return families, rng.normal(size=(3, 3)), rng.normal(size=(138, 6))


@pytest.mark.parametrize("cache_mb", [0, 0.25])
@pytest.mark.parametrize("transpose", [False, True])
def test_compact_complete_future_and_delayed_family(cache_mb, transpose):
    families, wrap, sources = model()
    kwargs = dict(cache_mb=cache_mb, fft_rows=2, transpose=transpose)
    for prefix in [0, 1, 2, 3, 7, 16, 31, 32, 33, 63, 64, 65, 66, 129, 138]:
        original, resumed = [
            HISTORY["CompactRadauConvolution"](*families, wrap, **kwargs) for _ in range(2)
        ]
        for value in sources[:prefix]:
            original.append(value)
        record = RESTORE["restore_compact_prefix"](resumed, sources[:prefix])
        assert record["accepted_inputs"] == prefix
        assert resumed.next_step == original.next_step == prefix
        np.testing.assert_array_equal(resumed.previous_f1, original.previous_f1)
        for a, b in zip(original.streams, resumed.streams, strict=True):
            np.testing.assert_array_equal(a.inputs, b.inputs)
            np.testing.assert_array_equal(a.pending[prefix:], b.pending[prefix:])
        for step in range(prefix, len(sources)):
            np.testing.assert_array_equal(resumed.past(step), original.past(step))
            original.append(sources[step])
            resumed.append(sources[step])
        np.testing.assert_array_equal(resumed.previous_f1, original.previous_f1)


@pytest.mark.parametrize("start", [None, 0, 5, 33, 64, 137])
def test_rebase_retains_original_inactive_diagnostic_and_future(start):
    families, wrap, sources = model()
    active = np.zeros(len(sources), dtype=bool)
    if start is not None:
        active[start] = True
    inactive = len(sources) if start is None else start
    sources[:inactive] *= 1e-10
    for prefix in [0, 1, 7, 32, 63, 64, 65, 129, 138]:
        original, resumed = [
            HISTORY["CompactCorrectionHistory"](*families, wrap, cache_mb=0.25, fft_rows=2)
            for _ in range(2)
        ]
        for k in range(prefix):
            original.append(k, sources[k], active=active[k])
        RESTORE["restore_correction_prefix"](resumed, sources[:prefix], active[:prefix])
        assert resumed.next_index == original.next_index == prefix
        assert resumed.start == original.start
        assert resumed.maximum_inactive_source_force == original.maximum_inactive_source_force
        for k in range(prefix, len(sources)):
            np.testing.assert_array_equal(original.past(k), resumed.past(k))
            original.append(k, sources[k], active=active[k])
            resumed.append(k, sources[k], active=active[k])


def test_batched_rejects_reuse_bad_shape_dtype_and_nonfinite():
    families, _, _ = model()

    def make():
        return HISTORY["BatchedOnlineConvolution"](families[0], 2)

    good = np.ones((3, 3, 2))
    for bad in [good.astype(np.float32), good[:, :2], np.full_like(good, np.nan)]:
        with pytest.raises(ValueError):
            RESTORE["restore_batched_prefix"](make(), bad)
    stream = make()
    RESTORE["restore_batched_prefix"](stream, good)
    with pytest.raises(ValueError, match="fresh"):
        RESTORE["restore_batched_prefix"](stream, good)


def test_correction_rejects_nonzero_discarded_source():
    families, wrap, sources = model()
    memory = HISTORY["CompactCorrectionHistory"](*families, wrap)
    with pytest.raises(AssertionError):
        RESTORE["restore_correction_prefix"](memory, sources[:3], np.zeros(3, dtype=bool))

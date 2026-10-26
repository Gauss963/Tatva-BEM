import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import BoundaryHistory, plan_cq

TOOLS = Path(__file__).resolve().parents[1] / "tools"
TOOL = runpy.run_path(str(TOOLS / "batched_history.py"))


@pytest.mark.parametrize("cache_mb", [0, 0.05])
@pytest.mark.parametrize("transpose", [False, True])
def test_all_causal_pairs_match_direct_nonsymmetric_history(cache_mb, transpose):
    rng = np.random.default_rng(734)
    n, d, rhs = 70, 4, 5
    weights = rng.normal(size=(n + 1, d, d))
    if transpose:
        weights = weights.transpose(0, 2, 1)
    weights[0] = 17.0  # Lag zero must remain implicit even in FFT updates.
    history = BoundaryHistory(plan_cq(d, 0.01, n, max_memory_mb=16), weights)
    data = rng.normal(size=(n + 1, d, rhs))
    actual = TOOL["BatchedBoundedConvolution"](
        history, right_hand_sides=rhs, cache_mb=cache_mb, fft_rows=2, direct_block=2
    )
    for i in range(n + 1):
        expected = sum((weights[i - j] @ data[j] for j in range(i)), start=np.zeros((d, rhs)))
        np.testing.assert_allclose(actual.past(i), expected, rtol=1e-12, atol=1e-12)
        actual.append(data[i])
    assert actual.statistics()["all_history_terms_retained"]
    assert actual.cache_peak_bytes <= actual.cache_limit_bytes
    assert actual.streamed_updates > 0
    with pytest.raises(ValueError, match="already complete"):
        actual.append(data[-1])


def test_batched_kernel_fft_is_built_once_for_all_five_directions():
    rng = np.random.default_rng(314)
    n, d = 70, 4
    weights = rng.normal(size=(n + 1, d, d))
    history = BoundaryHistory(plan_cq(d, 0.01, n, max_memory_mb=16), weights)
    data = rng.normal(size=(n + 1, d, 5))
    batched = TOOL["BatchedBoundedConvolution"](
        history, right_hand_sides=5, cache_mb=0, fft_rows=2, direct_block=2
    )
    scalar_cls = runpy.run_path(str(TOOLS / "bounded_history.py"))["BoundedOnlineConvolution"]
    scalars = [scalar_cls(history, cache_mb=0, fft_rows=2, direct_block=2) for _ in range(5)]
    for i in range(n + 1):
        expected = np.column_stack([s.past(i) for s in scalars])
        np.testing.assert_allclose(batched.past(i), expected, rtol=1e-12, atol=1e-12)
        batched.append(data[i])
        for j, scalar in enumerate(scalars):
            scalar.append(data[i, :, j])
    assert 5 * batched.kernel_fft_builds == sum(s.kernel_fft_builds for s in scalars)

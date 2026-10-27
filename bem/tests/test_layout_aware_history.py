import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import BoundaryHistory, plan_cq

TOOLS = Path(__file__).resolve().parents[1] / "tools"
TOOL = runpy.run_path(str(TOOLS / "layout_aware_history.py"))


@pytest.mark.parametrize("cache_mb", [0, 0.02])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("order", ["C", "F"])
def test_layout_streaming_retains_every_nonsymmetric_causal_pair(cache_mb, transposed, order):
    rng = np.random.default_rng(319)
    n, d = 130, 7
    weights = np.array(rng.normal(size=(n + 1, d, d)), order=order)
    if transposed:
        weights = weights.transpose(0, 2, 1)
    history = BoundaryHistory(plan_cq(d, 0.01, n, max_memory_mb=32), weights)
    memory = TOOL["LayoutAwareBoundedConvolution"](
        history, cache_mb=cache_mb, fft_rows=3, direct_block=2
    )
    inputs = rng.normal(size=(n + 1, d))
    for i in range(n + 1):
        expected = sum((weights[i - j] @ inputs[j] for j in range(i)), start=np.zeros(d))
        np.testing.assert_allclose(memory.past(i), expected, rtol=1e-12, atol=1e-12)
        memory.append(inputs[i])
    assert (memory.column_streamed_updates > 0) == (transposed != (order == "F"))
    assert memory.cache_peak_bytes <= memory.cache_limit_bytes
    assert memory.statistics()["all_history_terms_retained"]
    with pytest.raises(ValueError, match="already complete"):
        memory.append(inputs[-1])


@pytest.mark.parametrize("order", ["C", "F"])
def test_complete_forward_reverse_history_pairing_with_transposed_disk_layout(tmp_path, order):
    rng = np.random.default_rng(602)
    n, d = 95, 9
    path = tmp_path / "weights.npy"
    np.save(path, np.array(rng.normal(size=(n + 1, d, d)), order=order))
    weights = np.load(path, mmap_mode="r", allow_pickle=False)
    inputs, bars = rng.normal(size=(2, n + 1, d))
    plan = plan_cq(d, 0.01, n, max_memory_mb=32)
    forward = TOOL["LayoutAwareBoundedConvolution"](
        BoundaryHistory(plan, weights), cache_mb=0, fft_rows=4, direct_block=2
    )
    reverse = TOOL["LayoutAwareBoundedConvolution"](
        BoundaryHistory(plan, weights.transpose(0, 2, 1)), cache_mb=0, fft_rows=4, direct_block=2
    )
    action, transpose = np.zeros_like(inputs), np.zeros_like(inputs)
    for i in range(n + 1):
        action[i] = forward.past(i)
        forward.append(inputs[i])
        transpose[n - i] = reverse.past(i)
        reverse.append(bars[n - i])
    np.testing.assert_allclose(
        np.sum(bars * action), np.sum(inputs * transpose), rtol=1e-12, atol=1e-12
    )
    assert (reverse.column_streamed_updates > 0) == (order == "C")
    assert forward.prefer_input_columns == (order == "F")

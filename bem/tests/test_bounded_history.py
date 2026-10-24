import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import BoundaryHistory, OnlineConvolution, plan_cq

TOOLS = Path(__file__).resolve().parents[1] / "tools"
TOOL = runpy.run_path(str(TOOLS / "bounded_history.py"))


@pytest.mark.parametrize("steps,rows,capacity", [(257, 1, 0), (300, 4, 0.05), (513, 4, 4)])
@pytest.mark.parametrize("transposed", [False, True])
def test_bounded_disk_history_matches_direct_and_original(
    tmp_path, steps, rows, capacity, transposed
):
    plan = plan_cq(5, 0.01, steps, max_memory_mb=64)
    rng = np.random.default_rng(24)
    weights = rng.normal(size=(steps + 1, 5, 5)) / (1 + np.arange(steps + 1))[:, None, None]
    path = tmp_path / "weights.npy"
    np.save(path, weights, allow_pickle=False)
    saved = np.load(path, mmap_mode="r", allow_pickle=False)
    if transposed:
        saved, weights = saved.transpose(0, 2, 1), weights.transpose(0, 2, 1)
    history = BoundaryHistory(plan, saved)
    expected = OnlineConvolution(history)
    actual = TOOL["BoundedOnlineConvolution"](history, cache_mb=capacity, fft_rows=rows)
    inputs = rng.normal(size=(steps + 1, 5))
    for index in range(steps + 1):
        direct = history.past_traction(inputs, index)
        np.testing.assert_allclose(actual.past(index), direct, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(actual.past(index), expected.past(index))
        expected.append(inputs[index])
        actual.append(inputs[index])
    assert actual.statistics()["peak_retained_kernel_fft_bytes"] <= capacity * 1024**2
    if capacity < 1:
        assert actual.streamed_updates > 0
    assert saved.flags.writeable is False
    np.testing.assert_array_equal(saved, weights)
    with pytest.raises(ValueError, match="complete"):
        actual.append(inputs[0])


def test_bounded_correction_keeps_prefix_and_horizon_guards():
    original = runpy.run_path(str(TOOLS / "replay_resolvent.py"))["CorrectionHistory"]
    bounded = TOOL["bounded_correction_class"](original, cache_mb=0, fft_rows=1)
    plan = plan_cq(2, 0.01, 32, max_memory_mb=32)
    history = BoundaryHistory(plan, np.ones((33, 2, 2)))
    zero = np.zeros(2)
    first, second = original(history), bounded(history)
    for index in range(5):
        np.testing.assert_array_equal(first.past(index), second.past(index))
        first.append(index, zero, active=False)
        second.append(index, zero, active=False)
    for index in range(5, 38):
        np.testing.assert_array_equal(first.past(index), second.past(index))
        first.append(index, np.ones(2), active=True)
        second.append(index, np.ones(2), active=True)
    assert first.start == second.start == 5
    with pytest.raises(ValueError, match="too early"):
        second.past(38)
    with pytest.raises(AssertionError):
        bounded(history).append(0, np.ones(2), active=False)


def test_cache_preflight_rejects_invalid_budgets_and_does_not_count_disk_weights():
    plan = plan_cq(5, 0.01, 300, max_memory_mb=64)
    for capacity in [-1, np.nan, np.inf]:
        with pytest.raises(ValueError, match="capacity"):
            TOOL["cache_layout"](plan, capacity)
    for rows in [0, -1, None]:
        with pytest.raises(ValueError):
            TOOL["BoundedOnlineConvolution"](
                BoundaryHistory(plan, np.zeros((301, 5, 5))), cache_mb=0, fft_rows=rows
            )
    levels, size = TOOL["cache_layout"](plan, 0.05)
    assert size == sum(levels.values()) and size <= 0.05 * 1024**2
    assert TOOL["estimate_online_arrays"](plan, 300, 0, fft_rows=1) > 0

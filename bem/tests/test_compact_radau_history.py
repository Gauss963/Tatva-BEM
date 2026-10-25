"""Streamed/cached compact Radau histories preserve all fields and transposes."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import BoundaryHistory, plan_cq

TOOLS = Path(__file__).resolve().parents[1] / "tools"
TOOL = runpy.run_path(str(TOOLS / "compact_radau_history.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))


@pytest.mark.parametrize("capacity,rows", [(0, 1), (0.02, 2), (1, 4)])
def test_two_family_disk_history_and_reverse_dot_identity(tmp_path, capacity, rows):
    plan = plan_cq(3, 0.01, 257, max_memory_mb=64)
    rng = np.random.default_rng(572)
    weights = rng.normal(size=(2, plan.steps + 1, 3, 3))
    weights /= 1 + np.arange(plan.steps + 1)[None, :, None, None]
    wrap = rng.normal(size=(3, 3)) * 0.01
    dense = RADAU["expand_compact_weights"](*weights, wrap)
    families = []
    for i in range(2):
        path = tmp_path / f"weights{i}.npy"
        np.save(path, weights[i], allow_pickle=False)
        families.append(BoundaryHistory(plan, np.load(path, mmap_mode="r", allow_pickle=False)))
    x, cotangent = [rng.normal(size=(plan.steps + 1, 6)) for _ in range(2)]
    outputs = []
    for transposed, inputs in [(False, x), (True, cotangent[::-1])]:
        online = TOOL["CompactRadauConvolution"](
            *families, wrap, cache_mb=capacity, fft_rows=rows, transpose=transposed
        )
        kernels = dense.transpose(0, 2, 1) if transposed else dense
        y = np.empty_like(inputs)
        np.testing.assert_allclose(online.instantaneous, kernels[0], rtol=1e-13, atol=1e-13)
        for k in range(plan.steps + 1):
            past = online.past(k)
            # Repeated trial reads cannot advance or mutate physical history.
            np.testing.assert_array_equal(past, online.past(k))
            y[k] = online.instantaneous @ inputs[k] + past
            expected = np.einsum("kij,kj->i", kernels[: k + 1], inputs[k::-1])
            np.testing.assert_allclose(y[k], expected, rtol=2e-12, atol=2e-12)
            online.append(inputs[k])
        statistics = online.statistics()
        assert statistics["peak_retained_kernel_fft_bytes"] <= capacity * 1024**2
        assert not statistics["expanded_stage_time_weights_stored"]
        if capacity == 0:
            assert all(s["streamed_updates"] > 0 for s in statistics["families"])
        outputs.append(y)
        with pytest.raises(ValueError, match="complete"):
            online.append(inputs[-1])
    np.testing.assert_allclose(
        np.sum(outputs[0] * cotangent), np.sum(x * outputs[1][::-1]), rtol=2e-12
    )
    for i, history in enumerate(families):
        assert not history.weights.flags.writeable
        np.testing.assert_array_equal(history.weights, weights[i])


def test_stage_channels_share_streamed_kernel_ffts():
    plan = plan_cq(3, 0.01, 129, max_memory_mb=64)
    weights = np.random.default_rng(5).normal(size=(plan.steps + 1, 3, 3))
    results = []
    for channels in [1, 2, 5]:
        stream = TOOL["BatchedOnlineConvolution"](
            BoundaryHistory(plan, weights), channels, cache_mb=0, fft_rows=1
        )
        for _ in range(plan.steps + 1):
            stream.append(np.ones((plan.dofs, channels)))
        results.append(stream.statistics()["kernel_fft_builds"])
    assert len(set(results)) == 1
    assert results[0] > 0


def test_invalid_trial_does_not_advance_compact_history():
    plan = plan_cq(2, 0.1, 8, max_memory_mb=32)
    history = BoundaryHistory(plan, np.zeros((plan.steps + 1, 2, 2)))
    stream = TOOL["CompactRadauConvolution"](history, history, np.zeros((2, 2)))
    for value in [np.zeros(2), np.array([0, 0, 0, np.nan])]:
        with pytest.raises(ValueError, match="Invalid accepted"):
            stream.append(value)
        assert stream.next_step == 0
    with pytest.raises(ValueError, match="in order"):
        stream.past(1)

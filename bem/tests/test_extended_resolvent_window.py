import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import plan_cq

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/extend_resolvent_window.py"))


def test_extended_weights_follow_independent_bdf2_matrix_recurrence(tmp_path):
    rng = np.random.default_rng(95)
    raw = rng.normal(size=(5, 5))
    stiffness = raw.T @ raw + np.eye(5)
    mass = np.diag(np.linspace(0.5, 1.0, 5))
    plan = plan_cq(5, 0.02, 6, max_memory_mb=8)
    # Use an already available longer contour; changing the number of output
    # lags below does not change any frequency or damping radius.
    long = plan_cq(5, 0.02, 14, max_memory_mb=8)
    from dataclasses import replace

    plan = replace(plan, fft_size=long.fft_size, radius=long.radius)
    points = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    delta = (1.5 - 2 * points + 0.5 * points**2) / plan.dt
    physical = np.stack([np.linalg.inv(s * mass + stiffness) for s in delta])
    physical[[0, -1]] = physical[[0, -1]].real
    line = np.geomspace(0.2, 20.0, 5)
    spectrum = physical * np.sqrt(line)[None, :, None] * np.sqrt(line)[None, None, :]
    full, actual = TOOL["extend_weights"](
        spectrum, plan, 12, tmp_path / "weights.npy", line, fft_rows=2, max_workspace_mb=8
    )
    assert (full.dt, full.radius, full.fft_size) == (plan.dt, plan.radius, plan.fft_size)
    a0, a1, a2 = 1.5 / plan.dt * mass + stiffness, -2 / plan.dt * mass, 0.5 / plan.dt * mass
    expected = [np.linalg.inv(a0)]
    for index in range(1, full.steps + 1):
        rhs = a1 @ expected[-1]
        if index > 1:
            rhs += a2 @ expected[-2]
        expected.append(np.linalg.solve(a0, -rhs))
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-10)
    prefix = np.array(actual[: plan.steps + 1])
    audit = TOOL["compare_prefix"](prefix, actual, block_lags=2)
    assert audit["bitwise_equal"] and audit["maximum_absolute_difference"] == 0
    prefix[2, 0, 0] += 1e-5
    assert not TOOL["compare_prefix"](prefix, actual)["bitwise_equal"]
    actual._mmap.close()


@pytest.mark.parametrize("steps", [5, 6, 16, True])
def test_window_extension_cannot_silently_change_frequency_circle(steps):
    plan = plan_cq(5, 0.02, 6, max_memory_mb=8)
    with pytest.raises(ValueError, match="unchanged"):
        TOOL["extended_plan"](plan, steps)

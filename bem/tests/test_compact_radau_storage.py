"""Disk compact weights and startup equal the independent full RK convolution."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import invert_spectrum, plan_cq

TOOLS = Path(__file__).resolve().parents[1] / "tools"
STORAGE = runpy.run_path(str(TOOLS / "compact_radau_storage.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))


def samples(plan):
    mass = np.array([[2.0, 0.4], [0.4, 1.0]])
    stiffness = np.array([[0.7, -0.2], [-0.2, 0.5]])

    def transfer(s):
        return np.linalg.inv(s * mass + stiffness)

    nodes = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    compact = [RADAU["compact_matrix_function"](transfer, z, plan.dt) for z in nodes]
    dense = [RADAU["matrix_function"](transfer, RADAU["radau_symbol"](z) / plan.dt) for z in nodes]
    return np.array([v[0] for v in compact]), np.array([v[1] for v in compact]), np.array(dense)


def test_disk_ifft_and_full_startup_preserve_aliasing_and_causality(tmp_path):
    plan = plan_cq(2, 0.1, 5, max_memory_mb=32)
    # More startup than correction steps, with intentionally visible aliasing.
    plan = replace(plan, fft_size=64, radius=0.05 ** (1 / 64))
    f0, f1, dense = samples(plan)
    inputs = [v.copy() for v in [f0, f1]]
    paths, wrap, record = STORAGE["invert_families_to_files"](
        f0, f1, plan, tmp_path, fft_rows=1, max_workspace_mb=32
    )
    w0, w1 = [np.load(p, mmap_mode="r", allow_pickle=False) for p in paths]
    actual = RADAU["expand_compact_weights"](w0, w1, wrap)
    full_plan = replace(plan, steps=25, dofs=4)
    expected = invert_spectrum(dense, full_plan)
    np.testing.assert_allclose(actual, expected[: plan.steps + 1], rtol=2e-11, atol=2e-13)
    force = np.random.default_rng(98).normal(size=(full_plan.steps + 1, 2))
    force[:3] = 0
    force[-1] = [1e4, -2e4]  # A future load must not appear at early outputs.
    startup = STORAGE["causal_compact_startup"](
        f0[:, :, -1], f1[:, :, -1], wrap[:, -1], plan, force
    )
    source = np.zeros((len(force), 4))
    source[:, [1, 3]] = force
    for k in range(len(force)):
        target = np.einsum("kij,kj->i", expected[: k + 1], source[k::-1])
        np.testing.assert_allclose(startup[k].ravel(), target, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(startup[:3], 0, atol=2e-11)
    assert record["complete_requested_lags_retained"]
    assert not record["expanded_stage_time_weights_stored"]
    for value, original in zip([f0, f1], inputs, strict=True):
        np.testing.assert_array_equal(value, original)
    with pytest.raises(ValueError, match="unused"):
        STORAGE["invert_families_to_files"](f0, f1, plan, tmp_path)


def test_row_coefficients_match_the_full_matrix_function():
    values = np.array([[[2 + 1j, 1], [3 - 0.5j, 2]], [[1, 2], [4, 3]]])
    modes = np.array([[2 + 1j, 3 - 1j], [1, 4]])
    f0, f1 = STORAGE["compact_rows"](modes, values)
    for k in range(len(values)):
        for stage in range(2):
            np.testing.assert_allclose(
                f0[k] + modes[k, stage] * f1[k], values[k, stage], atol=1e-13
            )
    with pytest.raises(ValueError, match="coincident"):
        STORAGE["compact_rows"](np.ones((2, 2)), values)


def test_workspace_guard_precedes_weight_allocation(tmp_path):
    plan = plan_cq(2, 0.1, 5, max_memory_mb=32)
    f0, f1, _ = samples(plan)
    with pytest.raises(ValueError, match="array estimate"):
        STORAGE["invert_families_to_files"](f0, f1, plan, tmp_path, max_workspace_mb=1e-6)
    assert not list(tmp_path.iterdir())

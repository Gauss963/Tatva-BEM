"""Column streaming retains the complete original affine causal operator."""

import runpy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

TOOLS = Path(__file__).parents[1] / "tools"
STREAM = runpy.run_path(str(TOOLS / "stream_initial_residual.py"))
ORIGINAL = runpy.run_path(str(TOOLS / "radau_initialization.py"))


@pytest.mark.parametrize("order", ["C", "F"])
def test_streamed_projection_matches_original_full_history(tmp_path, order):
    rng = np.random.default_rng(11042)
    arrays = []
    for i in range(2):
        path = tmp_path / f"weights{i}.npy"
        np.save(path, np.array(rng.normal(size=(132, 7, 7)), order=order))
        arrays.append(np.load(path, mmap_mode="r"))
    force = rng.normal(size=7)
    for steps in [1, 2, 17, 131]:
        columns = []
        actual = STREAM["stream_initial_residual"](
            *arrays, force, steps, column_hook=columns.append, rows=3
        )
        expected = ORIGINAL["initial_residual_past"](*arrays, force, steps, rows=3)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
        assert columns == list(range(7))
        np.testing.assert_array_equal(actual[:2], 0)


def test_streamed_history_against_expanded_causal_operator():
    rng = np.random.default_rng(11043)
    arrays = [rng.normal(size=(18, 3, 3)) for _ in range(2)]
    force = rng.normal(size=3)
    expand = STREAM["RADAU"]["expand_compact_weights"]
    kernel = expand(*arrays, np.zeros((3, 3)))
    actual = STREAM["stream_initial_residual"](*arrays, force, 17)
    for step in range(1, 18):
        expected = sum((kernel[lag] @ np.tile(force, 2) for lag in range(1, step)), np.zeros(6))
        np.testing.assert_allclose(actual[step].ravel(), expected, rtol=2e-12, atol=2e-12)


def test_stream_rejects_invalid_force_and_horizon():
    arrays = [np.ones((4, 2, 2)) for _ in range(2)]
    for force in [np.ones(3), np.ones(2, dtype=np.float32), np.array([np.nan, 1])]:
        with pytest.raises(ValueError):
            STREAM["stream_initial_residual"](*arrays, force, 3)
    for steps in [0, 4, True]:
        with pytest.raises(ValueError):
            STREAM["stream_initial_residual"](*arrays, np.ones(2), steps)

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import bdf2_symbol, invert_spectrum, plan_cq

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/polynomial_spectrum.py"))


def test_polynomial_storage_preserves_nonsymmetric_cq_and_all_slices(tmp_path):
    plan = plan_cq(5, 0.02, 117, alias_tolerance=1e-12, max_memory_mb=32)
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    s = bdf2_symbol(zeta) / plan.dt
    a = np.random.default_rng(31).normal(size=(5, 5)) * 0.1 + 4 * np.eye(5)
    original = np.stack([np.linalg.inv(x * np.eye(5) + a) for x in s])
    original[[0, -1]] = original[[0, -1]].real
    compressed = TOOL["compress_spectrum"](original, tmp_path / "data", rtol=1e-13, atol=1e-18)
    difference = np.linalg.norm(compressed[:] - original, axis=(1, 2))
    tolerance = 1e-18 + 1e-13 * np.linalg.norm(original, axis=(1, 2))
    assert np.max(difference / tolerance) <= 1 + 1e-10
    np.testing.assert_allclose(compressed[:], original, rtol=1e-10, atol=1e-14)
    for key in [
        0,
        -1,
        (slice(5, 119), slice(1, 4), slice(None)),
        (slice(None), slice(None), -1),
        (slice(None), 2, 3),
        slice(10, 5),
    ]:
        np.testing.assert_allclose(compressed[key], original[key], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(
        invert_spectrum(compressed, plan),
        invert_spectrum(original.copy(), plan),
        rtol=1e-8,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="read-only"):
        compressed[1] = np.eye(5)


def test_every_grid_point_is_checked_and_nonsmooth_data_falls_back(tmp_path):
    original = np.random.default_rng(91).normal(size=(257, 3, 3)).astype(complex)
    original[80:160] = 0
    data = TOOL["compress_spectrum"](original, tmp_path / "data", degree=32, rtol=1e-13, atol=0)
    np.testing.assert_allclose(data[:], original, atol=1e-13)
    assert data.metadata["direct_grid_points"] > 0
    assert data.metadata["all_grid_points_checked"]
    assert data.metadata["direct_grid_points"] + data.metadata["polynomial_grid_points"] == len(
        original
    )
    path = tmp_path / "data/summary.json"
    metadata = json.loads(path.read_text())
    metadata["pieces"][0]["left"] = 1
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="cover"):
        TOOL["PolynomialSpectrum"](tmp_path / "data")


def test_storage_budget_does_not_silently_drop_data(tmp_path):
    original = np.ones((65, 3, 3), complex)
    with pytest.raises(ValueError, match="budget"):
        TOOL["compress_spectrum"](original, tmp_path / "data", max_storage_mb=0.0001)
    assert not (tmp_path / "data/summary.json").exists()


def test_nonfinite_source_cannot_be_recorded_as_verified(tmp_path):
    original = np.ones((257, 3, 3), complex)
    original[71, 0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        TOOL["compress_spectrum"](original, tmp_path / "data")
    assert not (tmp_path / "data/summary.json").exists()


def test_source_change_check_prevents_a_completed_manifest(tmp_path):
    def changed():
        raise ValueError("Source changed")

    with pytest.raises(ValueError, match="Source changed"):
        TOOL["compress_spectrum"](
            np.ones((65, 3, 3), complex), tmp_path / "data", source_check=changed
        )
    assert not (tmp_path / "data/summary.json").exists()


def test_unmeasurable_norm_falls_back_to_exact_data(tmp_path):
    original = np.ones((65, 3, 3), complex) * 1e200
    with np.errstate(over="ignore", invalid="ignore"):
        result = TOOL["compress_spectrum"](original, tmp_path / "data")
    np.testing.assert_array_equal(result[:], original)
    assert result.metadata["direct_grid_points"] == len(original)

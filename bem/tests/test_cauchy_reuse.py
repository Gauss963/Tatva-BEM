import runpy
from pathlib import Path

import numpy as np
import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/cauchy_reuse_experiment.py"))


def test_barycentric_values_and_derivatives_against_independent_dft_polynomial():
    rng = np.random.default_rng(14)
    n, radius = 64, 0.82
    circle = TOOL["CircleCauchy"](n, radius)
    coefficients = rng.normal(size=(n, 2, 3)) * 0.7 ** np.arange(n)[:, None, None]
    half = np.polynomial.polynomial.polyval(circle.sources[: n // 2 + 1] / radius, coefficients)
    half = np.moveaxis(half, -1, 0)
    targets = np.r_[circle.sources[[0, 7, n // 2]], 0, 0.5 - 0.25j, radius * 1.001]
    values, derivative = circle.evaluate(half, targets, derivative=True)
    expected = np.moveaxis(np.polynomial.polynomial.polyval(targets / radius, coefficients), -1, 0)
    deriv_coefficients = coefficients[1:] * np.arange(1, n)[:, None, None] / radius
    expected_derivative = np.moveaxis(
        np.polynomial.polynomial.polyval(targets / radius, deriv_coefficients), -1, 0
    )
    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-11, atol=1e-11)


def test_kernel_normalization_cancels_for_batched_cauchy_backend():
    circle = TOOL["CircleCauchy"](128, 0.9)
    half = (1 / (2 - circle.sources[:65]))[:, None]
    targets = np.array([0, 0.7 + 0.2j])

    def scaled(sources, densities, targets, derivative):
        value, grad = TOOL["_direct"](sources, densities, targets, derivative)
        return -value / (2 * np.pi), -grad / (2 * np.pi)

    value, grad = circle.evaluate(half, targets, derivative=True, backend=scaled)
    np.testing.assert_allclose(value[:, 0], 1 / (2 - targets), atol=1e-13)
    np.testing.assert_allclose(grad[:, 0], 1 / (2 - targets) ** 2, atol=1e-13)


def test_inverse_bdf2_map_preserves_frequency_and_rejects_bad_domains():
    source_dt, target_dt, n = 2e-6, 4e-6, 32768
    z = 1e-10 ** (1 / n) * np.exp(-2j * np.pi * np.arange(n // 2 + 1) / n)
    frequency = (1.5 - 2 * z + 0.5 * z**2) / target_dt
    mapped = TOOL["bdf2_preimage"](frequency, source_dt)
    np.testing.assert_allclose(
        (1.5 - 2 * mapped + 0.5 * mapped**2) / source_dt, frequency, rtol=1e-11, atol=2e-10
    )
    circle = TOOL["CircleCauchy"](2 * n, 1e-10 ** (1 / (2 * n)))
    circle.validate_targets(mapped)
    with pytest.raises(ValueError, match="right-half-plane"):
        TOOL["bdf2_preimage"](1j, source_dt)
    with pytest.raises(ValueError, match="unit disk"):
        circle.validate_targets(np.array([1.001]))
    with pytest.raises(ValueError, match="amplification"):
        circle.validate_targets(np.array([0.99999]))


def test_optional_fmm_matches_direct_cauchy_values_and_gradients():
    pytest.importorskip("fmm2dpy")
    circle = TOOL["CircleCauchy"](128, 0.95)
    half = np.stack((1 / (2 - circle.sources[:65]), 1 / (3 + circle.sources[:65])), axis=1)
    targets = np.array([0, 0.7 + 0.2j, 0.9 - 0.1j])
    actual = circle.evaluate(half, targets, derivative=True, backend=TOOL["fmm2d_backend"]())
    expected = circle.evaluate(half, targets, derivative=True)
    for candidate, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(candidate, reference, rtol=1e-11, atol=1e-12)

import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/cauchy_reuse_experiment.py"))


def test_spectrum_hash_uses_bounded_reads(tmp_path, monkeypatch):
    path = tmp_path / "spectrum.npy"
    data = b"finite-block spectrum" * 200000
    path.write_bytes(data)

    def denied(_):
        raise AssertionError("read_bytes would allocate the entire large spectrum")

    monkeypatch.setattr(Path, "read_bytes", denied)
    assert TOOL["stream_file_sha256"](path, chunk_bytes=8192) == hashlib.sha256(data).hexdigest()


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


def test_streamed_reference_reads_bounded_complete_matrices_and_preserves_samples():
    circle = TOOL["CircleCauchy"](128, 0.9)
    z = circle.sources[:65]
    half = np.stack((1 / (2 - z), 1 / (3 + z)), axis=1)[:, :, None]
    half[0] += 2j  # Same endpoint real projection as the original IRFFT path.
    original = half.copy()

    class BoundedMatrices:
        shape, dtype = half.shape, half.dtype

        def __getitem__(self, key):
            assert isinstance(key, (int, np.integer, slice))
            if isinstance(key, slice):
                assert len(range(*key.indices(65))) <= 7
            return half[key].copy()

    targets = np.r_[circle.sources[[0, 11, 64, 117]], 0, 0.7 + 0.1j, 0.90001]
    expected = circle.evaluate(half, targets)
    actual = circle.evaluate_streamed(BoundedMatrices(), targets, block_size=7)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(half, original)


@pytest.mark.parametrize("many_nodes", [False, True])
def test_derivative_remains_accurate_arbitrarily_close_to_sample_nodes(many_nodes):
    circle = TOOL["CircleCauchy"](512, 0.95)
    nodes = np.arange(64) if many_nodes else np.array([0, 7, 256])
    half = np.stack((1 / (2 - circle.sources[:257]), 1 / (3 + circle.sources[:257])), axis=1)
    targets = np.r_[circle.sources[nodes], circle.sources[nodes] + (1 + 1j) * 1e-11]
    values, derivatives = circle.evaluate(half, targets, derivative=True)
    expected = np.stack((1 / (2 - targets), 1 / (3 + targets)), axis=1)
    expected_derivatives = np.stack((1 / (2 - targets) ** 2, -1 / (3 + targets) ** 2), axis=1)
    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(derivatives, expected_derivatives, rtol=1e-11, atol=1e-11)


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

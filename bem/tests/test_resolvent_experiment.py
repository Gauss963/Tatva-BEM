"""Check the algebra and zero-source rebasing used by the resolvent prototype."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import BoundaryHistory, bdf2_symbol, invert_spectrum, plan_cq

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/replay_resolvent.py"))


def test_rebased_history_keeps_every_nonzero_source():
    rng = np.random.default_rng(91)
    plan = plan_cq(3, 0.01, 8, max_memory_mb=32)
    weights = rng.normal(size=(9, 3, 3))
    history = TOOL["CorrectionHistory"](BoundaryHistory(plan, weights))
    inputs = []
    for index in range(1, 65):
        actual = history.past(index)
        if index <= 57:
            np.testing.assert_array_equal(actual, np.zeros(3))
        else:
            expected = sum(weights[index - 57 - j] @ value for j, value in enumerate(inputs))
            np.testing.assert_allclose(actual, expected, atol=1e-13)
        value = np.zeros(3) if index < 57 else rng.normal(size=3)
        history.append(index, value, active=index >= 57)
        if index >= 57:
            inputs.append(value)
    assert history.start == 57
    with pytest.raises(ValueError, match="too early"):
        history.past(66)


def test_rebasing_refuses_to_discard_an_unexpected_nonzero_source():
    plan = plan_cq(2, 0.01, 8, max_memory_mb=32)
    history = TOOL["CorrectionHistory"](BoundaryHistory(plan, np.zeros((9, 2, 2))))
    with pytest.raises(AssertionError):
        history.append(1, np.ones(2), active=False)


def test_elastic_baseline_has_no_future_forcing_wraparound():
    plan = plan_cq(2, 0.04, 11, alias_tolerance=1e-4, max_memory_mb=32)
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    scalar = 1 / (bdf2_symbol(zeta) / plan.dt + 2)
    spectrum = np.column_stack((scalar, 2 * scalar))
    forcing = np.zeros(plan.steps + 1)
    forcing[8:] = np.arange(4) + 1
    actual = TOOL["causal_elastic_baseline"](spectrum, plan, forcing)
    np.testing.assert_allclose(actual[:8], 0, atol=1e-14)
    from scipy.fft import irfft

    weights = irfft(spectrum, n=plan.fft_size, axis=0)[: plan.steps + 1]
    weights /= plan.radius ** np.arange(plan.steps + 1)[:, None]
    expected = np.array(
        [sum(weights[j] * forcing[n - j] for j in range(n + 1)) for n in range(plan.steps + 1)]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize("version", [(1, 0), (2, 0)])
def test_noncommuting_matrix_inverse_has_cq_composition_property(tmp_path, version):
    plan = plan_cq(2, 0.04, 11, alias_tolerance=1e-12, max_memory_mb=32)
    mass = np.array([[1.0, 0.2], [0.2, 2]])
    stiffness = np.array([[4.0, 1.5], [1.5, 3]])
    contact = np.diag([9.0, 3])
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    s = bdf2_symbol(zeta) / plan.dt
    original = s[:, None, None] * mass + stiffness
    path = tmp_path / "source.npy"
    with path.open("wb") as stream:
        np.lib.format.write_array(stream, original, version=version, allow_pickle=False)
    forcing = np.arange(len(zeta)) * (1 + 2j)
    spectrum, response = TOOL["invert_elastic_spectrum"](
        path, plan, tmp_path / "inverse.npy", contact, forcing
    )
    expected = np.linalg.inv(original + contact)
    np.testing.assert_allclose(spectrum[:], expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(response, expected[:, :, -1] * forcing[:, None], rtol=1e-13)
    inverse = invert_spectrum(spectrum, plan)
    stat = spectrum.path.stat()
    reopened = TOOL["SavedDiskSpectrum"](spectrum.path, plan)
    np.testing.assert_array_equal(invert_spectrum(reopened, plan), inverse)
    assert spectrum.path.stat().st_mtime_ns == stat.st_mtime_ns
    with pytest.raises(ValueError, match="read-only"):
        reopened[1] = np.eye(2)
    coefficients = np.zeros_like(inverse)
    coefficients[0] = 1.5 * mass / plan.dt + stiffness + contact
    coefficients[1] = -2 * mass / plan.dt
    coefficients[2] = 0.5 * mass / plan.dt
    for index in range(plan.steps + 1):
        convolution = sum(inverse[j] @ coefficients[index - j] for j in range(index + 1))
        np.testing.assert_allclose(
            convolution, np.eye(2) if index == 0 else np.zeros((2, 2)), atol=2e-10
        )

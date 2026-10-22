import numpy as np
import pytest

from finiteblock_bem.cq import (
    BoundaryHistory,
    OnlineConvolution,
    ResourceBudgetError,
    bdf2_symbol,
    convolution_weights,
    plan_cq,
)


def test_bdf2_weights_against_exact_rational_recurrence():
    plan = plan_cq(2, 0.01, 80)
    weights = convolution_weights(lambda s: np.eye(2) / (s + 2), plan)
    exact = np.zeros(81)
    denominator = 1.5 + 2 * plan.dt
    exact[0] = plan.dt / denominator
    exact[1] = 2 * exact[0] / denominator
    for n in range(2, len(exact)):
        exact[n] = (2 * exact[n - 1] - 0.5 * exact[n - 2]) / denominator
    np.testing.assert_allclose(weights[:, 0, 0], exact, rtol=1e-7, atol=2e-11)
    np.testing.assert_allclose(weights[:, 0, 1], 0, atol=1e-14)


def test_derivative_transfer_retains_bdf2_sign_and_timing():
    plan = plan_cq(2, 0.02, 80)
    weights = convolution_weights(lambda s: np.eye(2) * s, plan)
    exact = np.zeros(81)
    exact[:3] = np.array([1.5, -2, 0.5]) / plan.dt
    np.testing.assert_allclose(weights[:, 0, 0], exact, atol=2e-9)


def test_past_traction_uses_every_previous_step():
    plan = plan_cq(2, 0.1, 4)
    weights = np.arange(1, 6)[:, None, None] * np.eye(2)
    history = BoundaryHistory(plan, weights)
    u = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
    np.testing.assert_allclose(history.past_traction(u, 3), [2 * 3 + 3 * 2 + 4 * 1, 0])
    with pytest.raises(ValueError):
        history.past_traction(u, 5)


def test_resource_guard_prevents_large_allocations():
    with pytest.raises(ResourceBudgetError, match="do not silently truncate history"):
        plan_cq(2790, 3e-7, 150000, max_memory_mb=512)
    with pytest.raises(ValueError):
        plan_cq(2, -0.1, 10)
    assert np.real(bdf2_symbol(0.8 + 0j)) > 0


@pytest.mark.parametrize("steps", [1, 31, 64, 137])
def test_online_fft_matches_every_dense_history_term(steps):
    rng = np.random.default_rng(83)
    plan = plan_cq(6, 0.01, steps)
    weights = rng.normal(size=(steps + 1, 6, 6))
    # Noncommuting matrix kernels and nonzero initial data detect indexing errors.
    reference = BoundaryHistory(plan, weights)
    online = OnlineConvolution(reference, direct_block=2)
    u = rng.normal(size=(steps + 1, 6))
    for step in range(steps + 1):
        np.testing.assert_allclose(online.past(step), reference.past_traction(u, step), atol=2e-13)
        online.append(u[step])
    with pytest.raises(ValueError, match="complete"):
        online.append(u[0])


def test_parallel_frequency_sampling_preserves_weights():
    plan = plan_cq(2, 0.03, 20)

    def transfer(s):
        return np.array([[1, 2], [-3, 4]]) / (s + 2)

    serial = convolution_weights(transfer, plan)
    parallel = convolution_weights(transfer, plan, workers=4)
    np.testing.assert_array_equal(serial, parallel)

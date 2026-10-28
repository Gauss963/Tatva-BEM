import runpy
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
FORWARD = runpy.run_path(str(TOOLS / "interior_radau.py"))["apply_weights"]
REVERSE = runpy.run_path(str(TOOLS / "interior_radau_adjoint.py"))["interior_pullback"]


def test_complete_rectangular_cq_transpose_matches_jax_gradient_and_all_stage_dot_product():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(22)
    weights = rng.normal(size=(9, 6, 4))
    stages, cotangent = rng.normal(size=(9, 4)), rng.normal(size=(9, 6))
    stages[0] = 0
    response = FORWARD(weights, stages)
    reverse = REVERSE(weights, cotangent)
    np.testing.assert_allclose(np.vdot(response, cotangent), np.vdot(stages, reverse), rtol=1e-14)
    np.testing.assert_array_equal(reverse[0], 0)
    target, time_weights = rng.normal(size=(9, 6)), rng.uniform(0.1, 1, size=(9, 1))
    target[0] = 0
    observed_gradient = REVERSE(weights, time_weights * (response - target))

    def objective(physical_stages):
        q = jnp.concatenate([jnp.zeros((1, 4)), physical_stages])
        values = [jnp.zeros(6)]
        for k in range(1, len(weights)):
            values.append(sum(jnp.asarray(weights[k - j]) @ q[j] for j in range(1, k + 1)))
        residual = jnp.stack(values) - jnp.asarray(target)
        return 0.5 * jnp.sum(jnp.asarray(time_weights) * residual**2)

    expected = jax.grad(objective)(jnp.asarray(stages[1:]))
    np.testing.assert_allclose(observed_gradient[1:], expected, rtol=1e-12, atol=1e-12)


def test_probe_cotangent_does_not_truncate_early_history_or_differentiate_artificial_time_zero():
    weights, cotangent = np.zeros((5, 2, 2)), np.zeros((5, 2))
    weights[3] = np.eye(2)
    cotangent[-1] = [2, 3]
    reverse = REVERSE(weights, cotangent)
    np.testing.assert_array_equal(reverse[1], [2, 3])
    np.testing.assert_array_equal(reverse[[0, 2, 3, 4]], 0)

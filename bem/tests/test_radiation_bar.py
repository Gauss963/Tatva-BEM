import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from finiteblock_bem.benchmarks import bar_displacement_exact
from finiteblock_bem.jax_backend import contact_point

BAR = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/radiation_bar_experiment.py"))


@pytest.mark.parametrize("force", [-3.0, -0.1, 0.1, 3.0])
def test_radiation_contact_step_matches_independent_scalar_equilibrium(force):
    theta = jnp.array([20.0, 12.0, 0.8, 0.4, 0.02])
    old_q, old_z, old_c = 0.03, 0.02, 0.005
    dt, radiation, pressure = 0.01, 0.4, 1.2
    q, response, residual = BAR["local_radiation_step"](
        old_q, old_z, old_c, force, dt, radiation, pressure, theta
    )

    def equation(x):
        tau = contact_point(jnp.array([pressure / theta[0], x]), old_z, old_c, theta)[1]
        return float(radiation / dt * (x - old_q) + tau - force)

    expected = brentq(equation, -1, 1, xtol=1e-14)
    assert float(q) == pytest.approx(expected, abs=1e-13)
    assert abs(float(residual)) < 1e-12
    assert response[3] >= old_c


def test_delay_recurrence_keeps_every_reflected_image():
    velocity = np.random.default_rng(4).normal(size=29)
    expected = np.zeros_like(velocity)
    for image in range(1, 8):
        expected[4 * image :] += 1.4 * velocity[: 29 - 4 * image]
    np.testing.assert_allclose(BAR["reflected_traction"](velocity, 4, 0.7), expected, atol=1e-15)


def test_reconstructed_wave_energy_before_first_reflection():
    # Outgoing waves have velocity v/2 in each bar and occupy length c*t.
    energy = BAR["reconstructed_bulk_energy"](
        np.full(8, 3.0), dt=0.01, round_trip=0.4, density=2.0, speed=5.0
    )
    assert energy == pytest.approx((2 * 5 / 2) * 0.08 * 3**2)


def test_zero_friction_delay_solution_converges_to_analytic_reflected_bar():
    errors = []
    for dt in [0.001, 0.0005]:
        fun = BAR["delay_rollout"](
            micro_dt=dt,
            macro_dt=dt,
            duration=1.2,
            round_trip=0.4,
            radiation=0.8,
            pressure=1.0,
            peak=1.0,
            rise=0.15,
        )
        data = np.asarray(fun(jnp.array([20.0, 12.0, 0.0, 0.0, 0.02])))
        time = (np.arange(len(data)) + 1) * dt
        exact = bar_displacement_exact(
            time, 0.0, length=0.2, speed=1.0, impedance=0.8, rise_time=0.15
        )
        errors.append(np.linalg.norm(data[:, 0] - exact) / np.linalg.norm(exact))
        np.testing.assert_allclose(
            data[:, 6], BAR["reflected_traction"](data[:, 4], round(0.4 / dt), 0.8), atol=5e-12
        )
        assert np.max(abs(data[:, -1])) < 1e-11
    assert errors[1] < 0.55 * errors[0]
    assert errors[1] < 0.01


def test_independent_volume_chain_converges_to_analytic_bar():
    errors = []
    dt = 0.0005
    for elements in [20, 40]:
        fun, _, _ = BAR["fem_rollout"](
            elements=elements,
            dt=dt,
            duration=1.2,
            length=0.2,
            shear_modulus=1.0,
            density=1.0,
            pressure=1.0,
            peak=1.0,
            rise=0.15,
        )
        data = np.asarray(fun(jnp.array([20.0, 12.0, 0.0, 0.0, 0.02]))[0])
        exact = bar_displacement_exact(
            (np.arange(len(data)) + 1) * dt,
            0.0,
            length=0.2,
            speed=1.0,
            impedance=0.5,
            rise_time=0.15,
        )
        errors.append(np.linalg.norm(data[:, 0] - exact) / np.linalg.norm(exact))
    assert errors[1] < 0.7 * errors[0]
    assert errors[1] < 0.02


def test_nonlinear_macro_history_ad_matches_forward_and_small_differences():
    fun = BAR["delay_rollout"](
        micro_dt=0.001,
        macro_dt=0.01,
        duration=1.2,
        round_trip=0.4,
        radiation=0.4,
        pressure=1.0,
        peak=1.1,
        rise=0.15,
    )
    theta = jnp.array([20.0, 12.0, 0.8, 0.4, 0.02])
    loss = jax.jit(lambda p: fun(p)[-1, 2])
    assert float(loss(theta)) > float(theta[-1])
    reverse = jax.grad(loss)(theta)
    np.testing.assert_allclose(reverse, jax.jacfwd(loss)(theta), rtol=1e-10, atol=1e-12)
    for i in range(1, 5):
        eps = 1e-6 * float(theta[i])
        direction = jnp.eye(5)[i] * eps
        fd = float((loss(theta + direction) - loss(theta - direction)) / (2 * eps))
        assert float(reverse[i]) == pytest.approx(fd, rel=1e-5, abs=1e-8)


@pytest.mark.parametrize("checkpoint_macro", [False, True])
def test_complete_fine_history_custom_vjp_retains_all_delayed_cotangents(checkpoint_macro):
    options = dict(
        micro_dt=0.001,
        duration=1.2,
        round_trip=0.4,
        radiation=0.4,
        pressure=1.0,
        peak=1.1,
        rise=0.15,
        checkpoint_macro=checkpoint_macro,
    )
    loss, raw = BAR["fine_delay_loss"](**options)
    theta = jnp.array([20.0, 12.0, 0.8, 0.4, 0.02])

    def reference(p):
        return raw(p)[-1, 2]

    np.testing.assert_allclose(
        jax.grad(loss)(theta), jax.jacfwd(reference)(theta), rtol=1e-10, atol=1e-12
    )
    # Small rings also check ordinary reverse with each checkpoint policy.
    np.testing.assert_allclose(
        jax.grad(loss)(theta), jax.grad(reference)(theta), rtol=1e-10, atol=1e-12
    )

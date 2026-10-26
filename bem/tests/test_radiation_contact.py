import runpy
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from finiteblock_bem.friction import ContactState  # noqa: E402
from finiteblock_bem.nodal import load_nodal_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LIMIT = runpy.run_path(str(ROOT / "tools/radiation_limit_experiment.py"))
CHAIN = runpy.run_path(str(ROOT / "tools/structured_radiation.py"))["RadiationChain"]
CONTACT = runpy.run_path(str(ROOT / "tools/radiation_contact.py"))["RadiationContactRoot"]
CASE = ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def setup_model(**options):
    model = load_nodal_model(CASE, **options)
    blocks = [
        LIMIT["body_limits"](m, model.reference.material)["mixed"]
        for m in (model.reference.moving, model.reference.stationary)
    ]
    reference = LIMIT["condense"](model, blocks)
    chain = CHAIN.from_dense(model, reference)
    shared = dict(platen_width=model.constraints.platen_width)
    fast = CONTACT(chain, model.constraints.contact_weights, **shared)
    dense = CONTACT(chain, model.constraints.contact_weights, dense_reference=reference, **shared)
    law = model.reference.contact
    theta = jnp.array(
        [
            law.normal_penalty,
            law.tangential_penalty,
            law.static_friction,
            law.kinetic_friction,
            law.characteristic_slip,
        ]
    )
    return model, reference, chain, fast, dense, theta


@pytest.fixture(scope="module")
def coarse():
    return setup_model(max_panel_length=100, contact_panel_length=60)


@pytest.fixture(scope="module")
def event():
    return setup_model(
        max_panel_length=20,
        contact_panel_length=5,
        contact_end_length=0.5,
        corner_panel_length=0.5,
        restart_knot_window=8,
        restart_knot_subdivisions=4,
    )


def problem(model, reference, theta, *, scale=1.0):
    """Independent manufactured nodal force, using NumPy production friction.

    Mixed sticking/sliding/opening roots have known displacements. This is not
    a physical event trajectory or a substitute for its retarded bulk forcing.
    """
    n = len(model.moving_contact)
    plastic = model.restart.plastic_slip.copy()
    cumulative = np.linspace(0.05, 0.9, n) * float(theta[4])
    base = np.c_[np.full(n, 2e-4), plastic]
    patterns = np.array([[1e-5, 1e-5], [2e-5, 0.003], [2e-5, -0.003], [-3e-4, 0.004]])
    target = np.tile(patterns, ((n + 3) // 4, 1))[:n] * scale
    q = np.r_[target[:, 0], target[:, 1], 0.002 * scale]
    previous = np.zeros_like(q)
    a, spring = 1 / 1.2492928633545953e-8, 25 * model.constraints.platen_width
    response = model.reference.contact.evaluate(
        base[:, 0] + q[:n], base[:, 1] + q[n : 2 * n], ContactState(plastic, cumulative)
    )
    forcing = -a * reference @ (q - previous)
    forcing[:n] -= model.constraints.contact_weights * response.pressure
    forcing[n : 2 * n] -= model.constraints.contact_weights * response.shear
    forcing[-1] -= spring * q[-1]
    return (
        tuple(jnp.asarray(v) for v in (plastic, cumulative, base, forcing, previous, q)),
        a,
        spring,
    )


def test_actual_mesh_nonlinear_roots_preserve_all_point_branches(event):
    model, reference, chain, fast, dense, theta = event
    assert chain.nodes == 137
    (z, c, base, forcing, previous, exact), a, spring = problem(model, reference, theta)
    answers = []
    for system in (fast, dense):
        q, info = system.solve(theta, z, c, base, forcing, previous, previous, a, spring)
        assert info[2] == 1 and info[1] < 1e-7
        np.testing.assert_allclose(q, exact, rtol=0, atol=2e-11)
        # Re-evaluate the original full dense matrix plus independent NumPy
        # friction. Checking only the represented sparse residual is insufficient.
        n = chain.nodes
        law = model.reference.contact.evaluate(
            np.asarray(base[:, 0] + q[:n]),
            np.asarray(base[:, 1] + q[n : 2 * n]),
            ContactState(z, c),
        )
        residual = a * reference @ np.asarray(q - previous) + np.asarray(forcing)
        residual[:n] += model.constraints.contact_weights * law.pressure
        residual[n : 2 * n] += model.constraints.contact_weights * law.shear
        residual[-1] += spring * float(q[-1])
        assert np.max(abs(residual / np.asarray(fast.residual_weights))) < 1e-7
        values = np.asarray(system.response(q, z, c, base, theta))
        np.testing.assert_allclose(values[:, 0], law.pressure, atol=1e-9)
        np.testing.assert_allclose(values[:, 1], law.shear, atol=1e-9)
        np.testing.assert_allclose(values[:, 2], law.state.plastic_slip, atol=1e-12)
        np.testing.assert_allclose(values[:, 3], law.state.cumulative_slip, atol=1e-12)
        assert values[0, 3] == c[0]  # Stick.
        assert values[1, 2] > z[1] and values[1, 3] > c[1]  # Positive sliding.
        assert values[2, 2] < z[2] and values[2, 3] > c[2]  # Negative sliding.
        assert values[3, 0] == 0 and values[3, 3] == c[3]  # Open.
        answers.append(np.asarray(q))
    np.testing.assert_allclose(*answers, rtol=0, atol=2e-11)


def test_nonlinear_root_ad_includes_parameters_previous_state_lift_and_history(coarse):
    model, reference, _, fast, dense, theta = coarse
    (z, c, base, forcing, previous, _), a, spring = problem(model, reference, theta)
    n = len(z)
    output_weights = jnp.linspace(0.4, 1.3, n)

    def loss(system, parameters, old_z, old_c, lift, history, old_q, scale, stiffness):
        q, _ = system.solve(
            parameters, old_z, old_c, lift, history, old_q, previous, scale, stiffness
        )
        response = system.response(q, old_z, old_c, lift, parameters)
        return jnp.dot(output_weights, response[:, 3]) + 0.01 * jnp.sum(q)

    gradients = []
    arguments = (theta, z, c, base, forcing, previous, jnp.array(a), jnp.array(spring))
    for system in (fast, dense):

        def fn(*args, system=system):
            return loss(system, *args)

        gradient = jax.grad(fn, argnums=tuple(range(8)))(*arguments)
        assert all(np.all(np.isfinite(g)) for g in gradient)
        gradients.append(gradient)
        # One independent simultaneous JVP checks all state/forcing inputs.
        rng = np.random.default_rng(81)
        directions = tuple(
            jnp.asarray(rng.normal(size=v.shape)) * jnp.maximum(abs(v), 1e-5) * 1e-3
            for v in arguments
        )
        tangent = jax.jvp(fn, arguments, directions)[1]
        reverse = sum(jnp.sum(g * d) for g, d in zip(gradient, directions, strict=True))
        np.testing.assert_allclose(tangent, reverse, rtol=2e-9, atol=1e-12)
    for left, right in zip(*gradients, strict=True):
        np.testing.assert_allclose(left, right, rtol=2e-8, atol=2e-10)
    for i, step in ((2, 1e-5), (4, 1e-6)):
        forward = loss(fast, theta.at[i].add(step), *arguments[1:])
        backward = loss(fast, theta.at[i].add(-step), *arguments[1:])
        np.testing.assert_allclose(
            gradients[0][0][i], (forward - backward) / (2 * step), rtol=2e-5, atol=2e-8
        )


def test_chained_updates_propagate_old_displacement_and_plastic_cotangents(coarse):
    model, reference, _, fast, dense, theta = coarse
    (z0, c0, base, _, previous, _), a, spring = problem(model, reference, theta)
    n = len(z0)
    # Construct three known targets from the independent NumPy friction update.
    # Forces remain frozen when differentiating theta; z/c and q are advanced.
    old_z, old_c, old_q = np.array(z0), np.array(c0), np.array(previous)
    loads = []
    for multiplier in (1.0, 1.15, 1.3):
        (_, _, _, _, _, target), _, _ = problem(model, reference, theta, scale=multiplier)
        target = np.asarray(target)
        response = model.reference.contact.evaluate(
            np.asarray(base[:, 0]) + target[:n],
            np.asarray(base[:, 1]) + target[n : 2 * n],
            ContactState(old_z, old_c),
        )
        force = -a * reference @ (target - old_q)
        force[:n] -= model.constraints.contact_weights * response.pressure
        force[n : 2 * n] -= model.constraints.contact_weights * response.shear
        force[-1] -= spring * target[-1]
        loads.append(force)
        old_q, old_z, old_c = target, response.state.plastic_slip, response.state.cumulative_slip
    loads = jnp.asarray(loads)

    def rollout(system, parameters):
        def step(carry, forcing):
            old_q, old_z, old_c = carry
            q, info = system.solve(parameters, old_z, old_c, base, forcing, old_q, old_q, a, spring)
            response = system.response(q, old_z, old_c, base, parameters)
            return (q, response[:, 2], response[:, 3]), info

        final, info = jax.lax.scan(step, (previous, z0, c0), loads)
        return jnp.mean(final[2]), info, final

    values, gradients = [], []
    for system in (fast, dense):
        value, info, final = rollout(system, theta)
        assert np.all(np.asarray(info[:, 2]) == 1)
        assert np.all(np.isfinite(final[0]))
        values.append(value)

        def fn(p, system=system):
            return rollout(system, p)[0]

        gradient = jax.grad(fn)(theta)
        forward = jnp.stack([jax.jvp(fn, (theta,), (direction,))[1] for direction in jnp.eye(5)])
        np.testing.assert_allclose(gradient, forward, rtol=2e-8, atol=1e-10)
        gradients.append(gradient)
    np.testing.assert_allclose(values[0], values[1], atol=2e-12)
    np.testing.assert_allclose(gradients[0], gradients[1], rtol=2e-7, atol=1e-9)


def test_failed_root_cannot_return_finite_looking_committed_state(coarse):
    model, reference, chain, _, _, theta = coarse
    (z, c, base, forcing, previous, _), a, spring = problem(model, reference, theta)
    system = CONTACT(
        chain,
        model.constraints.contact_weights,
        platen_width=model.constraints.platen_width,
        max_iterations=0,
    )
    q, info = system.solve(theta, z, c, base, forcing, previous, previous, a, spring)
    assert info[2] == 0 and info[1] > 1e-7
    assert np.all(np.isnan(q))
    assert np.all(np.isnan(system.response(q, z, c, base, theta)))


@pytest.mark.parametrize("argument", ["weights", "width", "tolerance", "iterations"])
def test_invalid_root_configuration_rejected(coarse, argument):
    model, _, chain, _, _, _ = coarse
    weights = model.constraints.contact_weights.copy()
    options = dict(platen_width=model.constraints.platen_width)
    if argument == "weights":
        weights[0] = 0
    elif argument == "width":
        options["platen_width"] = -1
    elif argument == "tolerance":
        options["tolerance"] = np.inf
    else:
        options["max_iterations"] = True
    with pytest.raises(ValueError):
        CONTACT(chain, weights, **options)

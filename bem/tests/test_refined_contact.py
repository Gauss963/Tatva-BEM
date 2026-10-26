import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.nodal import (
    NodalInterfaceEquilibrium,
    initialize_nodal_restart,
    load_nodal_model,
    nodal_body_stiffness,
)

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/refined_contact.py"))
SPACE = TOOL["ContactQuadrature"]
CASE = ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def test_positive_quadrature_preserves_virtual_work_and_exact_p1_moments():
    nodes = np.array([0.0, 0.1, 1.0, 2.4, 3.0])
    rng = np.random.default_rng(91)
    q = rng.normal(size=(2, len(nodes)))
    for count in (1, 2, 5):
        space = SPACE.build(nodes, count)
        assert space.points == count * (len(nodes) - 1) + 1
        assert np.all(space.weights > 0)
        np.testing.assert_allclose(space.interpolation.sum(axis=1), 1, atol=1e-15)
        np.testing.assert_allclose(space.interpolate(2 * nodes - 3), 2 * space.y - 3, atol=1e-15)
        normal, tangent = rng.normal(size=(2, space.points))
        force = space.pull_force(normal, tangent)
        coarse_work = np.dot(np.r_[q.reshape(-1), 0.3], force)
        fine_work = np.sum(
            space.weights * (space.interpolate(q[0]) * normal + space.interpolate(q[1]) * tangent)
        )
        np.testing.assert_allclose(coarse_work, fine_work, rtol=1e-13, atol=1e-14)
    identity = SPACE.build(nodes)
    np.testing.assert_array_equal(identity.interpolation, np.eye(len(nodes)))


def test_closed_elastic_energy_converges_to_independent_gauss_integral():
    nodes = np.array([0.0, 0.3, 0.9, 2.0])
    q = np.array([0.2, -0.1, 0.4, 0.1])
    gauss, weights = np.polynomial.legendre.leggauss(3)
    exact = 0.0
    for a, b, qa, qb in zip(nodes[:-1], nodes[1:], q[:-1], q[1:], strict=True):
        values = qa + (gauss + 1) / 2 * (qb - qa)
        exact += (b - a) / 2 * np.dot(weights, values**2)
    errors = []
    for count in (1, 2, 4, 8):
        space = SPACE.build(nodes, count)
        value = np.dot(space.weights, space.interpolate(q) ** 2)
        assert value > exact
        errors.append(value - exact)
    np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 4.0, rtol=2e-13)


@pytest.fixture(scope="module")
def static_case():
    model = load_nodal_model(CASE, max_panel_length=160)
    stiffness = nodal_body_stiffness(model, quadrature=12)
    reduction = TOOL["reduce_static_trace"](model, stiffness)
    return model, stiffness, reduction


def test_identity_quadrature_static_lift_matches_full_boundary_solver(static_case):
    model, stiffness, reduction = static_case
    space = SPACE.build(model.reference.moving.starts[model.moving_contact, 1])
    expected = initialize_nodal_restart(model, quadrature=12, stiffness=stiffness)
    actual = TOOL["initialize_refined_contact"](model, space, reduction, model.restart)
    np.testing.assert_allclose(actual.full_displacement, expected.displacement, rtol=0, atol=1e-9)
    assert actual.diagnostics["original_full_balance_mpa"] < 1e-7
    np.testing.assert_array_equal(actual.state.plastic_slip, expected.state.plastic_slip)
    # Compare both dynamic residuals on identical perturbations and old state.
    d = space.dofs
    impedance = np.eye(d) * 1e4
    base = NodalInterfaceEquilibrium(model, expected, impedance)
    refined = TOOL["dynamic_refined_reference"](model, space, actual, impedance)
    q = np.linspace(-0.0001, 0.01, d)
    old = model.restart
    a, ja, ca = base.evaluate(
        q,
        np.zeros(d),
        model.reference.normal_pressure,
        model.reference.restart_actuator + 0.01,
        old,
    )
    b, jb, cb = refined.evaluate(
        q,
        np.zeros(d),
        model.reference.normal_pressure,
        model.reference.restart_actuator + 0.01,
        old,
    )
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-7)
    np.testing.assert_allclose(ja, jb, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(
        ca.state.cumulative_slip, cb.state.cumulative_slip, rtol=0, atol=1e-9
    )


def test_refined_static_lift_preserves_full_body_force_and_seed(static_case):
    model, _, reduction = static_case
    space = SPACE.build(model.reference.moving.starts[model.moving_contact, 1], 3)
    state = space.seed_from_bulk_p1(model.restart)
    lift = TOOL["initialize_refined_contact"](model, space, reduction, state)
    assert lift.diagnostics["original_full_balance_mpa"] < 1e-7
    assert lift.diagnostics["maximum_strength_ratio"] <= 1 + 1e-6
    assert lift.diagnostics["mean_pressure_mpa"] == pytest.approx(16 * 500 / 480, rel=1e-9)
    np.testing.assert_array_equal(lift.state.plastic_slip, state.plastic_slip)
    np.testing.assert_array_equal(lift.state.cumulative_slip, state.cumulative_slip)


def test_refined_jax_local_tangents_and_state_derivatives_match_numpy_and_fd(static_case):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from finiteblock_bem.jax_backend import JaxContactEquilibrium

    model, _, reduction = static_case
    space = SPACE.build(model.reference.moving.starts[model.moving_contact, 1], 3)
    old = space.seed_from_bulk_p1(model.restart)
    lift = TOOL["initialize_refined_contact"](model, space, reduction, old)
    reference = TOOL["dynamic_refined_reference"](model, space, lift, np.eye(space.dofs) * 1e4)
    system = JaxContactEquilibrium(reference)
    n = space.bulk_nodes
    q = jnp.r_[jnp.full(n, 1e-5), jnp.linspace(-0.02, 0.02, n), 0.01]
    p, z, c = system.parameters, jnp.asarray(old.plastic_slip), jnp.asarray(old.cumulative_slip)
    forcing = system.forcing(
        model.reference.normal_pressure, model.reference.restart_actuator, np.zeros(space.dofs)
    )
    expected, tangent, contact = reference.evaluate(
        np.asarray(q),
        np.zeros(space.dofs),
        model.reference.normal_pressure,
        model.reference.restart_actuator,
        old,
    )
    np.testing.assert_allclose(
        system.residual(q, z, c, p, forcing), expected, rtol=1e-11, atol=1e-8
    )
    np.testing.assert_allclose(system.tangent(q, z, c, p), tangent, rtol=1e-11, atol=1e-8)
    response = np.asarray(system.local_response(q, z, c, p))
    np.testing.assert_allclose(
        response[:, 3], contact.state.cumulative_slip, rtol=1e-10, atol=1e-10
    )

    def loss(theta):
        return jnp.dot(jnp.asarray(space.weights), system.local_response(q, z, c, theta)[:, 3])

    reverse = jax.grad(loss)(p)
    forward = jnp.stack([jax.jvp(loss, (p,), (direction,))[1] for direction in jnp.eye(5)])
    np.testing.assert_allclose(reverse, forward, rtol=1e-10, atol=1e-11)
    for i, step in ((0, 0.01), (1, 0.001), (2, 1e-6)):
        fd = (loss(p.at[i].add(step)) - loss(p.at[i].add(-step))) / (2 * step)
        np.testing.assert_allclose(reverse[i], fd, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize(
    "nodes,count", [([0, 0], 1), ([1, 0], 2), ([0, np.nan], 2), ([0, 1], 0), ([0, 1], True)]
)
def test_invalid_quadrature_rejected(nodes, count):
    with pytest.raises(ValueError):
        SPACE.build(nodes, count)

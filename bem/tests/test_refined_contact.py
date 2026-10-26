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


def test_initial_body_residual_tail_excludes_instantaneous_lag_zero():
    from finiteblock_bem.cq import BoundaryHistory, plan_cq

    rng = np.random.default_rng(97)
    plan = plan_cq(5, 0.01, 9, max_memory_mb=8)
    weights = rng.normal(size=(10, 5, 5))
    residual = rng.normal(size=5)
    actual = TOOL["affine_initial_tail"](BoundaryHistory(plan, weights), residual, 9, block_lags=3)
    expected = np.array(
        [
            sum((weights[n - j] @ residual for j in range(1, n)), start=np.zeros(5))
            for n in range(10)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(actual[:2], 0)


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
    data = lift.diagnostics
    # The finite-quadrature body operator has a small rigid-translation defect.
    # Account for its independently computed internal resultant; do not loosen
    # the contact/root tolerance or silently regularize the bulk matrix.
    expected = (
        data["applied_normal_force_mpa_mm"] - data["moving_bulk_normal_force_mpa_mm"]
    ) / space.weights.sum()
    assert data["applied_normal_force_mpa_mm"] == pytest.approx(16 * 500, rel=1e-13)
    assert data["mean_pressure_mpa"] == pytest.approx(expected, rel=1e-11, abs=1e-10)
    assert abs(data["moving_normal_balance_defect_mpa_mm"]) / space.weights.sum() < 1e-9
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


@pytest.mark.parametrize("subdivisions", [1, 3])
def test_refined_resolvent_trajectory_matches_fresh_original_z_history(
    static_case, subdivisions, tmp_path
):
    pytest.importorskip("jax")
    from finiteblock_bem.cq import BoundaryHistory, convolution_weights, plan_cq
    from finiteblock_bem.jax_backend import JaxContactEquilibrium
    from finiteblock_bem.nodal import NodalInterfaceTransfer

    model, _, reduction = static_case
    space = SPACE.build(model.reference.moving.starts[model.moving_contact, 1], subdivisions)
    state = space.seed_from_bulk_p1(model.restart)
    lift = TOOL["initialize_refined_contact"](model, space, reduction, state)
    dt, steps = 8e-6, 8
    plan = plan_cq(space.dofs, dt, steps, max_memory_mb=64)
    transfer = NodalInterfaceTransfer(model, quadrature=12)
    samples = {}

    def impedance(s):
        if s not in samples:
            samples[s] = transfer.evaluate(s)
        return samples[s]

    original = convolution_weights(impedance, plan)
    law, wc = model.reference.contact, model.constraints.contact_weights
    diagonal = np.r_[wc * law.normal_penalty, wc * law.tangential_penalty, 0]
    linear_reference = np.diag(diagonal)
    linear_reference[-1, -1] += model.reference.spring_stiffness * model.constraints.platen_width
    resolvent = convolution_weights(
        lambda s: np.linalg.solve(impedance(s) + linear_reference, np.eye(space.dofs)), plan
    )
    time = np.arange(steps + 1) * dt
    ramp = model.reference.restart_actuator + 2 * time / time[-1]
    force = (ramp - ramp[0]) * linear_reference[-1, -1]
    elastic = np.array(
        [
            sum(
                (resolvent[i - j, :, -1] * force[j] for j in range(i + 1)),
                start=np.zeros(space.dofs),
            )
            for i in range(steps + 1)
        ]
    )
    reference = TOOL["dynamic_refined_reference"](model, space, lift, original[0])
    actual = TOOL["simulate_refined_prepared"](
        model,
        space,
        reference,
        state,
        BoundaryHistory(plan, resolvent),
        elastic,
        time,
        ramp,
        schedule_name="diagnostic_fresh_original_z_quadrature",
    )
    # Independently step the complete original Z coefficients with direct
    # causal sums, not the elastic correction convolution under test.
    system = JaxContactEquilibrium(reference)
    q = np.zeros((steps + 1, space.dofs))
    old, held = state, None
    expected = [
        reference.evaluate(
            q[0],
            np.zeros(space.dofs),
            model.reference.normal_pressure,
            ramp[0],
            state,
            jacobian=False,
        )[2]
    ]
    for i in range(1, steps + 1):
        past = sum((original[i - j] @ q[j] for j in range(i)), start=np.zeros(space.dofs))
        actuator = ramp[i] if held is None else held
        answer = system.solve(
            model.reference.normal_pressure, actuator, old, past=past, guess=q[i - 1]
        )
        assert answer.converged
        q[i] = answer.reduced_displacement
        old = answer.contact.state
        expected.append(answer.contact)
        if (
            held is None
            and np.average(old.cumulative_slip >= law.characteristic_slip, weights=space.weights)
            >= model.reference.payload["loading"]["stop_coverage"]
        ):
            held = actuator
    np.testing.assert_allclose(actual.bulk_displacement, q, rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(
        actual.pressure, np.stack([c.pressure for c in expected]), rtol=2e-7, atol=2e-6
    )
    np.testing.assert_allclose(
        actual.shear, np.stack([c.shear for c in expected]), rtol=2e-7, atol=2e-6
    )
    np.testing.assert_allclose(
        actual.cumulative_slip,
        np.stack([c.state.cumulative_slip for c in expected]),
        rtol=2e-7,
        atol=1e-8,
    )
    assert actual.cumulative_slip[-1].max() > 0.01
    assert actual.diagnostics["correction_start"] == 1
    assert actual.diagnostics["micro_friction_work_mpa_mm2"] > 0
    assert np.min(np.diff(actual.cumulative_slip, axis=0)) >= 0
    writer = runpy.run_path(str(ROOT / "tools/simulate_refined_contact.py"))
    writer["save_trajectory"](tmp_path, model, space, lift, actual, actual.diagnostics)
    with np.load(tmp_path / "trajectory.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["contact_y"], space.y)
        np.testing.assert_array_equal(saved["retained_contact_y"], space.bulk_y)
        assert saved["contact_jumps"].shape == (steps + 1, 2 * space.points)
        assert saved["retained_displacement"].shape == (steps + 1, space.dofs)
        np.testing.assert_allclose(
            saved["contact_jumps"][0, : space.points], reference.base_dn, rtol=0, atol=0
        )
        np.testing.assert_allclose(
            saved["contact_jumps"][0, space.points :], reference.base_ds, rtol=0, atol=0
        )
        np.testing.assert_array_equal(saved["cumulative_slip"], actual.cumulative_slip)
    if subdivisions > 1:
        ad = runpy.run_path(str(ROOT / "tools/refined_contact_adjoint.py"))
        forward = ad["forward_gradient"](reference, BoundaryHistory(plan, resolvent), actual)
        reverse = ad["reverse_gradient"](reference, BoundaryHistory(plan, resolvent), actual)
        np.testing.assert_allclose(reverse["gradient"], forward["gradient"], rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize(
    "nodes,count", [([0, 0], 1), ([1, 0], 2), ([0, np.nan], 2), ([0, 1], 0), ([0, 1], True)]
)
def test_invalid_quadrature_rejected(nodes, count):
    with pytest.raises(ValueError):
        SPACE.build(nodes, count)

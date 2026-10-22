# ruff: noqa: E402
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import block_diag

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from finiteblock_bem.case import load_case
from finiteblock_bem.constraints import build_constraints
from finiteblock_bem.friction import ContactState
from finiteblock_bem.jax_backend import (
    JaxContactEquilibrium,
    contact_point,
    contact_tangents,
    contact_values,
    differentiable_rollout,
)
from finiteblock_bem.nonlinear import ContactEquilibrium
from finiteblock_bem.operators import assemble_operators

CASE = Path(__file__).resolve().parents[1] / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def test_local_ad_matches_independent_analytic_tangent_in_open_stick_and_slip():
    case = load_case(CASE, max_panel_length=80)
    law = case.contact
    jumps = np.array([[-0.001, 0.01], [0.001, 0.001], [0.0001, 0.01], [0.0001, -0.01]])
    old = ContactState(np.zeros(4), np.array([0, 0.001, 0.002, 0.02]))
    ref = law.evaluate(jumps[:, 0], jumps[:, 1], old)
    params = jnp.array(
        [
            law.normal_penalty,
            law.tangential_penalty,
            law.static_friction,
            law.kinetic_friction,
            law.characteristic_slip,
        ]
    )
    values = contact_values(jumps, old.plastic_slip, old.cumulative_slip, params)
    d = contact_tangents(jumps, old.plastic_slip, old.cumulative_slip, params)
    np.testing.assert_allclose(values[:, 0], ref.pressure)
    np.testing.assert_allclose(values[:, 1], ref.shear)
    np.testing.assert_allclose(values[:, 2], ref.state.plastic_slip)
    np.testing.assert_allclose(d[:, 0, 0], ref.dp_dn)
    np.testing.assert_allclose(d[:, 1, 0], ref.dtau_dn)
    np.testing.assert_allclose(d[:, 1, 1], ref.dtau_ds)


def test_opening_switch_regression_converges_with_both_backends():
    case = load_case(CASE, max_panel_length=40)
    data = np.load(Path(__file__).with_name("fixtures") / "opening_switch.npz")
    ref = ContactEquilibrium(
        case,
        build_constraints(case),
        data["impedance"],
        base_displacement=data["base_u"],
        base_traction=data["base_t"],
    )
    old = ContactState(data["old_z"], data["old_c"])
    answers = [
        system.solve(16.0, float(data["actuator"]), old, past=data["past"], guess=data["guess"])
        for system in (ref, JaxContactEquilibrium(ref))
    ]
    for answer in answers:
        assert answer.converged and answer.iterations <= 4
        assert answer.residual_mpa < 1e-7
    np.testing.assert_allclose(
        answers[0].reduced_displacement, answers[1].reduced_displacement, atol=1e-12
    )


def test_custom_local_law_assembles_all_ad_couplings():
    case = load_case(CASE, max_panel_length=40)
    data = np.load(Path(__file__).with_name("fixtures") / "opening_switch.npz")
    ref = ContactEquilibrium(
        case,
        build_constraints(case),
        data["impedance"],
        base_displacement=data["base_u"],
        base_traction=data["base_t"],
    )

    def custom(jump, plastic, cumulative, parameters):
        # Exercise a normal/shear cross-coupling absent from standard LSW.
        adjusted = jump.at[0].multiply(1 + parameters[5] * jnp.tanh(jump[1] - plastic))
        return contact_point(adjusted, plastic, cumulative, parameters[:5])

    default = JaxContactEquilibrium(ref)
    params = jnp.concatenate((default.parameters, jnp.array([0.2])))
    system = JaxContactEquilibrium(ref, contact_law=custom, parameters=params)
    z, old_z, old_c = (jnp.array(data[k]) for k in ("guess", "old_z", "old_c"))
    forcing = system.forcing(16.0, float(data["actuator"]), data["past"])
    assembled = system.tangent(z, old_z, old_c, params)
    full_ad = jax.jacfwd(system.residual, argnums=0)(z, old_z, old_c, params, forcing)
    np.testing.assert_allclose(assembled, full_ad, atol=1e-10, rtol=1e-12)


def test_implicit_root_gradient_matches_finite_difference_and_primal_reference():
    case = load_case(CASE, max_panel_length=100)
    con = build_constraints(case)
    # Dynamic positive-real operator avoids static moving-body rigid modes.
    w = block_diag(
        *(
            assemble_operators(m, case.material, 3e5).impedance().real
            for m in (case.moving, case.stationary)
        )
    )
    ref = ContactEquilibrium(case, con, w)
    system = JaxContactEquilibrium(ref)
    old = ContactState(np.zeros(len(case.moving_contact)), np.full(len(case.moving_contact), 0.004))
    forcing = system.forcing(16.0, 1.0, np.zeros(len(w)))
    guess = jnp.zeros(con.size)
    params = system.parameters

    def objective(parameters):
        z, _ = system.implicit_displacement(
            parameters, old.plastic_slip, old.cumulative_slip, forcing, guess
        )
        return z[-1]

    result = system.solve(16.0, 1.0, old)
    expected = ref.solve(16.0, 1.0, old)
    assert result.converged and expected.converged
    np.testing.assert_allclose(
        result.reduced_displacement, expected.reduced_displacement, atol=2e-10
    )
    derivative = float(jax.grad(objective)(params)[2])
    epsilon = 1e-5
    finite = float(
        (objective(params.at[2].add(epsilon)) - objective(params.at[2].add(-epsilon)))
        / (2 * epsilon)
    )
    assert abs(derivative) > 1e-9  # Require an actual sliding sensitivity.
    np.testing.assert_allclose(derivative, finite, rtol=2e-4, atol=1e-9)
    # A multistep loss must include state and full-history sensitivities.
    # A synthetic stable linear memory isolates this from kernel discretization error.
    weights = np.zeros((5, con.size, con.size))
    weights[1:] = -np.eye(con.size)[None, ...] * np.array([2.0, 1.0, 0.5, 0.25])[:, None, None]
    forces = jnp.stack([forcing * (1 + 0.03 * n) for n in range(4)])
    rollout = differentiable_rollout(system, weights, forces, old.plastic_slip, old.cumulative_slip)

    def loss(parameters):
        _, response, _ = rollout(parameters)
        return jnp.mean(response[-1, :, 3])

    _, _, errors = rollout(params)
    assert float(jnp.max(errors)) < 1e-7
    grad = float(jax.grad(loss)(params)[2])
    finite = float(
        (loss(params.at[2].add(epsilon)) - loss(params.at[2].add(-epsilon))) / (2 * epsilon)
    )
    assert abs(grad) > 1e-9
    np.testing.assert_allclose(grad, finite, rtol=2e-4, atol=1e-9)

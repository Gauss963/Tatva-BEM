"""Independent ODE/composition checks before using the experimental RK-CQ."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from finiteblock_bem.cq import BoundaryHistory, plan_cq
from finiteblock_bem.friction import ContactState, SlipWeakeningContact

EXPERIMENT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/radau_experiment.py"))
A, B, C = (EXPERIMENT[name] for name in ["A", "B", "C"])
SYMBOL = EXPERIMENT["radau_symbol"]
WEIGHTS = EXPERIMENT["radau_weights"]


def test_radau_symbol_preserves_weighted_coercivity_on_complex_contour():
    scale = np.diag(np.sqrt(B))
    inverse = np.diag(1 / np.sqrt(B))
    for radius in [0.2, 0.9, 0.999]:
        for theta in np.linspace(0, 2 * np.pi, 13):
            z = radius * np.exp(1j * theta)
            delta = SYMBOL(z)
            expected = np.linalg.inv(A + z / (1 - z) * np.outer(np.ones(2), B))
            np.testing.assert_allclose(delta, expected, atol=1e-12)
            normalized = scale @ delta @ inverse
            minimum = np.linalg.eigvalsh((normalized + normalized.conj().T) / 2).min()
            assert minimum >= (1 - radius**2) / 2 - 1e-13


def test_matrix_function_has_polynomial_composition_property():
    symbol = SYMBOL(0.8 * np.exp(0.7j)) / 0.03
    value = EXPERIMENT["matrix_function"](lambda s: np.eye(2) * (s**2 + 2 * s + 3), symbol)
    expected = np.kron(symbol @ symbol + 2 * symbol + 3 * np.eye(2), np.eye(2))
    np.testing.assert_allclose(value, expected, rtol=2e-14, atol=1e-10)


@pytest.mark.parametrize("formulation", ["impedance", "compliance"])
def test_radau_cq_matches_independently_discretized_coupled_ode(formulation):
    dt, steps = 0.04, 23
    dynamic = np.array([[-1.2, 2.3], [-0.7, -0.9]])
    plan = plan_cq(4, dt, steps, alias_tolerance=1e-12, max_memory_mb=128)
    times = (np.arange(steps)[:, None] + C) * dt
    inputs = np.zeros((steps + 1, 4))
    inputs[1:] = np.stack((np.sin(3 * times), times**2), axis=-1).reshape(steps, 4)

    # Standard RK applied directly to x'=dynamic*x+u; no CQ weight formulas.
    expected = np.zeros_like(inputs)
    matrix = np.eye(4) - dt * np.kron(A, dynamic)
    previous = np.zeros(2)
    for index in range(1, steps + 1):
        expected[index] = np.linalg.solve(
            matrix, np.tile(previous, 2) + dt * np.kron(A, np.eye(2)) @ inputs[index]
        )
        previous = expected[index, -2:]
    if formulation == "impedance":
        weights = WEIGHTS(lambda s: s * np.eye(2) - dynamic, plan)
        actual = EXPERIMENT["linear_response"](BoundaryHistory(plan, weights), inputs)
    else:
        weights = WEIGHTS(lambda s: np.linalg.inv(s * np.eye(2) - dynamic), plan)
        actual = np.zeros_like(inputs)
        for index in range(1, steps + 1):
            actual[index] = np.einsum("kij,kj->i", weights[: index + 1], inputs[index::-1])
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-11)


def test_radau_rejects_wrong_stage_size_and_defective_eigenbasis():
    plan = plan_cq(3, 0.01, 8, max_memory_mb=128)
    with pytest.raises(ValueError, match="even"):
        WEIGHTS(lambda s: np.eye(1) / s, plan)
    with pytest.raises(ValueError, match="Near-defective"):
        EXPERIMENT["matrix_function"](lambda s: np.eye(2) / s, np.array([[1, 1], [0, 1]]))


def test_full_staged_history_and_state_update_have_implicit_gradients():
    jax = pytest.importorskip("jax")
    checker = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/check_radau_ad.py"))
    law = SlipWeakeningContact(100, 40, 0.8, 0.5, 0.2)
    con = SimpleNamespace(
        size=3,
        normal_jump=np.array([[1.0, 0, 0]]),
        tangential_jump=np.array([[0.0, 1, 0]]),
        contact_weights=np.ones(1),
        reduced_weights=np.ones(3),
    )
    reference = SimpleNamespace(
        constraints=con,
        case=SimpleNamespace(contact=law, restart_actuator=0),
        base_dn=np.array([0.1]),
        base_ds=np.array([0.03]),
        base_platen=0,
        base_t=np.array([-10.0, 0, 0]),
        spring=2,
    )
    lift = SimpleNamespace(state=ContactState(np.array([0.03]), np.zeros(1)))
    stiffness = np.array([[30.0, 0, 0], [0, 30, -20], [0, -20, 30]])
    plan = plan_cq(6, 0.05, 20, max_memory_mb=128)
    weights = WEIGHTS(lambda s: 0.2 * s * np.eye(3) + stiffness, plan)
    system = EXPERIMENT["staged_contact_system"](reference, weights[0])
    rollout = checker["make_rollout"](system, reference, weights, plan.dt, 15, lift)
    (value, output), gradient = jax.jit(jax.value_and_grad(rollout, has_aux=True))(
        system.parameters
    )
    assert value > law.characteristic_slip
    assert np.max(np.asarray(output[2])) < 1e-9
    epsilon = 1e-6
    plus, plus_output = rollout(system.parameters.at[3].add(epsilon))
    minus, minus_output = rollout(system.parameters.at[3].add(-epsilon))
    np.testing.assert_array_equal(plus_output[3], output[3])
    np.testing.assert_array_equal(minus_output[3], output[3])
    np.testing.assert_allclose(gradient[3], (plus - minus) / (2 * epsilon), rtol=2e-6, atol=1e-8)

# ruff: noqa: E402
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from finiteblock_bem.adjoint import trajectory_adjoint
from finiteblock_bem.case import load_case
from finiteblock_bem.condensation import InterfaceContactEquilibrium, build_condensed_history
from finiteblock_bem.dynamics import LoadingSchedule, initialize_restart, simulate
from finiteblock_bem.jax_backend import JaxContactEquilibrium, differentiable_rollout

CASE = Path(__file__).resolve().parents[1] / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def test_reverse_cq_adjoint_matches_scan_gradient_and_actuator_finite_difference(tmp_path):
    case = load_case(CASE, max_panel_length=80)
    lift = initialize_restart(case)
    dt, steps = 8e-6, 32
    schedule = LoadingSchedule(
        lambda t: case.normal_pressure,
        lambda t: case.restart_actuator + 2 * t / (steps * dt),
        "adjoint_test",
    )
    result = simulate(
        case,
        dt,
        steps,
        schedule=schedule,
        lift=lift,
        formulation="interface",
        cache_directory=tmp_path,
    )
    history = build_condensed_history(case, dt, steps, cache_directory=tmp_path)
    answer = trajectory_adjoint(case, lift, history, result)
    ref = InterfaceContactEquilibrium(case, lift, history.instantaneous)
    system = JaxContactEquilibrium(ref)
    force = jnp.stack(
        [
            system.forcing(
                case.normal_pressure, schedule.actuator(n * dt), np.zeros(history.plan.dofs)
            )
            for n in range(1, steps + 1)
        ]
    )
    rollout = differentiable_rollout(
        system, history.weights, force, lift.state.plastic_slip, lift.state.cumulative_slip
    )
    weights = jnp.array(ref.constraints.contact_weights)

    def loss(parameters):
        _, response, _ = rollout(parameters)
        return jnp.dot(response[-1, :, 3], weights) / weights.sum()

    expected = jax.grad(loss)(system.parameters)
    np.testing.assert_allclose(answer["value_mm"], loss(system.parameters), atol=1e-10)
    np.testing.assert_allclose(answer["gradient"], expected, rtol=2e-6, atol=1e-10)
    finite_mu = float(
        (loss(system.parameters.at[2].add(1e-5)) - loss(system.parameters.at[2].add(-1e-5))) / 2e-5
    )
    np.testing.assert_allclose(answer["gradient"][2], finite_mu, rtol=2e-5)
    index, eps = 5, 1e-5
    losses = []
    for sign in [-1, 1]:
        perturbed = LoadingSchedule(
            schedule.pressure,
            lambda t, s=sign: schedule.actuator(t) + (s * eps if round(t / dt) == index else 0.0),
            "fixed_test",
        )
        trial = simulate(
            case,
            dt,
            steps,
            schedule=perturbed,
            lift=lift,
            formulation="interface",
            cache_directory=tmp_path,
        )
        losses.append(np.average(trial.cumulative_slip[-1], weights=np.array(weights)))
    finite = (losses[1] - losses[0]) / (2 * eps)
    assert abs(finite) > 1e-7
    np.testing.assert_allclose(answer["actuator_gradient"][index], finite, rtol=2e-5, atol=1e-9)

    # Keep displacement AND bulk traction fixed when changing penalties. This
    # differs from rebuilding a new static equilibrium for each parameter.
    for index, field, epsilon in [
        (0, "normal_penalty", 1.0),
        (1, "tangential_penalty", 0.1),
    ]:
        losses = []
        for sign in [-1, 1]:
            varied = replace(
                case,
                contact=replace(
                    case.contact, **{field: getattr(case.contact, field) + sign * epsilon}
                ),
            )
            varied_reference = InterfaceContactEquilibrium(varied, lift, history.instantaneous)
            np.testing.assert_array_equal(varied_reference.base_t, ref.base_t)
            # Perturb the dynamic law after the fixed initial state. A separately
            # initialized physical penalty study must also differentiate its lift.
            losses.append(float(loss(system.parameters.at[index].add(sign * epsilon))))
        finite = (losses[1] - losses[0]) / (2 * epsilon)
        np.testing.assert_allclose(answer["gradient"][index], finite, rtol=2e-3, atol=1e-11)

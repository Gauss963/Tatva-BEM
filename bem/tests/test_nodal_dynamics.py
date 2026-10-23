# ruff: noqa: E402
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from finiteblock_bem.adjoint import trajectory_adjoint
from finiteblock_bem.dynamics import LoadingSchedule
from finiteblock_bem.io import save_dynamic
from finiteblock_bem.jax_backend import JaxContactEquilibrium, differentiable_rollout
from finiteblock_bem.nodal import (
    NodalInterfaceEquilibrium,
    initialize_nodal_restart,
    load_nodal_model,
)
from finiteblock_bem.nodal_dynamics import build_nodal_history, simulate_nodal

CASE = Path(__file__).resolve().parents[1] / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def test_nodal_dynamic_numpy_jax_adjoint_and_nodal_output_weights(tmp_path):
    model = load_nodal_model(CASE, max_panel_length=160)
    ref = model.reference
    dt, steps = 8e-6, 8
    lift = initialize_nodal_restart(model)
    schedule = LoadingSchedule(
        lambda _: ref.normal_pressure,
        lambda t: ref.restart_actuator + 2 * t / (steps * dt),
        "nodal_ad_diagnostic",
    )
    results = [
        simulate_nodal(
            model,
            dt,
            steps,
            lift=lift,
            schedule=schedule,
            backend=backend,
            cache_directory=tmp_path,
        )
        for backend in ["numpy", "jax"]
    ]
    for r in results:
        assert r.diagnostics["status"] == "completed_requested_horizon"
        assert r.cumulative_slip[-1].max() > 0.1
    for field in [
        "pressure",
        "shear",
        "plastic_slip",
        "cumulative_slip",
        "contact_jumps",
        "platen",
    ]:
        np.testing.assert_allclose(
            getattr(results[0], field), getattr(results[1], field), rtol=1e-7, atol=1e-8
        )
    history = build_nodal_history(model, dt, steps, cache_directory=tmp_path)
    equilibrium = NodalInterfaceEquilibrium(model, lift, history.instantaneous)
    system = JaxContactEquilibrium(equilibrium)
    answer = trajectory_adjoint(
        model.contact_case(), lift, history, results[1], equilibrium=equilibrium
    )
    force = jnp.stack(
        [
            system.forcing(
                ref.normal_pressure, schedule.actuator(n * dt), np.zeros(history.plan.dofs)
            )
            for n in range(1, steps + 1)
        ]
    )
    rollout = differentiable_rollout(
        system, history.weights, force, lift.state.plastic_slip, lift.state.cumulative_slip
    )
    wc = jnp.array(model.constraints.contact_weights)

    def loss(parameters):
        _, response, _ = rollout(parameters)
        return jnp.dot(response[-1, :, 3], wc) / wc.sum()

    np.testing.assert_allclose(answer["value_mm"], loss(system.parameters), atol=1e-10)
    np.testing.assert_allclose(
        answer["gradient"], jax.grad(loss)(system.parameters), rtol=1e-6, atol=1e-10
    )
    eps = 1e-5
    finite = float(
        (loss(system.parameters.at[2].add(eps)) - loss(system.parameters.at[2].add(-eps)))
        / (2 * eps)
    )
    np.testing.assert_allclose(answer["gradient"][2], finite, rtol=2e-5, atol=1e-9)
    output = tmp_path / "nodal-output"
    save_dynamic(
        output,
        model.contact_case(),
        results[1],
        CASE,
        contact_y=ref.moving.starts[model.moving_contact, 1],
        contact_weights=model.constraints.contact_weights,
    )
    with np.load(output / "trajectory.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["contact_weights"], model.constraints.contact_weights)
        np.testing.assert_array_equal(
            saved["contact_y"], ref.moving.starts[model.moving_contact, 1]
        )
        assert "displacement" not in saved
    table = np.genfromtxt(output / "history.csv", delimiter=",", names=True)
    np.testing.assert_allclose(table["mean_contact_pressure_mpa"][0], 16 * 500 / 480, rtol=1e-7)

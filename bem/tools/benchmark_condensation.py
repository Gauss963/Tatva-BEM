"""Compare full/condensed contact trajectories and their implicit AD gradients."""

import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag

from finiteblock_bem.case import load_case
from finiteblock_bem.condensation import InterfaceContactEquilibrium, build_condensed_history
from finiteblock_bem.constraints import build_constraints
from finiteblock_bem.cq import build_history
from finiteblock_bem.dynamics import initialize_restart
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.jax_backend import JaxContactEquilibrium, differentiable_rollout
from finiteblock_bem.nonlinear import ContactEquilibrium

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "outputs/condensation-ad"
    output.mkdir(parents=True, exist_ok=True)
    path = ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    case = load_case(path, max_panel_length=80)
    lift = initialize_restart(case)
    con = build_constraints(case)
    dt, steps = 8e-6, 32
    full = [build_history(m, case.material, dt, steps) for m in (case.moving, case.stationary)]
    interface = build_condensed_history(case, dt, steps)
    full_system = ContactEquilibrium(
        case,
        con,
        block_diag(*(h.instantaneous for h in full)),
        base_displacement=lift.displacement,
        base_traction=lift.traction,
    )
    interface_system = InterfaceContactEquilibrium(case, lift, interface.instantaneous)
    full_weights = np.stack(
        [
            con.prolongation.T
            @ (
                con.full_weights[:, None]
                * (block_diag(*(h.weights[n] for h in full)) @ con.prolongation)
            )
            for n in range(steps + 1)
        ]
    )
    records, answers = [], []
    for name, reference, weights in [
        ("full", full_system, full_weights),
        ("interface", interface_system, interface.weights),
    ]:
        system = JaxContactEquilibrium(reference)
        forcing = jnp.stack(
            [
                system.forcing(
                    case.normal_pressure,
                    case.restart_actuator + 2 * (n + 1) / steps,
                    np.zeros(len(reference.base_u)),
                )
                for n in range(steps)
            ]
        )
        rollout = differentiable_rollout(
            system, weights, forcing, lift.state.plastic_slip, lift.state.cumulative_slip
        )

        def loss(parameters):
            _, response, _ = rollout(parameters)
            return jnp.mean(response[-1, :, 3])

        value_grad = jax.jit(jax.value_and_grad(loss))
        started = perf_counter()
        value, gradient = jax.block_until_ready(value_grad(system.parameters))
        first = perf_counter() - started
        started = perf_counter()
        for _ in range(10):
            value, gradient = jax.block_until_ready(value_grad(system.parameters))
        elapsed = (perf_counter() - started) / 10
        _, response, error = rollout(system.parameters)
        if float(jnp.max(error)) >= 1e-7:
            raise RuntimeError("Unconverged roots invalidate the AD comparison")
        finite = float(
            (loss(system.parameters.at[2].add(1e-5)) - loss(system.parameters.at[2].add(-1e-5)))
            / 2e-5
        )
        records.append(
            {
                "formulation": name,
                "root_dofs": reference.constraints.size,
                "weight_bytes": weights.nbytes,
                "loss_mm": float(value),
                "gradient": np.asarray(gradient).tolist(),
                "mu_s_central_difference": finite,
                "first_value_grad_s": first,
                "warm_value_grad_s": elapsed,
                "max_root_residual_mpa": float(jnp.max(error)),
            }
        )
        answers.append(np.asarray(response))
    record = {
        "diagnostic": "32-step static-lift, constant-normal sliding; fixed bulk and lift",
        "results": records,
        "max_contact_trajectory_difference": float(np.max(abs(answers[0] - answers[1]))),
        "max_gradient_difference": float(
            np.max(abs(np.array(records[0]["gradient"]) - records[1]["gradient"]))
        ),
        "environment": environment_record(path),
    }
    write_json(output / "summary.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

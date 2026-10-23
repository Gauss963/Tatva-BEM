"""Check implicit JAX differentiation of the experimental staged CQ trajectory.

This bounded direct-history scan is an independent AD reference, not a scalable
RK adjoint. Bulk, lift, prescribed accelerated loading and branches are fixed.
"""

import argparse
import hashlib
import json
import runpy
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.nodal import (
    NodalInterfaceEquilibrium,
    initialize_nodal_restart,
    load_nodal_model,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = Path(__file__).with_name("radau_experiment.py")
EXPERIMENT = runpy.run_path(str(EXPERIMENT_PATH))


def make_rollout(system, reference, weights, dt, increment, lift):
    """All stage equations share old state; only last stage updates the carry."""
    count, stage_size = weights.shape[:2]
    steps = count - 1
    n = len(lift.state.plastic_slip)
    if steps > 512:
        raise ValueError("Direct-history AD reference is limited to 512 intervals")
    kernel = jnp.asarray(weights)
    indices = jnp.arange(count)
    times = (np.arange(steps)[:, None] + EXPERIMENT["C"]) * dt
    actuator = reference.case.restart_actuator + increment * times / (steps * dt)
    forcing = np.tile(reference.base_t, (steps, 2, 1))
    forcing[:, :, -1] += reference.spring * (reference.base_platen - actuator)
    forcing = jnp.asarray(forcing.reshape(steps, stage_size))
    contact_weights = jnp.asarray(reference.constraints.contact_weights)
    base_dn, base_ds = jnp.tile(reference.base_dn, 2), jnp.tile(reference.base_ds, 2)
    bn, bt = system.reference.constraints.normal_jump, system.reference.constraints.tangential_jump
    bn, bt = jnp.asarray(bn), jnp.asarray(bt)

    def rollout(parameters):
        def step(carry, index):
            history, old_z, old_c, guess = carry
            lag = jnp.clip(index - indices, 0, steps)
            past = jnp.einsum(
                "kij,kj->i", kernel[lag], jnp.where((indices < index)[:, None], history, 0)
            )
            plastic, cumulative = jnp.tile(old_z, 2), jnp.tile(old_c, 2)
            rhs = forcing[index - 1] + past
            q, _ = system.implicit_displacement(parameters, plastic, cumulative, rhs, guess)
            local = system.local_response(q, plastic, cumulative, parameters)
            endpoint = local[n:]
            error = jnp.max(abs(system.residual(q, plastic, cumulative, parameters, rhs)))
            # Exact predicates of contact_point, including the weakening clip.
            kn, kt, mus, muk, dc = parameters
            closure, relative = base_dn + bn @ q, base_ds + bt @ q
            active = closure > 0
            pressure = jnp.where(active, kn * closure, 0)
            mu = mus - (mus - muk) * jnp.minimum(cumulative / dc, 1)
            sliding = active & (abs(kt * (relative - plastic)) > mu * pressure)
            branches = active.astype(jnp.int8) + 2 * sliding + 4 * (cumulative < dc)
            return (history.at[index].set(q), endpoint[:, 2], endpoint[:, 3], q), (
                q,
                local,
                error,
                branches,
            )

        initial = (
            jnp.zeros((count, stage_size)),
            jnp.asarray(lift.state.plastic_slip),
            jnp.asarray(lift.state.cumulative_slip),
            jnp.zeros(stage_size),
        )
        final, trajectory = jax.lax.scan(jax.checkpoint(step), initial, jnp.arange(1, count))
        value = jnp.sum(final[2] * contact_weights) / jnp.sum(contact_weights)
        return value, trajectory

    return jax.jit(rollout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--refinement", type=int, default=8)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    options = summary["arguments"]
    current = environment_record(options["case"])
    for name in ["source_sha256", "case_sha256", "restart_data_sha256"]:
        if current[name] != summary["environment"][name]:
            raise ValueError(f"Saved RK run differs in {name}")
    if hashlib.sha256(EXPERIMENT_PATH.read_bytes()).hexdigest() != summary["prototype_sha256"]:
        raise ValueError("Experimental RK source differs from the saved trajectory")
    model = load_nodal_model(options["case"], max_panel_length=options["h"])
    lift = initialize_nodal_restart(model, quadrature=options["quadrature"])
    weights = np.load(args.run / f"weights-radau2-r{args.refinement}.npy", allow_pickle=False)
    d = weights.shape[1] // 2
    reference = NodalInterfaceEquilibrium(model, lift, np.zeros((d, d)))
    system = EXPERIMENT["staged_contact_system"](reference, weights[0])
    rollout = make_rollout(
        system, reference, weights, options["dt"] / args.refinement, options["increment"], lift
    )
    parameters = system.parameters
    start = perf_counter()
    (value, trajectory), gradient = jax.jit(jax.value_and_grad(rollout, has_aux=True))(parameters)
    value, gradient = float(value), np.asarray(gradient)
    stages, local, residual, branches = (np.asarray(a) for a in trajectory)
    seconds = perf_counter() - start
    n = len(model.moving_contact)
    with np.load(args.run / f"contact-radau2-r{args.refinement}.npz") as saved:
        for field, column in [("p", 0), ("tau", 1), ("z", 2), ("c", 3)]:
            np.testing.assert_allclose(
                local[:, n:, column], saved[field][1:], atol=1e-10, rtol=1e-10
            )
    with np.load(args.run / f"stages-radau2-r{args.refinement}.npz") as saved:
        np.testing.assert_allclose(stages, saved["stages"][1:], atol=1e-10, rtol=1e-10)
    if not np.isfinite(residual).all() or residual.max() >= 1e-7:
        raise ValueError("Reference rollout did not converge")
    checks = []
    # Penalty sensitivities are O(1e-9). Relative O(1e-6) perturbations
    # make their objective changes larger than solver roundoff.
    for index, (name, epsilon) in enumerate(
        zip(["kn", "kt", "mu_s", "mu_k", "Dc"], [1e-1, 1e-2, 1e-8, 1e-8, 1e-9], strict=True)
    ):
        values, branch_counts, max_residual = [], [], 0.0
        for sign in [1, -1]:
            altered = parameters.at[index].add(sign * epsilon)
            perturbed, output = rollout(altered)
            values.append(float(perturbed))
            max_residual = max(max_residual, float(np.max(np.asarray(output[2]))))
            branch_counts.append(int(np.count_nonzero(np.asarray(output[3]) != branches)))
        difference = (values[0] - values[1]) / (2 * epsilon)
        checks.append(
            dict(
                parameter=name,
                epsilon=epsilon,
                adjoint=float(gradient[index]),
                central_difference=difference,
                relative_difference=abs(difference - gradient[index])
                / max(abs(gradient[index]), 1e-15),
                changed_stage_branches=branch_counts,
                maximum_fd_residual_mpa=max_residual,
            )
        )
    record = dict(
        value_mm=value,
        gradient=gradient.tolist(),
        finite_differences=checks,
        maximum_residual_mpa=float(residual.max()),
        reverse_seconds_including_jit=seconds,
        environment=current,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prototype_sha256=summary["prototype_sha256"],
        scope="Conditional implicit AD of full short RK-CQ; fixed bulk/lift/prescribed loading; direct-history reference, not scalable production RK adjoint",
    )
    write_json(args.run / f"adjoint-r{args.refinement}.json", record)
    print(json.dumps(record, indent=2))
    for row in checks:
        if any(row["changed_stage_branches"]):
            raise ValueError(f"Finite difference crossed a branch: {row['parameter']}")
        if row["relative_difference"] > 1e-3:
            raise ValueError(f"Gradient discrepancy: {row['parameter']}")


if __name__ == "__main__":
    main()

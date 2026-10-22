#!/usr/bin/env python3
"""Measure synchronized JAX roots/gradients against NumPy and central differences."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag

from finiteblock_bem.case import load_case
from finiteblock_bem.constraints import build_constraints
from finiteblock_bem.cq import build_history
from finiteblock_bem.friction import ContactState
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.jax_backend import JaxContactEquilibrium, differentiable_rollout
from finiteblock_bem.nonlinear import ContactEquilibrium

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ad-benchmark")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    case_path = ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    case = load_case(case_path, max_panel_length=80)
    con = build_constraints(case)
    # This is an intentionally accelerated, zero-field gradient diagnostic.
    dt, steps = 4e-6, 32
    started = perf_counter()
    histories = [
        build_history(m, case.material, dt, steps, workers=4)
        for m in (case.moving, case.stationary)
    ]
    kernel_time = perf_counter() - started
    w = block_diag(*(h.instantaneous for h in histories))
    ref = ContactEquilibrium(case, con, w)
    system = JaxContactEquilibrium(ref)
    old = ContactState(np.zeros(len(case.moving_contact)), np.full(len(case.moving_contact), 0.004))
    p, d = 16.0, 1.0
    started = perf_counter()
    actual = system.solve(p, d, old)
    compile_solve = perf_counter() - started
    repetitions = 100
    started = perf_counter()
    for _ in range(repetitions):
        expected = ref.solve(p, d, old)
    numpy_time = (perf_counter() - started) / repetitions
    started = perf_counter()
    for _ in range(repetitions):
        actual = system.solve(p, d, old)
    jax_time = (perf_counter() - started) / repetitions
    if not actual.converged or not expected.converged:
        raise RuntimeError("Benchmark root failed")
    reduced_w = np.stack(
        [
            con.prolongation.T
            @ (
                con.full_weights[:, None]
                * (block_diag(*(h.weights[n] for h in histories)) @ con.prolongation)
            )
            for n in range(steps + 1)
        ]
    )
    forcing = np.stack(
        [system.forcing(p, d * (n + 1) / steps, np.zeros(len(w))) for n in range(steps)]
    )
    rollout = differentiable_rollout(
        system, reduced_w, forcing, old.plastic_slip, old.cumulative_slip
    )

    def objective(parameters):
        _, response, _ = rollout(parameters)
        return jnp.mean(response[-1, :, 3])

    grad_fn = jax.jit(jax.value_and_grad(objective))
    started = perf_counter()
    value, gradient = jax.block_until_ready(grad_fn(system.parameters))
    compile_gradient = perf_counter() - started
    started = perf_counter()
    for _ in range(10):
        value, gradient = jax.block_until_ready(grad_fn(system.parameters))
    grad_time = (perf_counter() - started) / 10
    finite = []
    for i, eps in enumerate([0.1, 0.01, 1e-5, 1e-5, 1e-6]):
        finite.append(
            float(
                (
                    objective(system.parameters.at[i].add(eps))
                    - objective(system.parameters.at[i].add(-eps))
                )
                / (2 * eps)
            )
        )
    _, _, errors = rollout(system.parameters)
    error = float(jnp.max(errors))
    if error >= 1e-7:
        raise RuntimeError(f"Rollout root residual {error} invalidates its gradient")
    result = {
        "diagnostic": "32-step fixed-bulk-CQ accelerated loading, not event16",
        "kernel_seconds": kernel_time,
        "jax_first_root_seconds": compile_solve,
        "numpy_root_seconds": numpy_time,
        "jax_warm_root_seconds": jax_time,
        "root_max_difference_mm": float(
            np.max(abs(actual.reduced_displacement - expected.reduced_displacement))
        ),
        "jax_first_value_and_grad_seconds": compile_gradient,
        "jax_warm_value_and_grad_seconds": grad_time,
        "loss_mean_cumulative_slip_mm": float(value),
        "parameter_order": ["normal_penalty", "tangential_penalty", "mu_s", "mu_k", "Dc"],
        "autodiff_gradient": np.asarray(gradient).tolist(),
        "central_difference": finite,
        "gradient_max_relative_error": float(
            np.max(
                abs(np.asarray(gradient) - finite) / np.maximum(abs(np.asarray(gradient)), 1e-10)
            )
        ),
        "rollout_max_residual_mpa": error,
        "environment": environment_record(case_path),
        "scope": "Initial state, geometry and bulk operators held fixed; active-branch gradients",
    }
    write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Experimental two-stage Radau IIA CQ, kept outside the production package.

Sources: Kramer et al. arXiv:2510.06804 and Banjai/Lubich arXiv:1702.08385.
Linear CQ has the RK composition property. The contact experiment freezes the
accepted old plastic/cumulative state at BOTH trial stages, then commits only
the stiffly accurate endpoint. No high-order or stability theorem is claimed
for that nonsmooth state update. This tool does not alter the production BDF2.
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.linalg import block_diag, lu_factor, lu_solve
from threadpoolctl import threadpool_limits

from finiteblock_bem.benchmarks import bar_displacement_exact, smooth_step
from finiteblock_bem.cq import (
    BoundaryHistory,
    OnlineConvolution,
    convolution_weights,
    invert_spectrum,
    plan_cq,
)
from finiteblock_bem.io import environment_record, write_json

ROOT = Path(__file__).resolve().parents[1]
A = np.array([[5 / 12, -1 / 12], [3 / 4, 1 / 4]])
B = np.array([3 / 4, 1 / 4])
C = np.array([1 / 3, 1.0])


def radau_symbol(zeta):
    """(A + z/(1-z) * 1*b.T)^-1, using stiff accuracy at z=1 too."""
    inverse = np.linalg.inv(A)
    return inverse - zeta * np.outer(inverse @ np.ones(2), [0, 1])


def matrix_function(transfer, symbol):
    """Stage-major matrix function, evaluated only on the CQ contour."""
    eigenvalues, eigenvectors = np.linalg.eig(symbol)
    if np.linalg.cond(eigenvectors) > 1e8:
        raise ValueError("Near-defective RK symbol requires a Schur/derivative evaluator")
    inverse = np.linalg.inv(eigenvectors)
    values = [np.atleast_2d(transfer(s)) for s in eigenvalues]
    size = values[0].shape[0]
    if any(v.shape != (size, size) or not np.all(np.isfinite(v)) for v in values):
        raise ValueError("Transfer must return finite, equally sized square matrices")
    result = np.zeros((2 * size, 2 * size), complex)
    for mode, value in enumerate(values):
        result += np.kron(np.outer(eigenvectors[:, mode], inverse[mode]), value)
    return result


def radau_weights(transfer, plan, *, workers=1):
    if plan.dofs % 2:
        raise ValueError("Radau stage DOFs must be even")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    count = plan.fft_size // 2 + 1
    spectrum = np.empty((count, plan.dofs, plan.dofs), complex)

    def sample(index):
        zeta = plan.radius * np.exp(-2j * np.pi * index / plan.fft_size)
        value = matrix_function(transfer, radau_symbol(zeta) / plan.dt)
        if value.shape != (plan.dofs, plan.dofs):
            raise ValueError("Transfer size differs from stage plan")
        return value

    with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(workers) as pool:
        for first in range(0, count, workers):
            ids = range(first, min(first + workers, count))
            values = map(sample, ids) if workers == 1 else pool.map(sample, ids)
            for index, value in zip(ids, values, strict=True):
                spectrum[index] = value
    return invert_spectrum(spectrum, plan)


def linear_response(history, forcing):
    """Solve W*q=f with a zero artificial block before the first RK interval."""
    online = OnlineConvolution(history)
    online.append(np.zeros(history.plan.dofs))
    factor = lu_factor(history.instantaneous)
    response = np.zeros_like(forcing)
    for index in range(1, len(forcing)):
        response[index] = lu_solve(factor, forcing[index] - online.past(index))
        online.append(response[index])
    return response


def analytic_bar(output):
    """Exact finite-bar transfer separates temporal error from spatial BEM error."""
    records = []
    for dt in [0.08, 0.04, 0.02, 0.01]:
        steps = round(3.6 / dt)
        endpoint = np.arange(steps + 1) * dt

        # tanh(s)/s is the loaded-face compliance of a unit fixed-end bar.
        # Repeat the scalar channel to satisfy the common CQ planner's d>=2.
        def transfer(s):
            return np.eye(2) * (np.tanh(s) / s)

        for method in ["bdf2", "radau2"]:
            stage_count = 1 if method == "bdf2" else 2
            plan = plan_cq(2 * stage_count, dt, steps, max_memory_mb=512)
            tick = perf_counter()
            weights = (
                convolution_weights(transfer, plan)
                if method == "bdf2"
                else radau_weights(transfer, plan)
            )
            time = endpoint[1:, None] if stage_count == 1 else (np.arange(steps)[:, None] + C) * dt
            forcing = np.zeros((steps + 1, 2 * stage_count))
            forcing[1:] = np.repeat(smooth_step(time, 0.3), 2, axis=1)
            # Direct apply compliance, rather than invert it, to avoid changing
            # the analytic bar's transfer to a polynomial truncation inverse.
            response = np.zeros_like(forcing)
            for index in range(1, steps + 1):
                response[index] = np.einsum("kij,kj->i", weights[: index + 1], forcing[index::-1])
            value = response[:, -1]
            exact = bar_displacement_exact(endpoint, 0)
            record = dict(
                method=method,
                dt=dt,
                endpoint_relative_l2=float(np.linalg.norm(value - exact) / np.linalg.norm(exact)),
                max_absolute_error=float(np.max(abs(value - exact))),
                seconds=perf_counter() - tick,
            )
            records.append(record)
            np.savez(
                output / f"analytic-{method}-{dt:g}.npz", time=endpoint, value=value, exact=exact
            )
    return records


def staged_contact_system(reference, weights):
    """Reuse production local AD/root code with two copies of the contact nodes."""
    from finiteblock_bem.jax_backend import JaxContactEquilibrium

    con = reference.constraints
    staged = SimpleNamespace(
        normal_jump=block_diag(con.normal_jump, con.normal_jump),
        tangential_jump=block_diag(con.tangential_jump, con.tangential_jump),
        contact_weights=np.tile(con.contact_weights, 2),
        reduced_weights=np.tile(con.reduced_weights, 2),
    )
    matrix = weights.copy()
    size = con.size
    matrix[[size - 1, 2 * size - 1], [size - 1, 2 * size - 1]] += reference.spring
    adapter = SimpleNamespace(
        constraints=staged,
        base_platen=reference.base_platen,
        case=reference.case,
        base_dn=np.tile(reference.base_dn, 2),
        base_ds=np.tile(reference.base_ds, 2),
        matrix=matrix,
    )
    return JaxContactEquilibrium(adapter)


def contact_experiment(output, args):
    """Two finite BEM blocks under explicitly accelerated diagnostic loading."""
    import jax.numpy as jnp

    from finiteblock_bem.dynamics import LoadingSchedule
    from finiteblock_bem.nodal import (
        NodalInterfaceEquilibrium,
        NodalInterfaceTransfer,
        initialize_nodal_restart,
        load_nodal_model,
    )
    from finiteblock_bem.nodal_dynamics import plan_nodal_history, simulate_nodal

    model = load_nodal_model(args.case, max_panel_length=args.h)
    lift = initialize_nodal_restart(model, quadrature=args.quadrature)
    d = 2 * len(model.moving_contact) + 1
    n = len(model.moving_contact)
    duration = args.dt * args.steps

    # This is NOT the original 20 mm/s event. The diagnostic must remain explicit.
    def actuator(t):
        return model.reference.restart_actuator + args.increment * t / duration

    schedule = LoadingSchedule(
        lambda t: model.reference.normal_pressure, actuator, "radau-accelerated-diagnostic"
    )
    records = []
    for refinement in args.refinements:
        dt, steps = args.dt / refinement, args.steps * refinement
        for method in ["bdf2", "radau2"]:
            tick = perf_counter()
            if method == "bdf2":
                result = simulate_nodal(
                    model,
                    dt,
                    steps,
                    lift=lift,
                    backend="jax",
                    quadrature=args.quadrature,
                    workers=args.workers,
                    schedule=schedule,
                    max_memory_mb=args.max_memory_mb,
                )
                if result.diagnostics["status"] != "completed_requested_horizon":
                    raise RuntimeError(result.diagnostics["failure"])
                z, c, p, tau, q = (
                    result.plastic_slip,
                    result.cumulative_slip,
                    result.pressure,
                    result.shear,
                    np.c_[result.contact_jumps, result.platen],
                )
                error = result.residual_mpa
                kernel_seconds = result.diagnostics["timing_seconds"]["kernel_build_or_cache_load"]
            else:
                plan = plan_cq(2 * d, dt, steps, max_memory_mb=args.max_memory_mb)
                base_plan, base_peak = plan_nodal_history(
                    model,
                    dt,
                    1,
                    workers=args.workers,
                    quadrature=args.quadrature,
                    max_memory_mb=args.max_memory_mb,
                )
                peak = plan.estimated_peak_bytes + base_peak - base_plan.estimated_peak_bytes
                if peak > args.max_memory_mb * 1024**2:
                    raise ValueError(f"Radau CQ/workspace needs at least {peak / 1024**2:.1f} MiB")
                transfer = NodalInterfaceTransfer(model, quadrature=args.quadrature)
                kernel_tick = perf_counter()
                weights = radau_weights(transfer.evaluate, plan, workers=args.workers)
                kernel_seconds = perf_counter() - kernel_tick
                del transfer
                np.save(output / f"weights-radau2-r{refinement}.npy", weights, allow_pickle=False)
                reference = NodalInterfaceEquilibrium(model, lift, np.zeros((d, d)))
                system = staged_contact_system(reference, weights[0])
                online = OnlineConvolution(BoundaryHistory(plan, weights))
                online.append(np.zeros(2 * d))
                z = np.zeros((steps + 1, n))
                c, p, tau = np.zeros_like(z), np.zeros_like(z), np.zeros_like(z)
                q = np.zeros((steps + 1, d))
                stages = np.zeros((steps + 1, 2 * d))
                z[0], c[0] = lift.state.plastic_slip, lift.state.cumulative_slip
                base = np.r_[reference.base_dn, reference.base_ds, reference.base_platen]
                q[0] = base
                initial = reference.case.contact.evaluate(
                    reference.base_dn, reference.base_ds, lift.state
                )
                p[0], tau[0] = initial.pressure, initial.shear
                error = np.zeros(steps + 1)
                guess = np.zeros(2 * d)
                for index in range(1, steps + 1):
                    forcing = np.tile(reference.base_t, (2, 1))
                    times = (index - 1 + C) * dt
                    forcing[:, -1] += reference.spring * (reference.base_platen - actuator(times))
                    forcing = jnp.asarray(forcing.ravel() + online.past(index))
                    plastic, cumulative = jnp.tile(z[index - 1], 2), jnp.tile(c[index - 1], 2)
                    accepted, _ = system.implicit_displacement(
                        system.parameters, plastic, cumulative, forcing, jnp.asarray(guess)
                    )
                    error[index] = float(
                        np.max(
                            abs(
                                system.residual(
                                    accepted, plastic, cumulative, system.parameters, forcing
                                )
                            )
                        )
                    )
                    if not np.isfinite(error[index]) or error[index] >= 1e-7:
                        raise RuntimeError(f"Radau contact failed at step {index}: {error[index]}")
                    local = np.asarray(
                        system.local_response(accepted, plastic, cumulative, system.parameters)
                    )
                    endpoint = local[n:]
                    p[index], tau[index], z[index], c[index] = endpoint[:, :4].T
                    q[index] = np.asarray(accepted)[d:] + base
                    stages[index] = np.asarray(accepted)
                    online.append(np.asarray(accepted))
                    guess = np.asarray(accepted)
                np.savez(output / f"stages-radau2-r{refinement}.npz", stages=stages)
            np.savez(output / f"contact-{method}-r{refinement}.npz", z=z, c=c, p=p, tau=tau, q=q)
            record = dict(
                method=method,
                refinement=refinement,
                dt=dt,
                steps=steps,
                max_c_mm=float(c[-1].max()),
                max_residual_mpa=float(error.max()),
                cumulative_non_decreasing=bool(np.min(np.diff(c, axis=0)) >= -1e-14),
                kernel_seconds=kernel_seconds,
                total_seconds=perf_counter() - tick,
            )
            records.append(record)
            print(json.dumps(record), flush=True)
    finest = max(args.refinements)
    reference = np.load(output / f"contact-bdf2-r{finest}.npz")
    field_weights = {
        "c": model.constraints.contact_weights,
        "p": model.constraints.contact_weights,
        "tau": model.constraints.contact_weights,
        "q": np.r_[
            model.constraints.contact_weights,
            model.constraints.contact_weights,
            model.constraints.platen_width,
        ],
    }
    for record in records:
        with np.load(output / f"contact-{record['method']}-r{record['refinement']}.npz") as data:
            for field in ["c", "q", "p", "tau"]:
                scale = np.sqrt(field_weights[field])
                denominator = reference[field][-1]
                if field == "q":
                    denominator = denominator - reference[field][0]
                record[f"final_{field}_relative_line_l2_vs_bdf2_r{finest}"] = float(
                    np.linalg.norm((data[field][-1] - reference[field][-1]) * scale)
                    / max(np.linalg.norm(denominator * scale), 1e-30)
                )
    reference.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/radau-experiment")
    parser.add_argument("--bem", action="store_true")
    parser.add_argument(
        "--case", type=Path, default=ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    )
    parser.add_argument("--h", type=float, default=80)
    parser.add_argument("--dt", type=float, default=8e-6)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--refinements", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--increment", type=float, default=2)
    parser.add_argument("--quadrature", type=int, default=12)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("FDBEM_CORES", "1")))
    parser.add_argument("--max-memory-mb", type=float, default=2048)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 64 or args.dt <= 0 or not np.isfinite(args.dt):
        raise ValueError("Short experiment requires 1..64 base steps and positive finite dt")
    if args.increment <= 0 or not np.isfinite(args.increment):
        raise ValueError("Diagnostic increment must be finite and positive")
    if any(r < 1 or r * args.steps > 512 for r in args.refinements):
        raise ValueError("Each refinement requires 1..512 time intervals")
    args.output.mkdir(parents=True, exist_ok=True)
    records = contact_experiment(args.output, args) if args.bem else analytic_bar(args.output)
    write_json(
        args.output / "summary.json",
        dict(
            records=records,
            arguments={
                key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            environment=environment_record(args.case),
            prototype_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope="Experimental RK-CQ; analytic linear transfer or accelerated short contact diagnostic; not original event-16; overlapping jobs invalidate timing comparison",
            constitutive_stages="Same accepted old state at both trial stages; commit only endpoint",
        ),
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

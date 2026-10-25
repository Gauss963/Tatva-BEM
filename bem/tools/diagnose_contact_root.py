"""Solve a captured local contact equation without committing physical state.

This diagnostic compares deterministic initial guesses, trust-region methods
and forcing continuation. Every accepted candidate must satisfy the unchanged
full microstep residual. Finding a root is not a trajectory or stability proof.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, root
from threadpoolctl import threadpool_limits

from finiteblock_bem.friction import ContactState, SlipWeakeningContact
from finiteblock_bem.io import fingerprint, write_json

LOCAL_PATH = Path(__file__).with_name("substepped_contact_experiment.py")
LOCAL = runpy.run_path(str(LOCAL_PATH))


def captured_system(data, substeps):
    """Reconstruct the canonical normal/shear/platen coordinates in the capture."""
    d = len(data["forcing"])
    n, eye = (d - 1) // 2, np.eye(d)
    if d != 2 * n + 1 or data["matrix"].shape != (d, d):
        raise ValueError("Expected a square normal/shear/platen contact equation")
    for key in ["base_dn", "base_ds", "plastic_slip", "cumulative_slip", "contact_weights"]:
        if data[key].shape != (n,):
            raise ValueError(f"Captured contact field has inconsistent shape: {key}")
    for key in ["forcing", "previous_displacement", "failed_displacement", "reduced_weights"]:
        if data[key].shape != (d,):
            raise ValueError(f"Captured equation field has inconsistent shape: {key}")
    if any(
        not np.all(np.isfinite(value))
        for key, value in data.items()
        if key != "recovery_events_json"
    ):
        raise ValueError("Captured numerical data must be finite")
    if np.any(data["contact_weights"] <= 0) or np.any(data["reduced_weights"] <= 0):
        raise ValueError("Captured work weights must be positive")
    constraints = SimpleNamespace(
        size=d,
        normal_jump=eye[:n],
        tangential_jump=eye[n : 2 * n],
        contact_weights=data["contact_weights"],
        reduced_weights=data["reduced_weights"],
        reduce_force=lambda value: value,
    )
    reference = SimpleNamespace(
        constraints=constraints,
        case=SimpleNamespace(contact=SlipWeakeningContact(*data["parameters"])),
        base_dn=data["base_dn"],
        base_ds=data["base_ds"],
        base_platen=0.0,
        base_u=np.zeros(d),
        base_t=data["forcing"],
        spring=0.0,
        matrix=data["matrix"],
        external_traction=lambda _: np.zeros(d),
    )
    system = LOCAL["SubsteppedContactEquilibrium"](reference, substeps=substeps)
    system.previous_displacement = data["previous_displacement"].copy()
    state = ContactState(data["plastic_slip"].copy(), data["cumulative_slip"].copy())
    return system, state


def probe_roots(system, state, data, *, methods, max_nfev=800, tolerance=1e-7):
    """Pure trial evaluations: no solve adapter, q/z/c update or work commit."""
    if max_nfev < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Require positive finite residual tolerance and evaluation budget")
    if not methods or any(m not in ["hybr", "lm", "trf", "forcing", "kinetic"] for m in methods):
        raise ValueError("Unknown or empty list of contact diagnostic methods")
    z, c, p = jnp.asarray(state.plastic_slip), jnp.asarray(state.cumulative_slip), system.parameters
    forcing, previous = jnp.asarray(data["forcing"]), jnp.asarray(data["previous_displacement"])

    def fun(q):
        return np.asarray(system.residual(jnp.asarray(q), z, c, p, forcing, previous))

    def jac(q):
        return np.asarray(system.tangent(jnp.asarray(q), z, c, p, previous))

    starts = dict(previous=data["previous_displacement"], failed=data["failed_displacement"])
    if "previous_previous_displacement" in data:
        starts["linear_predictor"] = 2 * starts["previous"] - data["previous_previous_displacement"]
    records, accepted = [], {}

    def report(name, vector, details):
        finite = bool(np.all(np.isfinite(vector)))
        error = float(np.max(abs(fun(vector)))) if finite else None
        if error is not None and not np.isfinite(error):
            error = None
        good = bool(error is not None and np.isfinite(error) and error < tolerance)
        row = dict(name=name, residual_mpa=error, accepted=good, **details)
        if good:
            response = np.asarray(system.local_response(vector, z, c, p, previous))
            if not np.all(np.isfinite(response)):
                raise ValueError("A candidate root has nonfinite local response")
            row.update(
                minimum_pressure_mpa=float(np.min(response[:, 0])),
                minimum_cumulative_increment_mm=float(
                    np.min(response[:, 3] - state.cumulative_slip)
                ),
                maximum_jump_increment_mm=float(np.max(abs(vector - starts["previous"]))),
                minimum_jacobian_singular_value=float(
                    np.linalg.svd(jac(vector), compute_uv=False)[-1]
                ),
            )
            accepted[name] = np.array(vector, copy=True)
        records.append(row)
        print(json.dumps(row), flush=True)

    for name, initial in starts.items():
        for method in methods:
            if method in ["forcing", "kinetic"]:
                continue
            start = perf_counter()
            try:
                if method == "hybr":
                    answer = root(
                        fun,
                        initial,
                        jac=jac,
                        method="hybr",
                        options=dict(xtol=1e-11, maxfev=max_nfev),
                    )
                else:
                    answer = least_squares(
                        fun,
                        initial,
                        jac=jac,
                        method=method,
                        x_scale="jac",
                        max_nfev=max_nfev,
                        ftol=1e-12,
                        xtol=1e-12,
                        gtol=1e-12,
                    )
                report(
                    f"{method}_{name}",
                    answer.x,
                    dict(
                        reported_success=bool(answer.success),
                        nfev=answer.nfev,
                        seconds=perf_counter() - start,
                        message=str(answer.message),
                    ),
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                records.append(
                    dict(
                        name=f"{method}_{name}",
                        accepted=False,
                        error=str(error),
                        seconds=perf_counter() - start,
                    )
                )
    if "kinetic" in methods:
        start = perf_counter()
        # A constant mu_k problem supplies a deterministic trial guess only.
        # Restore the original parameter/state arguments for the final root.
        kinetic = p.at[2].set(p[3])
        seed, _ = system.base.implicit_displacement(kinetic, z, c, forcing, previous)
        seed = np.asarray(seed)
        seed_error = float(
            np.max(abs(np.asarray(system.base.residual(seed, z, c, kinetic, forcing))))
        )
        if np.all(np.isfinite(seed)) and np.isfinite(seed_error) and seed_error < tolerance:
            for method in ["hybr", "lm"]:
                if method == "hybr":
                    answer = root(
                        fun, seed, jac=jac, method="hybr", options=dict(xtol=1e-11, maxfev=max_nfev)
                    )
                else:
                    answer = least_squares(
                        fun,
                        seed,
                        jac=jac,
                        method="lm",
                        x_scale="jac",
                        max_nfev=max_nfev,
                        ftol=1e-12,
                        xtol=1e-12,
                        gtol=1e-12,
                    )
                report(
                    f"{method}_kinetic_seed",
                    answer.x,
                    dict(
                        reported_success=bool(answer.success),
                        nfev=answer.nfev,
                        seed_residual_mpa=seed_error,
                        seconds=perf_counter() - start,
                        final_parameters_and_state_unchanged=True,
                    ),
                )
        else:
            records.append(
                dict(
                    name="kinetic_seed",
                    accepted=False,
                    seed_residual_mpa=seed_error if np.isfinite(seed_error) else None,
                )
            )
    if "forcing" in methods:
        start = perf_counter()
        seed, fraction, increment = starts["previous"].copy(), 0.0, 0.125
        offset, trials = fun(seed), []
        for _ in range(128):
            target = min(1.0, fraction + increment)
            answer = root(
                lambda q: fun(q) - (1 - target) * offset,
                seed,
                jac=jac,
                method="hybr",
                options=dict(xtol=1e-11, maxfev=max_nfev),
            )
            error = float(np.max(abs(fun(answer.x) - (1 - target) * offset)))
            passed = bool(
                np.all(np.isfinite(answer.x)) and np.isfinite(error) and error < tolerance
            )
            full_error = float(np.max(abs(fun(answer.x))))
            trials.append(
                dict(
                    fraction=target,
                    accepted=passed,
                    blended_residual_mpa=error if np.isfinite(error) else None,
                    full_residual_mpa=full_error if np.isfinite(full_error) else None,
                    nfev=answer.nfev,
                )
            )
            if passed:
                seed, fraction = np.asarray(answer.x), target
                if target == 1:
                    break
                increment = min(0.25, increment * 1.5)
            else:
                increment *= 0.5
                if increment < 1 / 65536:
                    break
        report(
            "forcing_continuation",
            seed,
            dict(reached_fraction=fraction, trials=trials, seconds=perf_counter() - start),
        )
    # Trial evaluations have no right to change the accepted physical state.
    np.testing.assert_array_equal(system.previous_displacement, data["previous_displacement"])
    np.testing.assert_array_equal(state.plastic_slip, data["plastic_slip"])
    np.testing.assert_array_equal(state.cumulative_slip, data["cumulative_slip"])
    if system.accepted_micro_work or system.accepted_integrated_strength_work:
        raise ValueError("Diagnostic unexpectedly committed trial contact work")
    distances = {
        f"{a} / {b}": float(np.max(abs(accepted[a] - accepted[b])))
        for i, a in enumerate(accepted)
        for b in list(accepted)[i + 1 :]
    }
    return records, accepted, distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--substeps", type=int, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["hybr", "lm", "trf", "forcing", "kinetic"],
        default=["hybr", "lm", "trf", "forcing", "kinetic"],
    )
    parser.add_argument("--max-nfev", type=int, default=800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new root diagnostic directory")
    paths = [args.fixture, Path(__file__), LOCAL_PATH]
    frozen = {str(path.resolve()): fingerprint(path) for path in paths}
    with np.load(args.fixture, allow_pickle=False) as source:
        data = {
            k: source[k]
            for k in source.files
            if not k.startswith("accepted_") and k != "recovery_events_json"
        }
    started = perf_counter()
    with threadpool_limits(limits=1):
        system, state = captured_system(data, args.substeps)
        initial_residual = float(
            np.max(
                abs(
                    np.asarray(
                        system.residual(
                            data["failed_displacement"],
                            state.plastic_slip,
                            state.cumulative_slip,
                            system.parameters,
                            data["forcing"],
                            data["previous_displacement"],
                        )
                    )
                )
            )
        )
        records, roots, distances = probe_roots(
            system, state, data, methods=args.methods, max_nfev=args.max_nfev
        )
    if any(fingerprint(path) != digest for path, digest in frozen.items()):
        raise ValueError("Captured equation or diagnostic implementation changed")
    args.output.mkdir(parents=True)
    if roots:
        np.savez_compressed(args.output / "candidate_roots.npz", **roots)
    write_json(
        args.output / "summary.json",
        dict(
            status="found_root_of_captured_equation" if roots else "no_root_recovered",
            substeps=args.substeps,
            captured_failed_residual_mpa=initial_residual,
            acceptance_tolerance_mpa=1e-7,
            records=records,
            accepted_root_distances_mm=distances,
            source_sha256=frozen,
            seconds_including_jit=perf_counter() - started,
            full_original_event_reproduced=False,
            scope="Unchanged captured local equation only; physical state not committed; no CQ restart, root uniqueness or trajectory stability assertion",
        ),
    )


if __name__ == "__main__":
    main()

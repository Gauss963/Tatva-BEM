"""Follow artificial forcing branches of a captured, unchanged stage equation.

Pseudo-arclength correctors can traverse folds of the artificial forcing
parameter. They are diagnostic trial equations only. A target candidate is
accepted solely by the original full residual at lambda=1, never by a curve
corrector or by a solver's success flag. No physical state is committed.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root
from threadpoolctl import threadpool_limits

from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
BASE = runpy.run_path(str(TOOLS / "diagnose_condensed_radau_root.py"))


def trace_forcing_curve(fun, jac, initial, accept, *, max_steps=240):
    """Try both orientations from F(x)=(1-lambda)*F(initial).

    This is a bounded root search, not a proof of existence or completeness.
    Unknowns and the forcing parameter are scaled before the arclength
    constraint. Every corrected point satisfies its own forcing equation.
    """
    initial = np.asarray(initial, dtype=float)
    offset = np.asarray(fun(initial))
    n = len(initial)
    force_scale = max(1.0, float(np.max(abs(offset))))
    xscale = np.maximum(1.0, abs(initial))
    trials, paths = [], []
    if accept(initial):
        return initial.copy(), dict(paths=paths, target_trials=trials)

    def equation(point):
        return (fun(xscale * point[:n]) - (1 - point[-1]) * offset) / force_scale

    def derivative(point):
        return np.column_stack((jac(xscale * point[:n]) * xscale[None, :], offset)) / force_scale

    for orientation in [1, -1]:
        point = np.r_[initial / xscale, 0.0]
        tangent = np.linalg.svd(derivative(point), full_matrices=True)[2][-1]
        tangent *= (1 if tangent[-1] >= 0 else -1) * orientation
        step, path = 0.05, [point.copy()]
        correctors = []
        for _ in range(max_steps):
            prediction = point + step * tangent

            def augmented(value):
                return np.r_[equation(value), tangent @ (value - prediction)]

            def augmented_jac(value):
                return np.vstack((derivative(value), tangent))

            answer = root(
                augmented,
                prediction,
                jac=augmented_jac,
                method="hybr",
                options=dict(xtol=1e-11, maxfev=120),
            )
            error = float(np.max(abs(equation(answer.x))) * force_scale)
            arc_error = float(abs(tangent @ (answer.x - prediction)))
            good = bool(np.all(np.isfinite(answer.x)) and error < 1e-7 and arc_error < 1e-9)
            correctors.append(
                dict(
                    step=step,
                    lambda_trial=float(answer.x[-1]),
                    residual_mpa=error,
                    arclength_error=arc_error,
                    accepted=good,
                    nfev=answer.nfev,
                )
            )
            if not good:
                step *= 0.5
                if step < 1e-5:
                    break
                continue
            old, point = point, answer.x
            path.append(point.copy())
            if (old[-1] - 1) * (point[-1] - 1) <= 0:
                target = root(
                    fun,
                    xscale * point[:n],
                    jac=jac,
                    method="hybr",
                    options=dict(xtol=1e-11, maxfev=400),
                )
                finite = bool(np.all(np.isfinite(target.x)))
                accepted = bool(finite and accept(target.x))
                trials.append(
                    dict(
                        orientation=orientation,
                        vector=target.x.tolist(),
                        residual_mpa=float(np.max(abs(fun(target.x)))) if finite else None,
                        accepted=accepted,
                        nfev=target.nfev,
                    )
                )
                if accepted:
                    paths.append(
                        dict(
                            orientation=orientation,
                            points=np.asarray(path).tolist(),
                            correctors=correctors,
                        )
                    )
                    return np.asarray(target.x), dict(paths=paths, target_trials=trials)
            tangent_next = np.linalg.svd(derivative(point), full_matrices=True)[2][-1]
            if tangent_next @ tangent < 0:
                tangent_next = -tangent_next
            tangent = tangent_next
            if abs(point[-1]) > 2:
                break
            step = min(0.2, step * (1.3 if answer.nfev < 20 else 1.0))
        paths.append(
            dict(orientation=orientation, points=np.asarray(path).tolist(), correctors=correctors)
        )
    return None, dict(paths=paths, target_trials=trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--retain-all-unweakened", action="store_true")
    parser.add_argument("--include-normal", action="store_true")
    parser.add_argument("--random-starts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    if args.max_steps < 1 or args.random_starts < 0:
        raise ValueError("Require positive curve budget and nonnegative random-start budget")
    capture, output = args.capture.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh owned continuation directory")
    failure = json.loads(capture.with_name("failure.json").read_text())
    if BASE["CAPTURE"]["digest"](capture) != failure["failed_equilibrium_sha256"]:
        raise ValueError("Captured equation changed")
    with np.load(capture, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files if not k.startswith("accepted_")}
    paths = [
        Path(__file__),
        TOOLS / "diagnose_condensed_radau_root.py",
        TOOLS / "diagnose_radau_root.py",
        TOOLS / "identity_stage_tangent.py",
        TOOLS / "radau_contact_experiment.py",
        TOOLS / "substepped_contact_experiment.py",
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    output.mkdir(parents=True)
    begun, records, candidates = perf_counter(), [], {}
    with threadpool_limits(limits=1):
        model = BASE["CondensedStage"](data, tuple(failure["segment_substeps"]))
        if args.retain_all_unweakened:
            nodes = np.union1d(
                model.nodes, np.flatnonzero(data["cumulative_slip"] < data["parameters"][-1])
            )
            if not np.array_equal(nodes, model.nodes):
                model = BASE["CondensedStage"](
                    data, tuple(failure["segment_substeps"]), nonlinear_nodes=nodes
                )
        m = len(model.nodes)
        # A closed-contact candidate has zero normal-force correction. Keep
        # only shear corrections in this search; the full gate detects any
        # opening or invalid eliminated branch, including at retained nodes.
        indices = (
            np.arange(4 * m)
            if args.include_normal
            else np.arange(4 * m).reshape(m, 2, 2)[:, :, 1].ravel()
        )

        def embed(vector):
            return jnp.zeros(4 * m).at[indices].set(vector)

        @jax.jit
        def residual(vector):
            return model.residual(embed(vector))[indices]

        @jax.jit
        def tangent(vector):
            return model.tangent(embed(vector))[indices[:, None], indices[None, :]]

        def fun(vector):
            return np.asarray(residual(jnp.asarray(vector)))

        def jac(vector):
            return np.asarray(tangent(jnp.asarray(vector)))

        def accept(vector):
            return model.check_candidate(np.asarray(embed(vector)))["accepted"]

        initial = model.correction_at(data["failed_displacement"])[indices]
        reference_error = model.full_error(data["failed_displacement"])
        np.testing.assert_allclose(reference_error, failure["residual_mpa"], rtol=1e-12, atol=1e-12)
        check = np.asarray(jax.jacfwd(residual)(jnp.asarray(initial)))
        tangent_error = float(
            np.linalg.norm(jac(initial) - check) / max(np.linalg.norm(check), 1e-30)
        )
        if tangent_error > 1e-11:
            raise ValueError("Reduced tangent failed independent AD check")
        write_json(
            output / "manifest.json",
            dict(
                source_sha256=frozen,
                capture_sha256=failure["failed_equilibrium_sha256"],
                nonlinear_nodes=model.nodes.tolist(),
                retained_unknowns=len(indices),
                normal_force_corrections_retained=args.include_normal,
                all_unweakened_nodes_retained=args.retain_all_unweakened,
                random_starts=args.random_starts,
                random_seed=args.seed,
                full_stage_dofs=len(data["forcing"]),
                original_failure_residual_mpa=reference_error,
                reduced_tangent_relative_error=tangent_error,
                linear_backward_error=model.linear_backward_error,
                state_not_committed=True,
            ),
        )
        if args.random_starts:
            rng = np.random.default_rng(args.seed)
            pressure = (
                data["parameters"][0] * model.jumps(data["failed_displacement"])[model.nodes, :, 0]
            )
            scale = np.stack((pressure, data["parameters"][3] * pressure), -1).ravel()[indices]
            scale = np.maximum(1, abs(scale))
            random_records = []
            for index in range(args.random_starts):
                seed = initial + rng.uniform(-1, 1, len(indices)) * scale
                answer = root(
                    fun, seed, jac=jac, method="hybr", options=dict(xtol=1e-11, maxfev=400)
                )
                unknown = np.asarray(embed(answer.x))
                checks = model.check_candidate(unknown)
                random_records.append(
                    dict(
                        index=index,
                        nfev=answer.nfev,
                        reported_success=bool(answer.success),
                        **checks,
                    )
                )
                if checks["accepted"]:
                    candidates["random_hybrid"] = model.expand(unknown)
                    break
            write_json(output / "random-starts.json", dict(seed=args.seed, records=random_records))
        for name, seed in dict(
            failed_correction=initial, zero_correction=np.zeros_like(initial)
        ).items():
            if candidates:
                break
            candidate, trace = trace_forcing_curve(fun, jac, seed, accept, max_steps=args.max_steps)
            write_json(output / f"{name}-trace.json", trace)
            row = dict(
                name=name,
                recovered_candidate=candidate is not None,
                corrected_points=sum(len(p["points"]) - 1 for p in trace["paths"]),
                target_trials=trace["target_trials"],
            )
            if candidate is not None:
                unknown = np.asarray(embed(candidate))
                row.update(model.check_candidate(unknown))
                candidates[name] = model.expand(unknown)
            records.append(row)
            write_json(output / "progress.json", dict(records=records))
            print(json.dumps(row), flush=True)
            if candidate is not None:
                break
    if candidates:
        np.savez(output / "candidate_roots.npz", **candidates)
    if BASE["CAPTURE"]["digest"](capture) != failure["failed_equilibrium_sha256"] or any(
        fingerprint(p) != h for p, h in frozen.items()
    ):
        raise ValueError("Captured equation or continuation sources changed")
    write_json(
        output / "summary.json",
        dict(
            status="found_root_of_original_captured_equation"
            if candidates
            else "no_root_recovered",
            records=records,
            seconds=perf_counter() - begun,
            source_sha256=frozen,
            full_original_event_reproduced=False,
            scope="Artificial-forcing continuation with original full-equation acceptance. No physical state commit, CQ restart, uniqueness or stability proof.",
        ),
    )


if __name__ == "__main__":
    main()

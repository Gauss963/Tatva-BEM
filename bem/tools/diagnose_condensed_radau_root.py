"""Diagnose a captured stage equation by exact linear elimination.

The candidate branch uses closed kinetic sliding at fully weakened outside
nodes and sticking at unweakened outside nodes.
Inside selected nodes the original two-stage local law, including opening,
is evaluated without approximation. Every accepted root must also satisfy
the original full equation; an invalid outside branch is never accepted.
No state, CQ history or trajectory is committed by this diagnostic.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import least_squares, root
from threadpoolctl import threadpool_limits

from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
CAPTURE = runpy.run_path(str(TOOLS / "diagnose_radau_root.py"))


class CondensedStage:
    """Fixed affine bulk elimination, with local AD on the remaining points."""

    def __init__(self, data, counts, *, nonlinear_nodes=None):
        self.data = data
        self.system, _, self.state, _ = CAPTURE["captured_stage"](data, counts)
        n, d = self.system.nodes, self.system.size
        self.z, self.c = jnp.asarray(data["plastic_slip"]), jnp.asarray(data["cumulative_slip"])
        self.previous = jnp.asarray(data["previous_displacement"])
        p = self.system.parameters
        kn, kt, _, muk, dc = np.asarray(p)
        previous = np.stack((data["base_dn"], data["base_ds"]), axis=-1)
        previous += np.stack(
            (data["previous_displacement"][:n], data["previous_displacement"][n : 2 * n]), -1
        )
        if nonlinear_nodes is None:
            q = data["failed_displacement"]
            values = np.asarray(self.system.local_response(q, self.z, self.c, p, self.previous))
            jumps = self.jumps(q)
            closed_pressure = kn * jumps[:, :, 0]
            different = np.any(abs(values[:, :, 0] - closed_pressure) > 1e-9, axis=1)
            baseline_shear = np.where(
                (data["cumulative_slip"] < dc)[:, None],
                kt * (jumps[:, :, 1] - data["plastic_slip"][:, None]),
                muk * closed_pressure,
            )
            different |= np.any(abs(values[:, :, 1] - baseline_shear) > 1e-9, axis=1)
            nonlinear_nodes = np.flatnonzero(different)
            if not len(nonlinear_nodes):
                # A purely affine candidate can also be verified by retaining
                # one local point. No constitutive branch is accepted on trust.
                nonlinear_nodes = np.array([0])
        nodes = np.asarray(nonlinear_nodes, dtype=int)
        if (
            nodes.ndim != 1
            or len(nodes) == 0
            or len(np.unique(nodes)) != len(nodes)
            or np.any((nodes < 0) | (nodes >= n))
        ):
            raise ValueError("Require unique, nonempty nonlinear contact node indices")
        self.nodes = nodes
        selected = np.zeros(n, dtype=bool)
        selected[nodes] = True
        sticking_baseline = selected | (data["cumulative_slip"] < dc)
        matrix = data["matrix"].copy()
        constant = np.zeros((2, d))
        weights = data["contact_weights"]
        for stage in range(2):
            normal = stage * d + np.arange(n)
            shear = normal + n
            matrix[normal, normal] += weights * kn
            matrix[shear[sticking_baseline], shear[sticking_baseline]] += (
                weights[sticking_baseline] * kt
            )
            matrix[shear[~sticking_baseline], normal[~sticking_baseline]] += (
                weights[~sticking_baseline] * muk * kn
            )
            constant[stage, :n] = weights * kn * data["base_dn"]
            constant[stage, n : 2 * n] = np.where(
                sticking_baseline,
                weights * kt * (data["base_ds"] - data["plastic_slip"]),
                weights * muk * kn * data["base_dn"],
            )
        # Unknown corrections are local MPa, ordered node/stage/(p,tau).
        i, a, component = np.indices((len(nodes), 2, 2))
        rows = (a * d + component * n + nodes[i]).ravel()
        injection = np.zeros((2 * d, len(rows)))
        injection[rows, np.arange(len(rows))] = weights[nodes[i]].ravel()
        rhs = np.column_stack((-data["forcing"] - constant.ravel(), injection))
        factor = lu_factor(matrix)
        solved = lu_solve(factor, rhs)
        for _ in range(2):
            solved += lu_solve(factor, rhs - matrix @ solved)
        self.linear_backward_error = float(
            np.linalg.norm(matrix @ solved - rhs) / max(np.linalg.norm(rhs), 1e-30)
        )
        self.q0, self.lift = solved[:, 0].copy(), solved[:, 1:].copy()
        q0, lift = self.q0.reshape(2, d), self.lift.reshape(2, d, len(rows))
        jump0 = self.jumps(q0.ravel())[nodes]
        jump_lift = np.stack((lift[:, nodes], lift[:, n + nodes]), axis=2).transpose(1, 0, 2, 3)
        jump0, jump_lift = jnp.asarray(jump0), jnp.asarray(jump_lift)
        old_z, old_c = self.z[nodes], self.c[nodes]
        previous = jnp.asarray(previous[nodes])
        point = CAPTURE["POLICY"]["BASE"]["staged_point"](counts)
        mapped = jax.vmap(point, in_axes=(0, 0, 0, 0, None))
        differentiated = jax.vmap(jax.jacfwd(point, argnums=0), in_axes=(0, 0, 0, 0, None))
        baseline_derivative = np.zeros((len(nodes), 2, 2, 2, 2))
        for stage in range(2):
            baseline_derivative[:, stage, 0, stage, 0] = kn
            baseline_derivative[:, stage, 1, stage, 1] = kt
        baseline_derivative = jnp.asarray(baseline_derivative)

        def local(unknown):
            jumps = jump0 - jnp.einsum("naik,k->nai", jump_lift, unknown)
            values = mapped(jumps, old_z, old_c, previous, p)
            baseline = jnp.stack((kn * jumps[:, :, 0], kt * (jumps[:, :, 1] - old_z[:, None])), -1)
            return jumps, values[:, :, :2] - baseline

        def residual(unknown):
            return unknown - local(unknown)[1].ravel()

        def tangent(unknown):
            jumps, _ = local(unknown)
            derivative = differentiated(jumps, old_z, old_c, previous, p)[:, :, :2]
            update = jnp.einsum("naibj,nbjk->naik", derivative - baseline_derivative, jump_lift)
            return jnp.eye(len(rows)) + update.reshape(len(rows), len(rows))

        self.residual, self.tangent = jax.jit(residual), jax.jit(tangent)

    def jumps(self, q):
        n, d = self.system.nodes, self.system.size
        stages = np.asarray(q).reshape(2, d)
        return np.stack(
            (
                self.data["base_dn"][:, None] + stages[:, :n].T,
                self.data["base_ds"][:, None] + stages[:, n : 2 * n].T,
            ),
            -1,
        )

    def expand(self, unknown):
        return self.q0 - self.lift @ np.asarray(unknown)

    def correction_at(self, q):
        values = np.asarray(
            self.system.local_response(q, self.z, self.c, self.system.parameters, self.previous)
        )[self.nodes, :, :2]
        jumps = self.jumps(q)[self.nodes]
        kn, kt = np.asarray(self.system.parameters)[:2]
        baseline = np.stack(
            (
                kn * jumps[:, :, 0],
                kt * (jumps[:, :, 1] - self.data["plastic_slip"][self.nodes, None]),
            ),
            -1,
        )
        return (values - baseline).ravel()

    def full_error(self, q):
        value = self.system.residual(
            q, self.z, self.c, self.system.parameters, self.data["forcing"], self.previous
        )
        return float(jnp.max(abs(value)))

    def check_candidate(self, unknown):
        q = self.expand(unknown)
        error = self.full_error(q)
        reduced_error = float(jnp.max(abs(self.residual(jnp.asarray(unknown)))))
        values = np.asarray(
            self.system.local_response(q, self.z, self.c, self.system.parameters, self.previous)
        )
        finite = bool(np.all(np.isfinite(q)) and np.all(np.isfinite(values)))
        return dict(
            reduced_residual_mpa=reduced_error,
            full_residual_mpa=error,
            all_stage_fields_finite=finite,
            accepted=bool(finite and error < 1e-7 and reduced_error < 1e-7),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=400)
    args = parser.parse_args()
    capture, output = args.capture.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh owned diagnostic directory")
    failure = json.loads(capture.with_name("failure.json").read_text())
    if CAPTURE["digest"](capture) != failure["failed_equilibrium_sha256"]:
        raise ValueError("Captured equation changed")
    with np.load(capture, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files if not key.startswith("accepted_")}
    source_paths = [
        Path(__file__),
        TOOLS / "diagnose_radau_root.py",
        TOOLS / "identity_stage_tangent.py",
        TOOLS / "radau_contact_experiment.py",
        TOOLS / "substepped_contact_experiment.py",
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in source_paths}
    output.mkdir(parents=True)
    started, records = perf_counter(), []
    with threadpool_limits(limits=1):
        model = CondensedStage(data, tuple(failure["segment_substeps"]))
        zeros = np.zeros(4 * len(model.nodes))

        def fun(x):
            return np.asarray(model.residual(jnp.asarray(x)))

        def jac(x):
            return np.asarray(model.tangent(jnp.asarray(x)))

        original_error = model.full_error(data["failed_displacement"])
        np.testing.assert_allclose(original_error, failure["residual_mpa"], rtol=1e-12, atol=1e-12)
        independent = np.asarray(jax.jacfwd(model.residual)(jnp.asarray(zeros)))
        tangent_error = float(
            np.linalg.norm(jac(zeros) - independent) / max(np.linalg.norm(independent), 1e-30)
        )
        if tangent_error > 1e-11:
            raise ValueError("Reduced local AD disagrees with independent full residual AD")
        write_json(
            output / "manifest.json",
            dict(
                capture=str(capture),
                capture_sha256=failure["failed_equilibrium_sha256"],
                source_sha256=frozen,
                nonlinear_nodes=model.nodes.tolist(),
                unknowns=len(zeros),
                full_stage_dofs=len(data["forcing"]),
                original_failure_residual_mpa=original_error,
                reduced_tangent_relative_error=tangent_error,
                linear_backward_error=model.linear_backward_error,
                state_not_committed=True,
            ),
        )
        starts = dict(
            zero_correction=zeros,
            failed_correction=model.correction_at(data["failed_displacement"]),
        )
        for name, initial in starts.items():
            for method in ["hybr", "lm", "trf"]:
                begin = perf_counter()
                if method == "hybr":
                    answer = root(
                        fun,
                        initial,
                        jac=jac,
                        method=method,
                        options=dict(xtol=1e-11, maxfev=args.max_nfev),
                    )
                else:
                    answer = least_squares(
                        fun,
                        initial,
                        jac=jac,
                        method=method,
                        x_scale="jac",
                        max_nfev=args.max_nfev,
                        ftol=1e-12,
                        xtol=1e-12,
                        gtol=1e-12,
                    )
                q = model.expand(answer.x)
                checks = model.check_candidate(answer.x)
                label = f"{method}_{name}"
                np.savez(output / f"{label}.npz", unknown=answer.x, displacement=q)
                row = dict(
                    name=label,
                    nfev=answer.nfev,
                    reported_success=bool(answer.success),
                    message=str(answer.message),
                    **checks,
                    seconds=perf_counter() - begin,
                )
                records.append(row)
                write_json(output / "progress.json", dict(records=records))
                print(json.dumps(row), flush=True)
                if checks["accepted"]:
                    break
            if records[-1]["accepted"]:
                break
    if CAPTURE["digest"](capture) != failure["failed_equilibrium_sha256"] or any(
        fingerprint(p) != h for p, h in frozen.items()
    ):
        raise ValueError("Capture or diagnostic sources changed")
    write_json(
        output / "summary.json",
        dict(
            status="found_root_of_original_captured_equation"
            if any(r["accepted"] for r in records)
            else "no_root_recovered",
            records=records,
            seconds=perf_counter() - started,
            source_sha256=frozen,
            full_original_event_reproduced=False,
            scope="Local candidate-branch elimination; all accepted roots satisfy the unchanged full equation. No CQ restart, state commit, uniqueness or stability proof.",
        ),
    )


if __name__ == "__main__":
    main()

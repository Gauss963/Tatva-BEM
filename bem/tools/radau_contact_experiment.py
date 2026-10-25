"""Coupled Radau stage contact on a piecewise-linear jump path.

The first stage is at dt/3. Its trial state feeds the second segment ending
at dt, and only the endpoint may be committed by a time integrator. JAX
differentiates the small local two-stage map and the original equilibrium
root, rather than all Newton iterations. No high-order friction theorem is
inferred from using Radau for the elastic bulk.
"""

import runpy
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag

from finiteblock_bem.jax_backend import JaxContactEquilibrium

TOOLS = Path(__file__).resolve().parent
LOCAL = runpy.run_path(str(TOOLS / "substepped_contact_experiment.py"))
ROBUST = runpy.run_path(str(TOOLS / "robust_contact_experiment.py"))


def staged_point(segment_substeps):
    if len(segment_substeps) != 2 or any(
        not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in segment_substeps
    ):
        raise ValueError("Require two positive integer segment substep counts")
    first, second = [LOCAL["substepped_point"](count) for count in segment_substeps]

    def point(jumps, plastic, cumulative, previous, parameters):
        a = first(jumps[0], plastic, cumulative, previous, parameters)
        b = second(jumps[1], a[2], a[3], jumps[0], parameters)
        return jnp.stack((a, b))

    return point


@dataclass(frozen=True)
class StageRootResult:
    displacement: np.ndarray
    response: np.ndarray
    residual_mpa: float
    iterations: int
    converged: bool
    recovery_attempts: list


class RadauContactEquilibrium:
    """Canonical single-stage kinematics, coupled stage bulk and local states."""

    def __init__(self, reference, matrix, *, segment_substeps=(1, 2)):
        local = staged_point(segment_substeps)
        con = reference.constraints
        d, n = con.size, len(con.contact_weights)
        matrix = np.array(matrix, copy=True)
        if matrix.shape != (2 * d, 2 * d) or not np.all(np.isfinite(matrix)):
            raise ValueError("Invalid coupled Radau equilibrium matrix")
        bn, bt = jnp.asarray(con.normal_jump), jnp.asarray(con.tangential_jump)
        wc, wr = jnp.asarray(con.contact_weights), jnp.tile(jnp.asarray(con.reduced_weights), 2)
        base = jnp.stack((jnp.asarray(reference.base_dn), jnp.asarray(reference.base_ds)), axis=-1)
        bulk = jnp.asarray(matrix)
        # The independent previous-state stage law supplies the unchanged
        # production Newton implementation and optional recovery trial seeds.
        stage_reference = SimpleNamespace(
            base_platen=reference.base_platen,
            case=reference.case,
            matrix=matrix,
            base_dn=np.tile(reference.base_dn, 2),
            base_ds=np.tile(reference.base_ds, 2),
            constraints=SimpleNamespace(
                normal_jump=block_diag(np.asarray(bn), np.asarray(bn)),
                tangential_jump=block_diag(np.asarray(bt), np.asarray(bt)),
                contact_weights=np.tile(con.contact_weights, 2),
                reduced_weights=np.tile(con.reduced_weights, 2),
            ),
        )
        independent = JaxContactEquilibrium(stage_reference)
        self.parameters = independent.parameters
        self.size, self.nodes = d, n
        self.segment_substeps = tuple(segment_substeps)

        def previous_jumps(previous):
            return base + jnp.stack((bn @ previous, bt @ previous), axis=-1)

        def jumps(q):
            stages = q.reshape(2, d)
            return base[:, None, :] + jnp.stack((stages @ bn.T, stages @ bt.T), axis=-1).transpose(
                1, 0, 2
            )

        mapped = jax.vmap(local, in_axes=(0, 0, 0, 0, None))
        differentiated = jax.vmap(jax.jacfwd(local, argnums=0), in_axes=(0, 0, 0, 0, None))

        def values(q, z, c, p, previous):
            return mapped(jumps(q), z, c, previous_jumps(previous), p)

        def residual(q, z, c, p, forcing, previous):
            response = values(q, z, c, p, previous)
            contact = [
                bn.T @ (wc * response[:, a, 0]) + bt.T @ (wc * response[:, a, 1]) for a in range(2)
            ]
            return (bulk @ q + forcing + jnp.concatenate(contact)) / wr

        def tangent(q, z, c, p, previous):
            derivative = differentiated(jumps(q), z, c, previous_jumps(previous), p)
            blocks = []
            for a in range(2):
                row = []
                for b in range(2):
                    v = derivative[:, a, :, b, :]
                    row.append(
                        bn.T @ ((wc * v[:, 0, 0])[:, None] * bn)
                        + bn.T @ ((wc * v[:, 0, 1])[:, None] * bt)
                        + bt.T @ ((wc * v[:, 1, 0])[:, None] * bn)
                        + bt.T @ ((wc * v[:, 1, 1])[:, None] * bt)
                    )
                blocks.append(jnp.concatenate(row, axis=1))
            return (bulk + jnp.concatenate(blocks, axis=0)) / wr[:, None]

        original = independent.implicit_displacement.__wrapped__
        if set(original.__code__.co_freevars) != {"residual", "tangent"}:
            raise ValueError("Production root closures changed; re-audit the stage adapter")

        def cell(value):
            return (lambda: value).__closure__[0]

        def root(parameters, plastic, cumulative, forcing, guess, previous, tolerance=1e-7):
            bindings = dict(
                residual=lambda q, z, c, p, f: residual(q, z, c, p, f, previous),
                tangent=lambda q, z, c, p: tangent(q, z, c, p, previous),
            )
            clone = FunctionType(
                original.__code__,
                original.__globals__,
                original.__name__,
                original.__defaults__,
                tuple(cell(bindings[name]) for name in original.__code__.co_freevars),
            )
            return clone(parameters, plastic, cumulative, forcing, guess, tolerance)

        self.local_response, self.residual, self.tangent = (
            jax.jit(values),
            jax.jit(residual),
            jax.jit(tangent),
        )
        self.implicit_displacement = jax.jit(root)
        # Existing recovery routines receive old z/c once, rather than two
        # independently committed stage states. Only this seed law tiles them.
        self.base = SimpleNamespace(
            residual=jax.jit(
                lambda q, z, c, p, f: independent.residual(q, jnp.tile(z, 2), jnp.tile(c, 2), p, f)
            ),
            tangent=jax.jit(
                lambda q, z, c, p: independent.tangent(q, jnp.tile(z, 2), jnp.tile(c, 2), p)
            ),
            implicit_displacement=jax.jit(
                lambda p, z, c, f, g: independent.implicit_displacement(
                    p, jnp.tile(z, 2), jnp.tile(c, 2), f, g
                )
            ),
        )

    def solve(self, old, forcing, guess, previous, *, recover=False, kinetic_seed=False):
        if kinetic_seed and not recover:
            raise ValueError("A kinetic trial seed requires explicit root recovery")
        forcing, guess, previous = [np.asarray(v) for v in [forcing, guess, previous]]
        if (
            forcing.shape != (2 * self.size,)
            or guess.shape != forcing.shape
            or previous.shape != (self.size,)
            or old.plastic_slip.shape != (self.nodes,)
            or not all(np.all(np.isfinite(v)) for v in [forcing, guess, previous])
        ):
            raise ValueError("Invalid stage forcing, displacement or accepted contact state")
        z, c = jnp.asarray(old.plastic_slip), jnp.asarray(old.cumulative_slip)
        q, count = self.implicit_displacement(self.parameters, z, c, forcing, guess, previous)
        q = np.asarray(q)

        def error(vector):
            return float(
                jnp.max(abs(self.residual(vector, z, c, self.parameters, forcing, previous)))
            )

        attempts = []
        value = error(q)
        if (not np.isfinite(value) or value >= 1e-7) and recover:
            view = SimpleNamespace(
                parameters=self.parameters,
                previous_displacement=previous,
                base=self.base,
                residual=self.residual,
                tangent=self.tangent,
            )
            candidate, trials = ROBUST["recover_root"](view, old, forcing, q)
            attempts.extend(trials)
            if candidate is None and kinetic_seed:
                candidate, trials = ROBUST["kinetic_seed_root"](view, old, forcing, q)
                attempts.extend(trials)
            if candidate is not None:
                q, _ = self.implicit_displacement(
                    self.parameters, z, c, forcing, candidate, previous
                )
                q, value = np.asarray(q), error(q)
        response = np.asarray(self.local_response(q, z, c, self.parameters, previous))
        passed = bool(
            np.all(np.isfinite(q))
            and np.all(np.isfinite(response))
            and np.isfinite(value)
            and value < 1e-7
        )
        return StageRootResult(q, response, value, int(count) + 1, passed, attempts)

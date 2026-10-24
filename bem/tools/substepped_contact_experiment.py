"""Previous-state contact updates on a linear path inside one bulk time step.

Each local substep calls the unchanged production contact_point. Bulk CQ stays
on its original time grid; only the constitutive path is refined. A selected
path is not a proof that bulk wave error is resolved. The implicit root keeps
the production Newton/line-search implementation through private closures.
"""

from types import FunctionType

import jax
import jax.numpy as jnp
import numpy as np

from finiteblock_bem.friction import ContactResponse, ContactState
from finiteblock_bem.jax_backend import JaxContactEquilibrium, contact_point
from finiteblock_bem.nonlinear import NewtonResult


def substepped_point(substeps):
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError("A positive integer constitutive substep count is required")

    def primitive(cumulative, parameters):
        _, _, mus, muk, dc = parameters
        weakening = jnp.minimum(cumulative, dc)
        return (
            mus * weakening
            - (mus - muk) * weakening**2 / (2 * dc)
            + muk * jnp.maximum(cumulative - dc, 0)
        )

    def point(jump, plastic, cumulative, previous_jump, parameters):
        if substeps == 1:
            response = contact_point(jump, plastic, cumulative, parameters)
            work = response[1] * (response[2] - plastic)
            integrated = response[0] * (
                primitive(response[3], parameters) - primitive(cumulative, parameters)
            )
            return jnp.r_[response, work, integrated]

        def update(carry, index):
            old_z, old_c, work, integrated, _ = carry
            local_jump = previous_jump + (index + 1) / substeps * (jump - previous_jump)
            response = contact_point(local_jump, old_z, old_c, parameters)
            work += response[1] * (response[2] - old_z)
            integrated += response[0] * (
                primitive(response[3], parameters) - primitive(old_c, parameters)
            )
            return (response[2], response[3], work, integrated, response), None

        zero = jnp.zeros_like(plastic)
        initial = (plastic, cumulative, zero, zero, jnp.zeros(5, dtype=jump.dtype))
        final, _ = jax.lax.scan(update, initial, jnp.arange(substeps))
        return jnp.r_[final[4], final[2], final[3]]

    return point


def _cell(value):
    return (lambda: value).__closure__[0]


class SubsteppedContactEquilibrium:
    """Local AD through all substeps; implicit AD bypasses Newton iterations.

    The explicit differentiable API includes previous displacement as an argument.
    The solve adapter commits that displacement only after a converged root. Full
    trajectory AD must propagate its cotangent as well as old plastic/cumulative
    state; the ordinary production adjoint does not include this extra argument.
    """

    def __init__(self, reference, *, substeps=1):
        local = substepped_point(substeps)
        self.substeps = substeps
        self.reference = reference
        self.base = JaxContactEquilibrium(reference)
        self.parameters = self.base.parameters
        con = reference.constraints
        bn, bt = jnp.asarray(con.normal_jump), jnp.asarray(con.tangential_jump)
        wc, wr = jnp.asarray(con.contact_weights), jnp.asarray(con.reduced_weights)
        matrix = jnp.asarray(reference.matrix)
        base = jnp.stack((jnp.asarray(reference.base_dn), jnp.asarray(reference.base_ds)), axis=1)
        mapped = jax.vmap(local, in_axes=(0, 0, 0, 0, None))
        differentiated = jax.vmap(jax.jacfwd(local, argnums=0), in_axes=(0, 0, 0, 0, None))

        def jumps(q):
            return base + jnp.stack((bn @ q, bt @ q), axis=1)

        def values(q, plastic, cumulative, parameters, previous):
            return mapped(jumps(q), plastic, cumulative, jumps(previous), parameters)

        def residual(q, plastic, cumulative, parameters, forcing, previous):
            response = values(q, plastic, cumulative, parameters, previous)
            return (
                matrix @ q + forcing + bn.T @ (wc * response[:, 0]) + bt.T @ (wc * response[:, 1])
            ) / wr

        def tangent(q, plastic, cumulative, parameters, previous):
            d = differentiated(jumps(q), plastic, cumulative, jumps(previous), parameters)
            return (
                matrix
                + bn.T @ ((wc * d[:, 0, 0])[:, None] * bn)
                + bn.T @ ((wc * d[:, 0, 1])[:, None] * bt)
                + bt.T @ ((wc * d[:, 1, 0])[:, None] * bn)
                + bt.T @ ((wc * d[:, 1, 1])[:, None] * bt)
            ) / wr[:, None]

        original = self.base.implicit_displacement.__wrapped__
        if set(original.__code__.co_freevars) != {"residual", "tangent"}:
            raise ValueError("Production root closures changed; re-audit the substep adapter")

        def root(parameters, plastic, cumulative, forcing, guess, previous, tolerance=1e-7):
            def residual_adapter(q, z, c, p, f):
                return residual(q, z, c, p, f, previous)

            def tangent_adapter(q, z, c, p):
                return tangent(q, z, c, p, previous)

            bindings = {"residual": residual_adapter, "tangent": tangent_adapter}
            clone = FunctionType(
                original.__code__,
                original.__globals__,
                original.__name__,
                original.__defaults__,
                tuple(_cell(bindings[name]) for name in original.__code__.co_freevars),
            )
            return clone(parameters, plastic, cumulative, forcing, guess, tolerance)

        self.local_response = jax.jit(values)
        self.response_tangents = jax.jit(
            lambda q, z, c, p, previous: differentiated(jumps(q), z, c, jumps(previous), p)
        )
        self.residual = jax.jit(residual)
        self.tangent = jax.jit(tangent)
        self.implicit_displacement = jax.jit(root)
        self.previous_displacement = np.zeros(con.size)
        self.accepted_micro_work = []
        self.accepted_integrated_strength_work = []

    def solve(self, normal_pressure, actuator, old, *, past=None, guess=None):
        con = self.reference.constraints
        if self.substeps == 1:
            answer = self.base.solve(normal_pressure, actuator, old, past=past, guess=guess)
        else:
            past = np.zeros(len(self.reference.base_u)) if past is None else past
            guess = np.zeros(con.size) if guess is None else guess
            forcing = self.base.forcing(normal_pressure, actuator, past)
            z, c = jnp.asarray(old.plastic_slip), jnp.asarray(old.cumulative_slip)
            previous = jnp.asarray(self.previous_displacement)
            q, iterations = self.implicit_displacement(
                self.parameters, z, c, forcing, jnp.asarray(guess), previous
            )
            q = np.asarray(q)
            error = float(jnp.max(abs(self.residual(q, z, c, self.parameters, forcing, previous))))
            values = np.asarray(self.local_response(q, z, c, self.parameters, previous))
            local = np.asarray(self.response_tangents(q, z, c, self.parameters, previous))
            contact = ContactResponse(
                values[:, 0],
                values[:, 1],
                local[:, 0, 0],
                local[:, 1, 0],
                local[:, 1, 1],
                values[:, 4],
                ContactState(values[:, 2], values[:, 3]),
            )
            answer = NewtonResult(
                q,
                contact,
                error,
                int(iterations) + 1,
                bool(np.all(np.isfinite(q)) and error < 1e-7),
            )
        if answer.converged:
            response = np.asarray(
                self.local_response(
                    answer.reduced_displacement,
                    old.plastic_slip,
                    old.cumulative_slip,
                    self.parameters,
                    self.previous_displacement,
                )
            )
            self.accepted_micro_work.append(response[:, 5].copy())
            self.accepted_integrated_strength_work.append(response[:, 6].copy())
            self.previous_displacement = answer.reduced_displacement.copy()
        return answer

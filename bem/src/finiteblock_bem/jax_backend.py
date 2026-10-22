"""JIT contact equilibrium with local AD tangents and implicit root gradients.

Bulk CQ matrices are fixed inputs built with SciPy. Gradients here cover the
contact parameters, old interface state and boundary forcing at one step.
They do not include E/rho/geometry derivatives through those offline matrices.
At opening/yield/weakening switches only the selected branch is differentiated.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .friction import ContactResponse, ContactState
from .nonlinear import NewtonResult

jax.config.update("jax_enable_x64", True)


def contact_point(jump, plastic, cumulative, parameters):
    """Pure local return map; parameters=(kn, kt, mu_s, mu_k, Dc)."""
    kn, kt, mus, muk, dc = parameters
    dn, ds = jump
    active = dn > 0
    pressure = jnp.where(active, kn * dn, 0.0)
    mu = mus - (mus - muk) * jnp.minimum(cumulative / dc, 1.0)
    trial = kt * (ds - plastic)
    sliding = active & (jnp.abs(trial) > pressure * mu)
    shear = jnp.where(active, jnp.where(sliding, jnp.sign(trial) * pressure * mu, trial), 0.0)
    increment = jnp.where(sliding, (trial - shear) / kt, 0.0)
    return jnp.array([pressure, shear, plastic + increment, cumulative + jnp.abs(increment), mu])


contact_values = jax.vmap(contact_point, in_axes=(0, 0, 0, None))
contact_tangents = jax.vmap(jax.jacfwd(contact_point, argnums=0), in_axes=(0, 0, 0, None))


class JaxContactEquilibrium:
    """Same primal interface as ContactEquilibrium, with a differentiable root API."""

    def __init__(self, reference, *, contact_law=contact_point, parameters=None):
        """Optionally accept a pure JAX local law with the contact_point signature.

        The five outputs remain (p, tau, plastic_new, cumulative_new, mu).
        Richer state vectors (e.g. RSF ageing state) need a separate adapter.
        """
        self.reference = reference
        self.base_platen = reference.base_platen
        law = reference.case.contact
        if law.weakening_evaluation != "previous":
            raise ValueError("JAX backend currently supports previous-state weakening")
        self.parameters = jnp.array(
            [
                law.normal_penalty,
                law.tangential_penalty,
                law.static_friction,
                law.kinetic_friction,
                law.characteristic_slip,
            ]
        )
        if parameters is not None:
            self.parameters = jnp.asarray(parameters)
        local_values = jax.vmap(contact_law, in_axes=(0, 0, 0, None))
        local_tangents = jax.vmap(jax.jacfwd(contact_law, argnums=0), in_axes=(0, 0, 0, None))
        con = reference.constraints
        bn, bt = jnp.array(con.normal_jump), jnp.array(con.tangential_jump)
        wc, wr = jnp.array(con.contact_weights), jnp.array(con.reduced_weights)
        matrix = jnp.array(reference.matrix)
        base_dn, base_ds = jnp.array(reference.base_dn), jnp.array(reference.base_ds)

        def values(z, plastic, cumulative, parameters):
            jumps = jnp.stack((base_dn + bn @ z, base_ds + bt @ z), axis=1)
            return local_values(jumps, plastic, cumulative, parameters)

        def residual(z, plastic, cumulative, parameters, forcing):
            response = values(z, plastic, cumulative, parameters)
            return (
                matrix @ z + forcing + bn.T @ (wc * response[:, 0]) + bt.T @ (wc * response[:, 1])
            ) / wr

        def tangent(z, plastic, cumulative, parameters):
            jumps = jnp.stack((base_dn + bn @ z, base_ds + bt @ z), axis=1)
            d = local_tangents(jumps, plastic, cumulative, parameters)
            # Differentiate only independent 2-component local maps, then assemble.
            # Avoid a full dense jacfwd through the linear bulk matrix multiply.
            return (
                matrix
                + bn.T @ ((wc * d[:, 0, 0])[:, None] * bn)
                + bn.T @ ((wc * d[:, 0, 1])[:, None] * bt)
                + bt.T @ ((wc * d[:, 1, 0])[:, None] * bn)
                + bt.T @ ((wc * d[:, 1, 1])[:, None] * bt)
            ) / wr[:, None]

        def root(parameters, plastic, cumulative, forcing, guess, tolerance=1e-7):
            def f(z):
                return residual(z, plastic, cumulative, parameters, forcing)

            def solve(f, initial):
                def condition(carry):
                    _, error, iteration, accepted = carry
                    return (error >= tolerance) & (iteration < 50) & accepted

                def iteration(carry):
                    z, _, count, _ = carry
                    direction = jnp.linalg.solve(tangent(z, plastic, cumulative, parameters), -f(z))
                    merit = jnp.linalg.norm(f(z))

                    def line_condition(state):
                        alpha, trial_merit, trial_error, tries = state
                        return (
                            (trial_merit >= (1 - 1e-4 * alpha) * merit)
                            & (trial_error >= tolerance)
                            & (tries < 24)
                        )

                    def line_iteration(state):
                        alpha, _, _, tries = state
                        alpha = alpha * 0.5
                        r = f(z + alpha * direction)
                        return alpha, jnp.linalg.norm(r), jnp.max(jnp.abs(r)), tries + 1

                    r = f(z + direction)
                    alpha, trial_merit, trial_error, _ = jax.lax.while_loop(
                        line_condition,
                        line_iteration,
                        (1.0, jnp.linalg.norm(r), jnp.max(jnp.abs(r)), 0),
                    )
                    accepted = (
                        (trial_merit < (1 - 1e-4 * alpha) * merit) | (trial_error < tolerance)
                    ) & jnp.isfinite(trial_error)
                    return (
                        jnp.where(accepted, z + alpha * direction, z),
                        trial_error,
                        count + 1,
                        accepted,
                    )

                z, _, count, _ = jax.lax.while_loop(
                    condition, iteration, (initial, jnp.max(jnp.abs(f(initial))), 0.0, True)
                )
                return z, count

            def tangent_solve(g, y):
                return jnp.linalg.solve(jax.jacfwd(g)(jnp.zeros_like(y)), y)

            return jax.lax.custom_root(f, guess, solve, tangent_solve, has_aux=True)

        self.implicit_displacement = jax.jit(root)
        self.local_response = jax.jit(values)
        self.response_tangents = jax.jit(
            lambda z, plastic, cumulative, parameters: local_tangents(
                jnp.stack((base_dn + bn @ z, base_ds + bt @ z), axis=1),
                plastic,
                cumulative,
                parameters,
            )
        )
        self.residual = jax.jit(residual)
        self.tangent = jax.jit(tangent)

    def forcing(self, normal_pressure, actuator, past):
        ref, con = self.reference, self.reference.constraints
        force = con.reduce_force(ref.base_t + past - ref.external_traction(normal_pressure))
        force[-1] += ref.spring * (self.base_platen - actuator)
        return jnp.array(force)

    def solve(self, normal_pressure, actuator, old, *, past=None, guess=None):
        con = self.reference.constraints
        past = np.zeros(len(self.reference.base_u)) if past is None else past
        guess = np.zeros(con.size) if guess is None else guess
        forcing = self.forcing(normal_pressure, actuator, past)
        plastic, cumulative = jnp.array(old.plastic_slip), jnp.array(old.cumulative_slip)
        z, iterations = self.implicit_displacement(
            self.parameters, plastic, cumulative, forcing, jnp.array(guess)
        )
        # NumPy conversion synchronizes dispatch; timings include actual execution.
        z = np.asarray(z)
        error = float(
            jnp.max(jnp.abs(self.residual(z, plastic, cumulative, self.parameters, forcing)))
        )
        contact = self.contact_response(z, old)
        return NewtonResult(
            z, contact, error, int(iterations) + 1, bool(np.isfinite(z).all() and error < 1e-7)
        )

    def contact_response(self, z, old):
        plastic, cumulative = jnp.array(old.plastic_slip), jnp.array(old.cumulative_slip)
        response = np.asarray(self.local_response(z, plastic, cumulative, self.parameters))
        local = np.asarray(self.response_tangents(z, plastic, cumulative, self.parameters))
        return ContactResponse(
            response[:, 0],
            response[:, 1],
            local[:, 0, 0],
            local[:, 1, 0],
            local[:, 1, 1],
            response[:, 4],
            ContactState(response[:, 2], response[:, 3]),
        )

    def evaluate(self, z, past, normal_pressure, actuator, old, *, frozen=False, jacobian=True):
        if frozen:
            raise ValueError(
                "JAX dynamic backend does not perform frozen-plastic static initialization"
            )
        force = self.forcing(normal_pressure, actuator, past)
        plastic, cumulative = jnp.array(old.plastic_slip), jnp.array(old.cumulative_slip)
        residual = np.asarray(self.residual(z, plastic, cumulative, self.parameters, force))
        tangent = (
            np.asarray(self.tangent(z, plastic, cumulative, self.parameters)) if jacobian else None
        )
        return residual, tangent, self.contact_response(z, old)


def differentiable_rollout(system, reduced_weights, forcing, plastic, cumulative):
    """Return a JIT trajectory function of contact parameters, with fixed CQ/lift.

    This bounded reference scan propagates gradients through the old contact
    state AND all past displacements. It uses quadratic direct history and is
    intended for gradient verification, not production-length inverse problems.
    The caller must check returned residuals before using any gradients.
    """
    weights, force = jnp.asarray(reduced_weights), jnp.asarray(forcing)
    steps, size = force.shape
    if steps > 256 or weights.shape != (steps + 1, size, size):
        raise ValueError("Differentiable reference rollout needs <=256 steps and matching weights")
    plastic, cumulative = jnp.asarray(plastic), jnp.asarray(cumulative)
    indices = jnp.arange(steps + 1)

    def rollout(parameters):
        def step(carry, n):
            history, old_z, old_c, guess = carry
            lag = jnp.clip(n - indices, 0, steps)
            past = jnp.einsum(
                "kij,kj->i", weights[lag], jnp.where((indices < n)[:, None], history, 0.0)
            )
            rhs = force[n - 1] + past
            z, _ = system.implicit_displacement(parameters, old_z, old_c, rhs, guess)
            response = system.local_response(z, old_z, old_c, parameters)
            error = jnp.max(jnp.abs(system.residual(z, old_z, old_c, parameters, rhs)))
            return (history.at[n].set(z), response[:, 2], response[:, 3], z), (z, response, error)

        initial = (jnp.zeros((steps + 1, size)), plastic, cumulative, jnp.zeros(size))
        # Recompute per-step intermediates in reverse mode instead of storing Newton traces.
        _, trajectory = jax.lax.scan(jax.checkpoint(step), initial, jnp.arange(1, steps + 1))
        return trajectory

    return jax.jit(rollout)

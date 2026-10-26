"""Implicit nonlinear contact root using the guarded two-block Cinf structure.

Only the instantaneous part is solved here. `forcing` must include the complete
retarded, external and lifted forces supplied by a future dynamic integrator.
This module does not generate those histories or replace Z(s) by s*Cinf.
Displacements are perturbations of `base_jump`, avoiding large restart values
in the radiation difference. Plastic/cumulative state is committed by callers
only after a converged root; Newton trials always use the same old state.
"""

import jax
import jax.numpy as jnp
import numpy as np

from finiteblock_bem.jax_backend import contact_point


class RadiationContactRoot:
    """Damped semismooth Newton with implicit JVP/VJP, in nodal-force balance.

    `dense_reference` enables an independent dense/autojacobian test path.
    Structured mode forms only local 2x2 point tangents. Its root derivative
    recovers those tangents from two linearized matvec probes, then solves the
    full represented nonsymmetric Jacobian and its transpose in O(n) work.
    """

    def __init__(
        self,
        chain,
        contact_weights,
        *,
        platen_width,
        tolerance=1e-7,
        max_iterations=50,
        dense_reference=None,
    ):
        weights = np.asarray(contact_weights, dtype=float)
        n = chain.nodes
        if (
            weights.shape != (n,)
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
            or not np.isfinite(platen_width)
            or platen_width <= 0
        ):
            raise ValueError("Require positive finite contact weights and platen width")
        if (
            not np.isfinite(tolerance)
            or tolerance <= 0
            or not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool)
            or max_iterations < 0
        ):
            raise ValueError("Require positive finite tolerance and a nonnegative iteration cap")
        if dense_reference is not None:
            dense_reference = np.asarray(dense_reference, dtype=float)
            if dense_reference.shape != (2 * n + 1, 2 * n + 1) or not np.all(
                np.isfinite(dense_reference)
            ):
                raise ValueError("Dense reference must be a finite matrix in jump/platen order")
        self.chain, self.tolerance = chain, tolerance
        self.dense_reference_mode = dense_reference is not None
        self.max_iterations = max_iterations
        wc = jnp.asarray(weights)
        wr = jnp.r_[wc, wc, platen_width]
        diag, off, border = (
            jnp.asarray(v) for v in (chain.diagonal, chain.off_diagonal, chain.border)
        )
        ids = chain.border_indices
        dense = None if dense_reference is None else jnp.asarray(dense_reference)
        linear_solve = chain.jax_solver()
        # A custom_root tangent solver must be linear in its tangent RHS.
        # Primal input validation and Newton-direction checks remain enabled.
        tangent_linear_solve = chain.jax_solver(check_rhs_finite=False)
        point_values = jax.vmap(contact_point, in_axes=(0, 0, 0, None))
        point_derivatives = jax.vmap(jax.jacfwd(contact_point, argnums=0), in_axes=(0, 0, 0, None))

        def bulk(q):
            if dense is not None:
                return dense @ q
            blocks = q[:-1].reshape(2, n)
            value = diag * blocks
            value = value.at[:, :-1].add(off * blocks[:, 1:])
            value = value.at[:, 1:].add(off * blocks[:, :-1])
            result = jnp.r_[value.reshape(-1), chain.platen * q[-1]]
            return result.at[ids].add(border @ q[ids])

        def jumps(q, base):
            return base + jnp.stack((q[:n], q[n : 2 * n]), axis=1)

        def values(q, plastic, cumulative, base, parameters):
            result = point_values(jumps(q, base), plastic, cumulative, parameters)
            # The production unilateral law can mask NaN jumps in an open
            # branch. Never expose a finite-looking state after a failed root.
            return jnp.where(jnp.all(jnp.isfinite(q)), result, jnp.full_like(result, jnp.nan))

        def forces(
            q, parameters, plastic, cumulative, base, forcing, previous, coefficient, spring
        ):
            response = values(q, plastic, cumulative, base, parameters)
            result = coefficient * bulk(q - previous) + forcing
            result = result.at[:n].add(wc * response[:, 0])
            result = result.at[n : 2 * n].add(wc * response[:, 1])
            return result.at[-1].add(spring * q[-1])

        # Conservative row-sum bound for cancellation in g(v)-known_bulk(v).
        rows = np.r_[abs(chain.diagonal).reshape(-1), abs(chain.platen)]
        for component in range(2):
            row = rows[component * n : (component + 1) * n]
            row[:-1] += abs(chain.off_diagonal[component])
            row[1:] += abs(chain.off_diagonal[component])
        rows[ids] += np.sum(abs(chain.border), axis=1)
        row_bound = jnp.asarray(rows)
        normal_probe = jnp.r_[jnp.ones(n), jnp.zeros(n + 1)]
        tangent_probe = jnp.r_[jnp.zeros(n), jnp.ones(n), 0.0]

        def root(
            parameters, plastic, cumulative, base, forcing, previous, guess, coefficient, spring
        ):
            if (
                parameters.shape != (5,)
                or plastic.shape != (n,)
                or cumulative.shape != (n,)
                or base.shape != (n, 2)
                or forcing.shape != (2 * n + 1,)
                or previous.shape != (2 * n + 1,)
                or guess.shape != (2 * n + 1,)
            ):
                raise ValueError(
                    "Root inputs must align with the contact nodes and jump/platen coordinates"
                )
            admissible = (
                jnp.all(jnp.isfinite(parameters))
                & jnp.all(parameters[:2] > 0)
                & (parameters[4] > 0)
                & (parameters[2] >= parameters[3])
                & (parameters[3] >= 0)
                & jnp.all(jnp.isfinite(plastic))
                & jnp.all(jnp.isfinite(cumulative))
                & jnp.all(cumulative >= 0)
                & jnp.all(jnp.isfinite(base))
                & jnp.all(jnp.isfinite(forcing))
                & jnp.all(jnp.isfinite(previous))
                & jnp.all(jnp.isfinite(guess))
                & jnp.isfinite(coefficient)
                & (coefficient > 0)
                & jnp.isfinite(spring)
                & (spring >= 0)
            )

            def f(q):
                return (
                    forces(
                        q,
                        parameters,
                        plastic,
                        cumulative,
                        base,
                        forcing,
                        previous,
                        coefficient,
                        spring,
                    )
                    / wr
                )

            def direction(q):
                if dense is not None:
                    return jnp.linalg.solve(jax.jacfwd(f)(q), -f(q))
                local = point_derivatives(jumps(q, base), plastic, cumulative, parameters)[:, :2]
                return linear_solve(-wr * f(q), coefficient, wc[:, None, None] * local, spring)

            def solve(fun, initial):
                def condition(carry):
                    _, error, count, accepted = carry
                    return admissible & (error >= tolerance) & (count < max_iterations) & accepted

                def iteration(carry):
                    q, _, count, _ = carry
                    step = direction(q)
                    merit = jnp.linalg.norm(fun(q))

                    def line_search(delta):
                        def line_condition(state):
                            alpha, trial_merit, error, tries = state
                            return (
                                (trial_merit >= (1 - 1e-4 * alpha) * merit)
                                & (error >= tolerance)
                                & (tries < 24)
                            )

                        def halve(state):
                            alpha, _, _, tries = state
                            alpha *= 0.5
                            residual = fun(q + alpha * delta)
                            return (
                                alpha,
                                jnp.linalg.norm(residual),
                                jnp.max(abs(residual)),
                                tries + 1,
                            )

                        residual = fun(q + delta)
                        alpha, trial_merit, error, _ = jax.lax.while_loop(
                            line_condition,
                            halve,
                            (1.0, jnp.linalg.norm(residual), jnp.max(abs(residual)), 0),
                        )
                        accepted = (
                            (trial_merit < (1 - 1e-4 * alpha) * merit) | (error < tolerance)
                        ) & jnp.isfinite(error)
                        return q + alpha * delta, error, accepted

                    trial, error, accepted = line_search(step)

                    def lookahead(_):
                        return line_search(step + direction(q + step))

                    trial, error, accepted = jax.lax.cond(
                        accepted, lambda _: (trial, error, accepted), lookahead, None
                    )
                    return jnp.where(accepted, trial, q), error, count + 1, accepted

                q, _, count, _ = jax.lax.while_loop(
                    condition, iteration, (initial, jnp.max(abs(fun(initial))), 0.0, True)
                )
                error = jnp.max(abs(fun(q)))
                converged = (
                    admissible
                    & jnp.all(jnp.isfinite(q))
                    & jnp.isfinite(error)
                    & (error < tolerance)
                )
                return q, jnp.array([count, error, converged.astype(q.dtype)])

            def tangent_solve(g, y):
                if dense is not None:
                    return jnp.linalg.solve(jax.jacfwd(g)(jnp.zeros_like(y)), y)
                # g maps perturbations to stress-normalized residuals. Two
                # constant probes recover all independent point derivatives.
                normal = wr * g(normal_probe) - coefficient * bulk(normal_probe)
                tangent = wr * g(tangent_probe) - coefficient * bulk(tangent_probe)
                bound = 256 * jnp.finfo(y.dtype).eps * (abs(coefficient) * row_bound + 1)
                valid = (
                    jnp.all(abs(tangent[:n]) <= bound[:n])
                    & (abs(normal[-1]) <= bound[-1])
                    & (abs(tangent[-1]) <= bound[-1])
                    & jnp.all(normal[:n] >= -bound[:n])
                    & jnp.all(tangent[n : 2 * n] >= -bound[n : 2 * n])
                )
                local = jnp.zeros((n, 2, 2), dtype=y.dtype)
                local = local.at[:, 0, 0].set(jnp.maximum(normal[:n], 0))
                local = local.at[:, 1, 0].set(normal[n : 2 * n])
                local = local.at[:, 1, 1].set(jnp.maximum(tangent[n : 2 * n], 0))
                result = tangent_linear_solve(wr * y, coefficient, local, spring)
                return jnp.where(valid, result, jnp.full_like(result, jnp.nan))

            q, info = jax.lax.custom_root(f, guess, solve, tangent_solve, has_aux=True)
            return jnp.where(info[2] > 0, q, jnp.full_like(q, jnp.nan)), info

        self.solve = jax.jit(root)
        self.force_residual = jax.jit(forces)
        self.response = jax.jit(values)
        self.bulk = jax.jit(bulk)
        self.residual_weights = wr

"""Trust-region recovery of the unchanged substepped contact equilibrium.

The primary Newton/line-search path remains unchanged. Recovery changes only
the root solver; its accepted state must satisfy the original microstep
residual. Implicit AD continues to use that residual and its local JAX tangent.
"""

import runpy
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import root

LOCAL = runpy.run_path(str(Path(__file__).with_name("substepped_contact_experiment.py")))


def recover_root(system, old, forcing, guess, *, tolerance=1e-7, max_evaluations=400):
    p = system.parameters
    z, c = jnp.asarray(old.plastic_slip), jnp.asarray(old.cumulative_slip)
    f, previous = jnp.asarray(forcing), jnp.asarray(system.previous_displacement)
    attempts = []

    def residual(q):
        return np.asarray(system.residual(jnp.asarray(q), z, c, p, f, previous))

    def tangent(q):
        return np.asarray(system.tangent(jnp.asarray(q), z, c, p, previous))

    def base_residual(q):
        return np.asarray(system.base.residual(jnp.asarray(q), z, c, p, f))

    def base_tangent(q):
        return np.asarray(system.base.tangent(jnp.asarray(q), z, c, p))

    def hybrid(fun, jac, initial, *, label, lam=1.0):
        try:
            answer = root(
                fun,
                initial,
                jac=jac,
                method="hybr",
                options={"xtol": 1e-11, "maxfev": max_evaluations},
            )
            vector = np.asarray(answer.x)
            error = float(np.max(abs(fun(vector))))
            finite = bool(np.all(np.isfinite(vector)) and np.isfinite(error))
            accepted = finite and error < tolerance
            attempts.append(
                dict(
                    method=label,
                    contact_fraction=lam,
                    reported_success=bool(answer.success),
                    residual_mpa=error if finite else None,
                    nfev=answer.nfev,
                    accepted=accepted,
                )
            )
            return vector, accepted
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            attempts.append(
                dict(
                    method=label,
                    contact_fraction=lam,
                    accepted=False,
                    error=str(error),
                    nfev=0,
                    residual_mpa=None,
                )
            )
            return np.asarray(initial), False

    candidate, accepted = hybrid(residual, tangent, guess, label="powell_hybrid")
    if accepted:
        return candidate, attempts
    # Start continuation from the positive-tangent previous-state contact law.
    seed, _ = system.base.implicit_displacement(p, z, c, f, jnp.asarray(guess))
    seed = np.asarray(seed)
    if not np.all(np.isfinite(seed)) or np.max(abs(base_residual(seed))) >= tolerance:
        attempts.append(dict(method="previous_state_seed", accepted=False, nfev=0))
        return None, attempts
    candidate, accepted = hybrid(residual, tangent, seed, label="hybrid_from_previous_state_root")
    if accepted:
        return candidate, attempts
    lam, increment = 0.0, 0.125
    for _ in range(64):
        target = min(1.0, lam + increment)

        def blended(q):
            return (1 - target) * base_residual(q) + target * residual(q)

        def blended_tangent(q):
            return (1 - target) * base_tangent(q) + target * tangent(q)

        candidate, accepted = hybrid(
            blended, blended_tangent, seed, label="contact_continuation", lam=target
        )
        if accepted:
            seed, lam = candidate, target
            if lam == 1:
                return seed, attempts
            increment = min(0.25, increment * 1.5)
        else:
            increment *= 0.5
            if increment < 1 / 65536:
                break
    return None, attempts


def kinetic_seed_root(system, old, forcing, guess, *, tolerance=1e-7, max_evaluations=400):
    """Try a constant-mu_k seed, accepting only the original full equation."""
    p = system.parameters
    z, c, f = (jnp.asarray(a) for a in [old.plastic_slip, old.cumulative_slip, forcing])
    previous = jnp.asarray(system.previous_displacement)
    kinetic = p.at[2].set(p[3])
    seed, _ = system.base.implicit_displacement(kinetic, z, c, f, jnp.asarray(guess))
    seed = np.asarray(seed)
    seed_error = float(np.max(abs(np.asarray(system.base.residual(seed, z, c, kinetic, f)))))
    seed_finite = bool(np.all(np.isfinite(seed)) and np.isfinite(seed_error))
    attempts = [
        dict(
            method="constant_kinetic_law_seed",
            accepted=False,
            seed_residual_mpa=seed_error if seed_finite else None,
            final_parameters_and_state_unchanged=True,
        )
    ]
    if not seed_finite or seed_error >= tolerance:
        return None, attempts

    def residual(q):
        return np.asarray(system.residual(jnp.asarray(q), z, c, p, f, previous))

    def tangent(q):
        return np.asarray(system.tangent(jnp.asarray(q), z, c, p, previous))

    # The seed may already lie on the fully weakened branch of the ORIGINAL
    # return map. No changed parameter or trial contact state is committed.
    error = float(np.max(abs(residual(seed))))
    if np.isfinite(error) and error < tolerance:
        attempts.append(
            dict(method="kinetic_seed_original_equation", accepted=True, residual_mpa=error, nfev=1)
        )
        return seed, attempts
    answer = root(
        residual,
        seed,
        jac=tangent,
        method="hybr",
        options={"xtol": 1e-11, "maxfev": max_evaluations},
    )
    vector = np.asarray(answer.x)
    error = float(np.max(abs(residual(vector))))
    finite = bool(np.all(np.isfinite(vector)) and np.isfinite(error))
    accepted = finite and error < tolerance
    attempts.append(
        dict(
            method="hybrid_from_kinetic_seed",
            accepted=accepted,
            reported_success=bool(answer.success),
            nfev=answer.nfev,
            residual_mpa=error if finite else None,
        )
    )
    return (vector if accepted else None), attempts


class RobustSubsteppedContactEquilibrium(LOCAL["SubsteppedContactEquilibrium"]):
    def __init__(self, reference, *, substeps, kinetic_seed_recovery=False):
        super().__init__(reference, substeps=substeps)
        self.recovery_events = []
        self.kinetic_seed_recovery = bool(kinetic_seed_recovery)

    def recover(self, old, forcing, guess):
        candidate, attempts = recover_root(self, old, forcing, guess)
        if candidate is None and self.kinetic_seed_recovery:
            try:
                candidate, additional = kinetic_seed_root(self, old, forcing, guess)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                candidate = None
                additional = [
                    dict(method="kinetic_seed_recovery", accepted=False, error=str(error))
                ]
            attempts.extend(additional)
        return candidate, attempts

    def solve(self, normal_pressure, actuator, old, *, past=None, guess=None):
        step = len(self.accepted_micro_work) + 1
        primary = super().solve(normal_pressure, actuator, old, past=past, guess=guess)
        if primary.converged:
            return primary
        forcing = self.base.forcing(
            normal_pressure,
            actuator,
            np.zeros(len(self.reference.base_u)) if past is None else past,
        )
        initial = self.previous_displacement if guess is None else guess
        candidate, attempts = self.recover(old, forcing, initial)
        event = dict(
            step=step,
            primary_residual_mpa=primary.residual_mpa,
            primary_newton_iterations=primary.iterations,
            attempts=attempts,
            recovered=False,
        )
        self.recovery_events.append(event)
        if candidate is None:
            return primary
        # Re-enter the original implicit-root API: only its converged solve
        # commits q/z/c and micro work. Recovery trials commit no physical state.
        accepted = super().solve(normal_pressure, actuator, old, past=past, guess=candidate)
        event.update(recovered=accepted.converged, final_residual_mpa=accepted.residual_mpa)
        return replace(accepted, iterations=primary.iterations + accepted.iterations)

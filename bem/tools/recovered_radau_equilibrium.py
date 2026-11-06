"""Primal recovery of the original stage equation using local JAX derivatives.

The condensed equation supplies trial guesses only. Final acceptance and
implicit derivatives belong to the original full residual, including all
parameter dependence of the eliminated elastic/contact blocks.
"""

import runpy
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import numpy as np
from scipy.optimize import root

TOOLS = Path(__file__).resolve().parent
POLICY = runpy.run_path(str(TOOLS / "identity_stage_tangent.py"))
CONDENSED = runpy.run_path(str(TOOLS / "diagnose_condensed_radau_root.py"))


def recover_original_stage(
    system,
    reference,
    matrix,
    old,
    forcing,
    previous,
    failed,
    *,
    max_starts=32,
    seed=20261104,
    on_recovery=None,
):
    """Return the original failure or a root accepted by the original system."""
    if failed.converged:
        return failed
    if not system.tangent_policy["applied"]:
        return failed
    if max_starts < 0:
        raise ValueError("Require a nonnegative trial budget")
    begun = perf_counter()
    trial_q = np.asarray(failed.displacement)
    if not np.all(np.isfinite(trial_q)):
        trial_q = np.tile(previous, 2)
    data = dict(
        matrix=np.asarray(matrix),
        forcing=np.asarray(forcing),
        previous_displacement=np.asarray(previous),
        failed_displacement=trial_q,
        plastic_slip=np.asarray(old.plastic_slip),
        cumulative_slip=np.asarray(old.cumulative_slip),
        parameters=np.asarray(system.parameters),
        base_dn=np.asarray(reference.base_dn),
        base_ds=np.asarray(reference.base_ds),
        contact_weights=np.asarray(reference.constraints.contact_weights),
        reduced_weights=np.asarray(reference.constraints.reduced_weights),
    )
    records, accepted, error_message = [], None, None
    try:
        model = CONDENSED["CondensedStage"](data, system.segment_substeps)
        nodes = np.union1d(
            model.nodes, np.flatnonzero(data["cumulative_slip"] < data["parameters"][-1])
        )
        if not np.array_equal(nodes, model.nodes):
            model = CONDENSED["CondensedStage"](
                data, system.segment_substeps, nonlinear_nodes=nodes
            )

        def fun(x):
            return np.asarray(model.residual(jnp.asarray(x)))

        def jac(x):
            return np.asarray(model.tangent(jnp.asarray(x)))

        initial = model.correction_at(trial_q)
        pressure = data["parameters"][0] * model.jumps(trial_q)[model.nodes, :, 0]
        scale = np.maximum(
            1, abs(np.stack((pressure, data["parameters"][3] * pressure), -1).ravel())
        )
        rng = np.random.default_rng(seed)
        starts = [initial, np.zeros_like(initial), model.correction_at(np.tile(previous, 2))]
        starts += [initial + rng.uniform(-1, 1, len(initial)) * scale for _ in range(max_starts)]
        for index, guess in enumerate(starts):
            answer = root(fun, guess, jac=jac, method="hybr", options=dict(xtol=1e-11, maxfev=400))
            if not np.all(np.isfinite(answer.x)):
                records.append(
                    dict(index=index, nfev=answer.nfev, accepted=False, reason="nonfinite_trial")
                )
                continue
            checks = model.check_candidate(answer.x)
            row = dict(
                index=index, nfev=answer.nfev, reported_success=bool(answer.success), **checks
            )
            records.append(row)
            if not checks["accepted"]:
                continue
            q = model.expand(answer.x)
            z, c = jnp.asarray(old.plastic_slip), jnp.asarray(old.cumulative_slip)
            # The unchanged full custom_root validates the seed. Its
            # derivatives never treat the condensed NumPy blocks as constant.
            q, _ = system.implicit_displacement(
                system.parameters, z, c, jnp.asarray(forcing), jnp.asarray(q), jnp.asarray(previous)
            )
            q = np.asarray(q)
            residual = float(
                jnp.max(abs(system.residual(q, z, c, system.parameters, forcing, previous)))
            )
            values = np.asarray(system.local_response(q, z, c, system.parameters, previous))
            good = bool(
                np.all(np.isfinite(q))
                and np.all(np.isfinite(values))
                and np.isfinite(residual)
                and residual < 1e-7
            )
            row.update(original_full_residual_mpa=residual, accepted=good)
            if good:
                accepted = (q, values, residual)
                break
        detail = dict(
            method="condensed_full_normal_coupling_multistart",
            seed=seed,
            nonlinear_nodes=model.nodes.tolist(),
            reduced_unknowns=len(initial),
            trials=records,
            accepted=accepted is not None,
            seconds=perf_counter() - begun,
            physical_state_not_committed_by_recovery=True,
            parameter_ad_uses_original_full_residual=True,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        error_message = str(error)
        detail = dict(
            method="condensed_full_normal_coupling_multistart",
            seed=seed,
            trials=records,
            accepted=False,
            error=error_message,
            seconds=perf_counter() - begun,
        )
    if on_recovery is not None:
        on_recovery(detail)
    attempts = [*failed.recovery_attempts, detail]
    if accepted is None:
        return replace(failed, recovery_attempts=attempts)
    q, values, residual = accepted
    return replace(
        failed,
        displacement=q,
        response=values,
        residual_mpa=residual,
        converged=True,
        recovery_attempts=attempts,
    )


def recovering_stage_equilibrium(
    reference, matrix, *, segment_substeps=(1, 2), max_starts=32, on_recovery=None
):
    if max_starts < 0:
        raise ValueError("Require a nonnegative random recovery budget")
    system = POLICY["stage_equilibrium"](reference, matrix, segment_substeps=segment_substeps)
    original = system.solve

    def solve(old, forcing, guess, previous, **kwargs):
        answer = original(old, forcing, guess, previous, **kwargs)
        return recover_original_stage(
            system,
            reference,
            matrix,
            old,
            forcing,
            previous,
            answer,
            max_starts=max_starts,
            on_recovery=on_recovery,
        )

    system.solve = solve
    return system

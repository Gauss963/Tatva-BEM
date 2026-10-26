"""Conditional full-history AD with independent bulk and contact-state grids.

The prepared resolvent is a fixed algebraic preconditioner. Its coarse
reference diagonal, affine lift, elastic forcing and recorded actuator are
held fixed. Differentiating the original root residual still includes all
five physical contact parameters, including kn/kt during sticking. No
assumption that the correction or its derivative vanishes before slip is
made. These derivatives describe the selected discrete root branches.
"""

import runpy
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from finiteblock_bem.cq import BoundaryHistory, OnlineConvolution
from finiteblock_bem.jax_backend import JaxContactEquilibrium

PARAMETERS = ["normal_penalty", "tangential_penalty", "mu_s", "mu_k", "Dc"]


def context(reference, history, result, constitutive_substeps):
    steps, d = len(result.time) - 1, history.plan.dofs
    m = len(reference.constraints.contact_weights)
    if (
        steps < 1
        or history.plan.steps < steps
        or reference.constraints.size != d
        or result.bulk_displacement.shape != (steps + 1, d)
        or result.plastic_slip.shape != (steps + 1, m)
        or result.cumulative_slip.shape != (steps + 1, m)
        or not np.array_equal(result.time, np.arange(steps + 1) * history.plan.dt)
        or not all(
            np.all(np.isfinite(v))
            for v in (result.bulk_displacement, result.plastic_slip, result.cumulative_slip)
        )
    ):
        raise ValueError("Complete finite retained/state histories on the same CQ grid required")
    if constitutive_substeps != 1:
        cls = runpy.run_path(str(Path(__file__).with_name("substepped_contact_experiment.py")))[
            "SubsteppedContactEquilibrium"
        ]
        system = cls(reference, substeps=constitutive_substeps)
    else:
        system = JaxContactEquilibrium(reference)
    con, law = reference.constraints, reference.case.contact
    bn, bt = jnp.asarray(con.normal_jump), jnp.asarray(con.tangential_jump)
    wc, wr = jnp.asarray(con.contact_weights), jnp.asarray(con.reduced_weights)
    # Dref belongs to the unchanged bulk interpolation, not the refined grid.
    coarse = reference.model.constraints.contact_weights
    diagonal = jnp.asarray(np.r_[coarse * law.normal_penalty, coarse * law.tangential_penalty, 0])
    a0 = jnp.asarray(reference.matrix) + jnp.diag(diagonal)
    composition = np.linalg.norm(history.instantaneous @ np.asarray(a0) - np.eye(d), ord=np.inf)
    if not np.isfinite(composition) or composition > 1e-6:
        raise ValueError("Lag-zero preconditioner does not match the retained root")

    def response(q, z, c, parameters, previous):
        return (
            system.local_response(q, z, c, parameters, previous)
            if constitutive_substeps != 1
            else system.local_response(q, z, c, parameters)
        )[:, :4]

    def equation(q, z, c, parameters, previous):
        return (
            system.residual(q, z, c, parameters, jnp.zeros(d), previous)
            if constitutive_substeps != 1
            else system.residual(q, z, c, parameters, jnp.zeros(d))
        )

    def tangent(q, z, c, parameters, previous):
        return (
            system.tangent(q, z, c, parameters, previous)
            if constitutive_substeps != 1
            else system.tangent(q, z, c, parameters)
        )

    def force(local):
        return bn.T @ (wc * local[..., 0]) + bt.T @ (wc * local[..., 1])

    return system, wc, wr, diagonal, a0, response, equation, tangent, force


def forward_gradient(
    reference, history, result, *, constitutive_substeps=1, convolution_factory=OnlineConvolution
):
    """Five independent forward JVP histories, including every sticking root."""
    begun = perf_counter()
    system, wc, wr, diagonal, a0, response, equation, tangent, force = context(
        reference, history, result, constitutive_substeps
    )
    m, d = len(wc), history.plan.dofs

    @jax.jit
    def step(q, z, c, previous, dz, dc, dprevious, memory):
        def root_source(z, c, parameters, previous):
            return equation(q, z, c, parameters, previous)

        def direct(dz, dc, dp, dprevious):
            return jax.jvp(
                root_source, (z, c, system.parameters, previous), (dz, dc, dp, dprevious)
            )[1]

        rhs = (a0 @ memory) / wr[:, None] - jax.vmap(direct)(dz.T, dc.T, jnp.eye(5), dprevious.T).T
        dq = jnp.linalg.solve(tangent(q, z, c, system.parameters, previous), rhs)

        def local(dq, dz, dc, dp, dprevious):
            return jax.jvp(
                response,
                (q, z, c, system.parameters, previous),
                (dq, dz, dc, dp, dprevious),
            )[1]

        values = jax.vmap(local)(dq.T, dz.T, dc.T, jnp.eye(5), dprevious.T)
        correction = diagonal[:, None] * dq - jax.vmap(force)(values).T
        return dq, values[:, :, 2].T, values[:, :, 3].T, correction

    streams = [convolution_factory(history) for _ in PARAMETERS]
    shared = {}
    for stream in streams:
        stream.transforms = shared
    dz, dc, dprevious = np.zeros((m, 5)), np.zeros((m, 5)), np.zeros((d, 5))
    for offset, index in enumerate(range(1, len(result.time))):
        memory = np.column_stack([stream.past(offset) for stream in streams])
        dprevious, dz, dc, correction = (
            np.asarray(v)
            for v in step(
                result.bulk_displacement[index],
                result.plastic_slip[index - 1],
                result.cumulative_slip[index - 1],
                result.bulk_displacement[index - 1],
                dz,
                dc,
                dprevious,
                memory,
            )
        )
        if not all(np.all(np.isfinite(v)) for v in (dprevious, dz, dc, correction)):
            raise ValueError(f"Nonfinite refined forward JVP at step {index}")
        for column, stream in enumerate(streams):
            stream.append(correction[:, column])
    return dict(
        gradient=np.average(dc, weights=np.asarray(wc), axis=0),
        parameter_order=PARAMETERS,
        seconds_including_jit=perf_counter() - begun,
        every_history_source_included=True,
        previous_displacement_tangent_included=constitutive_substeps != 1,
        bulk_dofs=d,
        contact_state_points=m,
        scope="Fixed bulk, static lift, initial state and recorded loading; selected discrete branches",
    )


def reverse_gradient(
    reference, history, result, *, constitutive_substeps=1, convolution_factory=OnlineConvolution
):
    """Reverse causal resolvent plus implicit root/state VJPs on the fine grid."""
    begun = perf_counter()
    system, wc, wr, diagonal, a0, response, equation, tangent, force = context(
        reference, history, result, constitutive_substeps
    )
    m, d = len(wc), history.plan.dofs

    @jax.jit
    def step(q, z, c, previous, state_bar, correction_bar, previous_bar):
        def local(q, z, c, parameters, previous):
            values = response(q, z, c, parameters, previous)
            return diagonal * q - force(values), values[:, 2:4]

        _, pull = jax.vjp(local, q, z, c, system.parameters, previous)
        qbar, zbar, cbar, parameter_bar, path_bar = pull((correction_bar, state_bar))
        multiplier = jnp.linalg.solve(
            tangent(q, z, c, system.parameters, previous).T, qbar + previous_bar
        )

        def root_source(z, c, parameters, previous):
            return equation(q, z, c, parameters, previous)

        _, root_pull = jax.vjp(root_source, z, c, system.parameters, previous)
        dz, dc, dp, root_path_bar = root_pull(-multiplier)
        raw = multiplier / wr
        return (
            a0.T @ raw,
            jnp.stack((zbar + dz, cbar + dc), axis=1),
            parameter_bar + dp,
            path_bar + root_path_bar,
            # The recorded actuator enters both the current physical root
            # and the correction source consumed by every future root.
            reference.spring * (raw[-1] + correction_bar[-1]),
        )

    reverse = convolution_factory(BoundaryHistory(history.plan, history.weights.transpose(0, 2, 1)))
    state_bar = np.c_[np.zeros(m), np.asarray(wc) / np.sum(wc)]
    previous_bar, gradient = np.zeros(d), np.zeros(5)
    actuator = np.zeros(len(result.time))
    for offset, index in enumerate(range(len(result.time) - 1, 0, -1)):
        hbar, state_bar, direct, previous_bar, actuator[index] = (
            np.asarray(v)
            for v in step(
                result.bulk_displacement[index],
                result.plastic_slip[index - 1],
                result.cumulative_slip[index - 1],
                result.bulk_displacement[index - 1],
                state_bar,
                reverse.past(offset),
                previous_bar,
            )
        )
        if not all(np.all(np.isfinite(v)) for v in (hbar, state_bar, direct, previous_bar)):
            raise ValueError(f"Nonfinite refined reverse VJP at step {index}")
        gradient += direct
        reverse.append(hbar)
    return dict(
        gradient=gradient,
        actuator_gradient=actuator,
        parameter_order=PARAMETERS,
        objective="final contact-weighted mean cumulative slip",
        value_mm=float(np.average(result.cumulative_slip[-1], weights=np.asarray(wc))),
        seconds_including_jit=perf_counter() - begun,
        every_history_source_included=True,
        previous_displacement_cotangent_included=constitutive_substeps != 1,
        bulk_dofs=d,
        contact_state_points=m,
        scope="Fixed bulk, static lift, initial state and recorded loading; selected discrete branches; no event-time derivative",
    )

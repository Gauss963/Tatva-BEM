"""Conditional implicit-root AD with both Radau stages and complete history.

The bulk, measured initial lift/state/force, reference diagonal and affine
startup stay fixed. Parameters and recorded stage actuators are independent
inputs. These are selected discrete-branch derivatives, not derivatives of
the healed preload, geometry, materials or endogenous stopping time.
"""

import runpy
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from finiteblock_bem.cq import cq_plan_signature

TOOLS = Path(__file__).resolve().parent
CONTACT = runpy.run_path(str(TOOLS / "radau_contact_experiment.py"))
HISTORY = runpy.run_path(str(TOOLS / "compact_radau_history.py"))
PARAMETERS = ["normal_penalty", "tangential_penalty", "mu_s", "mu_k", "Dc"]


def compact_instantaneous(first, second, wrap):
    if cq_plan_signature(first.plan) != cq_plan_signature(second.plan):
        raise ValueError("RK tangent families require the same CQ plan")
    d = first.plan.dofs
    if np.shape(wrap) != (d, d) or not np.all(np.isfinite(wrap)):
        raise ValueError("Invalid RK tangent shift wrap")
    delta0 = np.linalg.inv(HISTORY["RADAU"]["A"])
    delta1 = -np.outer(delta0 @ np.ones(2), [0, 1])
    instantaneous = np.kron(np.eye(2), first.instantaneous)
    instantaneous += np.kron(delta0, second.instantaneous) + np.kron(delta1, wrap)
    return delta0, delta1, instantaneous


class BatchedCompactTangents:
    """Share each family FFT across stages and every tangent direction."""

    def __init__(self, first, second, wrap, directions, *, cache_mb=0, fft_rows=4):
        self.plan, self.directions = first.plan, directions
        self.delta0, self.delta1, _ = compact_instantaneous(first, second, wrap)
        self.streams = [
            HISTORY["BatchedOnlineConvolution"](
                family, 2 * directions, cache_mb=cache_mb / 2, fft_rows=fft_rows
            )
            for family in [first, second]
        ]
        self.previous_f1 = np.zeros((first.plan.dofs, 2 * directions))

    def stages(self, value):
        return value.reshape(self.plan.dofs, 2, self.directions).transpose(1, 0, 2)

    def past(self, index):
        first, second = [self.stages(stream.past(index)) for stream in self.streams]
        return (
            first
            + np.einsum("ab,bip->aip", self.delta0, second)
            + np.einsum("ab,bip->aip", self.delta1, self.stages(self.previous_f1))
        ).reshape(2 * self.plan.dofs, self.directions)

    def append(self, value):
        value = np.asarray(value)
        if value.shape != (2 * self.plan.dofs, self.directions):
            raise ValueError("Tangent sources must align with both stages and directions")
        stages = value.reshape(2, self.plan.dofs, self.directions).transpose(1, 0, 2)
        stages = stages.reshape(self.plan.dofs, 2 * self.directions)
        second = self.streams[1]
        self.previous_f1 = second.history.instantaneous @ stages + second.past(second.next_step)
        for stream in self.streams:
            stream.append(stages)

    def statistics(self):
        return dict(
            directions=self.directions,
            families=[stream.statistics() for stream in self.streams],
            shared_family_ffts_across_stages_and_directions=True,
            every_history_source_included=True,
        )


def context(reference, first, second, wrap, impedance, stages, segment_substeps):
    con = reference.constraints
    d, m = first.plan.dofs, len(con.contact_weights)
    q = np.asarray(stages["displacement"]).reshape(-1, 2 * d)
    response = np.asarray(stages["response"])
    if (
        con.size != d
        or q.shape[0] < 2
        or q.shape[0] - 1 > first.plan.steps
        or response.shape != (len(q), m, 2, 7)
        or np.shape(stages["actuator"]) != (len(q), 2)
        or np.shape(stages["stage_time"]) != (len(q), 2)
        or np.shape(impedance) != (2 * d, 2 * d)
        or not np.all(np.isfinite(impedance))
        or not np.all(np.isfinite(stages["actuator"]))
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(response))
        or np.any(q[0] != 0)
    ):
        raise ValueError("Require complete aligned finite stage displacement/state histories")
    expected_time = np.zeros((len(q), 2))
    expected_time[1:] = (np.arange(len(q) - 1)[:, None] + HISTORY["RADAU"]["C"]) * first.plan.dt
    if not np.allclose(stages["stage_time"], expected_time, rtol=1e-12, atol=1e-15):
        raise ValueError("Saved RK stage clock differs from the prepared CQ plan")
    matrix = np.array(impedance, copy=True)
    matrix[[d - 1, 2 * d - 1], [d - 1, 2 * d - 1]] += reference.spring
    system = CONTACT["RadauContactEquilibrium"](
        reference, matrix, segment_substeps=segment_substeps
    )
    law = reference.case.contact
    wc = jnp.asarray(con.contact_weights)
    wr = jnp.tile(jnp.asarray(con.reduced_weights), 2)
    diagonal = jnp.tile(jnp.r_[wc * law.normal_penalty, wc * law.tangential_penalty, 0], 2)
    a0 = jnp.asarray(matrix) + jnp.diag(diagonal)
    _, _, instantaneous = compact_instantaneous(first, second, wrap)
    composition = np.linalg.norm(instantaneous @ np.asarray(a0) - np.eye(2 * d), ord=np.inf)
    if not np.isfinite(composition) or composition > 1e-6:
        raise ValueError("Lag-zero RK preconditioner does not match the original root")
    bn, bt = jnp.asarray(con.normal_jump), jnp.asarray(con.tangential_jump)

    def force(local):
        return jnp.concatenate(
            [bn.T @ (wc * local[:, a, 0]) + bt.T @ (wc * local[:, a, 1]) for a in range(2)]
        )

    def equation(q, z, c, parameters, previous):
        return system.residual(q, z, c, parameters, jnp.zeros(2 * d), previous)

    return SimpleNamespace(
        system=system,
        q=q,
        response=response,
        wc=wc,
        wr=wr,
        diagonal=diagonal,
        a0=a0,
        force=force,
        equation=equation,
        d=d,
        m=m,
    )


def forward_directions(
    reference,
    first,
    second,
    wrap,
    impedance,
    stages,
    parameter_directions,
    actuator_directions,
    *,
    segment_substeps=(1, 2),
    cache_mb=0,
    fft_rows=4,
):
    """Batched JVPs of every root, stage path, state and causal source."""
    begun = perf_counter()
    ctx = context(reference, first, second, wrap, impedance, stages, segment_substeps)
    system, d, m = ctx.system, ctx.d, ctx.m
    dp = np.asarray(parameter_directions, dtype=float)
    da = np.asarray(actuator_directions, dtype=float)
    if (
        dp.ndim != 2
        or dp.shape[0] != 5
        or dp.shape[1] < 1
        or da.shape != (len(ctx.q), 2, dp.shape[1])
        or not np.all(np.isfinite(dp))
        or not np.all(np.isfinite(da))
        or np.any(da[0] != 0)
    ):
        raise ValueError("Require finite parameter/loading directions with fixed initial actuator")
    directions = dp.shape[1]

    @jax.jit
    def step(q, z, c, previous, dz, dc, dprevious, memory, dp, da):
        def direct(dz, dc, dp, dprevious):
            return jax.jvp(
                lambda z, c, p, previous: ctx.equation(q, z, c, p, previous),
                (z, c, system.parameters, previous),
                (dz, dc, dp, dprevious),
            )[1]

        loading = (
            jnp.zeros((2 * d, directions))
            .at[jnp.array([d - 1, 2 * d - 1])]
            .set(reference.spring * da)
        )
        rhs = (ctx.a0 @ memory + loading) / ctx.wr[:, None]
        rhs -= jax.vmap(direct)(dz.T, dc.T, dp.T, dprevious.T).T
        dq = jnp.linalg.solve(system.tangent(q, z, c, system.parameters, previous), rhs)

        def local(dq, dz, dc, dp, dprevious):
            return jax.jvp(
                system.local_response,
                (q, z, c, system.parameters, previous),
                (dq, dz, dc, dp, dprevious),
            )[1]

        values = jax.vmap(local)(dq.T, dz.T, dc.T, dp.T, dprevious.T)
        correction = ctx.diagonal[:, None] * dq - jax.vmap(ctx.force)(values).T + loading
        return dq[d:], values[:, :, 1, 2].T, values[:, :, 1, 3].T, correction

    memory = BatchedCompactTangents(
        first, second, wrap, directions, cache_mb=cache_mb, fft_rows=fft_rows
    )
    dz, dc, previous = (
        np.zeros((m, directions)),
        np.zeros((m, directions)),
        np.zeros((d, directions)),
    )
    for offset, index in enumerate(range(1, len(ctx.q))):
        previous, dz, dc, correction = [
            np.asarray(v)
            for v in step(
                ctx.q[index],
                ctx.response[index - 1, :, 1, 2],
                ctx.response[index - 1, :, 1, 3],
                ctx.q[index - 1, d:],
                dz,
                dc,
                previous,
                memory.past(offset),
                dp,
                da[index],
            )
        ]
        if not all(np.all(np.isfinite(v)) for v in [previous, dz, dc, correction]):
            raise ValueError(f"Nonfinite complete Radau tangent at step {index}")
        memory.append(correction)
    return dict(
        objective_directional_derivative=np.average(dc, weights=np.asarray(ctx.wc), axis=0),
        parameter_order=PARAMETERS,
        seconds_including_jit=perf_counter() - begun,
        history=memory.statistics(),
        scope=__doc__,
    )


def reverse_gradient(
    reference,
    first,
    second,
    wrap,
    impedance,
    stages,
    *,
    segment_substeps=(1, 2),
    cache_mb=0,
    fft_rows=4,
):
    """One transposed compact history plus local VJPs and implicit root solves."""
    begun = perf_counter()
    ctx = context(reference, first, second, wrap, impedance, stages, segment_substeps)
    system, d, m = ctx.system, ctx.d, ctx.m

    @jax.jit
    def step(q, z, c, previous, state_bar, correction_bar, previous_bar):
        def local(q, z, c, parameters, previous):
            values = system.local_response(q, z, c, parameters, previous)
            return ctx.diagonal * q - ctx.force(values), values[:, 1, 2:4]

        _, pull = jax.vjp(local, q, z, c, system.parameters, previous)
        qbar, zbar, cbar, parameter_bar, path_bar = pull((correction_bar, state_bar))
        qbar = qbar.at[d:].add(previous_bar)
        multiplier = jnp.linalg.solve(system.tangent(q, z, c, system.parameters, previous).T, qbar)
        _, root_pull = jax.vjp(
            lambda z, c, p, previous: ctx.equation(q, z, c, p, previous),
            z,
            c,
            system.parameters,
            previous,
        )
        dz, dc, dp, root_path_bar = root_pull(-multiplier)
        raw = multiplier / ctx.wr
        return (
            ctx.a0.T @ raw,
            jnp.stack((zbar + dz, cbar + dc), axis=1),
            parameter_bar + dp,
            path_bar + root_path_bar,
            reference.spring
            * (raw[jnp.array([d - 1, 2 * d - 1])] + correction_bar[jnp.array([d - 1, 2 * d - 1])]),
        )

    reverse = HISTORY["CompactRadauConvolution"](
        first, second, wrap, cache_mb=cache_mb, fft_rows=fft_rows, transpose=True
    )
    state_bar = np.c_[np.zeros(m), np.asarray(ctx.wc) / np.sum(ctx.wc)]
    previous_bar, gradient = np.zeros(d), np.zeros(5)
    actuator_bar = np.zeros((len(ctx.q), 2))
    for offset, index in enumerate(range(len(ctx.q) - 1, 0, -1)):
        hbar, state_bar, direct, previous_bar, actuator_bar[index] = [
            np.asarray(v)
            for v in step(
                ctx.q[index],
                ctx.response[index - 1, :, 1, 2],
                ctx.response[index - 1, :, 1, 3],
                ctx.q[index - 1, d:],
                state_bar,
                reverse.past(offset),
                previous_bar,
            )
        ]
        if not all(np.all(np.isfinite(v)) for v in [hbar, state_bar, direct, previous_bar]):
            raise ValueError(f"Nonfinite complete Radau adjoint at step {index}")
        gradient += direct
        reverse.append(hbar)
    return dict(
        gradient=gradient,
        actuator_gradient=actuator_bar,
        initial_state_cotangent=state_bar,
        initial_path_cotangent=previous_bar,
        parameter_order=PARAMETERS,
        objective="final contact-weighted mean cumulative slip",
        value_mm=float(np.average(ctx.response[-1, :, 1, 3], weights=np.asarray(ctx.wc))),
        seconds_including_jit=perf_counter() - begun,
        history=reverse.statistics(),
        every_history_source_included=True,
        endpoint_state_and_previous_path_cotangents_included=True,
        full_initialized_gradient=False,
        event_time_derivative_included=False,
        scope=__doc__,
    )

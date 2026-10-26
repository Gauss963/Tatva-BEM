"""Experimental five-direction JVP with a single complete CQ history stream."""

import runpy
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

TOOLS = Path(__file__).resolve().parent
AD = runpy.run_path(str(TOOLS / "refined_contact_adjoint.py"))
CONVOLUTION = runpy.run_path(str(TOOLS / "batched_history.py"))["BatchedBoundedConvolution"]


def forward_gradient_batched(
    reference, history, result, *, constitutive_substeps=1, cache_mb=256, fft_rows=4
):
    begun = perf_counter()
    system, wc, wr, diagonal, a0, response, equation, tangent, force = AD["context"](
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
                response, (q, z, c, system.parameters, previous), (dq, dz, dc, dp, dprevious)
            )[1]

        values = jax.vmap(local)(dq.T, dz.T, dc.T, jnp.eye(5), dprevious.T)
        correction = diagonal[:, None] * dq - jax.vmap(force)(values).T
        return dq, values[:, :, 2].T, values[:, :, 3].T, correction

    memory = CONVOLUTION(history, right_hand_sides=5, cache_mb=cache_mb, fft_rows=fft_rows)
    dz, dc, dprevious = np.zeros((m, 5)), np.zeros((m, 5)), np.zeros((d, 5))
    for offset, index in enumerate(range(1, len(result.time))):
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
                memory.past(offset),
            )
        )
        if not all(np.all(np.isfinite(v)) for v in (dprevious, dz, dc, correction)):
            raise ValueError(f"Nonfinite batched refined forward JVP at step {index}")
        memory.append(correction)
    return dict(
        gradient=np.average(dc, weights=np.asarray(wc), axis=0),
        parameter_order=AD["PARAMETERS"],
        seconds_including_jit=perf_counter() - begun,
        every_history_source_included=True,
        previous_displacement_tangent_included=constitutive_substeps != 1,
        bulk_dofs=d,
        contact_state_points=m,
        history=memory.statistics(),
        scope="Fixed bulk, static lift, initial state and recorded loading; selected discrete branches",
    )

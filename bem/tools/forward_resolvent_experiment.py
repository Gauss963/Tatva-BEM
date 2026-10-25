"""Independent forward JVP check of conditional resolvent parameter gradients.

The closed/sticking prefix has zero primal correction but nonzero kn/kt
derivatives. Its complete causal response is evaluated in frequency blocks.
Only outputs after the last prefix input are used, avoiding future-input
circular wrap without storing full-horizon matrix weights. Post-activation
tangents use forward history, independently of the reverse adjoint recurrence.
Bulk, static lift, recorded loading and selected branches remain fixed.
"""

import runpy
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import irfft, rfft

from finiteblock_bem.cq import OnlineConvolution
from finiteblock_bem.jax_backend import JaxContactEquilibrium


def prefix_forward_response(spectrum, plan, sources, full_steps, *, coordinate_scale=None):
    """Causal H*source outputs starting at the last supplied prefix time.

    Earlier DFT-product outputs can contain circular wrap and are deliberately
    excluded. Each source has shape (prefix_steps+1, dofs, directions).
    """
    sources = np.asarray(sources)
    if (
        sources.ndim != 3
        or not 1 <= len(sources) <= full_steps + 1
        or sources.shape[1] != plan.dofs
        or sources.shape[2] < 1
        or not np.all(np.isfinite(sources))
        or 2 * (full_steps + 1) > plan.fft_size
        or spectrum.shape != (plan.fft_size // 2 + 1, plan.dofs, plan.dofs)
    ):
        raise ValueError("Invalid complete contour, prefix sources or output horizon")
    scale = np.ones(plan.dofs) if coordinate_scale is None else np.asarray(coordinate_scale)
    if scale.shape != (plan.dofs,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Finite positive coordinate scales required")
    damping = plan.radius ** np.arange(full_steps + 1)
    transformed = rfft(
        sources * damping[: len(sources), None, None] / scale[None, :, None],
        n=plan.fft_size,
        axis=0,
    )
    product = np.empty_like(transformed)
    for first in range(0, len(product), 64):
        last = min(first + 64, len(product))
        product[first:last] = np.einsum(
            "kij,kjp->kip", spectrum[first:last], transformed[first:last]
        )
    first_output = len(sources) - 1
    response = irfft(product, n=plan.fft_size, axis=0)[first_output : full_steps + 1]
    response = response / damping[first_output:, None, None] / scale[None, :, None]
    if not np.all(np.isfinite(response)):
        raise ValueError("Nonfinite complete-prefix forward response")
    return response


def forward_parameter_gradient(
    reference,
    history,
    result,
    spectrum,
    start,
    *,
    coordinate_scale=None,
    constitutive_substeps=1,
    convolution_factory=None,
):
    begun = perf_counter()
    steps, d = len(result.time) - 1, history.plan.dofs
    n = result.plastic_slip.shape[1]
    if d != 2 * n + 1 or not 1 <= start <= steps or steps - start > history.plan.steps:
        raise ValueError("Correction weights must cover the activated forward source")
    law = reference.case.contact
    if np.any(result.plastic_slip[:start] != result.plastic_slip[0]) or np.any(
        result.cumulative_slip[:start] != result.cumulative_slip[0]
    ):
        raise ValueError("The omitted prefix must have unchanged contact state")
    dn, ds = result.contact_jumps[:start, :n], result.contact_jumps[:start, n : 2 * n]
    if np.any(dn <= 0) or np.any(
        abs(law.tangential_penalty * (ds - result.plastic_slip[0]))
        > law.coefficient(result.cumulative_slip[:start]) * law.normal_penalty * dn
    ):
        raise ValueError("The omitted prefix must remain closed and exactly sticking")
    substepped = constitutive_substeps != 1
    if substepped:
        tool = runpy.run_path(str(Path(__file__).with_name("substepped_contact_experiment.py")))
        system = tool["SubsteppedContactEquilibrium"](reference, substeps=constitutive_substeps)
    else:
        system = JaxContactEquilibrium(reference)
    wc = jnp.asarray(reference.constraints.contact_weights)
    wr = jnp.asarray(reference.constraints.reduced_weights)
    cdiag = jnp.r_[wc * law.normal_penalty, wc * law.tangential_penalty, 0]
    a0 = jnp.asarray(reference.matrix) + jnp.diag(cdiag)
    q = np.c_[
        result.contact_jumps - np.r_[reference.base_dn, reference.base_ds],
        result.platen - reference.base_platen,
    ]
    # Only kn and kt create prefix sources. Constant Fc0, initial q and
    # recorded hold forcing remain fixed, as in the existing conditional AD.
    source = np.zeros((start, d, 2))
    source[1:, :n, 0] = -np.asarray(wc) * dn[1:]
    source[1:, n : 2 * n, 1] = -np.asarray(wc) * (ds[1:] - result.plastic_slip[0])
    prefix = prefix_forward_response(
        spectrum, history.plan, source, steps, coordinate_scale=coordinate_scale
    )

    def response(q, z, c, p, previous):
        return (
            system.local_response(q, z, c, p, previous)
            if substepped
            else system.local_response(q, z, c, p)
        )[:, :4]

    @jax.jit
    def step(current, old_z, old_c, previous, z_dot, c_dot, previous_dot, memory_dot):
        def equation(z, c, p, previous):
            return (
                system.residual(current, z, c, p, jnp.zeros(d), previous)
                if substepped
                else system.residual(current, z, c, p, jnp.zeros(d))
            )

        def source_dot(dz, dc, dp, dqprev):
            return jax.jvp(
                equation,
                (old_z, old_c, system.parameters, previous),
                (dz, dc, dp, dqprev),
            )[1]

        direct = jax.vmap(source_dot)(z_dot.T, c_dot.T, jnp.eye(5), previous_dot.T).T
        tangent = (
            system.tangent(current, old_z, old_c, system.parameters, previous)
            if substepped
            else system.tangent(current, old_z, old_c, system.parameters)
        )
        q_dot = jnp.linalg.solve(tangent, (a0 @ memory_dot) / wr[:, None] - direct)

        def local_dot(dq, dz, dc, dp, dqprev):
            return jax.jvp(
                response,
                (current, old_z, old_c, system.parameters, previous),
                (dq, dz, dc, dp, dqprev),
            )[1]

        contact_dot = jax.vmap(local_dot)(q_dot.T, z_dot.T, c_dot.T, jnp.eye(5), previous_dot.T)
        force_dot = jnp.concatenate(
            (
                wc[:, None] * contact_dot[:, :, 0].T,
                wc[:, None] * contact_dot[:, :, 1].T,
                jnp.zeros((1, 5)),
            ),
            axis=0,
        )
        return (
            q_dot,
            contact_dot[:, :, 2].T,
            contact_dot[:, :, 3].T,
            cdiag[:, None] * q_dot - force_dot,
        )

    factory = OnlineConvolution if convolution_factory is None else convolution_factory
    online = [factory(history) for _ in range(5)]
    shared = {}
    for stream in online:
        stream.transforms = shared
    z_dot, c_dot, previous_dot = np.zeros((n, 5)), np.zeros((n, 5)), np.zeros((d, 5))
    previous_dot[:, :2] = prefix[0]
    for offset, index in enumerate(range(start, steps + 1)):
        past = np.column_stack([stream.past(offset) for stream in online])
        past[:, :2] += prefix[offset + 1]
        previous_dot, z_dot, c_dot, h_dot = (
            np.asarray(value)
            for value in step(
                q[index],
                result.plastic_slip[index - 1],
                result.cumulative_slip[index - 1],
                q[index - 1],
                z_dot,
                c_dot,
                previous_dot,
                past,
            )
        )
        if not all(np.all(np.isfinite(a)) for a in [previous_dot, z_dot, c_dot, h_dot]):
            raise ValueError(f"Nonfinite forward JVP at step {index}")
        for column, stream in enumerate(online):
            stream.append(h_dot[:, column])
    return dict(
        gradient=np.average(c_dot, weights=np.asarray(wc), axis=0),
        seconds_including_jit=perf_counter() - begun,
        parameter_order=["normal_penalty", "tangential_penalty", "mu_s", "mu_k", "Dc"],
        complete_prefix_derivatives_included=True,
        previous_displacement_tangent_included=substepped,
        shared_fft_kernel_bytes=sum(a.nbytes for a in shared.values()),
        scope="Five conditional parameter JVPs; no material/geometry, lift, initial-state or event-time derivatives",
    )

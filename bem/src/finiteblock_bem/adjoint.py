"""Discrete adjoint of the complete retained-coordinate CQ/contact trajectory.

Local derivatives come from JAX VJPs. Future bulk contributions use the exact
transpose CQ convolution in reverse time. No Newton iterates are retained.
Bulk matrices, static lift and the recorded actuator schedule are FIXED.
Switch/event-time derivatives and material/geometry/lift derivatives are not
included. Gradients are conditional on the converged active branches.
"""

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from .condensation import InterfaceContactEquilibrium
from .cq import BoundaryHistory, OnlineConvolution
from .jax_backend import JaxContactEquilibrium


def trajectory_adjoint(case, lift, history, result, *, equilibrium=None):
    """Differentiate final contact-weighted mean cumulative slip.

    Returns all five LSW parameter derivatives and per-step prescribed actuator
    derivatives, plus initial interface-state cotangents. The forward trajectory
    must contain all accepted steps and retained physical contact jumps.
    """
    started = perf_counter()
    steps = history.plan.steps
    if (
        result.diagnostics["status"] != "completed_requested_horizon"
        or result.contact_jumps is None
        or len(result.time) != steps + 1
        or np.max(result.residual_mpa) >= 1e-7
    ):
        raise ValueError("Adjoint requires a complete, converged interface trajectory")
    reference = (
        InterfaceContactEquilibrium(case, lift, history.instantaneous)
        if equilibrium is None
        else equilibrium
    )
    system = JaxContactEquilibrium(reference)
    scale = jnp.array(reference.constraints.reduced_weights)
    n = len(case.moving_contact)
    q = np.column_stack(
        (
            result.contact_jumps - np.r_[reference.base_dn, reference.base_ds],
            result.platen - reference.base_platen,
        )
    )

    @jax.jit
    def step(q, plastic, cumulative, parameters, eta, future):
        def local(q, plastic, cumulative, parameters):
            residual = system.residual(q, plastic, cumulative, parameters, jnp.zeros_like(q))
            state = system.local_response(q, plastic, cumulative, parameters)[:, 2:4]
            return residual, state

        _, pullback = jax.vjp(local, q, plastic, cumulative, parameters)
        state_q = pullback((jnp.zeros_like(q), eta))[0]
        jacobian = system.tangent(q, plastic, cumulative, parameters)
        mu = jnp.linalg.solve(jacobian.T, state_q - future)
        _, old_z, old_c, parameter_gradient = pullback((-mu, eta))
        return mu / scale, jnp.stack((old_z, old_c), axis=1), parameter_gradient

    transpose = BoundaryHistory(history.plan, history.weights.transpose(0, 2, 1))
    reverse = OnlineConvolution(transpose)
    reverse.append(np.zeros(history.plan.dofs))
    weights = reference.constraints.contact_weights
    eta = np.column_stack((np.zeros(n), weights / np.sum(weights)))
    gradient = np.zeros(5)
    actuator_gradient = np.zeros(steps + 1)
    for reverse_step, index in enumerate(range(steps, 0, -1), start=1):
        raw, eta, increment = step(
            jnp.asarray(q[index]),
            jnp.asarray(result.plastic_slip[index - 1]),
            jnp.asarray(result.cumulative_slip[index - 1]),
            system.parameters,
            jnp.asarray(eta),
            jnp.asarray(reverse.past(reverse_step)),
        )
        raw, eta, increment = (np.asarray(value) for value in (raw, eta, increment))
        if not all(np.all(np.isfinite(value)) for value in (raw, eta, increment)):
            raise FloatingPointError(f"Nonfinite adjoint at step {index}")
        gradient += increment
        actuator_gradient[index] = reference.spring * raw[-1]
        reverse.append(raw)
    return {
        "objective": "final contact-weighted mean cumulative slip",
        "value_mm": float(np.average(result.cumulative_slip[-1], weights=weights)),
        "parameter_order": ["normal_penalty", "tangential_penalty", "mu_s", "mu_k", "Dc"],
        "gradient": gradient,
        "actuator_gradient": actuator_gradient,
        "initial_plastic_slip_gradient": eta[:, 0],
        "initial_cumulative_slip_gradient": eta[:, 1],
        "seconds_including_jit": perf_counter() - started,
        "steps": steps,
        "scope": "Fixed bulk, static lift and recorded loading; active-branch discrete derivatives",
        "event_time_derivative_included": False,
    }

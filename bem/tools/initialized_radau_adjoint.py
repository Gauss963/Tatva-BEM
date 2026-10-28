"""Connect selected healed-preload sensitivities to the complete RK adjoint.

With measured static force F0 and initial platen u0, the physical affine
contact term is fc - fc0 + r0 = fc + F0 + spring*(u0-a0). Differentiating
this combined term in both the current root and every past source includes
the measured residual derivative without pretending r0 is exactly zero.
"""

import runpy
from pathlib import Path
from types import FunctionType, SimpleNamespace

import jax.numpy as jnp
import numpy as np

BASE = runpy.run_path(str(Path(__file__).with_name("radau_contact_adjoint.py")))


def initialized_context(static_sensitivity):
    dq0 = np.asarray(static_sensitivity["displacement_jacobian"])
    dz0 = np.asarray(static_sensitivity["plastic_jacobian"])
    df0 = np.asarray(static_sensitivity["retained_bulk_force_jacobian"])
    parameters0 = np.asarray(static_sensitivity["parameters"])

    def context(reference, first, second, wrap, impedance, stages, segment_substeps):
        ctx = BASE["context"](reference, first, second, wrap, impedance, stages, segment_substeps)
        d, m = ctx.d, ctx.m
        if (
            dq0.shape != (d, 5)
            or dz0.shape != (m, 5)
            or df0.shape != (d, 5)
            or not all(np.all(np.isfinite(a)) for a in [dq0, dz0, df0, parameters0])
            or not np.array_equal(parameters0, np.asarray(ctx.system.parameters))
        ):
            raise ValueError("Measured static sensitivities must align with RK grids/parameters")
        displacement, plastic, bulk = map(jnp.asarray, [dq0, dz0, df0])
        p0 = jnp.asarray(parameters0)
        matrix = ctx.a0 - jnp.diag(ctx.diagonal)
        original = ctx.system

        def shifts(parameters):
            change = parameters - p0
            return displacement @ change, plastic @ change, change

        def affine_force(change):
            # The spring reference is fixed; only the measured static platen
            # moves with the healed preload. The recorded actuator stays fixed.
            value = bulk @ change
            return value.at[-1].add(reference.spring * (displacement @ change)[-1])

        def response(q, z, c, parameters, previous):
            shift, offset, change = shifts(parameters)
            values = original.local_response(
                q + jnp.tile(shift, 2), z + offset, c, parameters, previous + shift
            )
            # State coordinates subtract the initial plastic sensitivity, so
            # the tested complete AD loops retain zero initial tangents.
            values = values.at[:, :, 2].add(-offset[:, None])
            # Carry the five affine parameter coordinates through the same
            # local JVP/VJP. Their force map is linear, including its platen.
            carried = jnp.broadcast_to(change, (m, 2, 5))
            return jnp.concatenate((values, carried), axis=-1)

        def force(values):
            return ctx.force(values[:, :, :7]) + jnp.tile(affine_force(values[0, 0, 7:]), 2)

        def equation(q, z, c, parameters, previous):
            shift, offset, change = shifts(parameters)
            tiled = jnp.tile(shift, 2)
            correction = jnp.tile(affine_force(change), 2) - matrix @ tiled
            return (
                ctx.equation(q + tiled, z + offset, c, parameters, previous + shift)
                + correction / ctx.wr
            )

        def tangent(q, z, c, parameters, previous):
            shift, offset, _ = shifts(parameters)
            return original.tangent(
                q + jnp.tile(shift, 2), z + offset, c, parameters, previous + shift
            )

        return SimpleNamespace(
            **{
                **ctx.__dict__,
                "system": SimpleNamespace(
                    parameters=original.parameters, local_response=response, tangent=tangent
                ),
                "force": force,
                "equation": equation,
            }
        )

    return context


def initialized_gradient(method, static_sensitivity, *arguments, **keywords):
    """Reuse the tested complete causal loops with a measured static context."""
    if method not in ("forward_directions", "reverse_gradient"):
        raise ValueError("Choose the complete RK forward or reverse derivative engine")
    original = BASE[method]
    engine = FunctionType(
        original.__code__,
        {**original.__globals__, "context": initialized_context(static_sensitivity)},
        original.__name__,
        original.__defaults__,
    )
    engine.__kwdefaults__ = original.__kwdefaults__
    answer = engine(*arguments, **keywords)
    answer.update(
        full_initialized_gradient=True,
        selected_healed_static_displacement_sensitivity_included=True,
        initial_plastic_offset_sensitivity_included=True,
        measured_static_force_and_residual_sensitivities_included=True,
        every_initial_force_history_source_included=True,
        event_time_derivative_included=False,
        scope="Fixed bulk/restart input/recorded stage loading; selected healed static and transient branches; no material/geometry or stopping-time derivative",
    )
    return answer

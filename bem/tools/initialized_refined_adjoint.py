"""Contact-parameter AD including the healed static lift and plastic offset.

The bulk, original restart input and recorded actuator schedule stay fixed.
The converged static active set and transient branches are selected. State
coordinates subtract the initial-offset sensitivity, allowing the existing
complete causal forward/reverse engines to retain zero initial tangents.
"""

import runpy
from pathlib import Path
from types import FunctionType

import jax.numpy as jnp
import numpy as np

TOOLS = Path(__file__).resolve().parent
BASE = runpy.run_path(str(TOOLS / "refined_contact_adjoint.py"))


def initialized_context(static_system, static_sensitivity):
    dq0 = jnp.asarray(static_sensitivity["displacement_jacobian"])
    dz0 = jnp.asarray(static_sensitivity["plastic_jacobian"])
    parameters0 = jnp.asarray(static_sensitivity["parameters"])

    def context(reference, history, result, constitutive_substeps):
        values = BASE["context"](reference, history, result, constitutive_substeps)
        system, wc, wr, diagonal, a0, response, equation, tangent, _ = values
        d, m = history.plan.dofs, len(wc)
        if dq0.shape != (d, 5) or dz0.shape != (m, 5):
            raise ValueError("Static sensitivities must align with bulk and contact grids")
        if not np.array_equal(np.asarray(parameters0), np.asarray(system.parameters)):
            raise ValueError("Static and transient physical parameters differ")
        bn = jnp.asarray(reference.constraints.normal_jump)
        bt = jnp.asarray(reference.constraints.tangential_jump)
        # Shifting absolute contact kinematics must add the STATIC bulk
        # force, not the instantaneous dynamic impedance times that shift.
        difference = jnp.asarray(static_system.matrix - reference.matrix)
        initial_z = jnp.asarray(result.plastic_slip[0])
        dn0, ds0 = jnp.asarray(reference.base_dn), jnp.asarray(reference.base_ds)

        def shifted_response(q, z, c, parameters, previous):
            shift, offset = dq0 @ (parameters - parameters0), dz0 @ (parameters - parameters0)
            local = response(q + shift, z + offset, c, parameters, previous + shift)
            local = local.at[:, 2].add(-offset)
            # Include the initial contact force derivative in every future
            # correction source. Omitting it would be another history-source bug.
            kn, kt = parameters[:2]
            dn = dn0 + bn @ shift
            pressure0 = kn * jnp.maximum(dn, 0.0)
            shear0 = jnp.where(dn > 0, kt * (ds0 + bt @ shift - initial_z - offset), 0.0)
            return jnp.column_stack((local, pressure0, shear0))

        def shifted_equation(q, z, c, parameters, previous):
            shift, offset = dq0 @ (parameters - parameters0), dz0 @ (parameters - parameters0)
            return (
                equation(q + shift, z + offset, c, parameters, previous + shift)
                + (difference @ shift) / wr
            )

        def correction_force(local):
            return bn.T @ (wc * (local[..., 0] - local[..., 4])) + bt.T @ (
                wc * (local[..., 1] - local[..., 5])
            )

        return (
            system,
            wc,
            wr,
            diagonal,
            a0,
            shifted_response,
            shifted_equation,
            tangent,
            correction_force,
        )

    return context


def initialized_gradient(
    method, static_system, static_sensitivity, reference, history, result, **keywords
):
    if method not in ("forward_gradient", "reverse_gradient"):
        raise ValueError("Choose the complete forward or reverse gradient engine")
    original = BASE[method]
    # Explicitly inject only the mathematical context; reuse the independently
    # tested causal loop, previous-path cotangents and actuator contributions.
    engine = FunctionType(
        original.__code__,
        {**original.__globals__, "context": initialized_context(static_system, static_sensitivity)},
        original.__name__,
        original.__defaults__,
    )
    engine.__kwdefaults__ = original.__kwdefaults__
    answer = engine(reference, history, result, **keywords)
    answer.update(
        initial_static_contact_sensitivity_included=True,
        initial_plastic_offset_sensitivity_included=True,
        initial_contact_force_history_source_included=True,
        scope="Fixed bulk/restart input/recorded loading; selected healed static and transient branches",
    )
    return answer

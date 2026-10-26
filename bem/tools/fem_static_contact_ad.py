"""Selected-active-set AD of Tatva's healed Coulomb static initialization.

This differentiates a converged linear equilibrium and its output plastic
offset, rather than active-set iterations. It is a building block: the
ordinary transient conditional adjoint still holds its static lift fixed.
"""

import jax
import jax.numpy as jnp
import numpy as np


def healed_static_map(system, input_state, root):
    """Build p -> (static q, initial plastic offset) on the selected active set."""
    con, law = system.constraints, system.case.contact
    bn, bt = jnp.asarray(con.normal_jump), jnp.asarray(con.tangential_jump)
    wc = jnp.asarray(con.contact_weights)
    seed, cumulative = (
        jnp.asarray(input_state.plastic_slip),
        jnp.asarray(input_state.cumulative_slip),
    )
    base_dn, base_ds = jnp.asarray(system.base_dn), jnp.asarray(system.base_ds)
    p0 = jnp.array(
        [
            law.normal_penalty,
            law.tangential_penalty,
            law.static_friction,
            law.kinetic_friction,
            law.characteristic_slip,
        ]
    )
    closure = np.asarray(base_dn + bn @ root)
    trial = law.tangential_penalty * np.asarray(base_ds + bt @ root - seed)
    mu = law.coefficient(input_state.cumulative_slip)
    active = closure > 0
    sliding = active & (abs(trial) > mu * law.normal_penalty * closure)
    direction = jnp.asarray(np.where(sliding, np.sign(trial), 0.0))
    active, sliding = jnp.asarray(active), jnp.asarray(sliding)

    def solve(parameters, normal, actuator):
        kn, kt, mus, muk, dc = parameters
        coefficient = mus - (mus - muk) * jnp.minimum(cumulative / dc, 1.0)
        n = wc * kn * active
        t = wc * kt * (active & ~sliding)
        s = wc * coefficient * kn * direction * (active & sliding)
        matrix = (
            jnp.asarray(system.matrix)
            + bn.T @ (n[:, None] * bn)
            + bt.T @ (t[:, None] * bt)
            + bt.T @ (s[:, None] * bn)
        )
        constant = bn.T @ (n * base_dn) + bt.T @ (t * (base_ds - seed) + s * base_dn)
        # The external normal-load API is discrete/NumPy. This function
        # differentiates contact parameters and actuator only, holding normal fixed.
        external = jnp.asarray(con.reduce_force(system.external_traction(normal) - system.base_t))
        external = external.at[-1].add(system.spring * (actuator - system.base_platen))
        q = jnp.linalg.solve(matrix, external - constant)
        pressure = kn * (base_dn + bn @ q)
        shear = direction * coefficient * pressure
        plastic = jnp.where(sliding, base_ds + bt @ q - shear / kt, seed)
        return q, plastic

    return (
        solve,
        p0,
        dict(
            active_points=int(np.asarray(active).sum()),
            sliding_points=int(np.asarray(sliding).sum()),
            scope="Selected static active set; input plastic offset and healed memory fixed",
            active_set_iterations_differentiated=False,
            full_transient_initialization_gradient_connected=False,
        ),
    )


def healed_static_parameter_jacobian(system, normal, actuator, input_state, root):
    solve, parameters, diagnostic = healed_static_map(system, input_state, root)

    def mapping(parameters):
        return solve(parameters, normal, actuator)

    q, plastic = mapping(parameters)
    if np.max(abs(np.asarray(q) - root)) > 1e-8:
        raise ValueError("Selected static AD map differs from the converged active-set root")
    dq, dz = jax.jacfwd(mapping)(parameters)
    if not all(np.all(np.isfinite(v)) for v in (dq, dz, plastic)):
        raise ValueError("Nonfinite selected-active-set static sensitivity")
    return dict(
        displacement_jacobian=np.asarray(dq),
        plastic_jacobian=np.asarray(dz),
        plastic_slip=np.asarray(plastic),
        parameters=np.asarray(parameters),
        parameter_order=["normal_penalty", "tangential_penalty", "mu_s", "mu_k", "Dc"],
        **diagnostic,
    )

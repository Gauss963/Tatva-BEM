"""Exact nodal contact-tangent assembly using the original local JAX AD.

Only identity normal/tangential jump maps use this optional specialization.
The dense elastic matrix, original residual, Newton/line search and local law
remain unchanged. General interpolation maps retain the original tangent.
"""

import runpy
from pathlib import Path
from types import FunctionType

import jax
import jax.numpy as jnp
import numpy as np

BASE = runpy.run_path(str(Path(__file__).with_name("radau_contact_experiment.py")))


def closure_values(function):
    return {
        name: cell.cell_contents
        for name, cell in zip(
            function.__code__.co_freevars, function.__closure__ or (), strict=True
        )
    }


def identity_tangent(system):
    """Return a finite-value fast path plus the unchanged nonfinite fallback."""
    original = system.tangent.__wrapped__
    values = closure_values(original)
    required = {"bn", "bt", "bulk", "differentiated", "jumps", "previous_jumps", "wc", "wr"}
    if set(values) != required:
        raise ValueError("Original stage tangent closures changed; re-audit the specialization")
    n, d = system.nodes, system.size
    if d != 2 * n + 1:
        return None
    normal = np.pad(np.eye(n), ((0, 0), (0, n + 1)))
    tangent = np.pad(np.eye(n), ((0, 0), (n, 1)))
    if not np.array_equal(values["bn"], normal) or not np.array_equal(values["bt"], tangent):
        return None
    i, a, alpha, b, beta = np.indices((n, 2, 2, 2, 2))
    rows, columns = (a * d + alpha * n + i).ravel(), (b * d + beta * n + i).ravel()
    if len(np.unique(rows * (2 * d) + columns)) != 16 * n:
        raise ValueError("Contact scatter indices must be unique")
    rows, columns = jnp.asarray(rows), jnp.asarray(columns)
    bulk, weights, reduced = (values[k] for k in ["bulk", "wc", "wr"])
    differentiate, jumps, previous_jumps = (
        values[k] for k in ["differentiated", "jumps", "previous_jumps"]
    )

    def evaluate(q, z, c, parameters, previous):
        derivative = differentiate(jumps(q), z, c, previous_jumps(previous), parameters)
        force_derivative = derivative[:, :, :2, :, :]
        updates = weights[:, None, None, None, None] * force_derivative

        def finite(_):
            contact = (
                jnp.zeros_like(bulk).at[rows, columns].set(updates.ravel(), unique_indices=True)
            )
            return (bulk + contact) / reduced[:, None]

        return jax.lax.cond(
            jnp.all(jnp.isfinite(force_derivative)),
            finite,
            lambda _: original(q, z, c, parameters, previous),
            operand=None,
        )

    return evaluate


def stage_equilibrium(reference, matrix, *, segment_substeps=(1, 2)):
    """Construct a private original system, then bind only its contact tangent."""
    system = BASE["RadauContactEquilibrium"](reference, matrix, segment_substeps=segment_substeps)
    tangent = identity_tangent(system)
    if tangent is None:
        system.tangent_policy = dict(applied=False, reason="nonidentity_contact_maps")
        return system
    root = system.implicit_displacement.__wrapped__
    if set(root.__code__.co_freevars) != {"cell", "original", "residual", "tangent"}:
        raise ValueError("Original stage root closures changed; re-audit the specialization")

    def cell(value):
        return (lambda: value).__closure__[0]

    replaced = tuple(
        cell(tangent) if name == "tangent" else old
        for name, old in zip(root.__code__.co_freevars, root.__closure__, strict=True)
    )
    rebound = FunctionType(
        root.__code__, root.__globals__, root.__name__, root.__defaults__, replaced
    )
    rebound.__kwdefaults__ = root.__kwdefaults__
    system.tangent, system.implicit_displacement = jax.jit(tangent), jax.jit(rebound)
    system.tangent_policy = dict(
        applied=True,
        contact_nodes=system.nodes,
        stage_dofs=2 * system.size,
        contact_scatter_entries=16 * system.nodes,
        dense_contact_products_removed=16,
        local_ad_unchanged=True,
        original_dense_bulk_retained=True,
        original_residual_and_root_unchanged=True,
        nonfinite_uses_original_tangent=True,
    )
    return system

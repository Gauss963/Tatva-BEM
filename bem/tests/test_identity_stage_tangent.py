"""Structured tangent equals the original residual derivative and implicit root."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from finiteblock_bem.friction import ContactState, SlipWeakeningContact  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1] / "tools"
POLICY = runpy.run_path(str(TOOLS / "identity_stage_tangent.py"))


def reference(n=3):
    d = 2 * n + 1
    return SimpleNamespace(
        base_dn=np.full(n, 0.1),
        base_ds=np.zeros(n),
        base_platen=0.0,
        case=SimpleNamespace(contact=SlipWeakeningContact(100, 40, 0.8, 0.5, 0.2)),
        constraints=SimpleNamespace(
            size=d,
            normal_jump=np.pad(np.eye(n), ((0, 0), (0, n + 1))),
            tangential_jump=np.pad(np.eye(n), ((0, 0), (n, 1))),
            contact_weights=np.linspace(0.3, 2, n),
            reduced_weights=np.r_[np.linspace(0.3, 2, n), np.linspace(0.3, 2, n), 3],
        ),
    )


@pytest.mark.parametrize("path", ["stick", "slide", "reverse", "open"])
def test_tangent_matches_original_and_independent_full_residual_jacfwd(path):
    jax.config.update("jax_enable_x64", True)
    ref = reference()
    n, d = len(ref.base_dn), ref.constraints.size
    matrix = 0.01 * np.random.default_rng(44).normal(size=(2 * d, 2 * d)) + np.eye(2 * d)
    original = POLICY["BASE"]["RadauContactEquilibrium"](ref, matrix, segment_substeps=(4, 8))
    candidate = POLICY["stage_equilibrium"](ref, matrix, segment_substeps=(4, 8))
    q = np.zeros((2, d))
    if path != "stick":
        q[0, n : 2 * n], q[1, n : 2 * n] = 0.36, 0.42
    if path == "reverse":
        q[1, n : 2 * n] = -0.2
    if path == "open":
        q[1, :n] = -0.2
    q = jnp.asarray(q.ravel())
    z, c, previous = jnp.zeros(n), jnp.zeros(n), jnp.zeros(d)
    args = (q, z, c, candidate.parameters, previous)
    actual = np.asarray(candidate.tangent(*args))
    expected = np.asarray(original.tangent(*args))
    oracle = jax.jacfwd(
        lambda value: original.residual(
            value, z, c, original.parameters, jnp.zeros(2 * d), previous
        )
    )(q)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-12)
    np.testing.assert_allclose(actual, oracle, rtol=2e-13, atol=2e-12)
    assert candidate.tangent_policy["contact_scatter_entries"] == 16 * n


def test_implicit_root_and_selected_parameter_derivative_preserved():
    ref = reference(1)
    ref.constraints.contact_weights[:] = 1
    ref.constraints.reduced_weights[:] = 1
    matrix = np.diag([50.0, 30.0, 1.0, 50.0, 30.0, 1.0])
    matrix[1, 4], matrix[4, 1] = -0.4, 0.7
    systems = [
        factory(ref, matrix, segment_substeps=(4, 8))
        for factory in [POLICY["BASE"]["RadauContactEquilibrium"], POLICY["stage_equilibrium"]]
    ]
    forcing = jnp.array([-10.0, -16.3, 0.0, -10.0, -14.6, 0.0])
    previous, guess, z, c = jnp.zeros(3), jnp.zeros(6), jnp.zeros(1), jnp.zeros(1)
    roots, gradients = [], []
    for system in systems:
        answer = system.solve(ContactState(np.zeros(1), np.zeros(1)), forcing, guess, previous)
        assert answer.converged and answer.residual_mpa < 1e-7
        roots.append(answer.displacement)

        def loss(parameters):
            q, _ = system.implicit_displacement(parameters, z, c, forcing, guess, previous)
            return system.local_response(q, z, c, parameters, previous)[0, 1, 3]

        gradients.append(np.asarray(jax.grad(loss)(system.parameters)))
    np.testing.assert_allclose(roots[0], roots[1], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(gradients[0], gradients[1], rtol=1e-10, atol=1e-12)
    assert abs(gradients[0][-1]) > 1e-6


def test_general_contact_maps_keep_original_tangent():
    ref = reference(2)
    ref.constraints.normal_jump[0, :2] = [0.75, 0.25]
    system = POLICY["stage_equilibrium"](ref, np.eye(10))
    assert not system.tangent_policy["applied"]
    assert system.tangent_policy["reason"] == "nonidentity_contact_maps"

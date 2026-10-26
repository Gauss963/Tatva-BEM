import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import block_diag

from finiteblock_bem.nodal import load_nodal_model

ROOT = Path(__file__).resolve().parents[1]
LIMIT = runpy.run_path(str(ROOT / "tools/radiation_limit_experiment.py"))
CHAIN = runpy.run_path(str(ROOT / "tools/structured_radiation.py"))["RadiationChain"]
CASE = ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def reference_model(**options):
    model = load_nodal_model(CASE, **options)
    body = [
        LIMIT["body_limits"](m, model.reference.material)["mixed"]
        for m in (model.reference.moving, model.reference.stationary)
    ]
    reference = LIMIT["condense"](model, body)
    return model, body, reference


@pytest.fixture(scope="module")
def coarse_reference():
    return reference_model(max_panel_length=100, contact_panel_length=60)


@pytest.fixture(scope="module")
def event_reference():
    # Same 250+254 faces and 137 contact nodes as the existing Q12 event kernel.
    return reference_model(
        max_panel_length=20,
        contact_panel_length=5,
        contact_end_length=0.5,
        corner_panel_length=0.5,
        restart_knot_window=8,
        restart_knot_subdivisions=4,
    )


@pytest.mark.parametrize("fixture", ["coarse_reference", "event_reference"])
def test_structured_limit_matches_dense_constrained_energy(fixture, request):
    model, body, reference = request.getfixturevalue(fixture)
    chain = CHAIN.from_dense(model, reference)
    if fixture == "event_reference":
        assert chain.nodes == 137
        assert [model.reference.moving.count, model.reference.stationary.count] == [250, 254]
    assert chain.diagnostics["reference_relative_energy_frobenius"] < 1e-11
    assert chain.stored_bytes < reference.nbytes
    p = model.constraints.prolongation
    jump = np.vstack(
        (model.constraints.normal_jump, model.constraints.tangential_jump, np.eye(p.shape[1])[-1])
    )
    k = p.T @ block_diag(*body) @ p
    q = np.random.default_rng(31).normal(size=len(reference))
    # Independent primal constrained trace minimizes the complete body energy.
    trace = np.linalg.solve(k, jump.T @ reference @ q)
    np.testing.assert_allclose(jump @ trace, q, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(q @ chain.dense() @ q, trace @ k @ trace, rtol=2e-12)


def mixed_tangent(model):
    # Populate independently selected sticking, positive/negative sliding and
    # open branches of the actual previous-state LSW law, including d tau/dn.
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from finiteblock_bem.jax_backend import contact_tangents

    n = len(model.moving_contact)
    law = model.reference.contact
    theta = jnp.array(
        [
            law.normal_penalty,
            law.tangential_penalty,
            law.static_friction,
            law.kinetic_friction,
            law.characteristic_slip,
        ]
    )
    jumps = np.tile(
        [[1e-4, 1e-5], [1e-4, 0.003], [1e-4, -0.003], [-1e-4, 0.002]], ((n + 3) // 4, 1)
    )[:n]
    c = np.linspace(0, 2 * law.characteristic_slip, n)
    tangent = np.asarray(contact_tangents(jnp.array(jumps), jnp.zeros(n), jnp.array(c), theta))[
        :, :2
    ]
    return tangent * model.constraints.contact_weights[:, None, None]


def test_real_friction_nonsymmetric_forward_and_transpose_solves(event_reference):
    model, _, reference = event_reference
    chain = CHAIN.from_dense(model, reference)
    tangent = mixed_tangent(model)
    coefficient, spring = 1 / 1.2492928633545953e-8, 25 * model.constraints.platen_width
    matrix = chain.dense_step(coefficient, tangent, spring)
    assert np.max(abs(matrix - matrix.T)) > 1e3
    rhs = np.random.default_rng(34).normal(size=(len(reference), 3))
    for transpose in (False, True):
        a = matrix.T if transpose else matrix
        actual = chain.solve(rhs, coefficient, tangent, spring, transpose=transpose)
        expected = np.linalg.solve(a, rhs)
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-15)
        residual = (a @ actual - rhs) / np.sqrt(np.diag(matrix))[:, None]
        assert np.max(abs(residual)) < 1e-11
    f, g = rhs[:, 0], rhs[:, 1]
    np.testing.assert_allclose(
        g @ chain.solve(f, coefficient, tangent, spring),
        f @ chain.solve(g, coefficient, tangent, spring, transpose=True),
        rtol=2e-12,
    )
    import jax
    import jax.numpy as jnp

    fast = chain.jax_solver()
    actual = fast(jnp.asarray(f), coefficient, jnp.asarray(tangent), spring)
    np.testing.assert_allclose(actual, np.linalg.solve(matrix, f), rtol=2e-10, atol=1e-15)
    reverse = jax.grad(
        lambda b: jnp.asarray(g) @ fast(b, coefficient, jnp.asarray(tangent), spring)
    )(jnp.asarray(f))
    np.testing.assert_allclose(reverse, np.linalg.solve(matrix.T, g), rtol=2e-10, atol=1e-15)


def test_reconstruction_guard_rejects_nonendpoint_coupling(coarse_reference):
    model, _, reference = coarse_reference
    bad = reference.copy()
    n = len(model.moving_contact)
    # This remains a small symmetric SPD perturbation but cannot be represented
    # by the analytically derived endpoint/platen border. Never discard it.
    bad[2, n + 2] += 1e-4 * np.sqrt(reference[2, 2] * reference[n + 2, n + 2])
    bad[n + 2, 2] = bad[2, n + 2]
    assert np.linalg.eigvalsh(bad)[0] > 0
    with pytest.raises(ValueError, match="five-coordinate structure"):
        CHAIN.from_dense(model, bad)


@pytest.mark.parametrize("change", ["pressure_shear", "negative", "nonfinite", "scale", "spring"])
def test_unsupported_step_inputs_are_rejected(coarse_reference, change):
    model, _, reference = coarse_reference
    chain = CHAIN.from_dense(model, reference)
    tangent = np.zeros((chain.nodes, 2, 2))
    coefficient, spring = 1.0, 1.0
    if change == "pressure_shear":
        tangent[0, 0, 1] = 1.0
    elif change == "negative":
        tangent[0, 1, 1] = -1.0
    elif change == "nonfinite":
        tangent[0, 0, 0] = np.nan
    elif change == "scale":
        coefficient = 0.0
    else:
        spring = -1.0
    with pytest.raises(ValueError):
        chain.solve(np.ones(len(reference)), coefficient, tangent, spring)


def test_jax_implicit_derivatives_include_scale_tangent_border_and_spring(coarse_reference):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    # Import the pure production law to use the same enabled float64 policy.
    from finiteblock_bem.jax_backend import contact_point

    assert callable(contact_point)
    model, _, reference = coarse_reference
    chain = CHAIN.from_dense(model, reference)
    solve = chain.jax_solver()
    n = chain.nodes
    rhs = jnp.asarray(np.random.default_rng(37).normal(size=len(reference)))
    weights = jnp.linspace(0.4, 1.2, len(reference))
    d = jnp.linspace(1.0, 2.0, n)

    def inputs(parameters):
        tangent = jnp.zeros((n, 2, 2)).at[:, 0, 0].set(parameters[1] * d)
        tangent = tangent.at[:, 1, 1].set(parameters[2] * d)
        tangent = tangent.at[:, 1, 0].set(parameters[3] * d)
        return parameters[0], tangent, parameters[4]

    def loss(parameters, dense):
        a, tangent, spring = inputs(parameters)
        if dense:
            matrix = a * jnp.asarray(reference)
            matrix = matrix.at[jnp.arange(n), jnp.arange(n)].add(tangent[:, 0, 0])
            matrix = matrix.at[n + jnp.arange(n), n + jnp.arange(n)].add(tangent[:, 1, 1])
            matrix = matrix.at[n + jnp.arange(n), jnp.arange(n)].add(tangent[:, 1, 0])
            matrix = matrix.at[-1, -1].add(spring)
            value = jnp.linalg.solve(matrix, rhs)
        else:
            value = solve(rhs, a, tangent, spring)
        return weights @ value

    theta = jnp.array([1e5, 300.0, 10.0, 90.0, 100.0])
    a, tangent, spring = inputs(theta)
    np.testing.assert_allclose(
        solve(rhs, a, tangent, spring), chain.solve(rhs, a, tangent, spring), rtol=2e-12, atol=1e-13
    )
    for dense in (False, True):

        def fn(p, dense=dense):
            return loss(p, dense)

        gradient = np.asarray(jax.grad(fn)(theta))
        forward = np.array([jax.jvp(fn, (theta,), (direction,))[1] for direction in jnp.eye(5)])
        np.testing.assert_allclose(gradient, forward, rtol=2e-11, atol=1e-13)
        if not dense:
            structured = gradient
        else:
            np.testing.assert_allclose(structured, gradient, rtol=2e-10, atol=1e-13)
    for i in range(5):
        step = float(theta[i]) * 1e-4
        fd = float(
            (loss(theta.at[i].add(step), False) - loss(theta.at[i].add(-step), False)) / (2 * step)
        )
        np.testing.assert_allclose(structured[i], fd, rtol=2e-6, atol=1e-12)
    # The derivative with respect to arbitrary boundary forcing uses J^{-T}.
    matrix = chain.dense_step(float(a), np.asarray(tangent), float(spring))
    actual = jax.grad(lambda b: weights @ solve(b, a, tangent, spring))(rhs)
    np.testing.assert_allclose(
        actual, np.linalg.solve(matrix.T, np.asarray(weights)), rtol=2e-11, atol=1e-13
    )
    bad = tangent.at[0, 0, 1].set(1.0)
    assert np.all(np.isnan(solve(rhs, a, bad, spring)))

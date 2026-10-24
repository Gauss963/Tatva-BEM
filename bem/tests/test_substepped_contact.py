import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from finiteblock_bem.friction import ContactState, SlipWeakeningContact  # noqa: E402
from finiteblock_bem.jax_backend import JaxContactEquilibrium, contact_point  # noqa: E402

TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/substepped_contact_experiment.py")
)


def reference():
    con = SimpleNamespace(
        size=3,
        normal_jump=np.array([[1.0, 0, 0]]),
        tangential_jump=np.array([[0.0, 1, 0]]),
        contact_weights=np.ones(1),
        reduced_weights=np.ones(3),
        reduce_force=lambda x: x,
    )
    return SimpleNamespace(
        constraints=con,
        case=SimpleNamespace(contact=SlipWeakeningContact(100, 40, 0.8, 0.5, 0.2)),
        base_dn=np.array([0.1]),
        base_ds=np.zeros(1),
        base_platen=0.0,
        base_u=np.zeros(3),
        base_t=np.array([-10.0, -14.2, 0]),
        matrix=np.diag([50.0, 30.0, 1.0]),
        spring=1.0,
        external_traction=lambda _: np.zeros(3),
    )


def test_one_substep_is_the_identical_local_law_and_solver():
    parameters = jnp.array([100.0, 40.0, 0.8, 0.5, 0.2])
    law = TOOL["substepped_point"](1)
    for jump in [[0.1, 0.03], [0.1, 0.3], [0.1, -0.3], [-0.1, 0.3]]:
        expected = contact_point(jnp.array(jump), 0.01, 0.02, parameters)
        actual = law(jnp.array(jump), 0.01, 0.02, jnp.array([0.1, 0]), parameters)
        np.testing.assert_array_equal(actual[:5], expected)
    ref = reference()
    old = ContactState(np.zeros(1), np.zeros(1))
    baseline = JaxContactEquilibrium(ref).solve(0.0, 0.0, old)
    candidate = TOOL["SubsteppedContactEquilibrium"](ref, substeps=1).solve(0.0, 0.0, old)
    np.testing.assert_array_equal(candidate.reduced_displacement, baseline.reduced_displacement)
    np.testing.assert_array_equal(
        candidate.contact.state.cumulative_slip, baseline.contact.state.cumulative_slip
    )


def test_prescribed_path_converges_to_the_continuous_softening_solution():
    parameters = jnp.array([100.0, 40.0, 0.8, 0.5, 0.2])
    # Constant pressure p=16; yield at ds=.32. Before Dc, the exact
    # rate-independent positive-slip solution is z=(kt*ds-mus*p)/(kt-p*a).
    exact = (40 * 0.37 - 0.8 * 16) / (40 - 16 * (0.8 - 0.5) / 0.2)
    errors = []
    work_gaps = []
    for count in [1, 8, 32, 128]:
        value = np.asarray(
            jax.jit(TOOL["substepped_point"](count))(
                jnp.array([0.16, 0.37]), 0.0, 0.0, jnp.array([0.16, 0.0]), parameters
            )
        )
        errors.append(abs(value[3] - exact))
        work_gaps.append(value[5] - value[6])
        assert 0 <= value[3] < 0.2 and value[5] >= 0
        assert abs(value[1]) <= value[0] * value[4] + 1e-12
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < 0.012
    assert np.all(np.asarray(work_gaps) >= -1e-12)
    assert work_gaps[-1] < work_gaps[0]


def test_implicit_substep_ad_includes_previous_displacement():
    ref = reference()
    system = TOOL["SubsteppedContactEquilibrium"](ref, substeps=16)
    old = jnp.zeros(1)
    # At least two sliding microsteps are needed for previous jump to affect c.
    forcing = jnp.array([-10.0, -14.6, 0.0])
    guess = jnp.array([0.0, 0.23, 0.0])

    def loss(parameters, previous):
        q, _ = system.implicit_displacement(parameters, old, old, forcing, guess, previous)
        assert q.shape == (3,)
        return system.local_response(q, old, old, parameters, previous)[0, 3]

    previous = jnp.zeros(3)
    gp, gq = jax.grad(loss, argnums=(0, 1))(system.parameters, previous)
    eps = 1e-6
    finite = (
        loss(system.parameters, previous.at[1].add(eps))
        - loss(system.parameters, previous.at[1].add(-eps))
    ) / (2 * eps)
    assert abs(gq[1]) > 1e-6
    np.testing.assert_allclose(gq[1], finite, rtol=1e-5, atol=1e-8)
    finite = (
        loss(system.parameters.at[2].add(eps), previous)
        - loss(system.parameters.at[2].add(-eps), previous)
    ) / (2 * eps)
    np.testing.assert_allclose(gp[2], finite, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("count", [0, -1, 1.5])
def test_invalid_substep_count_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        TOOL["substepped_point"](count)

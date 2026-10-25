"""Stage paths commit once and retain previous/state/parameter derivatives."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from finiteblock_bem.friction import ContactState, SlipWeakeningContact  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1] / "tools"
TOOL = runpy.run_path(str(TOOLS / "radau_contact_experiment.py"))


def reference():
    return SimpleNamespace(
        base_dn=np.array([0.1]),
        base_ds=np.zeros(1),
        base_platen=0.0,
        case=SimpleNamespace(contact=SlipWeakeningContact(100, 40, 0.8, 0.5, 0.2)),
        constraints=SimpleNamespace(
            size=3,
            normal_jump=np.array([[1.0, 0, 0]]),
            tangential_jump=np.array([[0.0, 1, 0]]),
            contact_weights=np.ones(1),
            reduced_weights=np.ones(3),
        ),
    )


@pytest.mark.parametrize(
    "jumps",
    [
        np.array([[0.16, 0.36], [0.13, 0.42]]),
        np.array([[0.16, 0.36], [-0.1, -0.2]]),
        np.array([[0.16, -0.36], [0.13, 0.2]]),
    ],
)
def test_stage_map_matches_independent_numpy_path(jumps):
    law = reference().case.contact
    point = jax.jit(TOOL["staged_point"]((4, 8)))
    previous = np.array([0.16, 0.0])
    actual = np.asarray(point(jumps, 0.0, 0.0, previous, jnp.array([100, 40, 0.8, 0.5, 0.2])))
    state = ContactState(np.zeros(1), np.zeros(1))
    start = previous
    for stage, count in enumerate([4, 8]):
        work = 0.0
        for k in range(count):
            jump = start + (k + 1) / count * (jumps[stage] - start)
            result = law.evaluate(jump[:1], jump[1:], state)
            work += result.shear[0] * (result.state.plastic_slip[0] - state.plastic_slip[0])
            state = result.state
        expected = [
            result.pressure[0],
            result.shear[0],
            state.plastic_slip[0],
            state.cumulative_slip[0],
            result.friction_coefficient[0],
            work,
        ]
        np.testing.assert_allclose(actual[stage, :6], expected, rtol=1e-12, atol=1e-13)
        start = jumps[stage]


def test_endpoint_stage_depends_on_first_stage_trial_state():
    point = TOOL["staged_point"]((4, 8))
    p = jnp.array([100.0, 40.0, 0.8, 0.5, 0.2])
    jumps, previous = jnp.array([[0.16, 0.36], [0.16, 0.40]]), jnp.array([0.16, 0.0])
    derivative = jax.jacfwd(lambda q: point(q, 0.0, 0.0, previous, p)[1, 3])(jumps)
    assert abs(derivative[0, 1]) > 1e-5
    h = 1e-6
    finite = (
        point(jumps.at[0, 1].add(h), 0.0, 0.0, previous, p)[1, 3]
        - point(jumps.at[0, 1].add(-h), 0.0, 0.0, previous, p)[1, 3]
    ) / (2 * h)
    np.testing.assert_allclose(derivative[0, 1], finite, rtol=1e-5, atol=1e-8)


def test_coupled_root_implicit_ad_and_single_accepted_endpoint():
    ref = reference()
    matrix = np.diag([50.0, 30.0, 1.0, 50.0, 30.0, 1.0])
    matrix[1, 4], matrix[4, 1] = -0.4, 0.7
    system = TOOL["RadauContactEquilibrium"](ref, matrix, segment_substeps=(4, 8))
    # First stage weakens on multiple microsteps; the endpoint unloads and
    # retains that state. A fully weakened endpoint would erase this particular
    # previous-displacement derivative on the monotone constant-pressure path.
    forcing, previous = jnp.array([-10.0, -16.3, 0.0, -10.0, -14.6, 0.0]), jnp.zeros(3)
    z, c, guess = jnp.zeros(1), jnp.zeros(1), jnp.zeros(6)

    def loss(p, prev):
        q, _ = system.implicit_displacement(p, z, c, forcing, guess, prev)
        return system.local_response(q, z, c, p, prev)[0, 1, 3]

    gp, gprev = jax.grad(loss, argnums=(0, 1))(system.parameters, previous)
    for index in [2, 3, 4]:
        h = 1e-6
        finite = (
            loss(system.parameters.at[index].add(h), previous)
            - loss(system.parameters.at[index].add(-h), previous)
        ) / (2 * h)
        np.testing.assert_allclose(gp[index], finite, rtol=2e-5, atol=1e-8)
    h = 1e-6
    finite = (
        loss(system.parameters, previous.at[1].add(h))
        - loss(system.parameters, previous.at[1].add(-h))
    ) / (2 * h)
    assert abs(gprev[1]) > 1e-6
    np.testing.assert_allclose(gprev[1], finite, rtol=2e-5, atol=1e-8)
    old = ContactState(np.zeros(1), np.zeros(1))
    answer = system.solve(old, np.asarray(forcing), np.asarray(guess), np.asarray(previous))
    assert answer.converged and answer.residual_mpa < 1e-7
    np.testing.assert_array_equal(old.plastic_slip, 0)
    np.testing.assert_array_equal(old.cumulative_slip, 0)
    assert answer.response[0, 1, 3] >= answer.response[0, 0, 3] >= 0
    with pytest.raises(ValueError, match="explicit root recovery"):
        system.solve(
            old, np.asarray(forcing), np.asarray(guess), np.asarray(previous), kinetic_seed=True
        )


@pytest.mark.parametrize("counts", [(0, 2), (1, 0), (True, 2), (1.5, 2), (1,)])
def test_invalid_stage_microsteps_rejected(counts):
    with pytest.raises(ValueError, match="positive integer"):
        TOOL["staged_point"](counts)

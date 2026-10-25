"""Regression for a real 275-coordinate root that stalled Newton/L2."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from finiteblock_bem.friction import ContactState, SlipWeakeningContact

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/robust_contact_experiment.py")
)


def problem():
    path = Path(__file__).with_name("fixtures") / "contact4_event16_equilibrium.npz"
    with np.load(path, allow_pickle=False) as source:
        data = {k: source[k] for k in source.files}
    d = len(data["forcing"])
    n, eye = (d - 1) // 2, np.eye(d)
    con = SimpleNamespace(
        size=d,
        normal_jump=eye[:n],
        tangential_jump=eye[n : 2 * n],
        contact_weights=data["contact_weights"],
        reduced_weights=data["reduced_weights"],
        reduce_force=lambda value: value,
    )
    reference = SimpleNamespace(
        constraints=con,
        case=SimpleNamespace(contact=SlipWeakeningContact(*data["parameters"])),
        base_dn=data["base_dn"],
        base_ds=data["base_ds"],
        base_platen=0.0,
        base_u=np.zeros(d),
        base_t=data["forcing"],
        spring=0.0,
        matrix=data["matrix"],
        external_traction=lambda _: np.zeros(d),
    )
    system = TOOL["RobustSubsteppedContactEquilibrium"](reference, substeps=4)
    system.previous_displacement = data["previous_displacement"].copy()
    state = ContactState(data["plastic_slip"].copy(), data["cumulative_slip"].copy())
    return system, state, data


def test_failed_real_equilibrium_recovers_without_committing_trial_states():
    system, state, data = problem()
    failed_residual = system.residual(
        data["failed_displacement"],
        state.plastic_slip,
        state.cumulative_slip,
        system.parameters,
        data["forcing"],
        data["previous_displacement"],
    )
    np.testing.assert_allclose(np.max(abs(failed_residual)), 17.388634458455527, rtol=1e-10)
    answer = system.solve(0.0, 0.0, state, guess=data["previous_displacement"])
    assert answer.converged and answer.residual_mpa < 1e-7
    np.testing.assert_allclose(
        answer.reduced_displacement, data["independent_root"], rtol=1e-8, atol=1e-9
    )
    assert len(system.accepted_micro_work) == 1
    assert len(system.accepted_integrated_strength_work) == 1
    np.testing.assert_array_equal(state.plastic_slip, data["plastic_slip"])
    np.testing.assert_array_equal(state.cumulative_slip, data["cumulative_slip"])
    np.testing.assert_array_equal(system.previous_displacement, answer.reduced_displacement)


def test_recovered_root_keeps_implicit_parameter_derivative():
    system, state, data = problem()
    q, attempts = TOOL["recover_root"](
        system, state, data["forcing"], data["previous_displacement"]
    )
    assert q is not None and attempts[-1]["accepted"]
    original = system.parameters
    previous = jnp.asarray(data["previous_displacement"])
    weights = jnp.asarray(data["contact_weights"] / np.sum(data["contact_weights"]))

    def value(parameters):
        current, _ = system.implicit_displacement(
            parameters, state.plastic_slip, state.cumulative_slip, data["forcing"], q, previous
        )
        response = system.local_response(
            current, state.plastic_slip, state.cumulative_slip, parameters, previous
        )
        return jnp.dot(weights, response[:, 3])

    derivative = jax.grad(value)(original)[3]
    eps = 1e-6
    values = []
    for change in [-eps, eps]:
        system.parameters = original.at[3].add(change)
        shifted, _ = TOOL["recover_root"](system, state, data["forcing"], q)
        assert shifted is not None
        response = system.local_response(
            shifted, state.plastic_slip, state.cumulative_slip, system.parameters, previous
        )
        values.append(float(jnp.dot(weights, response[:, 3])))
    system.parameters = original
    finite = (values[1] - values[0]) / (2 * eps)
    np.testing.assert_allclose(derivative, finite, rtol=1e-5, atol=1e-8)

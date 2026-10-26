"""Full refined-state AD against original-Z solves, not only local tangents."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
from finiteblock_bem.cq import BoundaryHistory, plan_cq
from finiteblock_bem.friction import ContactState, SlipWeakeningContact
from finiteblock_bem.jax_backend import JaxContactEquilibrium

TOOLS = Path(__file__).resolve().parents[1] / "tools"
REFINED = runpy.run_path(str(TOOLS / "refined_contact.py"))
AD = runpy.run_path(str(TOOLS / "refined_contact_adjoint.py"))
SUBSTEP = runpy.run_path(str(TOOLS / "substepped_contact_experiment.py"))


def problem():
    space = REFINED["ContactQuadrature"].build(np.array([0.0, 0.4, 1.0]), 3)
    con = space.constraints(1.0)
    law = SlipWeakeningContact(100, 40, 0.8, 0.5, 0.2)
    n, d = space.bulk_nodes, space.dofs
    bulk = np.diag(np.r_[np.full(n, 30.0), np.full(n, 5.0), 2.0])
    for node in range(n):
        link = np.zeros(d)
        link[n + node], link[-1] = 1, -1
        bulk += 25 * np.outer(link, link)
    coupled = np.linspace(-0.2, 0.2, d)
    bulk += 3 * np.outer(coupled, coupled)
    mass = 0.01 * (np.eye(d) + np.outer(coupled, coupled))
    dt, steps, spring = 0.01, 16, 20.0
    original = np.zeros((steps + 1, d, d))
    original[:3] = [bulk + 1.5 * mass / dt, -2 * mass / dt, 0.5 * mass / dt]
    matrix = original[0].copy()
    matrix[-1, -1] += spring
    coarse = REFINED["line_weights"](space.bulk_y)
    reference = SimpleNamespace(
        constraints=con,
        case=SimpleNamespace(contact=law),
        model=SimpleNamespace(constraints=SimpleNamespace(contact_weights=coarse)),
        matrix=matrix,
        base_u=np.zeros(d),
        base_dn=np.full(space.points, 0.1),
        base_ds=np.zeros(space.points),
        base_platen=0.0,
        base_t=-space.pull_force(np.full(space.points, 10.0), np.zeros(space.points)),
        spring=spring,
        external_traction=lambda _: np.zeros(d),
    )
    diagonal = np.r_[coarse * law.normal_penalty, coarse * law.tangential_penalty, 0]
    inverse = np.linalg.inv(matrix + np.diag(diagonal))
    weights = np.zeros_like(original)
    weights[0] = inverse
    for i in range(1, steps + 1):
        weights[i] = -inverse @ sum(
            (original[k] @ weights[i - k] for k in range(1, min(i, 2) + 1)),
            start=np.zeros((d, d)),
        )
    time = np.arange(steps + 1) * dt
    actuator = np.linspace(0, 1.4, steps + 1)
    history = BoundaryHistory(plan_cq(d, dt, steps, max_memory_mb=16), weights)
    return reference, original, history, time, actuator


def direct_trajectory(reference, original, time, actuator, substeps, parameters=None):
    system = (
        JaxContactEquilibrium(reference, parameters=parameters)
        if substeps == 1
        else SUBSTEP["SubsteppedContactEquilibrium"](reference, substeps=substeps)
    )
    if parameters is not None:
        import jax.numpy as jnp

        system.parameters = jnp.asarray(parameters)
    m, d = len(reference.base_dn), reference.constraints.size
    q = np.zeros((len(time), d))
    z, c = np.zeros((len(time), m)), np.zeros((len(time), m))
    for i in range(1, len(time)):
        past = sum((original[i - j] @ q[j] for j in range(i)), start=np.zeros(d))
        answer = system.solve(
            0.0, actuator[i], ContactState(z[i - 1], c[i - 1]), past=past, guess=q[i - 1]
        )
        assert answer.converged
        q[i], z[i], c[i] = (
            answer.reduced_displacement,
            answer.contact.state.plastic_slip,
            answer.contact.state.cumulative_slip,
        )
    return SimpleNamespace(time=time, bulk_displacement=q, plastic_slip=z, cumulative_slip=c)


@pytest.mark.parametrize("substeps", [1, 4])
def test_full_refined_reverse_matches_independent_jvp_and_original_z_finite_differences(substeps):
    reference, original, history, time, actuator = problem()
    result = direct_trajectory(reference, original, time, actuator, substeps)
    reverse = AD["reverse_gradient"](reference, history, result, constitutive_substeps=substeps)
    forward = AD["forward_gradient"](reference, history, result, constitutive_substeps=substeps)
    assert result.cumulative_slip[-1].max() > 0.2
    np.testing.assert_allclose(reverse["gradient"], forward["gradient"], rtol=1e-10, atol=1e-11)
    assert reverse["contact_state_points"] != (reverse["bulk_dofs"] - 1) // 2
    assert reverse["every_history_source_included"]
    assert reverse["previous_displacement_cotangent_included"] == (substeps != 1)
    parameters = np.array([100.0, 40.0, 0.8, 0.5, 0.2])
    wc = reference.constraints.contact_weights

    def loss(parameters, actuator=actuator):
        trajectory = direct_trajectory(reference, original, time, actuator, substeps, parameters)
        return np.average(trajectory.cumulative_slip[-1], weights=wc)

    for column, epsilon in enumerate([1e-3, 1e-3, 1e-5, 1e-5, 1e-5]):
        direction = np.eye(5)[column] * epsilon
        difference = (loss(parameters + direction) - loss(parameters - direction)) / (2 * epsilon)
        np.testing.assert_allclose(reverse["gradient"][column], difference, rtol=2e-4, atol=2e-7)
    index, epsilon = len(time) // 2, 1e-5
    direction = np.eye(len(time))[index] * epsilon
    difference = (
        loss(parameters, actuator + direction) - loss(parameters, actuator - direction)
    ) / (2 * epsilon)
    np.testing.assert_allclose(
        reverse["actuator_gradient"][index], difference, rtol=2e-4, atol=2e-7
    )


def test_refined_ad_rejects_incomplete_history_before_differentiation():
    reference, original, history, time, actuator = problem()
    result = direct_trajectory(reference, original, time, actuator, 1)
    from dataclasses import replace

    incomplete = BoundaryHistory(replace(history.plan, steps=3), history.weights[:4])
    with pytest.raises(ValueError, match="Complete"):
        AD["reverse_gradient"](reference, incomplete, result)

import json
import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from finiteblock_bem.cq import (
    BoundaryHistory,
    bdf2_symbol,
    convolution_weights,
    invert_spectrum,
    plan_cq,
)
from finiteblock_bem.friction import ContactState, SlipWeakeningContact

jax = pytest.importorskip("jax")
TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/adjoint_resolvent.py"))


def test_reference_loads_all_saved_cotangents_and_rejects_unrelated_runs(tmp_path):
    environment = dict(source_sha256="source", case_sha256="case", restart_data_sha256="state")
    path = tmp_path / "adjoint.json"
    record = dict(
        environment=environment,
        parameter_order=["normal_penalty", "tangential_penalty", "mu_s", "mu_k", "Dc"],
        objective="final contact-weighted mean cumulative slip",
        steps=4,
        value_mm=0.2,
        gradient=[1, 2, 3, 4, 5],
    )
    path.write_text(json.dumps(record))
    arrays = dict(
        gradient=np.array(record["gradient"]),
        actuator_gradient=np.arange(5),
        initial_plastic_slip_gradient=np.arange(3),
        initial_cumulative_slip_gradient=np.arange(3) + 1,
    )
    np.savez(tmp_path / "adjoint.npz", **arrays)
    load = TOOL["load_gradient_reference"]
    answer = load(path, tmp_path, environment, steps=4, value=0.2, nodes=3)
    for name, value in arrays.items():
        np.testing.assert_array_equal(answer[name], value)
    with pytest.raises(ValueError, match="objective value"):
        load(path, tmp_path, environment, steps=4, value=0.3, nodes=3)
    with pytest.raises(ValueError, match="source run"):
        load(path, tmp_path / "other", environment, steps=4, value=0.2, nodes=3)
    arrays["gradient"] = arrays["gradient"] + 1
    np.savez(tmp_path / "adjoint.npz", **arrays)
    with pytest.raises(ValueError, match="different gradients"):
        load(path, tmp_path, environment, steps=4, value=0.2, nodes=3)


@pytest.mark.parametrize("work_coordinates", [False, True])
def test_prefix_frequency_transpose_matches_causal_nonsymmetric_history(work_coordinates):
    from scipy.fft import rfft

    rng = np.random.default_rng(37)
    plan = plan_cq(7, 0.02, 37, max_memory_mb=32)
    weights = rng.normal(size=(38, 7, 7))
    hbar = rng.normal(size=(38, 7))
    start = 21
    hbar[:start] = 0
    scale = np.geomspace(0.25, 20, 7) if work_coordinates else np.ones(7)
    spectrum = rfft(
        weights
        * scale[None, :, None]
        * scale[None, None, :]
        * plan.radius ** np.arange(38)[:, None, None],
        n=plan.fft_size,
        axis=0,
    )
    actual = TOOL["inactive_prefix_transpose"](spectrum, plan, hbar, start, coordinate_scale=scale)
    expected = np.array(
        [sum(weights[j - i].T @ hbar[j] for j in range(start, 38)) for i in range(start)]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_correction_adjoint_keeps_kn_kt_and_initial_state_prefix_derivatives():
    import jax.numpy as jnp

    from finiteblock_bem.adjoint import trajectory_adjoint
    from finiteblock_bem.jax_backend import JaxContactEquilibrium, differentiable_rollout

    law = SlipWeakeningContact(100, 40, 0.8, 0.5, 0.2)
    con = SimpleNamespace(
        size=3,
        normal_jump=np.array([[1.0, 0, 0]]),
        tangential_jump=np.array([[0.0, 1, 0]]),
        contact_weights=np.ones(1),
        reduced_weights=np.ones(3),
    )
    case = SimpleNamespace(contact=law, moving_contact=np.array([0]))
    plan = plan_cq(3, 0.05, 20, alias_tolerance=1e-12, max_memory_mb=32)
    stiffness = np.array([[30.0, 0.2, 0.1], [0.2, 30, -20], [0.1, -20, 30]])

    def zfun(s):
        return 0.2 * s * np.eye(3) + stiffness

    zweights = convolution_weights(zfun, plan)
    matrix = zweights[0].copy()
    matrix[-1, -1] += 2
    reference = SimpleNamespace(
        constraints=con,
        case=case,
        base_dn=np.array([0.1]),
        base_ds=np.array([0.03]),
        base_platen=0,
        base_u=np.zeros(3),
        matrix=matrix,
        spring=2,
    )
    system = JaxContactEquilibrium(reference)
    state = ContactState(np.array([0.03]), np.zeros(1))
    time = np.arange(21) * plan.dt
    force = np.tile([-10.0, 0, 0], (20, 1))
    force[:, -1] -= 2 * 15 * time[1:]
    rollout = differentiable_rollout(
        system, zweights, force, state.plastic_slip, state.cumulative_slip
    )
    q, response, residual = (np.asarray(a) for a in rollout(system.parameters))
    q = np.vstack((np.zeros(3), q))
    plastic = np.vstack((state.plastic_slip, response[:, :, 2]))
    cumulative = np.vstack((state.cumulative_slip, response[:, :, 3]))
    result = SimpleNamespace(
        time=time,
        contact_jumps=q[:, :2] + [0.1, 0.03],
        platen=q[:, -1],
        plastic_slip=plastic,
        cumulative_slip=cumulative,
        diagnostics={"status": "completed_requested_horizon"},
        residual_mpa=np.r_[0, residual],
    )
    original = trajectory_adjoint(
        case,
        SimpleNamespace(state=state),
        BoundaryHistory(plan, zweights),
        result,
        equilibrium=reference,
    )
    start = int(np.flatnonzero(cumulative[:, 0] > 0)[0])
    assert 2 <= start < 20
    tangent = np.diag([100.0, 40, 2])
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    spectrum = np.stack([np.linalg.inv(zfun(s) + tangent) for s in bdf2_symbol(zeta) / plan.dt])
    lweights = invert_spectrum(spectrum, plan)
    # The standalone integrator has no dependency on an original-grid kernel
    # or saved trajectory. Compare it with the independent original CQ scan.
    simulator = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools/simulate_resolvent.py")
    )
    reference.base_t = np.array([-10.0, 0, 0])
    reference.external_traction = lambda _: np.zeros(3)
    con.reduce_force = lambda value: value

    def evaluate(q, past, pressure, actuator, old, *, jacobian):
        forcing = system.forcing(pressure, actuator, past)
        residual = np.asarray(
            system.residual(q, old.plastic_slip, old.cumulative_slip, system.parameters, forcing)
        )
        return residual, None, system.contact_response(q, old)

    reference.evaluate = evaluate
    model = SimpleNamespace(
        moving_contact=np.array([0]),
        constraints=con,
        reference=SimpleNamespace(
            contact=law,
            normal_pressure=0,
            moving=SimpleNamespace(count=1),
            stationary=SimpleNamespace(count=1),
        ),
    )
    elastic = simulator["HELPER"]["causal_elastic_baseline"](
        spectrum[:, :, -1], plan, 2 * 15 * time
    )
    standalone = simulator["simulate_prepared"](
        model,
        reference,
        state,
        BoundaryHistory(plan, lweights),
        elastic,
        time,
        15 * time,
        diagnostic=True,
    )
    for name in ["contact_jumps", "platen", "plastic_slip", "cumulative_slip"]:
        np.testing.assert_allclose(
            getattr(standalone, name), getattr(result, name), rtol=1e-7, atol=1e-9
        )
    assert standalone.diagnostics["status"] == "completed_requested_horizon"
    prefix_plan = replace(plan, steps=20 - start)
    answer = TOOL["resolvent_adjoint"](
        reference,
        BoundaryHistory(prefix_plan, lweights[: prefix_plan.steps + 1]),
        result,
        spectrum,
        start,
    )
    for name in [
        "gradient",
        "actuator_gradient",
        "initial_plastic_slip_gradient",
        "initial_cumulative_slip_gradient",
    ]:
        np.testing.assert_allclose(answer[name], original[name], rtol=3e-7, atol=1e-9)
    assert np.linalg.norm(answer["prefix_parameter_gradient"][:2]) > 1e-8
    # Also compare to AD through the independent original direct-history scan.
    gradient = jax.grad(lambda p: jnp.mean(rollout(p)[1][-1, :, 3]))(system.parameters)
    np.testing.assert_allclose(answer["gradient"], gradient, rtol=3e-7, atol=1e-9)

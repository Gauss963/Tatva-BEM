"""RK power quadrature matches an independent mechanical energy identity."""

import runpy
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
WORK = runpy.run_path(str(TOOLS / "radau_port_work.py"))
RADAU = WORK["RADAU"]


@pytest.mark.parametrize("physical_damping", [0.0, 0.3])
def test_port_work_equals_mechanical_energy_plus_physical_and_rk_dissipation(physical_damping):
    dt, steps, d = 0.08, 45, 2
    mass = np.array([[2.0, 0.2], [0.2, 1.0]])
    stiffness = np.array([[4.0, -0.5], [-0.5, 2.0]])
    damping = physical_damping * np.array([[1.0, 0.1], [0.1, 0.8]])
    generator = np.block(
        [
            [np.zeros((d, d)), np.eye(d)],
            [-np.linalg.solve(mass, stiffness), -np.linalg.solve(mass, damping)],
        ]
    )
    stage_system = np.eye(4 * d) - dt * np.kron(RADAU["A"], generator)
    state = np.zeros((steps + 1, 2, 2 * d))
    force = np.zeros((steps + 1, 2, d))
    velocity = np.zeros((steps + 1, 2, d))
    acceleration = np.zeros_like(velocity)
    for k in range(1, steps + 1):
        time = dt * (k - 1 + RADAU["C"])
        force[k] = np.c_[np.sin(1.7 * time), 0.3 * np.cos(2.1 * time)]
        forcing = np.c_[np.zeros((2, d)), np.linalg.solve(mass, force[k].T).T]
        right = np.tile(state[k - 1, 1], (2, 1)) + dt * RADAU["A"] @ forcing
        state[k] = np.linalg.solve(stage_system, right.ravel()).reshape(2, 2 * d)
        velocity[k] = state[k, :, d:]
        acceleration[k] = (state[k] @ generator.T + forcing)[:, d:]
    q = state[:, :, :d]
    np.testing.assert_allclose(WORK["stage_derivative"](q, dt), velocity, rtol=2e-12, atol=2e-13)
    history, report = WORK["bulk_port_work"](force, q, dt, np.ones(d))
    physical = dt * np.einsum("a,tai,ij,taj->t", RADAU["B"], velocity, damping, velocity)
    weights = (
        np.diag(RADAU["B"]) @ RADAU["A"]
        + RADAU["A"].T @ np.diag(RADAU["B"])
        - np.outer(RADAU["B"], RADAU["B"])
    )
    assert np.linalg.eigvalsh(weights).min() >= -1e-15
    numerical = (
        0.5
        * dt**2
        * (
            np.einsum("ab,tai,ij,tbj->t", weights, velocity, stiffness, velocity)
            + np.einsum("ab,tai,ij,tbj->t", weights, acceleration, mass, acceleration)
        )
    )
    endpoint = state[:, 1]
    energy = 0.5 * (
        np.einsum("ti,ij,tj->t", endpoint[:, :d], stiffness, endpoint[:, :d])
        + np.einsum("ti,ij,tj->t", endpoint[:, d:], mass, endpoint[:, d:])
    )
    np.testing.assert_allclose(
        history["cumulative_work"], energy + np.cumsum(physical + numerical), rtol=2e-12, atol=2e-13
    )
    assert np.min(numerical) >= -1e-15
    assert report["final_work_mpa_mm2"] > 0


def test_residual_work_bound_encloses_changed_forces_and_preserves_negative_work():
    rng = np.random.default_rng(904)
    q = rng.normal(size=(20, 2, 3))
    q[0] = 0
    velocity = WORK["stage_derivative"](q, 0.1)
    force = -velocity
    line = np.array([0.2, 2.0, 5.0])
    epsilon = 0.03
    nominal, report = WORK["bulk_port_work"](
        force, q, 0.1, line, residual_bound_mpa=epsilon, radius=0.9
    )
    perturbed = force + epsilon * line * rng.uniform(-1, 1, force.shape)
    actual, _ = WORK["bulk_port_work"](perturbed, q, 0.1, line, radius=0.9)
    error = abs(actual["cumulative_work"] - nominal["cumulative_work"])
    assert np.all(error <= nominal["cumulative_residual_error_bound"] + 1e-10)
    weighted = abs(
        actual["radius_weighted_cumulative_work"] - nominal["radius_weighted_cumulative_work"]
    )
    assert np.all(weighted <= nominal["radius_weighted_residual_error_bound"] + 1e-10)
    assert report["minimum_prefix_upper_bound_mpa_mm2"] < 0
    assert report["minimum_radius_weighted_prefix_upper_bound_mpa_mm2"] < 0
    assert report["final_work_mpa_mm2"] < 0


def test_force_reconstruction_has_the_original_spring_and_initial_force_signs():
    rng = np.random.default_rng(910)
    q = rng.normal(size=(9, 2, 5))
    q[0] = 0
    response = rng.normal(size=(9, 2, 2, 7))
    actuator = rng.normal(size=(9, 2))
    initial, contact = rng.normal(size=(2, 5))
    line = np.array([2, 3, 2, 3, 7])
    spring = 6
    force = WORK["equilibrium_bulk_force"](q, response, actuator, initial, contact, line, spring)
    fc = np.zeros_like(q)
    fc[:, :, :2] = response[:, :, :, 0].transpose(0, 2, 1) * line[:2]
    fc[:, :, 2:4] = response[:, :, :, 1].transpose(0, 2, 1) * line[2:4]
    residual = force + fc - contact + initial
    residual[:, :, -1] += spring * (q[:, :, -1] - actuator + actuator[0])
    np.testing.assert_allclose(residual, 0, atol=1e-13)
    with pytest.raises(ValueError, match="zero-initial"):
        WORK["bulk_port_work"](force, q + 1, 0.1, line)

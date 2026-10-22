from pathlib import Path

import numpy as np

from finiteblock_bem.case import load_case
from finiteblock_bem.constraints import build_constraints
from finiteblock_bem.friction import ContactState, SlipWeakeningContact
from finiteblock_bem.nonlinear import ContactEquilibrium

CASE = Path(__file__).resolve().parents[1] / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def test_contact_return_map_open_stick_softening_kinetic_and_reversal():
    law = SlipWeakeningContact(76620, 7662, 0.8, 0.6, 0.0125959, weakening_evaluation="updated")
    state = ContactState(np.zeros(5), np.zeros(5))
    ds = np.array([0.1, 0.0005, 0.003, 0.05, -0.003])
    dn = np.array([-1e-3, *([16 / 76620] * 4)])
    response = law.evaluate(dn, ds, state)
    assert response.pressure[0] == response.shear[0] == 0
    assert response.state.cumulative_slip[0] == 0
    assert response.state.cumulative_slip[1] == 0
    assert 0 < response.state.cumulative_slip[2] < law.characteristic_slip
    assert response.state.cumulative_slip[3] > law.characteristic_slip
    assert response.shear[4] < 0 and response.state.plastic_slip[4] < 0
    strength = response.pressure * law.coefficient(response.state.cumulative_slip)
    assert np.all(abs(response.shear) <= strength + 1e-10)
    assert np.all(response.shear * response.state.plastic_slip >= 0)
    np.testing.assert_array_equal(state.plastic_slip, 0)
    np.testing.assert_array_equal(state.cumulative_slip, 0)


def test_unloading_does_not_heal_cumulative_slip_implicitly():
    law = SlipWeakeningContact(76620, 7662, 0.8, 0.6, 0.0125959)
    old = ContactState(np.array([0.02]), np.array([0.03]))
    result = law.evaluate(np.array([16 / 76620]), np.array([0.0201]), old)
    assert result.state.cumulative_slip[0] == 0.03
    assert result.state.plastic_slip[0] == 0.02


def test_default_previous_state_update_matches_tatva_and_is_unique_at_high_pressure():
    law = SlipWeakeningContact(76620, 7662, 0.8, 0.6, 0.0125959)
    old = ContactState(np.array([0.0]), np.array([0.003]))
    pressure = 600.0
    result = law.evaluate(np.array([pressure / law.normal_penalty]), np.array([0.2]), old)
    mu = law.coefficient(old.cumulative_slip)[0]
    assert result.friction_coefficient[0] == mu
    np.testing.assert_allclose(result.shear[0], pressure * mu)
    np.testing.assert_allclose(
        result.state.plastic_slip[0], 0.2 - pressure * mu / law.tangential_penalty
    )
    np.testing.assert_allclose(
        result.state.cumulative_slip[0], 0.003 + result.state.plastic_slip[0]
    )


def test_global_nonlinear_tangent_and_work_conjugate_pairing():
    case = load_case(CASE, max_panel_length=80)
    con = build_constraints(case)
    size = case.moving.dofs + case.stationary.dofs
    system = ContactEquilibrium(case, con, np.eye(size) * 3000)
    state = ContactState(np.zeros(len(case.moving_contact)), np.zeros(len(case.moving_contact)))
    target = np.concatenate(
        (np.full(len(state.plastic_slip), 16 / 76620), np.full(len(state.plastic_slip), 0.003))
    )
    z = np.linalg.lstsq(np.vstack((con.normal_jump, con.tangential_jump)), target, rcond=None)[0]
    past = np.zeros(size)
    _, jacobian, response = system.evaluate(z, past, 16, 0, state)
    direction = np.random.default_rng(18).normal(size=con.size)
    epsilon = 1e-8
    forward = system.evaluate(z + epsilon * direction, past, 16, 0, state, jacobian=False)[0]
    backward = system.evaluate(z - epsilon * direction, past, 16, 0, state, jacobian=False)[0]
    np.testing.assert_allclose(
        jacobian @ direction, (forward - backward) / (2 * epsilon), rtol=2e-7, atol=2e-5
    )
    generalized = con.normal_jump.T @ (con.contact_weights * response.pressure)
    generalized += con.tangential_jump.T @ (con.contact_weights * response.shear)
    work = z @ generalized
    expected = np.sum(
        con.contact_weights
        * ((con.normal_jump @ z) * response.pressure + (con.tangential_jump @ z) * response.shear)
    )
    np.testing.assert_allclose(work, expected, atol=1e-12)


def test_constraints_preserve_stationary_supports_and_rigid_platen():
    case = load_case(CASE, max_panel_length=80)
    con = build_constraints(case)
    reduced = np.random.default_rng(4).normal(size=con.size)
    full = con.prolongation @ reduced
    np.testing.assert_array_equal(full[con.fixed_dofs], 0)
    np.testing.assert_allclose(full[2 * case.moving.indices("platen") + 1], reduced[-1])

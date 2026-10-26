import runpy
from pathlib import Path

import numpy as np

from finiteblock_bem.friction import ContactState
from finiteblock_bem.nonlinear import ContactEquilibrium

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/fem_static_contact.py"))
LOCAL = runpy.run_path(str(ROOT / "tests/test_substepped_contact.py"))


def reference():
    result = LOCAL["reference"]()
    result.evaluate = lambda *a, **kw: ContactEquilibrium.evaluate(result, *a, **kw)
    return result


def test_healed_preload_slides_at_static_strength_without_weakening_memory():
    ref = reference()
    ref.base_t[1] = -20.0
    old = ContactState(np.zeros(1), np.zeros(1))
    q, state, diagnostic = TOOL["solve_healed_static_contact"](ref, 0.0, 0.0, old)
    # Independently solve the saturated static equations for this one-point case.
    con, law = ref.constraints, ref.case.contact
    pressure = law.normal_penalty * (ref.base_dn + con.normal_jump @ q)
    trial = law.tangential_penalty * (ref.base_ds + con.tangential_jump @ q)
    shear = law.static_friction * pressure
    assert trial[0] > shear[0]
    expected = np.linalg.solve(
        ref.matrix, -ref.base_t - con.normal_jump.T @ pressure - con.tangential_jump.T @ shear
    )
    np.testing.assert_allclose(q, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        state.plastic_slip, ref.base_ds + con.tangential_jump @ q - shear / law.tangential_penalty
    )
    np.testing.assert_array_equal(state.cumulative_slip, old.cumulative_slip)
    np.testing.assert_array_equal(old.plastic_slip, [0.0])
    assert diagnostic["static_sliding_points"] == 1
    assert not diagnostic["input_seed_unchanged"]
    assert diagnostic["original_force_residual_mpa"] < 1e-7


def test_admissible_static_seed_is_preserved_and_open_points_carry_no_force():
    ref = reference()
    old = ContactState(np.full(1, 0.01), np.zeros(1))
    _, state, diagnostic = TOOL["solve_healed_static_contact"](ref, 0.0, 0.0, old)
    np.testing.assert_array_equal(state.plastic_slip, old.plastic_slip)
    assert diagnostic["input_seed_unchanged"]
    ref.base_t[0] = 100.0
    q, state, diagnostic = TOOL["solve_healed_static_contact"](ref, 0.0, 0.0, old)
    response = ref.case.contact.evaluate(
        ref.base_dn + ref.constraints.normal_jump @ q,
        ref.base_ds + ref.constraints.tangential_jump @ q,
        state,
        frozen=True,
    )
    np.testing.assert_array_equal(response.pressure, [0.0])
    np.testing.assert_array_equal(response.shear, [0.0])
    np.testing.assert_array_equal(state.plastic_slip, old.plastic_slip)
    assert diagnostic["static_open_points"] == 1

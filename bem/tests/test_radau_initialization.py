import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CHECK = runpy.run_path(str(TOOLS / "radau_initialization.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))


@pytest.mark.parametrize("order", ["C", "F"])
def test_measured_initial_residual_tail_matches_complete_stage_weight_sum(order):
    rng = np.random.default_rng(27)
    weights = [np.array(rng.normal(size=(18, 3, 3)), order=order) for _ in range(2)]
    wrap = rng.normal(size=(3, 3))
    force = np.array([0.1, -0.07, 0.03])
    full = RADAU["expand_compact_weights"](*weights, wrap)
    actual = CHECK["initial_residual_past"](*weights, force, 17, rows=1)
    expected = np.zeros_like(actual)
    for k in range(1, 18):
        expected[k] = sum((full[lag] @ np.tile(force, 2)).reshape(2, 3) for lag in range(1, k))
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(actual[:2], 0)
    # The finite-DFT wrap is exclusively lag zero, which the current root owns.
    again = RADAU["expand_compact_weights"](*weights, 100 * wrap)
    np.testing.assert_array_equal(again[1:], full[1:])


def test_body_initializer_retains_the_measured_force_and_original_healed_state():
    case = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    model = load_nodal_model(case, max_panel_length=160, contact_panel_length=160)
    seed = model.restart.plastic_slip.copy()
    lift, reference = CHECK["initialize_radau_body"](model, quadrature=12)
    np.testing.assert_array_equal(model.restart.plastic_slip, seed)
    np.testing.assert_array_equal(lift.state.cumulative_slip, model.restart.cumulative_slip)
    np.testing.assert_array_equal(reference.base_t, lift.retained_bulk_force)
    n = len(model.moving_contact)
    np.testing.assert_array_equal(reference.base_dn, lift.retained_displacement[:n])
    np.testing.assert_array_equal(reference.base_ds, lift.retained_displacement[n:-1])
    value, _, _ = reference.evaluate(
        np.zeros(reference.constraints.size),
        np.zeros(reference.constraints.size),
        model.reference.normal_pressure,
        model.reference.restart_actuator,
        lift.state,
        jacobian=False,
    )
    assert np.max(abs(value)) < 1e-7
    assert lift.diagnostics["original_full_balance_mpa"] < 1e-7


def test_initial_residual_history_rejects_truncated_weights():
    weights = np.zeros((4, 3, 3))
    with pytest.raises(ValueError):
        CHECK["initial_residual_past"](weights, weights, np.ones(3), 5)

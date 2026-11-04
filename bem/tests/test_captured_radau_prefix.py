"""Recover the original endpoint loading stop and discarded zero source prefix."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from finiteblock_bem.friction import ContactState, SlipWeakeningContact

pytest.importorskip("jax")

TOOL = runpy.run_path(str(Path(__file__).parents[1] / "tools/validate_captured_radau_prefix.py"))


def prefix():
    law = SlipWeakeningContact(100, 40, 0.8, 0.5, 0.2)
    reference = SimpleNamespace(
        constraints=SimpleNamespace(size=3, contact_weights=np.ones(1)),
        base_dn=np.array([0.1]),
        base_ds=np.zeros(1),
        spring=2.0,
        case=SimpleNamespace(contact=law),
    )
    state = ContactState(np.zeros(1), np.zeros(1))
    q, values = np.zeros((4, 2, 3)), np.zeros((4, 1, 2, 7))
    q[1:, :, 1] = [[0.05, 0.1], [0.3, 0.4], [0.45, 0.5]]
    old = state
    for k in range(4):
        for stage in range(2):
            response = law.evaluate(reference.base_dn, q[k, stage, 1:2], old)
            values[k, 0, stage, :5] = [
                response.pressure[0],
                response.shear[0],
                response.state.plastic_slip[0],
                response.state.cumulative_slip[0],
                response.friction_coefficient[0],
            ]
            values[k, 0, stage, 5] = response.shear[0] * (
                response.state.plastic_slip[0] - old.plastic_slip[0]
            )
            old = response.state
    time = np.zeros((4, 2))
    time[1:] = np.arange(3)[:, None] + [1 / 3, 1]
    return reference, state, q, values, time, 20 * time


@pytest.mark.parametrize("diagnostic", [False, True])
def test_sources_and_endpoint_feedback_are_recovered(diagnostic):
    ref, state, q, values, time, ramp = prefix()
    actual = TOOL["reconstruct_prefix"](ref, state, q, values, time, ramp, diagnostic=diagnostic)
    np.testing.assert_array_equal(actual["active"], [False, False, True, True])
    np.testing.assert_array_equal(actual["sources"][:2], 0)
    if diagnostic:
        assert actual["held"] is None and actual["hold_time"] is None
        np.testing.assert_array_equal(actual["actuator"], ramp)
        np.testing.assert_array_equal(actual["sources"][:, [2, 5]], 0)
    else:
        assert actual["held"] == 40 and actual["hold_time"] == 2
        np.testing.assert_array_equal(actual["actuator"][:3], ramp[:3])
        np.testing.assert_array_equal(actual["actuator"][3], [40, 40])
        np.testing.assert_allclose(
            actual["sources"][3, [2, 5]], 2 * (40 - ramp[3]), rtol=0, atol=1e-14
        )
    # The original elastic correction vanishes before first plastic flow;
    # afterward it equals kt times the updated plastic offset at each stage.
    np.testing.assert_allclose(
        actual["sources"][:, [1, 4]], 40 * values[:, 0, :, 2], rtol=0, atol=2e-14
    )


def test_modified_initial_contact_state_is_rejected():
    ref, state, q, values, time, ramp = prefix()
    values[0, 0, 0, 3] = 1e-6
    with pytest.raises(AssertionError):
        TOOL["reconstruct_prefix"](ref, state, q, values, time, ramp)

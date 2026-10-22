from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
from finiteblock_bem.case import load_case
from finiteblock_bem.condensation import InterfaceContactEquilibrium
from finiteblock_bem.dynamics import StaticLift
from finiteblock_bem.friction import ContactState
from finiteblock_bem.jax_backend import JaxContactEquilibrium


def test_safeguarded_active_set_lookahead_resolves_graded_switch():
    root = Path(__file__).resolve().parents[1]
    case = load_case(
        root / "src/finiteblock_bem/resources/cases/pmma_event16.toml",
        max_panel_length=40,
        contact_panel_length=20,
        contact_end_length=2,
        corner_panel_length=2,
    )
    data = np.load(Path(__file__).with_name("fixtures") / "graded_switch.npz", allow_pickle=False)
    lift = StaticLift(data["lift_u"], data["lift_t"], case.restart, {})
    reference = InterfaceContactEquilibrium(case, lift, data["impedance"])
    old = ContactState(data["old_z"], data["old_c"])
    answers = [
        system.solve(16.0, float(data["actuator"]), old, past=data["past"], guess=data["guess"])
        for system in (reference, JaxContactEquilibrium(reference))
    ]
    for answer in answers:
        assert answer.converged and answer.iterations <= 5
        assert answer.residual_mpa < 1e-7
    np.testing.assert_allclose(
        answers[0].reduced_displacement, answers[1].reduced_displacement, atol=2e-12
    )

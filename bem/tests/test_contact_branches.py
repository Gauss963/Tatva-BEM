import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from finiteblock_bem.friction import ContactState, SlipWeakeningContact


def test_branch_audit_detects_tiny_sliding_and_previous_state_clip():
    audit = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/run_adjoint.py"))[
        "exact_contact_branches"
    ]
    law = SlipWeakeningContact(100, 40, 0.8, 0.6, 0.1)
    old = ContactState(np.zeros(1), np.zeros(1))
    contact = law.evaluate(np.array([0.1]), np.array([0.2 + 1e-13]), old)
    increment = contact.state.plastic_slip[0]
    assert 0 < increment < 1e-10  # The former increment proxy missed this branch.
    result = SimpleNamespace(
        contact_jumps=np.array([[0.1, 0.2], [0.1, 0.2 + 1e-13], [0, 1.0]]),
        plastic_slip=np.array([[0], [increment], [increment]]),
        cumulative_slip=np.array([[0], [0.1], [0.1]]),
    )
    answer = audit(SimpleNamespace(contact=law), result)
    np.testing.assert_array_equal(answer["exact_sliding_entries"], [[True], [False]])
    np.testing.assert_array_equal(answer["exact_opening_entries"], [[False], [True]])
    np.testing.assert_array_equal(answer["weakening_clip_entries"], [[True], [False]])
    # JAX minimum uses a tie derivative at Dc; equality must not be silently
    # grouped with the fully weakened branch when comparing perturbations.
    np.testing.assert_array_equal(answer["weakening_clip_tie_entries"], [[False], [True]])

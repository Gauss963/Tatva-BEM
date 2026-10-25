"""Root diagnostics must solve the captured equation, not optimizer success."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from finiteblock_bem.friction import ContactState

pytest.importorskip("jax")
TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/diagnose_contact_root.py"))


def test_all_methods_reproduce_a_closed_sticking_equilibrium_without_state_commit():
    data = dict(
        matrix=np.diag([10.0, 10.0, 2.0]),
        forcing=np.array([-4.0, -0.01, -0.2]),
        previous_displacement=np.zeros(3),
        failed_displacement=np.zeros(3),
        plastic_slip=np.zeros(1),
        cumulative_slip=np.zeros(1),
        parameters=np.array([100.0, 40.0, 0.8, 0.6, 0.2]),
        base_dn=np.array([0.1]),
        base_ds=np.zeros(1),
        contact_weights=np.ones(1),
        reduced_weights=np.ones(3),
    )
    system, state = TOOL["captured_system"](data, 8)
    records, roots, distances = TOOL["probe_roots"](
        system, state, data, methods=["hybr", "lm", "trf", "forcing", "kinetic"], max_nfev=100
    )
    assert len(roots) == 9 and all(r["accepted"] for r in records)
    expected = np.array([-6 / 110, 0.01 / 50, 0.1])
    for q in roots.values():
        np.testing.assert_allclose(q, expected, rtol=1e-10, atol=1e-12)
    assert max(distances.values()) < 1e-10
    np.testing.assert_array_equal(system.previous_displacement, np.zeros(3))
    np.testing.assert_array_equal(state.plastic_slip, np.zeros(1))
    np.testing.assert_array_equal(state.cumulative_slip, np.zeros(1))
    assert not system.accepted_micro_work
    np.testing.assert_array_equal(system.parameters, data["parameters"])


def test_least_squares_success_at_nonzero_minimum_is_rejected():
    zeros = np.zeros(3)
    data = dict(
        forcing=zeros,
        previous_displacement=zeros,
        failed_displacement=zeros,
        plastic_slip=np.zeros(1),
        cumulative_slip=np.zeros(1),
    )
    system = SimpleNamespace(
        parameters=np.zeros(5),
        previous_displacement=zeros.copy(),
        accepted_micro_work=[],
        accepted_integrated_strength_work=[],
        residual=lambda q, *_: np.array([float(q[0]) ** 2 + 1, float(q[1]), float(q[2])]),
        tangent=lambda q, *_: np.diag([2 * float(q[0]), 1.0, 1.0]),
    )
    state = ContactState(np.zeros(1), np.zeros(1))
    records, roots, _ = TOOL["probe_roots"](system, state, data, methods=["lm"], max_nfev=20)
    assert not roots
    assert all(r["reported_success"] and not r["accepted"] for r in records)
    assert all(r["residual_mpa"] == 1 for r in records)

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/affine_stick_contact.py"))
SYNTHETIC = runpy.run_path(str(ROOT / "tests/test_refined_contact_adjoint.py"))
LOCAL = runpy.run_path(str(ROOT / "tests/test_substepped_contact.py"))


@pytest.mark.parametrize("substeps", [1, 4])
def test_affine_shortcut_matches_coupled_original_history_through_sliding(substeps):
    from finiteblock_bem.friction import ContactState

    ref, original, history, time, actuator = SYNTHETIC["problem"]()
    expected = SYNTHETIC["direct_trajectory"](ref, original, time, actuator, substeps)
    system = TOOL["AffineStickContactEquilibrium"](ref, substeps=substeps)
    q = np.zeros_like(expected.bulk_displacement)
    plastic = np.zeros_like(expected.plastic_slip)
    cumulative = np.zeros_like(expected.cumulative_slip)
    old = ContactState(expected.plastic_slip[0], expected.cumulative_slip[0])
    for i in range(1, len(time)):
        past = sum((original[i - j] @ q[j] for j in range(i)), start=np.zeros(q.shape[1]))
        answer = system.solve(0.0, actuator[i], old, past=past, guess=q[i - 1])
        assert answer.converged and answer.residual_mpa < 1e-7
        q[i], old = answer.reduced_displacement, answer.contact.state
        plastic[i], cumulative[i] = old.plastic_slip, old.cumulative_slip
        np.testing.assert_allclose(
            old.plastic_slip, expected.plastic_slip[i], rtol=1e-11, atol=1e-12
        )
        np.testing.assert_allclose(
            old.cumulative_slip, expected.cumulative_slip[i], rtol=1e-11, atol=1e-12
        )
    np.testing.assert_allclose(q, expected.bulk_displacement, rtol=1e-11, atol=1e-12)
    assert system.fast_stick_steps > 0
    assert system.nonlinear_steps > 0
    assert len(system.accepted_micro_work) == len(time) - 1
    actual = SimpleNamespace(
        time=time, bulk_displacement=q, plastic_slip=plastic, cumulative_slip=cumulative
    )
    for method in ("forward_gradient", "reverse_gradient"):
        baseline = SYNTHETIC["AD"][method](ref, history, expected, constitutive_substeps=substeps)
        candidate = SYNTHETIC["AD"][method](ref, history, actual, constitutive_substeps=substeps)
        np.testing.assert_allclose(
            candidate["gradient"], baseline["gradient"], rtol=1e-10, atol=1e-11
        )


def test_inside_final_cone_is_insufficient_after_previous_state_weakening():
    from finiteblock_bem.friction import ContactState

    ref = LOCAL["reference"]()
    ref.base_t[1] = -7.0
    old = ContactState(np.zeros(1), np.full(1, 0.2))
    expected = LOCAL["TOOL"]["SubsteppedContactEquilibrium"](ref, substeps=4)
    actual = TOOL["AffineStickContactEquilibrium"](ref, substeps=4)
    for system in (expected, actual):
        system.previous_displacement[1] = 0.175
    a, b = [system.solve(0.0, 0.0, old) for system in (expected, actual)]
    assert a.converged and b.converged
    assert actual.fast_stick_steps == 0
    assert b.contact.state.plastic_slip[0] > 0
    np.testing.assert_allclose(
        a.reduced_displacement, b.reduced_displacement, rtol=1e-13, atol=1e-13
    )


def test_affine_penalty_factorization_is_refreshed_when_parameters_change():
    import jax.numpy as jnp

    from finiteblock_bem.friction import ContactState
    from finiteblock_bem.jax_backend import JaxContactEquilibrium

    ref = LOCAL["reference"]()
    system = TOOL["AffineStickContactEquilibrium"](ref, substeps=1)
    system.parameters = jnp.array([120.0, 40.0, 0.8, 0.5, 0.2])
    old = ContactState(np.zeros(1), np.zeros(1))
    actual = system.solve(0.0, 0.0, old)
    expected = JaxContactEquilibrium(ref, parameters=system.parameters).solve(0.0, 0.0, old)
    assert actual.converged and system.fast_stick_steps == 1
    np.testing.assert_allclose(
        actual.reduced_displacement, expected.reduced_displacement, rtol=1e-12, atol=1e-12
    )

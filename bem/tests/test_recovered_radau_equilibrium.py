"""The actual failed finite-block equation is recovered at its original gate."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
from finiteblock_bem.friction import ContactState, SlipWeakeningContact

ROOT = Path(__file__).parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/recovered_radau_equilibrium.py"))


def test_actual_captured_failure_recovers_without_committing_old_state():
    with np.load(
        ROOT / "docs/diagnostics/radau_step14451/equation.npz", allow_pickle=False
    ) as saved:
        data = {k: saved[k] for k in saved.files}
    n = len(data["plastic_slip"])
    ref = SimpleNamespace(
        base_dn=data["base_dn"],
        base_ds=data["base_ds"],
        base_platen=0.0,
        case=SimpleNamespace(contact=SlipWeakeningContact(*data["parameters"])),
        constraints=SimpleNamespace(
            size=2 * n + 1,
            normal_jump=np.pad(np.eye(n), ((0, 0), (0, n + 1))),
            tangential_jump=np.pad(np.eye(n), ((0, 0), (n, 1))),
            contact_weights=data["contact_weights"],
            reduced_weights=data["reduced_weights"],
        ),
    )
    system = TOOL["POLICY"]["stage_equilibrium"](ref, data["matrix"], segment_substeps=(64, 128))
    state = ContactState(data["plastic_slip"].copy(), data["cumulative_slip"].copy())
    original_z, original_c = state.plastic_slip.copy(), state.cumulative_slip.copy()
    failed = TOOL["POLICY"]["BASE"]["StageRootResult"](
        data["failed_displacement"], np.zeros((n, 2, 7)), float(data["residual_mpa"]), 50, False, []
    )
    reports = []
    answer = TOOL["recover_original_stage"](
        system,
        ref,
        data["matrix"],
        state,
        data["forcing"],
        data["previous_displacement"],
        failed,
        on_recovery=reports.append,
    )
    assert answer.converged and answer.residual_mpa < 1e-7
    assert np.all(np.isfinite(answer.response))
    assert np.min(answer.response[:, :, 3] - original_c[:, None]) >= 0
    assert np.min(answer.response[:, :, 5]) >= 0
    np.testing.assert_array_equal(state.plastic_slip, original_z)
    np.testing.assert_array_equal(state.cumulative_slip, original_c)
    np.testing.assert_array_equal(failed.displacement, data["failed_displacement"])
    assert reports[-1]["accepted"]
    assert reports[-1]["nonlinear_nodes"] == list(range(252, 265))
    assert reports[-1]["parameter_ad_uses_original_full_residual"]
    assert answer.iterations == failed.iterations


def test_nonidentity_maps_keep_original_failure():
    failure = SimpleNamespace(converged=False)
    system = SimpleNamespace(tangent_policy=dict(applied=False))
    assert TOOL["recover_original_stage"](system, None, None, None, None, None, failure) is failure

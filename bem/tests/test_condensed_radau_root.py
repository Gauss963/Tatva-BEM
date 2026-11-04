"""Exact branch elimination must still pass the full original stage equation."""

import runpy
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

TOOL = runpy.run_path(str(Path(__file__).parents[1] / "tools/diagnose_condensed_radau_root.py"))


def captured(end_normal=0.12, *, diagonal=False, zero_shear=False):
    n, d = 3, 7
    rng = np.random.default_rng(11031)
    matrix = 50 * np.eye(2 * d)
    if not diagonal:
        matrix += 0.2 * rng.normal(size=matrix.shape)
    q = np.zeros((2, d))
    q[:, :n] = [[0.03, 0.04, 0.02], [0.04, 0.03, end_normal - 0.1]]
    if not zero_shear:
        q[:, n : 2 * n] = [[0.30, 0.30, 0.35], [0.34, 0.35, 0.42]]
    data = dict(
        matrix=matrix,
        forcing=np.zeros(2 * d),
        previous_displacement=np.zeros(d),
        failed_displacement=q.ravel(),
        plastic_slip=np.zeros(n),
        cumulative_slip=np.array([0.5, 0.5, 0.0]),
        parameters=np.array([100, 40, 0.8, 0.5, 0.2]),
        base_dn=np.full(n, 0.1),
        base_ds=np.zeros(n),
        contact_weights=np.array([0.7, 1.2, 0.9]),
        reduced_weights=np.r_[np.array([0.7, 1.2, 0.9]).repeat(2).reshape(3, 2).T.ravel(), 1.0],
    )
    counts = (4, 8)
    system, _, state, _ = TOOL["CAPTURE"]["captured_stage"](data, counts)
    values = np.asarray(
        system.local_response(
            q.ravel(),
            jnp.asarray(state.plastic_slip),
            jnp.asarray(state.cumulative_slip),
            system.parameters,
            jnp.zeros(d),
        )
    )
    weights = data["contact_weights"]
    force = np.stack(
        [np.r_[weights * values[:, a, 0], weights * values[:, a, 1], 0.0] for a in range(2)]
    )
    data["forcing"] = -matrix @ q.ravel() - force.ravel()
    return data, counts


@pytest.mark.parametrize("end_normal", [0.12, -0.05])
def test_exact_recovery_and_local_ad_for_closed_or_open_selected_point(end_normal):
    data, counts = captured(end_normal)
    model = TOOL["CondensedStage"](data, counts, nonlinear_nodes=[2])
    unknown = model.correction_at(data["failed_displacement"])
    np.testing.assert_allclose(
        model.expand(unknown), data["failed_displacement"], rtol=0, atol=2e-15
    )
    np.testing.assert_allclose(model.residual(unknown), 0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        model.tangent(unknown),
        jax.jacfwd(model.residual)(jnp.asarray(unknown)),
        rtol=1e-12,
        atol=1e-12,
    )
    assert model.check_candidate(unknown)["accepted"]
    assert model.linear_backward_error < 1e-14


def test_reduced_root_on_invalid_outside_branch_is_rejected():
    data, counts = captured(diagonal=True, zero_shear=True)
    model = TOOL["CondensedStage"](data, counts, nonlinear_nodes=[2])
    unknown = np.zeros(4)
    checks = model.check_candidate(unknown)
    assert checks["reduced_residual_mpa"] < 1e-12
    assert checks["full_residual_mpa"] > 1
    assert not checks["accepted"]


def test_unweakened_outside_points_use_stick_candidate_and_full_validation():
    data, counts = captured(diagonal=True, zero_shear=True)
    data["cumulative_slip"][:] = 0
    model = TOOL["CondensedStage"](data, counts, nonlinear_nodes=[2])
    unknown = np.zeros(4)
    assert model.check_candidate(unknown)["accepted"]

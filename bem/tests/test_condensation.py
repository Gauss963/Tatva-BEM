from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import block_diag

from finiteblock_bem.case import load_case
from finiteblock_bem.condensation import InterfaceTransfer, plan_interface
from finiteblock_bem.constraints import build_constraints
from finiteblock_bem.cq import ResourceBudgetError
from finiteblock_bem.dynamics import (
    LoadingSchedule,
    RestartProjectionError,
    initialize_restart,
    simulate,
)
from finiteblock_bem.kernels import _radial, green_and_gradient, source_traction
from finiteblock_bem.operators import (
    PreparedBoundaryOperators,
    _angular_geometry,
    _green_and_traction,
    assemble_operators,
)
from finiteblock_bem.statics import assemble_static_operators

CASE = Path(__file__).resolve().parents[1] / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def test_direct_radial_traction_matches_full_gradient_tensor():
    case = load_case(CASE)
    rng = np.random.default_rng(71)
    vector = rng.normal(size=(100, 2)) * 50
    normal = rng.normal(size=(100, 2))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    for s in [1000 + 300j, 1e5 + 4e4j]:
        green, gradient = green_and_gradient(vector, s, case.material)
        radial = _radial(np.linalg.norm(vector, axis=1), s, case.material)
        g, t = _green_and_traction(
            radial, np.arange(100), _angular_geometry(vector, normal), case.material
        )
        np.testing.assert_allclose(g, green, rtol=2e-13, atol=1e-15)
        np.testing.assert_allclose(
            t, source_traction(gradient, normal, case.material), rtol=2e-12, atol=1e-13
        )


def test_streamed_static_operators_match_prepared_graded_operators():
    case = load_case(CASE, max_panel_length=80, contact_panel_length=20, contact_end_length=2)
    for mesh in (case.moving, case.stationary):
        streamed = assemble_static_operators(mesh, case.material)
        prepared = PreparedBoundaryOperators(mesh, case.material).evaluate()
        np.testing.assert_allclose(streamed.single_layer, prepared.single_layer, atol=2e-15)
        np.testing.assert_allclose(
            streamed.trace_double_layer, prepared.trace_double_layer, atol=2e-13
        )


def test_near_panel_subdivision_agrees_with_high_order_on_strong_grading():
    case = load_case(CASE, max_panel_length=80, contact_panel_length=20, contact_end_length=1)
    low = PreparedBoundaryOperators(case.moving, case.material, quadrature=12)
    high = PreparedBoundaryOperators(case.moving, case.material, quadrature=48)
    assert len(low.near[3]) > 0
    for s in [None, 2e4 + 1e4j]:
        a, b = low.evaluate(s), high.evaluate(s)
        # Check only the subdivided pairs, independently of coarse far-panel error.
        i, j = low.near[3].T
        i = 2 * i[:, None] + np.arange(2)
        j = 2 * j[:, None] + np.arange(2)
        np.testing.assert_allclose(
            a.single_layer[i[:, :, None], j[:, None, :]],
            b.single_layer[i[:, :, None], j[:, None, :]],
            rtol=1e-8,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            a.trace_double_layer[i[:, :, None], j[:, None, :]],
            b.trace_double_layer[i[:, :, None], j[:, None, :]],
            rtol=1e-7,
            atol=1e-9,
        )


def test_frequency_condensation_preserves_trace_constraints_and_force_work():
    case = load_case(CASE, max_panel_length=80)
    transfer = InterfaceTransfer(case)
    con = build_constraints(case)
    rng = np.random.default_rng(82)
    for s in [1000.0, 2e4 + 3e4j, 4e5 + 2e5j]:
        z, reconstruct = transfer.evaluate(s, reconstruction=True)
        q = rng.normal(size=transfer.dofs)
        u = reconstruct @ q
        w = block_diag(
            *(
                assemble_operators(m, case.material, s).impedance()
                for m in (case.moving, case.stationary)
            )
        )
        reduced = con.prolongation.T @ u / np.sum(con.prolongation**2, axis=0)
        np.testing.assert_allclose(transfer.jump @ reduced, q, rtol=1e-10, atol=1e-10)
        generalized = con.reduce_force(w @ u)
        np.testing.assert_allclose(generalized, transfer.jump.T @ z @ q, rtol=1e-8, atol=1e-5)
        assert np.max(abs(u[con.fixed_dofs])) == 0
        np.testing.assert_allclose(
            np.vdot(q, z @ q), np.vdot(u, con.full_weights * (w @ u)), rtol=1e-9
        )


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_condensed_cq_matches_full_nonlinear_sliding_trajectory(tmp_path, backend):
    if backend == "jax":
        pytest.importorskip("jax")
    case = load_case(CASE, max_panel_length=80)
    lift = initialize_restart(case)
    # Constant normal pressure, accelerated actuator: both schemes use the same lift.
    schedule = LoadingSchedule(
        lambda t: case.normal_pressure,
        lambda t: case.restart_actuator + 4000 * t,
        "constant_normal_sliding_comparison",
    )
    results = [
        simulate(
            case,
            8e-6,
            128,
            lift=lift,
            schedule=schedule,
            formulation=mode,
            backend=backend,
            cache_directory=tmp_path,
        )
        for mode in ("full", "interface")
    ]
    for result in results:
        assert result.diagnostics["status"] == "completed_requested_horizon"
        assert result.cumulative_slip[-1].max() > 0.1
    for name in ("pressure", "shear", "plastic_slip", "cumulative_slip", "platen"):
        np.testing.assert_allclose(
            getattr(results[0], name), getattr(results[1], name), rtol=2e-6, atol=2e-7
        )
    assert results[1].displacement is None
    assert results[1].contact_jumps.shape == (129, 2 * len(case.moving_contact))


def test_graded_contact_pairs_and_exact_coverage():
    case = load_case(CASE, max_panel_length=40, contact_panel_length=20, contact_end_length=2)
    i, j = case.moving_contact, case.stationary_contact
    np.testing.assert_allclose(case.moving.centers[i], case.stationary.centers[j], atol=1e-10)
    np.testing.assert_allclose(case.moving.lengths[i], case.stationary.lengths[j], atol=1e-10)
    np.testing.assert_allclose(case.moving.lengths[i][[0, -1]], 2.0)
    assert np.max(case.moving.lengths[i]) <= 20 + 1e-10
    np.testing.assert_allclose(case.moving.lengths[i].sum(), 480.0)


def test_interface_preflight_still_budgets_eliminated_spatial_unknowns():
    case = load_case(CASE, max_panel_length=2, contact_panel_length=40)
    with pytest.raises(ResourceBudgetError, match="spatial assembly"):
        plan_interface(case, 8e-6, 5625, workers=4, max_memory_mb=512)


def test_contact_end_grading_needs_consistent_corner_resolution_without_slip_relaxation():
    invalid = load_case(CASE, max_panel_length=40, contact_panel_length=20, contact_end_length=2)
    with pytest.raises(RestartProjectionError, match="friction bound"):
        initialize_restart(invalid)
    resolved = load_case(
        CASE,
        max_panel_length=40,
        contact_panel_length=20,
        contact_end_length=2,
        corner_panel_length=2,
    )
    lift = initialize_restart(resolved, quadrature=12)
    assert lift.diagnostics["initial_max_strength_ratio"] < 0.8
    assert lift.diagnostics["introduced_cumulative_slip_mm"] == 0

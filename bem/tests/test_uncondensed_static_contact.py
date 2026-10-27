import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.nodal import load_nodal_model, nodal_body_stiffness

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


@pytest.mark.parametrize("subdivisions", [1, 2])
def test_body_root_preserves_original_force_and_refined_work_maps(subdivisions):
    refined = runpy.run_path(str(TOOLS / "refined_contact.py"))
    static = runpy.run_path(str(TOOLS / "fem_static_contact.py"))
    direct = runpy.run_path(str(TOOLS / "uncondensed_static_contact.py"))
    model = load_nodal_model(CASE, max_panel_length=160, contact_panel_length=160)
    space = refined["ContactQuadrature"].build(
        model.reference.moving.starts[model.moving_contact, 1], subdivisions
    )
    state = space.seed_from_bulk_p1(model.restart)
    k = nodal_body_stiffness(model, quadrature=4)
    reduction = refined["reduce_static_trace"](model, k)
    old = static["initialize_fem_static_contact"](model, space, reduction, state)
    answer = direct["initialize_uncondensed_contact"](model, space, reduction, state)
    np.testing.assert_allclose(answer.pressure, old.pressure, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(answer.shear, old.shear, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(answer.full_displacement, old.full_displacement, atol=1e-10)
    np.testing.assert_array_equal(answer.state.cumulative_slip, state.cumulative_slip)
    body_coordinates = np.linalg.lstsq(
        model.constraints.prolongation, answer.full_displacement, rcond=None
    )[0]
    physical_body = reduction.reduced_stiffness @ body_coordinates - reduction.reduced_load
    projected_body = reduction.jump.T @ answer.retained_bulk_force
    assert np.max(abs(physical_body - projected_body) / model.constraints.reduced_weights) < 1e-7
    assert answer.diagnostics["original_full_balance_mpa"] < 1e-7
    assert answer.diagnostics["actual_port_equilibrium_residual_mpa"] < 1e-7
    # Perturb only the reconstruction, leaving the original body equations
    # untouched. The direct solve must not inherit the bad Schur reconstruction.
    broken = replace(reduction, displacement_response=reduction.displacement_response * 1.01)
    with pytest.raises(ValueError, match="full-body balance"):
        static["initialize_fem_static_contact"](model, space, broken, state)
    again = direct["initialize_uncondensed_contact"](model, space, broken, state)
    np.testing.assert_array_equal(again.full_displacement, answer.full_displacement)
    np.testing.assert_array_equal(again.retained_bulk_force, answer.retained_bulk_force)


def test_body_static_system_rejects_a_different_normal_load():
    refined = runpy.run_path(str(TOOLS / "refined_contact.py"))
    direct = runpy.run_path(str(TOOLS / "uncondensed_static_contact.py"))
    model = load_nodal_model(CASE, max_panel_length=160, contact_panel_length=160)
    space = refined["ContactQuadrature"].build(
        model.reference.moving.starts[model.moving_contact, 1]
    )
    reduction = refined["reduce_static_trace"](model, nodal_body_stiffness(model, quadrature=4))
    system = direct["BodyStaticEquilibrium"](model, space, reduction)
    with pytest.raises(ValueError, match="original constant normal load"):
        system.external_traction(model.reference.normal_pressure * 2)

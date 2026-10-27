import runpy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
TRANSFER = runpy.run_path(str(ROOT / "tools/schur_condensation_experiment.py"))[
    "ConstrainedSchurTransfer"
]
TRACE = runpy.run_path(str(ROOT / "tools/trace_schur_experiment.py"))["TraceSchurTransfer"]


@pytest.mark.parametrize("factory", [TRANSFER, TRACE])
def test_schur_retains_contact_and_full_boundary_equilibrium_without_rigid_inverse(factory):
    model = load_nodal_model(CASE, max_panel_length=80)
    operator = factory(model, quadrature=12)
    reference = NodalInterfaceTransfer(model, quadrature=12)
    for s in [None, 1e-3, 100 + 2000j, 1e6 + 3e5j]:
        k, _ = operator.assemble_reduced(s)
        z, u, eliminated = operator.condense(k)
        np.testing.assert_allclose(operator.jump @ u, np.eye(operator.dofs), atol=1e-12)
        error = np.linalg.norm(k @ u - operator.jump.T @ z)
        assert error / max(np.linalg.norm(k @ u), 1) < 1e-11
        assert np.linalg.cond(eliminated) < 1e6
        assert np.linalg.norm(z - z.T) / np.linalg.norm(z) < 1e-11
        if s is None or np.isreal(s):
            # The static retained map has rigid translation modes, not a
            # positive artificial spring. Positivity is tested in work form.
            assert np.linalg.eigvalsh((z + z.T.conj()) / 2).min() > -1e-7
        if s is not None and abs(s) > 100:
            expected, full = reference.evaluate(s, reconstruction=True)
            result, actual = operator.evaluate(s, reconstruction=True)
            assert np.linalg.norm(result - expected) / np.linalg.norm(expected) < 1e-10
            assert np.linalg.norm(actual - full) / np.linalg.norm(full) < 1e-10


@pytest.mark.parametrize("factory", [TRANSFER, TRACE])
def test_static_schur_material_homogeneity_is_not_lost_to_rigid_body_inversion(factory):
    model = load_nodal_model(CASE, max_panel_length=80)
    material = replace(model.reference.material, young=2.5 * model.reference.material.young)
    varied = replace(model, reference=replace(model.reference, material=material))
    reference = factory(model).evaluate(None)
    actual = factory(varied).evaluate(None)
    assert np.linalg.norm(actual - 2.5 * reference) / np.linalg.norm(reference) < 1e-11


def test_two_stage_trace_matches_joint_schur_including_full_reconstruction():
    model = load_nodal_model(CASE, max_panel_length=80)
    joint, trace = TRANSFER(model), TRACE(model)
    assert all(not value.flags.writeable for value in trace.body_interiors)
    for s in [None, 1e-3, 100 + 2000j, 1e6 + 3e5j]:
        k, _ = joint.assemble_reduced(s)
        a, u, _ = joint.condense(k)
        b, v, final = trace.condense(k)
        assert np.linalg.norm(a - b) / np.linalg.norm(a) < 1e-11
        assert np.linalg.norm(u - v) / np.linalg.norm(u) < 1e-11
        assert final.shape[0] < joint.interior.shape[1]


def test_four_workers_share_readonly_trace_bases_and_preserve_results_bitwise():
    model = load_nodal_model(CASE, max_panel_length=80)
    operator = TRACE(model)
    for value in [operator.trace_indices, operator.trace_lifting, operator.trace_nullspace]:
        assert not value.flags.writeable
    frequencies = [100 + 2000j, 1e6 + 3e5j] * 2
    expected = [operator.evaluate(s) for s in frequencies]
    with ThreadPoolExecutor(4) as pool:
        actual = list(pool.map(operator.evaluate, frequencies))
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("factory", [TRANSFER, TRACE])
def test_explicit_raw_operators_reuse_same_elimination_without_geometry_allocation(factory):
    model = load_nodal_model(CASE, max_panel_length=160)
    original = factory(model, quadrature=4)
    independent = factory(model, quadrature=4, prepared=False)
    assert independent.prepared is None
    for s in [100 + 2000j, 1e6 + 3e5j]:
        packets = [p.assemble(s) for p in original.prepared]
        expected, full = original.evaluate(s, reconstruction=True)
        actual, reconstruction = independent.evaluate_operators(packets, reconstruction=True)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(reconstruction, full)

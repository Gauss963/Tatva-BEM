import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np

from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
TRANSFER = runpy.run_path(str(ROOT / "tools/schur_condensation_experiment.py"))[
    "ConstrainedSchurTransfer"
]


def test_schur_retains_contact_and_full_boundary_equilibrium_without_rigid_inverse():
    model = load_nodal_model(CASE, max_panel_length=80)
    operator = TRANSFER(model, quadrature=12)
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


def test_static_schur_material_homogeneity_is_not_lost_to_rigid_body_inversion():
    model = load_nodal_model(CASE, max_panel_length=80)
    material = replace(model.reference.material, young=2.5 * model.reference.material.young)
    varied = replace(model, reference=replace(model.reference, material=material))
    reference = TRANSFER(model).evaluate(None)
    actual = TRANSFER(varied).evaluate(None)
    assert np.linalg.norm(actual - 2.5 * reference) / np.linalg.norm(reference) < 1e-11

"""Independent bulk material identities for future material-sensitive CQ."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model


@pytest.mark.parametrize("young_scale,density_scale", [(2.5, 1.0), (1.0, 1.7), (4.0, 9.0)])
def test_bulk_material_homogeneity_matches_independent_constitutive_reassembly(
    young_scale, density_scale
):
    case = (
        Path(__file__).resolve().parents[1]
        / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    )
    model = load_nodal_model(case, max_panel_length=80)
    material = model.reference.material
    varied = replace(
        material, young=young_scale * material.young, density=density_scale * material.density
    )
    candidate = replace(model, reference=replace(model.reference, material=varied))
    baseline = NodalInterfaceTransfer(model, quadrature=12)
    actual = NodalInterfaceTransfer(candidate, quadrature=12)
    # Static free-body compliance is singular: test the full boundary stiffness,
    # rather than inverting rigid modes in the dynamic condensation at s=0.
    for mesh in [model.reference.moving, model.reference.stationary]:
        expected = (
            young_scale * PreparedGalerkinOperators(mesh, material).assemble().force_stiffness()
        )
        result = PreparedGalerkinOperators(mesh, varied).assemble().force_stiffness()
        assert np.linalg.norm(result - expected) / np.linalg.norm(expected) < 1e-11
    for s in [20 + 2000j, 1e6 + 3e5j]:
        mapped = s * np.sqrt(density_scale / young_scale)
        expected = young_scale * baseline.evaluate(mapped)
        result = actual.evaluate(s)
        assert np.linalg.norm(result - expected) / np.linalg.norm(expected) < 1e-11

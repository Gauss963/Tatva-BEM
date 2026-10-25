"""Density enrichment must preserve the physical P1 trace and affine patch."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.material import ElasticMaterial
from finiteblock_bem.mesh import polygon_boundary

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/galerkin_density.py"))


def polygon():
    return polygon_boundary([[0, 0], [1.7, 0.1], [1.3, 1.1], [-0.1, 0.7]], ["free"] * 4, 0.4)


def test_enriched_density_retains_rigid_modes_and_independent_affine_traction():
    mesh, material = polygon(), ElasticMaterial(10, 0.25, 1)
    original_coordinates = mesh.starts.copy()
    rotation = np.column_stack((-mesh.starts[:, 1], mesh.starts[:, 0])).ravel()
    rigid = np.column_stack((np.tile([1, 0], mesh.count), np.tile([0, 1], mesh.count), rotation))
    gradient = np.array([[0.3, 0.7], [-0.2, 0.4]])
    displacement = (mesh.starts @ gradient.T).ravel()
    stress = material.mu * (gradient + gradient.T) + material.lam * np.trace(gradient) * np.eye(2)
    traction = mesh.normals @ stress.T
    expected = np.zeros((mesh.count, 2))
    for edge in range(mesh.count):
        expected[edge] += mesh.lengths[edge] / 2 * traction[edge]
        expected[(edge + 1) % mesh.count] += mesh.lengths[edge] / 2 * traction[edge]
    baseline = None
    for factor in [1, 2]:
        stiffness, d, record = TOOL["restricted_density_stiffness"](
            mesh,
            material,
            factor,
            radius=0.5,
            quadrature=16,
            reference_derivative=baseline,
            max_memory_mb=512,
        )
        if factor == 1:
            baseline = d
        else:
            assert record["density_panels"] > mesh.count
        assert stiffness.shape == (mesh.dofs, mesh.dofs)
        np.testing.assert_allclose(stiffness, stiffness.T, atol=1e-11)
        np.testing.assert_allclose(stiffness @ rigid, 0, atol=1e-7)
        np.testing.assert_allclose(stiffness @ displacement, expected.ravel(), rtol=1e-7, atol=1e-8)
    np.testing.assert_array_equal(mesh.starts, original_coordinates)


@pytest.mark.parametrize("factor,radius", [(0, 1), (2.5, 1), (True, 1), (2, np.nan), (2, 0)])
def test_invalid_density_partition_is_rejected(factor, radius):
    with pytest.raises(ValueError):
        TOOL["refine_density_mesh"](polygon(), factor, radius)


def test_density_workspace_is_checked_before_preparing_quadrature(monkeypatch):
    function = TOOL["restricted_density_stiffness"]

    def unexpected(*args, **kwargs):
        raise AssertionError("Quadrature must not be allocated beyond the preflight budget")

    monkeypatch.setitem(function.__globals__, "PreparedGalerkinOperators", unexpected)
    with pytest.raises(ValueError, match="array estimate"):
        function(polygon(), ElasticMaterial(10, 0.25, 1), 2, max_memory_mb=1e-6)

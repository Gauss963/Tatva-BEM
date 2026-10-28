import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem import ElasticMaterial, polygon_boundary
from finiteblock_bem.kernels import green_and_gradient, static_green_and_gradient

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/interior_fields.py"))
MATERIAL = ElasticMaterial(3.0, 0.23, 1.2)


def square(h):
    return polygon_boundary([[0, 0], [1, 0], [1, 1], [0, 1]], ["b", "r", "t", "l"], h)


@pytest.mark.parametrize("radius", [1e-7, 0.03, 0.2, 2.0])
@pytest.mark.parametrize("s", [None, 1.4 + 0.9j])
def test_analytic_hessian_matches_independent_gradient_differences_and_index_symmetry(radius, s):
    point = radius * np.array([0.8, 0.6])
    actual = TOOL["green_hessian"](point, MATERIAL, s)
    step = radius * 2e-5
    exact = np.empty_like(actual)

    def gradient(x):
        return (
            static_green_and_gradient(x, MATERIAL)
            if s is None
            else green_and_gradient(x, s, MATERIAL)
        )[1]

    for ell in range(2):
        shift = np.eye(2)[ell] * step
        exact[ell] = (gradient(point + shift) - gradient(point - shift)) / (2 * step)
    np.testing.assert_allclose(actual, exact, rtol=3e-6, atol=1e-7 / radius**2)
    np.testing.assert_allclose(actual, actual.swapaxes(0, 1), rtol=1e-11, atol=1e-11 / radius**2)
    np.testing.assert_allclose(actual, actual.swapaxes(2, 3), rtol=1e-11, atol=1e-11 / radius**2)


def test_hessian_has_static_limit_and_laplace_conjugate_symmetry():
    point = np.array([[0.4, 0.3], [0.7, -0.2]])
    static = TOOL["green_hessian"](point, MATERIAL)
    dynamic = TOOL["green_hessian"](point, MATERIAL, 1e-5 + 2e-6j)
    np.testing.assert_allclose(dynamic, static, rtol=1e-7, atol=1e-8)
    np.testing.assert_array_equal(
        TOOL["green_hessian"](point, MATERIAL, 1.4 - 0.9j),
        TOOL["green_hessian"](point, MATERIAL, 1.4 + 0.9j).conj(),
    )


def test_auxiliary_density_recovery_passes_affine_patch_and_is_not_lumped_nodal_force():
    from finiteblock_bem.galerkin import assemble_galerkin_operators

    mesh = square(0.5)
    points = np.array([[0.4, 0.3], [0.7, 0.6]])
    gradient = np.array([[0.2, 0.3], [-0.1, -0.4]])
    displacement = (mesh.starts @ gradient.T).ravel()
    potential = TOOL["InteriorPotentials"](mesh, MATERIAL, points)
    operators = assemble_galerkin_operators(mesh, MATERIAL, quadrature=16)
    correct = potential.from_galerkin_trace(displacement, operators=operators)
    np.testing.assert_allclose(correct["displacement"], points @ gradient.T, atol=2e-8)
    np.testing.assert_allclose(correct["gradient"], np.broadcast_to(gradient, (2, 2, 2)), atol=2e-7)
    weights = (mesh.lengths + np.roll(mesh.lengths, 1)) / 2
    nodal_force = operators.force_stiffness() @ displacement
    wrong = potential.evaluate(displacement, nodal_force / np.repeat(weights, 2))
    assert np.linalg.norm(wrong["displacement"] - correct["displacement"]) > 1e-3


def test_static_affine_patch_recovers_displacement_gradient_and_plane_strain_stress():
    mesh = square(0.5)
    points = np.array([[0.4, 0.3], [0.9, 0.8], [1e-5, 0.6]])
    matrix, shift = np.array([[0.1, -0.3], [0.4, -0.2]]), np.array([1.2, -0.8])
    gradient = np.broadcast_to(matrix, (len(points), 2, 2))
    sigma = MATERIAL.lam * np.trace(matrix) * np.eye(2) + MATERIAL.mu * (matrix + matrix.T)
    data = TOOL["InteriorPotentials"](mesh, MATERIAL, points, quadrature=20)
    actual = data.evaluate(
        (mesh.starts @ matrix.T + shift).ravel(), (mesh.normals @ sigma.T).ravel()
    )
    np.testing.assert_allclose(actual["displacement"], points @ matrix.T + shift, atol=2e-9)
    np.testing.assert_allclose(actual["gradient"], gradient, atol=2e-7)
    np.testing.assert_allclose(
        actual["stress"], TOOL["plane_strain_stress"](gradient, MATERIAL), atol=2e-6
    )


def manufactured_error(h, kind):
    mesh, s = square(h), 1.4 + 0.9j
    points = np.array([[0.3, 0.4], [0.7, 0.6]])
    polarization = np.eye(2)[0 if kind == "P" else 1]
    k = s / (MATERIAL.cp if kind == "P" else MATERIAL.cs)
    u = np.exp(-k * mesh.starts[:, 0])[:, None] * polarization
    boundary_gradient = np.zeros((mesh.count, 2, 2), complex)
    boundary_gradient[:, :, 0] = -k * np.exp(-k * mesh.centers[:, 0])[:, None] * polarization
    stress = TOOL["plane_strain_stress"](boundary_gradient, MATERIAL)
    traction = np.column_stack(
        (
            stress[:, 0] * mesh.normals[:, 0] + stress[:, 2] * mesh.normals[:, 1],
            stress[:, 2] * mesh.normals[:, 0] + stress[:, 1] * mesh.normals[:, 1],
        )
    )
    actual = TOOL["InteriorPotentials"](mesh, MATERIAL, points, quadrature=16).evaluate(
        u.ravel(), traction.ravel(), s
    )
    exact_u = np.exp(-k * points[:, 0])[:, None] * polarization
    exact_gradient = np.zeros((len(points), 2, 2), complex)
    exact_gradient[:, :, 0] = -k * exact_u
    exact_stress = TOOL["plane_strain_stress"](exact_gradient, MATERIAL)
    return {
        key: float(np.linalg.norm(actual[key] - exact) / np.linalg.norm(exact))
        for key, exact in [
            ("displacement", exact_u),
            ("gradient", exact_gradient),
            ("stress", exact_stress),
        ]
    }


@pytest.mark.parametrize("kind", ["P", "S"])
def test_independent_manufactured_p_and_s_waves_converge_in_displacement_and_stress(kind):
    coarse, fine = manufactured_error(0.125, kind), manufactured_error(0.0625, kind)
    for key in coarse:
        assert fine[key] < coarse[key] / 3
        assert fine[key] < 0.005


@pytest.mark.parametrize("point", [[0, 0.5], [1.01, 0.5], [0, 0]])
def test_boundary_and_exterior_targets_are_rejected_without_interior_jump_substitution(point):
    with pytest.raises(ValueError, match="strictly interior"):
        TOOL["InteriorPotentials"](square(0.5), MATERIAL, [point])


def test_near_panel_depth_limit_rejects_underresolved_stress_instead_of_silently_integrating():
    with pytest.raises(ValueError, match="subdivision limit"):
        TOOL["InteriorPotentials"](square(1), MATERIAL, [[1e-5, 0.5]], max_depth=1)


def test_retained_interior_map_matches_original_two_body_condensation_and_recovered_density():
    from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

    case = (
        Path(__file__).resolve().parents[1]
        / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    )
    model = load_nodal_model(case, max_panel_length=160)
    points = [np.array([[100, 240]]), np.array([[270, 240]])]
    transfer = TOOL["InteriorRetainedTransfer"](model, points, quadrature=8)
    original = NodalInterfaceTransfer(model, quadrature=8, prepared=False)
    s = 200 + 2j * np.pi * 1000
    result = transfer.evaluate(s)
    z, boundary = original.evaluate(s, reconstruction=True)
    np.testing.assert_allclose(result["impedance"], z, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(result["boundary_displacement"], boundary, rtol=1e-12, atol=1e-12)
    rng = np.random.default_rng(5)
    direction = rng.normal(size=z.shape[0])
    actual = result["displacement"] @ direction
    first, values = 0, []
    for mesh, probe in zip(
        (model.reference.moving, model.reference.stationary), points, strict=True
    ):
        field = TOOL["InteriorPotentials"](mesh, model.reference.material, probe)
        values.append(
            field.from_galerkin_trace(
                boundary[first : first + mesh.dofs] @ direction, s, quadrature=8
            )["displacement"].ravel()
        )
        first += mesh.dofs
    np.testing.assert_allclose(actual, np.concatenate(values), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result["velocity"], s * result["displacement"])
    with pytest.raises(ValueError, match="static lift separately"):
        transfer.evaluate(0)

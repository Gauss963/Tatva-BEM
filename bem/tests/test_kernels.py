import numpy as np
import pytest

from finiteblock_bem import ElasticMaterial, polygon_boundary
from finiteblock_bem.kernels import green_and_gradient, source_traction, static_green_and_gradient
from finiteblock_bem.operators import assemble_operators


@pytest.mark.parametrize("radius", [1e-7, 0.03, 0.2, 2.0])
def test_green_gradient_and_reciprocity(radius):
    material = ElasticMaterial(3.0, 0.23, 1.2)
    vector = radius * np.array([0.8, 0.6])
    s = 1.4 + 0.9j
    green, gradient = green_and_gradient(vector, s, material)
    np.testing.assert_allclose(green, green.T, atol=1e-14)
    np.testing.assert_allclose(green, green_and_gradient(-vector, s, material)[0], atol=1e-14)
    step = radius * 2e-5
    for k in range(2):
        shift = np.eye(2)[k] * step
        difference = (
            green_and_gradient(vector + shift, s, material)[0]
            - green_and_gradient(vector - shift, s, material)[0]
        ) / (2 * step)
        np.testing.assert_allclose(gradient[k], difference, rtol=2e-6, atol=1e-7)


def test_green_satisfies_laplace_navier_equation():
    material = ElasticMaterial(3.0, 0.23, 1.2)
    s = 1.4 + 0.9j
    point = np.array([0.7, 0.4])
    step = 2e-5
    second = np.empty((2, 2, 2, 2), dtype=complex)
    for k in range(2):
        shift = np.eye(2)[k] * step
        second[k] = (
            green_and_gradient(point + shift, s, material)[1]
            - green_and_gradient(point - shift, s, material)[1]
        ) / (2 * step)
    laplacian = second[0, 0] + second[1, 1]
    grad_div = np.zeros((2, 2), complex)
    for i in range(2):
        for j in range(2):
            grad_div[i, j] = sum(second[i, k, k, j] for k in range(2))
    green = green_and_gradient(point, s, material)[0]
    result = material.density * s * s * green - material.mu * laplacian
    result -= (material.lam + material.mu) * grad_div
    assert np.linalg.norm(result) / np.linalg.norm(material.density * s * s * green) < 2e-7


def _manufactured_wave_error(h, kind):
    mesh = polygon_boundary([[0, 0], [1, 0], [1, 1], [0, 1]], ["b", "r", "t", "l"], h)
    material = ElasticMaterial(3.0, 0.23, 1.2)
    s = 1.4 + 0.9j
    op = assemble_operators(mesh, material, s)
    wave_number = s / (material.cp if kind == "P" else material.cs)
    amplitude = np.exp(-wave_number * mesh.centers[:, 0])
    derivative = -wave_number * amplitude
    if kind == "P":
        u = np.column_stack((amplitude, np.zeros_like(amplitude))).ravel()
        t = np.column_stack(
            (
                (material.lam + 2 * material.mu) * derivative * mesh.normals[:, 0],
                material.lam * derivative * mesh.normals[:, 1],
            )
        ).ravel()
    else:
        u = np.column_stack((np.zeros_like(amplitude), amplitude)).ravel()
        t = np.column_stack(
            (
                material.mu * derivative * mesh.normals[:, 1],
                material.mu * derivative * mesh.normals[:, 0],
            )
        ).ravel()
    residual = op.trace_double_layer @ u - op.single_layer @ t
    return np.linalg.norm(residual) / np.linalg.norm(u)


@pytest.mark.parametrize("kind", ["P", "S"])
def test_boundary_representation_manufactured_wave(kind):
    coarse = _manufactured_wave_error(0.1, kind)
    fine = _manufactured_wave_error(0.05, kind)
    assert fine < coarse
    assert fine < 0.005


def test_static_point_force_normalization_by_traction_flux():
    material = ElasticMaterial(3.0, 0.23, 1.2)
    angle = np.arange(128) * 2 * np.pi / 128
    normals = np.column_stack((np.cos(angle), np.sin(angle)))
    radius = 0.01
    _, gradient = static_green_and_gradient(radius * normals, material)
    flux = (
        np.sum(source_traction(gradient, normals, material), axis=0)
        * radius
        * 2
        * np.pi
        / len(angle)
    )
    np.testing.assert_allclose(flux, -np.eye(2), atol=2e-14)


def test_static_rigid_translations_have_zero_traction():
    mesh = polygon_boundary([[0, 0], [1, 0], [1, 1], [0, 1]], ["b", "r", "t", "l"], 0.1)
    op = assemble_operators(mesh, ElasticMaterial(3.0, 0.23, 1.2))
    for component in [0, 1]:
        u = np.tile(np.eye(2)[component], (mesh.count, 1)).ravel()
        assert np.linalg.norm(op.trace_double_layer @ u) < 1e-11


def test_density_changes_dynamic_response():
    mesh = polygon_boundary([[0, 0], [1, 0], [1, 1], [0, 1]], ["b", "r", "t", "l"], 0.25)
    first = assemble_operators(mesh, ElasticMaterial(3.0, 0.23, 1.2), 20 + 10j).impedance()
    second = assemble_operators(mesh, ElasticMaterial(3.0, 0.23, 2.4), 20 + 10j).impedance()
    assert np.linalg.norm(first - second) / np.linalg.norm(first) > 0.1

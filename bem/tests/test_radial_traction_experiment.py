import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.kernels import (
    _difference_derivatives_series,
    _radial,
    _tensor_from_radial,
    source_traction,
)
from finiteblock_bem.material import ElasticMaterial
from finiteblock_bem.mesh import polygon_boundary

TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/radial_traction_experiment.py")
)


@pytest.mark.parametrize("s", [10 + 3j, 1e4 + 2e5j, 2e6 + 3e5j])
def test_radial_traction_matches_full_gradient_for_general_normals(s):
    rng = np.random.default_rng(19)
    vector = rng.normal(size=(3, 17, 2)) * np.geomspace(1e-4, 300, 17)[None, :, None]
    normals = rng.normal(size=(3, 1, 2))
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    radius = np.linalg.norm(vector, axis=-1)
    material = ElasticMaterial(young=3e3, poisson=0.35, density=1.2e-9)
    a, b, da, db = [x.reshape(radius.shape) for x in _radial(radius.ravel(), s, material)]
    green, full = _tensor_from_radial(vector, radius, a, b, da, db)
    new_green, radial = TOOL["radial_green"](vector, radius, a, b, da, db)
    np.testing.assert_array_equal(new_green, green)
    for sign in [1, -1]:
        actual = TOOL["radial_source_traction"](radial if sign == 1 else -radial, normals, material)
        expected = source_traction(sign * full, normals, material)
        np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=1e-10)


@pytest.mark.parametrize("series", ["reference", "horner"])
def test_isolated_assembler_preserves_globals_and_full_operators(series):
    material = ElasticMaterial(young=3e3, poisson=0.35, density=1.2e-9)
    mesh = polygon_boundary([[0, 0], [10, 1], [8, 20], [-2, 18]], ["free"] * 4, 10)
    original = PreparedGalerkinOperators(mesh, material, quadrature=6)
    bindings = dict(PreparedGalerkinOperators.assemble.__globals__)
    variant = TOOL["RadialTractionPrepared"](mesh, material, quadrature=6, series=series)
    for s in [None, 10 + 2j, 1e5 + 2e6j]:
        expected, actual = original.assemble(s), variant.assemble(s)
        for field in ["single_layer", "trace_double_layer", "hypersingular", "mixed_mass"]:
            np.testing.assert_allclose(
                getattr(actual, field), getattr(expected, field), rtol=1e-11, atol=1e-9
            )
        np.testing.assert_allclose(
            actual.force_stiffness(), expected.force_stiffness(), rtol=1e-10, atol=1e-8
        )
    for name, value in bindings.items():
        assert PreparedGalerkinOperators.assemble.__globals__[name] is value


@pytest.mark.parametrize("s", [100 + 200j, 1e6 + 5e5j, 1e8 - 3e8j])
def test_horner_preserves_the_existing_nine_term_derivative_series(s):
    material = ElasticMaterial(young=3e3, poisson=0.35, density=1.2e-9)
    ks, kp = s / material.cs, s / material.cp
    r = np.geomspace(1e-12, 0.19 / max(abs(ks), abs(kp)), 117)
    expected = _difference_derivatives_series(r, ks, kp)
    actual = TOOL["horner_difference_series"](r, ks, kp)
    for target, answer in zip(expected, actual, strict=True):
        np.testing.assert_allclose(answer, target, rtol=5e-13, atol=1e-22)


def test_radial_contraction_preserves_jax_forward_and_reverse_derivatives():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    def with_jax_arrays(function):
        return FunctionType(
            function.__code__,
            {**function.__globals__, "np": jnp},
            function.__name__,
            function.__defaults__,
        )

    dense_green = with_jax_arrays(_tensor_from_radial)
    dense_traction = with_jax_arrays(source_traction)
    new_green = with_jax_arrays(TOOL["radial_green"])
    new_traction = with_jax_arrays(TOOL["radial_source_traction"])
    normal = jnp.array([0.4, -0.9]) / jnp.linalg.norm(jnp.array([0.4, -0.9]))

    def evaluate(parameters, variant):
        vector = parameters[:2]
        material = SimpleNamespace(mu=parameters[2], lam=parameters[3])
        coefficients = parameters[4::2] + 1j * parameters[5::2]
        _, gradient = (new_green if variant else dense_green)(
            vector, jnp.linalg.norm(vector), *coefficients
        )
        return (new_traction if variant else dense_traction)(gradient, normal, material)

    parameters = jnp.array([2.0, 0.9, 100, 500, 0.2, 0.1, 0.1, -0.3, -0.2, 0.4, 0.7, -0.6])
    direction = jnp.asarray(np.random.default_rng(39).normal(size=len(parameters)))
    reference, actual = lambda p: evaluate(p, False), lambda p: evaluate(p, True)
    old_value, old_tangent = jax.jvp(reference, (parameters,), (direction,))
    new_value, new_tangent = jax.jvp(actual, (parameters,), (direction,))
    np.testing.assert_allclose(new_value, old_value, rtol=1e-13, atol=1e-11)
    np.testing.assert_allclose(new_tangent, old_tangent, rtol=1e-13, atol=1e-11)
    old_gradient = jax.grad(lambda p: jnp.real(jnp.sum(reference(p) * (1 + 2j))))(parameters)
    new_gradient = jax.grad(lambda p: jnp.real(jnp.sum(actual(p) * (1 + 2j))))(parameters)
    np.testing.assert_allclose(new_gradient, old_gradient, rtol=1e-13, atol=1e-11)


def test_shared_prepared_geometry_and_series_cache_are_deterministic_with_four_workers():
    material = ElasticMaterial(young=3e3, poisson=0.35, density=1.2e-9)
    mesh = polygon_boundary([[0, 0], [10, 1], [8, 20], [-2, 18]], ["free"] * 4, 10)
    variant = TOOL["RadialTractionPrepared"](mesh, material, quadrature=6, series="horner")
    frequencies = [10 + 2j, 1e4 + 1e5j, 10 + 2j, 1e4 + 1e5j]
    expected = [variant.assemble(s).force_stiffness() for s in frequencies]
    TOOL["series_coefficients"].cache_clear()
    with ThreadPoolExecutor(4) as pool:
        actual = list(pool.map(lambda s: variant.assemble(s).force_stiffness(), frequencies))
    for target, value in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(value, target)

"""Independent raw, condensed and causal checks of the optional JIT adapter."""

import runpy
from pathlib import Path
from types import FunctionType

import numpy as np
import pytest
from scipy.special import kv

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
TOOLS = Path(__file__).resolve().parents[1] / "tools"
JIT = runpy.run_path(str(TOOLS / "jax_ray_galerkin.py"))
VARIANT = JIT["JaxRayGalerkin"]
PAIRED_TESTS = runpy.run_path(str(TOOLS.parent / "tests/test_paired_ray_galerkin.py"))


@pytest.fixture(scope="module")
def transfer():
    return PAIRED_TESTS["transfer"].__wrapped__()


@pytest.mark.parametrize("s", [None, 175 + 0j, 175 + 31400j, 3e6 + 0j])
def test_original_raw_and_full_schur_gates(transfer, s):
    original = PAIRED_TESTS["test_both_full_raw_bodies_and_complete_schur_maps_keep_original_gates"]
    check = FunctionType(original.__code__, {**original.__globals__, "VARIANT": VARIANT})
    # The reused control additionally checks that original globals remain intact.
    check(transfer, s)


def test_whole_frequency_exact_fallback_when_bins_rejected(transfer):
    variant = VARIANT(transfer.prepared[0], degree=8, check_rtol=1e-30)
    evaluate, table = variant.prepare(175 + 31400j)
    assert not np.any(table.accepted)
    exact = (
        JIT["FACTORED"]["FactoredGalerkinOperators"]
        .share(transfer.prepared[0])
        .assemble(175 + 31400j)
    )
    actual = evaluate()
    for name in ["single_layer", "trace_double_layer", "hypersingular"]:
        np.testing.assert_array_equal(getattr(actual, name), getattr(exact, name))


def test_complete_causal_history_and_future_transpose(transfer):
    original = PAIRED_TESTS["test_all_small_contour_causal_weights_forward_and_future_transpose"]
    check = FunctionType(original.__code__, {**original.__globals__, "VARIANT": VARIANT})
    check(transfer)


def test_changed_radial_source_requires_reaudit():
    def incompatible(r, s, material):
        return r

    with pytest.raises(ValueError, match="re-audit"):
        JIT["_radial_code"](incompatible)


@jax.jit
def value_and_ray_scale_jvp(z, coefficients, first):
    one = jax.numpy.array(1.0, dtype=jax.numpy.float64)
    return jax.jvp(lambda scale: JIT["_pair"](scale * z, coefficients, first), (one,), (one,))


@pytest.mark.parametrize("angle", [0, 0.7, -0.7, np.pi / 2 - 1e-5, -np.pi / 2 + 1e-5])
def test_ray_scale_derivative_matches_independent_bessel_identities_at_seams(angle):
    # d/dt K0(t*z)=-z*K1(t*z), d/dt K1(t*z)=-z*K0(t*z)-K1(t*z) at t=1.
    # These independent SciPy identities test the derivative, not just an
    # optimization result or JVP/VJP duality of the same approximation.
    seam = np.ldexp(np.ones(42), np.arange(-32, 10))
    radius = np.r_[
        np.geomspace(1e-10, 512, 1237), seam, np.nextafter(seam, 0), np.nextafter(seam, np.inf)
    ]
    table = JIT["RAY"]["RayBesselTable"].prepare(
        np.exp(1j * angle), radius.min() / 2, radius.max() * 2
    )
    z = table.direction * radius
    assert np.all(table._coordinates(z)[0])
    values, tangent = value_and_ray_scale_jvp(z, table.coefficients, np.int32(table.first_exponent))
    exact_values = np.stack([kv(0, z), kv(1, z)])
    exact_tangent = np.stack([-z * kv(1, z), -z * kv(0, z) - kv(1, z)])
    np.testing.assert_allclose(values, exact_values, rtol=2e-13, atol=1e-280)
    np.testing.assert_allclose(tangent, exact_tangent, rtol=2e-11, atol=1e-280)

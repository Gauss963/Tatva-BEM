"""Pair reuse, exact fallback, private assembly and whole-history controls."""

import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from scipy.special import kv

from finiteblock_bem import kernels
from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parents[1] / "tools"
PAIRED = runpy.run_path(str(TOOLS / "paired_ray_galerkin.py"))
TABLE = PAIRED["RAY"]["RayBesselTable"]
VARIANT = PAIRED["PairedRayGalerkin"]
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


@pytest.mark.parametrize("angle", [0, 0.7, -0.7, np.pi / 2 - 1e-8, -np.pi / 2 + 1e-8])
def test_pair_matches_independent_orders_and_scipy_on_fresh_points(angle):
    table = TABLE.prepare(np.exp(1j * angle), 1e-12, 1e4)
    seam = np.ldexp(np.ones(53), np.arange(-40, 13))
    radius = np.r_[np.geomspace(1e-12, 1e4, 1237), seam, np.nextafter(seam, 0)]
    z = table.direction * radius
    pair = PAIRED["bessel_pair"](table, z)
    for order, value in enumerate(pair):
        np.testing.assert_array_equal(value, table(order, z))
        np.testing.assert_allclose(value, kv(order, z), rtol=2e-13, atol=1e-300)
    assert all(np.ndim(a) == 0 for a in PAIRED["bessel_pair"](table, complex(z[500])))
    assert PAIRED["bessel_pair"](table, np.empty((2, 0), complex))[0].shape == (2, 0)


@pytest.mark.parametrize("rejected", [False, True])
def test_exact_fallback_for_unsupported_arguments_and_rejected_bins(rejected):
    table = TABLE.prepare(1 + 2j, 1e-3, 100, check_rtol=1e-30 if rejected else 5e-14)
    z = np.array([0j, -2 + 1j, 1 + 0j, 1e-7 + 2e-7j, 1e6 + 2e6j, np.nan + 1j])
    if rejected:
        assert not np.any(table.accepted)
        z = np.r_[z, (1 + 2j) * np.geomspace(1e-3, 10, 127)]
    for order, value in enumerate(PAIRED["bessel_pair"](table, z)):
        np.testing.assert_array_equal(value, kv(order, z))
    real = np.array([[1.0, 2.0]])
    for order, value in enumerate(PAIRED["bessel_pair"](table, real)):
        assert not np.iscomplexobj(value)
        np.testing.assert_array_equal(value, kv(order, real))


def test_adapter_rejects_changed_reference_bessel_assignment():
    def altered(r, s, material):
        return r

    with pytest.raises(ValueError, match="re-audit"):
        PAIRED["paired_radial_code"](altered)


@pytest.fixture(scope="module")
def transfer():
    model = load_nodal_model(
        CASE,
        max_panel_length=160,
        contact_panel_length=160,
        contact_end_length=40,
        corner_panel_length=40,
    )
    return runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"](
        model, quadrature=8
    )


@pytest.mark.parametrize("s", [None, 175 + 0j, 175 + 31400j, 3e6 + 0j])
def test_both_full_raw_bodies_and_complete_schur_maps_keep_original_gates(transfer, s):
    saved = kernels.kv, kernels._radial, PreparedGalerkinOperators.assemble
    exact = [a.assemble(s) for a in transfer.prepared]
    variants = [VARIANT(a) for a in transfer.prepared]
    actual = [a.assemble(s) for a in variants]
    for a, b in zip(actual, exact, strict=True):
        for name in ["single_layer", "trace_double_layer", "hypersingular"]:
            x, y = getattr(a, name), getattr(b, name)
            assert np.linalg.norm(x - y) <= 1e-14 + 1e-11 * np.linalg.norm(y)
    if s is not None:
        a = transfer.evaluate_operators(actual, reconstruction=True)
        b = transfer.evaluate_operators(exact, reconstruction=True)
        for x, y in zip(a, b, strict=True):
            assert np.linalg.norm(x - y) / np.linalg.norm(y) < 1e-8
            assert np.linalg.norm(x - y, np.inf) / np.linalg.norm(y, np.inf) < 1e-8
        evaluate, table = variants[0].prepare(s)
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda _: evaluate(), range(4)))
        for result in results:
            np.testing.assert_array_equal(result.hypersingular, actual[0].hypersingular)
        assert not table.coefficients.flags.writeable
    assert (kernels.kv, kernels._radial, PreparedGalerkinOperators.assemble) == saved
    assert saved[2].__globals__["_radial"] is saved[1]
    assert variants[0].statistics()["original_small_argument_series_retained"]


def test_all_small_contour_causal_weights_forward_and_future_transpose(transfer):
    # Reuse the independently checked full-contour experiment, changing only
    # its private backend constructor. No original test/module is modified.
    from types import FunctionType

    original = runpy.run_path(str(TOOLS.parent / "tests/test_ray_bessel_galerkin.py"))[
        "test_complete_small_radau_causal_history_and_future_transpose"
    ]
    check = FunctionType(original.__code__, {**original.__globals__, "VARIANT": VARIANT})
    check()

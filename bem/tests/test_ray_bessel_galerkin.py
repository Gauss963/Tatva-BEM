import json
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
RAY = runpy.run_path(str(TOOLS / "ray_bessel_galerkin.py"))
TABLE, VARIANT = RAY["RayBesselTable"], RAY["RayBesselGalerkin"]
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


@pytest.mark.parametrize("angle", [0, 0.7, -0.7, np.pi / 2 - 1e-8, -np.pi / 2 + 1e-8])
def test_fresh_points_and_dyadic_seams_on_positive_real_rays(angle):
    direction = np.exp(1j * angle)
    table = TABLE.prepare(direction, 1e-12, 1e4)
    seam = np.ldexp(np.ones(53), np.arange(-40, 13))
    radius = np.r_[
        np.geomspace(1e-12, 1e4, 1237),
        seam,
        np.nextafter(seam, 0),
        np.nextafter(seam, np.inf),
    ]
    z = direction * radius
    for order in [0, 1]:
        np.testing.assert_allclose(table(order, z), kv(order, z), rtol=2e-13, atol=1e-300)
    assert table.statistics()["accepted_bins"] == len(table.accepted)
    assert not table.coefficients.flags.writeable and not table.accepted.flags.writeable
    assert not table.statistics()["uniform_error_certificate"]


def test_unsupported_argument_and_rejected_bin_use_exact_scipy_fallback():
    table = TABLE.prepare(1 + 2j, 1e-3, 100)
    z = np.array([0j, -2 + 1j, 1 + 0j, 1e-7 + 2e-7j, 1e6 + 2e6j, np.nan + 1j])
    for order in [0, 1, 2]:
        np.testing.assert_array_equal(table(order, z), kv(order, z))
    np.testing.assert_array_equal(table(0, np.array([1.0, 2.0])), kv(0, [1.0, 2.0]))
    np.testing.assert_array_equal(table([0, 1], np.array([1j, 2j])), kv([0, 1], [1j, 2j]))
    assert np.ndim(table(0, 1 + 2j)) == 0
    # Deliberately impossible sampled tolerance disables bins without using a weak fit.
    rejected = TABLE.prepare(1 + 2j, 1e-3, 100, degree=8, check_rtol=1e-30)
    assert not np.any(rejected.accepted)
    z = (1 + 2j) * np.geomspace(1e-3, 10, 127)
    np.testing.assert_array_equal(rejected(1, z), kv(1, z))


@pytest.mark.parametrize("s,minimum,maximum", [(0, 1, 2), (-1 + 2j, 1, 2), (1, 0, 2), (1, 2, 1)])
def test_invalid_domain_is_rejected_before_table_allocation(s, minimum, maximum):
    with pytest.raises(ValueError):
        TABLE.prepare(s, minimum, maximum)


def test_nonfinite_scaled_library_bin_falls_back_with_serializable_metadata(monkeypatch):
    namespace = TABLE.prepare.__func__.__globals__
    original = namespace["kve"]

    def unavailable(order, z):
        value = original(order, z)
        return np.where(abs(z) > 4, complex(np.nan, np.nan), value)

    monkeypatch.setitem(namespace, "kve", unavailable)
    table = TABLE.prepare(1 + 2j, 1, 8)
    z = table.direction * np.linspace(4.01, 7.99, 37)
    for order in [0, 1]:
        np.testing.assert_array_equal(table(order, z), kv(order, z))
    json.dumps(table.statistics(), allow_nan=False)


def test_mutable_shared_geometry_is_rejected():
    model = load_nodal_model(
        CASE,
        max_panel_length=160,
        contact_panel_length=160,
        contact_end_length=40,
        corner_panel_length=40,
    )
    prepared = PreparedGalerkinOperators(
        model.reference.moving, model.reference.material, quadrature=4
    )
    prepared.mass.flags.writeable = True
    with pytest.raises(ValueError, match="immutable"):
        VARIANT(prepared)


@pytest.mark.parametrize("s", [None, 175 + 0j, 175 + 31400j, 3e6 + 0j])
def test_complete_both_body_and_condensed_maps_keep_original_tolerances(s):
    model = load_nodal_model(
        CASE,
        max_panel_length=160,
        contact_panel_length=160,
        contact_end_length=40,
        corner_panel_length=40,
    )
    transfer = runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"](
        model, quadrature=8
    )
    original_kv, original_radial = kernels.kv, kernels._radial
    original_assemble = PreparedGalerkinOperators.assemble
    variants = [VARIANT(a) for a in transfer.prepared]
    exact = [a.assemble(s) for a in transfer.prepared]
    actual = [a.assemble(s) for a in variants]
    for a, b in zip(actual, exact, strict=True):
        for field in ["single_layer", "trace_double_layer", "hypersingular"]:
            x, y = getattr(a, field), getattr(b, field)
            assert np.linalg.norm(x - y) / (1e-14 + 1e-11 * np.linalg.norm(y)) < 1
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
    assert kernels.kv is original_kv and kernels._radial is original_radial
    assert PreparedGalerkinOperators.assemble is original_assemble
    assert original_assemble.__globals__["_radial"] is original_radial


def test_complete_small_radau_causal_history_and_future_transpose():
    from finiteblock_bem.cq import plan_cq

    rk = runpy.run_path(str(TOOLS / "radau_experiment.py"))
    model = load_nodal_model(
        CASE,
        max_panel_length=160,
        contact_panel_length=160,
        contact_end_length=40,
        corner_panel_length=40,
    )
    transfer = runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"](
        model, quadrature=8
    )
    original = transfer.prepared
    variants = [VARIANT(a) for a in original]
    plan = plan_cq(transfer.dofs, 2e-6, 9)
    weights = []
    for providers in [original, variants]:
        families = [
            np.empty((plan.fft_size // 2 + 1, plan.dofs, plan.dofs), complex) for _ in range(2)
        ]

        def evaluate(s):
            return transfer.evaluate_operators([a.assemble(s) for a in providers])

        for k in range(plan.fft_size // 2 + 1):
            z = plan.radius * np.exp(-2j * np.pi * k / plan.fft_size)
            families[0][k], families[1][k] = rk["compact_matrix_function"](evaluate, z, plan.dt)
        compact = rk["invert_compact_spectrum"](*families, plan)
        weights.append(rk["expand_compact_weights"](*compact))
    assert plan.fft_size == 32
    rng = np.random.default_rng(26030)
    x, cotangent = rng.normal(size=(2, plan.steps + 1, 2 * plan.dofs))
    histories, pullbacks = [], []
    for w in weights:
        histories.append(
            np.stack([sum(w[n - k] @ x[k] for k in range(n + 1)) for n in range(len(x))])
        )
        pullbacks.append(
            np.stack(
                [sum(w[n - k].T @ cotangent[n] for n in range(k, len(x))) for k in range(len(x))]
            )
        )
    for actual, exact in [
        (weights[1], weights[0]),
        (histories[1], histories[0]),
        (pullbacks[1], pullbacks[0]),
    ]:
        assert np.linalg.norm(actual - exact) / np.linalg.norm(exact) < 1e-9
    lhs, rhs = np.vdot(histories[1], cotangent), np.vdot(x, pullbacks[1])
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1) < 1e-12

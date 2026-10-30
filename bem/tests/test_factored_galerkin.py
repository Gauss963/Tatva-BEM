import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parents[1] / "tools"
FAST = runpy.run_path(str(TOOLS / "factored_galerkin.py"))
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


@pytest.mark.parametrize("equation", FAST["SIGNATURES"])
def test_factored_contractions_match_complex_bilinear_quadrature(equation):
    rng = np.random.default_rng(76)
    p, q = 3, 144
    args = [rng.random((p, q)), rng.normal(size=(p, q, 2, 2)) + 1j * rng.normal(size=(p, q, 2, 2))]
    for item in equation.split("->")[0].split(",")[2:]:
        args.append(rng.normal(size=(p, q, 2) if "q" in item else (p, 2)))
    expected = np.einsum(equation, *args)
    actual = FAST["factored_einsum"](equation, *args)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-12)


@pytest.mark.parametrize("s", [None, 200 + 0j, 200 + 2e4j, 5e5 + 2e5j])
def test_both_body_operators_condensation_and_shared_geometry_remain_equivalent(s):
    model = load_nodal_model(
        CASE,
        max_panel_length=160,
        contact_panel_length=160,
        contact_end_length=40,
        corner_panel_length=40,
    )
    schur = runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"](
        model, quadrature=8
    )
    original_einsum = np.einsum
    original_namespace = PreparedGalerkinOperators.assemble.__globals__["np"]
    variants = [FAST["FactoredGalerkinOperators"].share(a) for a in schur.prepared]
    reference, actual = [], []
    for a, b in zip(schur.prepared, variants, strict=True):
        assert a.batches is b.batches and a.mass is b.mass
        reference.append(a.assemble(s))
        actual.append(b.assemble(s))
        for name in ["single_layer", "trace_double_layer", "hypersingular"]:
            x, y = getattr(reference[-1], name), getattr(actual[-1], name)
            np.testing.assert_allclose(y, x, rtol=2e-11, atol=2e-8)
    if s is not None:
        x = schur.evaluate_operators(reference, reconstruction=True)
        y = schur.evaluate_operators(actual, reconstruction=True)
        for a, b in zip(x, y, strict=True):
            np.testing.assert_allclose(b, a, rtol=2e-9, atol=1e-6)
    assert np.einsum is original_einsum and original_namespace is np
    assert PreparedGalerkinOperators.assemble.__globals__["np"] is np
    with ThreadPoolExecutor(4) as pool:
        shared = list(pool.map(variants[0].assemble, [s] * 4))
    for result in shared:
        np.testing.assert_array_equal(result.hypersingular, actual[0].hypersingular)
    assert variants[0].statistics()["additional_workspace_estimate_bytes"] < 1024**2


def test_geometry_mutation_is_rejected_before_assembly():
    model = load_nodal_model(
        CASE,
        max_panel_length=160,
        contact_panel_length=160,
        contact_end_length=160,
        corner_panel_length=160,
    )
    prepared = PreparedGalerkinOperators(
        model.reference.moving, model.reference.material, quadrature=4
    )
    prepared.mass.flags.writeable = True
    with pytest.raises(ValueError, match="immutable"):
        FAST["FactoredGalerkinOperators"].share(prepared)


def test_complete_radau_causal_history_and_transpose_match_reference():
    """Both physical branches, every contour node and every retained lag."""
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
    variant = [FAST["FactoredGalerkinOperators"].share(a) for a in original]
    plan = plan_cq(transfer.dofs, 2e-6, 9)
    weights = []
    for providers in [original, variant]:
        families = [
            np.empty((plan.fft_size // 2 + 1, plan.dofs, plan.dofs), dtype=complex)
            for _ in range(2)
        ]

        def evaluate(s):
            return transfer.evaluate_operators([a.assemble(s) for a in providers])

        for k in range(plan.fft_size // 2 + 1):
            z = plan.radius * np.exp(-2j * np.pi * k / plan.fft_size)
            families[0][k], families[1][k] = rk["compact_matrix_function"](evaluate, z, plan.dt)
        compact = rk["invert_compact_spectrum"](*families, plan)
        weights.append(rk["expand_compact_weights"](*compact))
    assert plan.fft_size == 32
    assert weights[0].shape == (plan.steps + 1, 2 * plan.dofs, 2 * plan.dofs)
    assert np.linalg.norm(weights[1] - weights[0]) / np.linalg.norm(weights[0]) < 1e-10
    rng = np.random.default_rng(431)
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
    for a, b in [(histories[1], histories[0]), (pullbacks[1], pullbacks[0])]:
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-10
    left, right = np.vdot(histories[1], cotangent), np.vdot(x, pullbacks[1])
    assert abs(left - right) / max(abs(left), abs(right), 1) < 1e-12
    assert transfer.prepared is original

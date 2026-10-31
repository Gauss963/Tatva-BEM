"""Independent raw, condensed and causal checks of the optional JIT adapter."""

import runpy
from pathlib import Path
from types import FunctionType

import numpy as np
import pytest

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

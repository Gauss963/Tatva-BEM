import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parents[1] / "tools"
PACKED = runpy.run_path(str(TOOLS / "packed_galerkin_blocks.py"))
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
FIELDS = ["single_layer", "trace_double_layer", "hypersingular", "mixed_mass"]


@pytest.mark.parametrize("s", [None, 175 + 31400j, 3e6 + 0j])
def test_packed_fibers_and_blocks_match_independent_complete_integrals(s):
    model = load_nodal_model(
        CASE,
        max_panel_length=80,
        contact_panel_length=80,
        contact_end_length=10,
        corner_panel_length=10,
    )
    for mesh in [model.reference.moving, model.reference.stationary]:
        exact = PreparedGalerkinOperators(mesh, model.reference.material, quadrature=8).assemble(s)
        provider = PACKED["PackedBlockProvider"](mesh, model.reference.material, quadrature=8)
        for r, c in [
            (np.arange(mesh.dofs), np.array([1])),
            (np.array([0, 1, 6, 7]), np.array([4, 5, 20, 21])),
        ]:
            result = provider.request(r, c).evaluate(s)
            for name in FIELDS:
                np.testing.assert_allclose(
                    getattr(result, name),
                    getattr(exact, name)[np.ix_(r, c)],
                    rtol=2e-12,
                    atol=1e-10,
                )


def test_packing_reduces_fiber_batch_count_without_changing_pairs_or_quadrature():
    model = load_nodal_model(
        CASE,
        max_panel_length=80,
        contact_panel_length=80,
        contact_end_length=10,
        corner_panel_length=10,
    )
    mesh = model.reference.moving
    provider = PACKED["BASE"]["GalerkinBlockProvider"](mesh, model.reference.material, quadrature=8)
    original = provider.request(np.arange(mesh.dofs), [1])
    packed = PACKED["pack_request"](original)
    assert len(packed.batches) < len(original.batches)
    for key in ["prepared_panel_pairs", "prepared_quadrature_points", "prepared_geometry_bytes"]:
        assert original.statistics()[key] == packed.statistics()[key]
    assert packed.mass is original.mass


def test_frequency_local_pair_cache_reuses_values_and_rejects_cross_frequency():
    model = load_nodal_model(
        CASE,
        max_panel_length=80,
        contact_panel_length=80,
        contact_end_length=10,
        corner_panel_length=10,
    )
    mesh = model.reference.moving
    provider = PACKED["PackedBlockProvider"](mesh, model.reference.material, quadrature=8)
    request = provider.request(np.arange(mesh.dofs), [1])
    cache = PACKED["PairValueCache"](provider, 200 + 10000j)
    for b in request.batches:
        a = cache.evaluate(provider.geometry, b, 200 + 10000j)
        expected = PACKED["BASE"]["_PAIR"](provider.geometry, b, 200 + 10000j)
        for x, y in zip(a, expected, strict=True):
            if x is None:
                assert y is None
            else:
                np.testing.assert_array_equal(x, y)
    count = cache.evaluations
    for b in request.batches:
        cache.evaluate(provider.geometry, b, 200 + 10000j)
    assert cache.evaluations == count
    with pytest.raises(ValueError):
        cache.evaluate(provider.geometry, request.batches[0], 201 + 10000j)


def test_packed_cached_complete_operators_pass_reference_audit():
    model = load_nodal_model(
        CASE,
        max_panel_length=80,
        contact_panel_length=80,
        contact_end_length=10,
        corner_panel_length=10,
    )
    mesh = model.reference.moving
    s = 175 + 31400j
    method = PACKED["PackedHierarchicalGalerkin"](
        mesh, model.reference.material, quadrature=8, leaf_panels=4
    )
    actual, record = method.evaluate(s)
    exact = PreparedGalerkinOperators(mesh, model.reference.material, quadrature=8).assemble(s)
    for field in FIELDS:
        a, b = getattr(actual, field), getattr(exact, field)
        assert np.linalg.norm(a - b) / (1e-14 + 1e-11 * np.linalg.norm(b)) < 1
    assert record["source_pair_evaluations"] <= record["original_complete_panel_pairs"]
    assert record["pair_cache_enabled"]


def test_pair_cache_capacity_limit_uses_uncached_evaluator():
    model = load_nodal_model(
        CASE,
        max_panel_length=160,
        contact_panel_length=160,
        contact_end_length=40,
        corner_panel_length=40,
    )
    mesh = model.reference.moving
    provider = PACKED["PackedBlockProvider"](mesh, model.reference.material, quadrature=8)
    request = provider.request([0, 1], [2, 3])
    batch = request.batches[0]
    cache = PACKED["PairValueCache"](provider, 200 + 10000j, max_bytes=0)
    assert not cache.enabled and cache.data is None and cache.nbytes == 0
    a = cache.evaluate(provider.geometry, batch, 200 + 10000j)
    b = PACKED["BASE"]["_PAIR"](provider.geometry, batch, 200 + 10000j)
    for x, y in zip(a, b, strict=True):
        if x is None:
            assert y is None
        else:
            np.testing.assert_array_equal(x, y)

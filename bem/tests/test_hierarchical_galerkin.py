import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parents[1] / "tools"
H = runpy.run_path(str(TOOLS / "hierarchical_galerkin.py"))
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
FIELDS = ["single_layer", "trace_double_layer", "hypersingular"]


@pytest.fixture
def model():
    return load_nodal_model(
        CASE,
        max_panel_length=80,
        contact_panel_length=80,
        contact_end_length=10,
        corner_panel_length=10,
    )


def test_shared_support_tree_covers_both_bases_and_partition_covers_all_entries(model):
    mesh = model.reference.moving
    root = H["cluster_tree"](mesh, 4)
    coverage = np.zeros((mesh.dofs, mesh.dofs), int)
    for a, b, far in H["partition"](root, 1):
        coverage[np.ix_(a.dofs, b.dofs)] += 1
        if a is not b:
            coverage[np.ix_(b.dofs, a.dofs)] += 1
        for c in [a, b]:
            for i in c.labels:
                points = np.vstack(
                    [
                        mesh.starts[i],
                        mesh.ends[i],
                        mesh.starts[(i - 1) % mesh.count],
                        mesh.ends[(i - 1) % mesh.count],
                    ]
                )
                assert np.all(points >= c.lower) and np.all(points <= c.upper)
        if far:
            assert not np.intersect1d(a.labels, b.labels).size
    np.testing.assert_array_equal(coverage, 1)


@pytest.mark.parametrize("s", [None, 200 + 21000j, 3e6 + 0j])
def test_paired_oracle_matches_both_directions_with_distinct_double_layers(model, s):
    mesh, material = model.reference.moving, model.reference.material
    original = PreparedGalerkinOperators(mesh, material, quadrature=8).assemble(s)
    provider = H["BLOCK"]["GalerkinBlockProvider"](mesh, material, quadrature=8)
    rows = np.array([0, 1, 10, 11])
    columns = np.array([4, 5, 18, 19])
    oracle = H["PairedOracle"](provider, rows, columns, s)
    actual = oracle.full()
    for k, name in enumerate(FIELDS):
        np.testing.assert_allclose(
            actual[k], getattr(original, name)[np.ix_(rows, columns)], rtol=2e-12, atol=1e-10
        )
    np.testing.assert_allclose(
        actual[3], original.trace_double_layer[np.ix_(columns, rows)].T, rtol=2e-12, atol=1e-10
    )
    assert np.linalg.norm(actual[1] - actual[3]) > 1e-7
    calls = oracle.calls
    np.testing.assert_array_equal(oracle.get([0, 1], [2, 3]), actual[:, :2, 2:])
    assert oracle.calls == calls


class SyntheticOracle:
    def __init__(self, matrix):
        self.values = np.stack([matrix] * 4)
        self.known = np.zeros(matrix.shape, bool)

    def get(self, rows, columns):
        self.known[np.ix_(rows, columns)] = True
        return self.values[np.ix_(np.arange(4), rows, columns)]


def test_complex_aca_and_complete_bilinear_transpose():
    rng = np.random.default_rng(13)
    left = rng.normal(size=(64, 2)) + 1j * rng.normal(size=(64, 2))
    right = rng.normal(size=(2, 72)) + 1j * rng.normal(size=(2, 72))
    matrix = left @ right
    factor = H["adaptive_cross"](SyntheticOracle(matrix), 0, 1e-9)
    assert factor is not None
    np.testing.assert_allclose(factor.dense(), matrix, rtol=1e-11, atol=1e-10)
    x = rng.normal(size=72) + 1j * rng.normal(size=72)
    y = rng.normal(size=64) + 1j * rng.normal(size=64)
    np.testing.assert_allclose(y @ (factor.dense() @ x), factor.transpose_apply(y) @ x, rtol=1e-12)


def test_high_rank_aca_returns_explicit_dense_fallback():
    rng = np.random.default_rng(71)
    factor = H["adaptive_cross"](SyntheticOracle(rng.normal(size=(48, 48))), 0, 1e-12, max_rank=2)
    assert factor is None


@pytest.mark.parametrize("s", [None, 175 + 31400j, 3e6 + 0j])
def test_complete_hierarchical_raw_operators_match_independent_reference(model, s):
    for mesh in [model.reference.moving, model.reference.stationary]:
        exact = PreparedGalerkinOperators(mesh, model.reference.material, quadrature=8).assemble(s)
        method = H["HierarchicalGalerkin"](
            mesh, model.reference.material, quadrature=8, leaf_panels=4, eta=1
        )
        actual, record = method.evaluate(s)
        for name in FIELDS:
            a, b = getattr(actual, name), getattr(exact, name)
            assert np.linalg.norm(a - b) / (1e-14 + 1e-11 * np.linalg.norm(b)) < 1
        np.testing.assert_allclose(
            actual.force_stiffness(), exact.force_stiffness(), rtol=1e-8, atol=1e-6
        )
        assert record["certified_error_bound"] is False
        assert any(r["kind"] == "exact_near" for r in record["blocks"])


def test_rectangular_small_entries_do_not_stall_zero_row_lookahead():
    matrix = np.full((8, 128), 0.02)
    factor = H["adaptive_cross"](SyntheticOracle(matrix), 0, 1.0)
    assert factor is not None
    np.testing.assert_allclose(factor.dense(), matrix, atol=1e-15)


def test_partition_and_operator_metadata_are_plain_json_values(model):
    import json

    mesh = model.reference.moving
    method = H["HierarchicalGalerkin"](mesh, model.reference.material, quadrature=8, leaf_panels=4)
    _, record = method.evaluate(200 + 10000j)
    json.dumps(
        dict(
            near=sum(not far for _, _, far in method.blocks),
            far=sum(far for _, _, far in method.blocks),
            record=record,
        ),
        allow_nan=False,
    )


def test_support_boxes_enclose_previous_endpoint_with_allowed_closure_roundoff():
    from finiteblock_bem.mesh import BoundaryMesh

    starts = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    ends = np.roll(starts, -1, axis=0).copy()
    ends[2, 0] -= 1e-11
    mesh = BoundaryMesh(starts, ends, np.array(["a", "b", "c"]))
    root = H["cluster_tree"](mesh, 2)
    first = root.children[0]
    np.testing.assert_array_equal(first.labels, [0])
    assert first.lower[0] <= ends[2, 0]

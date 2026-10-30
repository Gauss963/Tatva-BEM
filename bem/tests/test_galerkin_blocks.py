import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parents[1] / "tools"
BLOCKS = runpy.run_path(str(TOOLS / "galerkin_blocks.py"))
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
FIELDS = ["single_layer", "trace_double_layer", "hypersingular", "mixed_mass"]


@pytest.mark.parametrize("count", [3, 4, 7, 11])
def test_pair_ordinals_reproduce_every_original_group_and_batch_boundary(count):
    groups = [[], [], []]
    for i in range(count):
        for j in range(i, count):
            kind = 0 if i == j else (1 if j == i + 1 or (i == 0 and j == count - 1) else 2)
            groups[kind].append((i, j))
    for kind, pairs in enumerate(groups):
        for ordinal, (i, j) in enumerate(pairs):
            assert BLOCKS["pair_order"](i, j, count) == (kind, ordinal)


@pytest.fixture
def model():
    return load_nodal_model(
        CASE,
        max_panel_length=80,
        contact_panel_length=80,
        contact_end_length=10,
        corner_panel_length=10,
    )


@pytest.mark.parametrize("s", [None, 200 + 0j, 200 + 2e4j, 3e6 + 8e5j])
def test_both_bodies_selected_entries_fibers_and_unsorted_blocks_match_complete_assembly(model, s):
    rng = np.random.default_rng(142)
    for mesh in [model.reference.moving, model.reference.stationary]:
        reference = PreparedGalerkinOperators(
            mesh, model.reference.material, quadrature=8
        ).assemble(s)
        provider = BLOCKS["GalerkinBlockProvider"](mesh, model.reference.material, quadrature=8)
        n = mesh.dofs
        selections = [
            ([n - 1], [0]),
            (np.arange(n), [3]),
            ([5, 0, n - 2, 9], [n - 1, 7, 0]),
            (rng.permutation(n), rng.permutation(n)),
        ]
        for rows, columns in selections:
            request = provider.request(rows, columns)
            result = request.evaluate(s)
            for name in FIELDS:
                expected = getattr(reference, name)[np.ix_(rows, columns)]
                np.testing.assert_allclose(getattr(result, name), expected, rtol=2e-12, atol=1e-10)
            assert all(
                not a.flags.writeable
                for b in request.batches
                for a in vars(b).values()
                if isinstance(a, np.ndarray)
            )
        assert not hasattr(provider.geometry, "mass")
        assert not hasattr(provider.geometry, "batches")


def test_tiled_complete_operators_and_complex_bilinear_reciprocity(model):
    mesh, material = model.reference.moving, model.reference.material
    provider = BLOCKS["GalerkinBlockProvider"](mesh, material, quadrature=8)
    s = 230 + 24000j
    reference = PreparedGalerkinOperators(mesh, material, quadrature=8).assemble(s)
    full = {k: np.empty((mesh.dofs, mesh.dofs), complex) for k in FIELDS}
    parts = np.array_split(np.arange(mesh.dofs), 3)
    for rows in parts:
        for columns in parts:
            result = provider.request(rows, columns).evaluate(s)
            for name in FIELDS:
                full[name][np.ix_(rows, columns)] = getattr(result, name)
    for name in FIELDS:
        np.testing.assert_allclose(full[name], getattr(reference, name), rtol=2e-12, atol=1e-10)
    a = provider.request([0, 7, 11], [4, 15]).evaluate(s)
    b = provider.request([4, 15], [0, 7, 11]).evaluate(s)
    for name in ["single_layer", "hypersingular"]:
        np.testing.assert_allclose(getattr(a, name), getattr(b, name).T, rtol=2e-12, atol=1e-10)
    assert np.linalg.norm(a.single_layer - b.single_layer.conj().T) > 1e-7


def test_a_single_entry_never_constructs_full_matrices_or_prepared_geometry(model, monkeypatch):
    mesh, material = model.reference.moving, model.reference.material
    original_zeros = np.zeros

    def forbidden_init(*args, **kwargs):
        raise AssertionError("Full preparation was called")

    def bounded_zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and shape == (mesh.dofs, mesh.dofs):
            raise AssertionError("A complete matrix was allocated")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(PreparedGalerkinOperators, "__init__", forbidden_init)
    monkeypatch.setattr(np, "zeros", bounded_zeros)
    provider = BLOCKS["GalerkinBlockProvider"](mesh, material, quadrature=8)
    request = provider.request([3], [mesh.dofs - 1])
    result = request.evaluate(200 + 10000j)
    assert result.hypersingular.shape == (1, 1)
    assert request.statistics()["prepared_panel_pairs"] <= 4
    fiber = provider.request(np.arange(mesh.dofs), [3])
    assert fiber.statistics()["prepared_panel_pairs"] <= 2 * mesh.count
    assert fiber.statistics()["prepared_panel_pairs"] < fiber.statistics()["total_panel_pairs"]


def test_requests_are_reentrant_and_preserve_original_bindings(model):
    mesh = model.reference.moving
    original = PreparedGalerkinOperators.assemble
    provider = BLOCKS["GalerkinBlockProvider"](mesh, model.reference.material, quadrature=8)
    request = provider.request([0, 3, 5], [7, 9])
    with ThreadPoolExecutor(4) as pool:
        values = list(pool.map(request.evaluate, [200 + 10000j] * 4))
    for v in values[1:]:
        for name in FIELDS:
            np.testing.assert_array_equal(getattr(v, name), getattr(values[0], name))
    assert PreparedGalerkinOperators.assemble is original


@pytest.mark.parametrize(
    "rows, columns", [([0, 0], [1]), ([-1], [0]), ([0.5], [1]), ([], [0]), ([0], [100000])]
)
def test_invalid_dof_selections_are_rejected(model, rows, columns):
    provider = BLOCKS["GalerkinBlockProvider"](model.reference.moving, model.reference.material)
    with pytest.raises(ValueError):
        provider.request(rows, columns)


def test_pair_budget_rejects_before_geometry_preparation(model, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Geometry allocated before the budget check")

    monkeypatch.setattr(PreparedGalerkinOperators, "_prepare", forbidden)
    mesh = model.reference.moving
    provider = BLOCKS["GalerkinBlockProvider"](mesh, model.reference.material)
    with pytest.raises(ValueError, match="budget"):
        provider.request(np.arange(mesh.dofs), np.arange(mesh.dofs), max_pairs=1)


@pytest.mark.parametrize("s", [175 + 31400j, 3e6 + 0j])
def test_selected_blocks_preserve_both_body_condensation_and_full_boundary_response(s):
    from finiteblock_bem.galerkin import GalerkinOperators

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
    originals = [a.assemble(s) for a in transfer.prepared]
    selected = []
    for mesh in [model.reference.moving, model.reference.stationary]:
        provider = BLOCKS["GalerkinBlockProvider"](mesh, model.reference.material, quadrature=8)
        result = provider.request(np.arange(mesh.dofs), np.arange(mesh.dofs)).evaluate(s)
        selected.append(GalerkinOperators(*(getattr(result, name) for name in FIELDS)))
    original = transfer.evaluate_operators(originals, reconstruction=True)
    actual = transfer.evaluate_operators(selected, reconstruction=True)
    for a, b in zip(actual, original, strict=True):
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-9

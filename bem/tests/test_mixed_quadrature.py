import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.material import ElasticMaterial
from finiteblock_bem.mesh import polygon_boundary
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/mixed_quadrature_experiment.py")
)
PREPARED = TOOL["MixedQuadraturePrepared"]


def test_segment_distance_handles_crossing_parallel_and_unequal_segments():
    a = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], float)
    b = np.array([[2, 2], [2, 0], [1, 0], [1, 0]], float)
    c = np.array([[0, 2], [0, 1], [2, 0], [0.2, 0]], float)
    d = np.array([[2, 0], [10, 1], [10, 0], [0.3, 0]], float)
    np.testing.assert_allclose(TOOL["segment_distances"](a, b, c, d), [0, 1, 1, 0])


@pytest.mark.parametrize("far_order,separation", [(12, 2.0), (6, 1e6)])
def test_reference_order_limits_preserve_every_operator_bitwise(far_order, separation):
    mesh = polygon_boundary([[0, 0], [1.7, 0.1], [1.3, 1.1], [-0.1, 0.7]], ["free"] * 4, 0.4)
    material = ElasticMaterial(10, 0.25, 1)
    expected = PreparedGalerkinOperators(mesh, material, quadrature=12, batch_pairs=7)
    actual = PREPARED(mesh, material, far_order=far_order, separation=separation, batch_pairs=7)
    for s in [None, 0.01 + 0.02j, 1 + 2j, 100 + 300j]:
        a, b = expected.assemble(s), actual.assemble(s)
        for name in ["single_layer", "trace_double_layer", "hypersingular", "mixed_mass"]:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_mixed_rules_are_readonly_reciprocal_and_shared_across_four_workers():
    mesh = polygon_boundary([[0, 0], [1.7, 0.1], [1.3, 1.1], [-0.1, 0.7]], ["free"] * 4, 0.2)
    material = ElasticMaterial(10, 0.25, 1)
    expected = PreparedGalerkinOperators(mesh, material, quadrature=12)
    actual = PREPARED(mesh, material, near_order=12, far_order=8, separation=2)
    record = actual.statistics()
    assert record["near_pairs"] > 0 and record["far_pairs"] > 0
    assert record["quadrature_points"] < record["uniform_near_order_points"]
    for batch in actual.batches:
        for value in vars(batch).values():
            if isinstance(value, np.ndarray):
                assert not value.flags.writeable
    frequencies = [0.1 + 0.2j, 1 + 2j, 0.1 + 0.2j, 1 + 2j]
    answers = [actual.assemble(s).force_stiffness() for s in frequencies]
    with ThreadPoolExecutor(4) as pool:
        parallel = list(pool.map(lambda s: actual.assemble(s).force_stiffness(), frequencies))
    for s, answer, threaded in zip(frequencies, answers, parallel, strict=True):
        np.testing.assert_array_equal(answer, threaded)
        reference = expected.assemble(s).force_stiffness()
        assert np.linalg.norm(answer - reference) / np.linalg.norm(reference) < 1e-6
        np.testing.assert_allclose(answer, answer.T, rtol=1e-12, atol=1e-11)
        admittance = answer / s
        assert np.linalg.eigvalsh((admittance + admittance.conj().T) / 2).min() >= -1e-9
    assert actual.statistics() == record


@pytest.mark.parametrize("options", [{"near_order": 7}, {"far_order": 14}, {"separation": np.inf}])
def test_invalid_mixed_orders_or_separation_rejected(options):
    mesh = polygon_boundary([[0, 0], [1, 0], [1, 1], [0, 1]], ["free"] * 4, 0.5)
    with pytest.raises(ValueError):
        PREPARED(mesh, ElasticMaterial(10, 0.25, 1), **options)


def test_mixed_actual_finite_block_condensation_retains_the_same_trace():
    case = (
        Path(__file__).resolve().parents[1]
        / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    )
    model = load_nodal_model(case, max_panel_length=80)
    reference = NodalInterfaceTransfer(model, quadrature=12)
    actual = TOOL["mixed_transfer"](model, near_order=12, far_order=8, separation=2)
    for s in [100 + 2e4j, 1e6 + 3e5j]:
        a, b = reference.evaluate(s), actual.evaluate(s)
        assert a.shape == b.shape == (2 * len(model.moving_contact) + 1,) * 2
        assert np.linalg.norm(a - b) / np.linalg.norm(a) < 1e-6

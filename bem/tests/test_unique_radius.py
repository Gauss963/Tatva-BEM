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
    str(Path(__file__).resolve().parents[1] / "tools/unique_radius_experiment.py")
)
PREPARED = TOOL["UniqueRadiusPrepared"]


def geometry():
    return polygon_boundary([[0, 0], [2, 0], [2, 2], [0, 2]], ["free"] * 4, 0.5)


def test_exact_radius_reuse_preserves_each_weak_operator_and_static_limit():
    material, mesh = ElasticMaterial(10, 0.25, 1), geometry()
    reference = PreparedGalerkinOperators(mesh, material, quadrature=6, batch_pairs=7)
    variant = PREPARED(mesh, material, quadrature=6, batch_pairs=7, radial_chunk_size=13)
    bindings = dict(PreparedGalerkinOperators.assemble.__globals__)
    stats = variant.statistics()
    assert stats["exactly_unique_radii"] < stats["quadrature_radii"]
    assert not stats["radius_rounding_or_interpolation"]
    for s in [None, 0.001 + 0.002j, 0.3 + 2j, 100 + 300j]:
        expected, actual = reference.assemble(s), variant.assemble(s)
        for name in ["single_layer", "trace_double_layer", "hypersingular", "mixed_mass"]:
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name, value in bindings.items():
        assert PreparedGalerkinOperators.assemble.__globals__[name] is value
    for value in [variant.unique_radii, *variant.radius_maps]:
        assert not value.flags.writeable


def test_frequency_tables_are_thread_local_and_condensation_is_unchanged():
    case = (
        Path(__file__).resolve().parents[1]
        / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    )
    model = load_nodal_model(case, max_panel_length=100)
    reference = NodalInterfaceTransfer(model, quadrature=6)
    variant = TOOL["unique_radius_transfer"](model, quadrature=6, radial_chunk_size=113)
    frequencies = [100 + 2e4j, 1e6 + 3e5j, 100 + 2e4j, 1e6 + 3e5j]
    expected = [reference.evaluate(s) for s in frequencies]
    with ThreadPoolExecutor(4) as pool:
        actual = list(pool.map(variant.evaluate, frequencies))
    for a, b in zip(expected, actual, strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=1e-8)
    np.testing.assert_array_equal(actual[0], actual[2])
    np.testing.assert_array_equal(actual[1], actual[3])


@pytest.mark.parametrize(
    "options,error",
    [
        ({"radial_chunk_size": 0}, ValueError),
        ({"max_radial_workspace_mb": np.inf}, ValueError),
        ({"max_radial_workspace_mb": 1e-6}, MemoryError),
    ],
)
def test_extra_workspace_is_rejected_explicitly(options, error):
    with pytest.raises(error):
        PREPARED(geometry(), ElasticMaterial(10, 0.25, 1), quadrature=6, **options)

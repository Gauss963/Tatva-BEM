import ast
import hashlib
import json
import math
import runpy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = runpy.run_path(str(ROOT / "bem/tools/chamfer_mesh_refinement.py"))


def geometry_module():
    """Run the pinned reference's exact pure mesh functions without FEM/JAX setup."""
    names = {
        "_axis_coordinates",
        "_graded_segment",
        "_graded_axis_coordinates",
        "_structured_2d_boundary",
        "_structured_2d_grid",
        "create_structured_quad_block",
    }
    folder = ROOT / "bem/tests/fixtures"
    fixture = folder / "tatva_geometry_933ae1f.py.txt"
    manifest = json.loads((folder / "tatva_geometry_933ae1f.json").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    text = fixture.read_text()
    source = ast.parse(text)
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    for node in nodes:
        assert (
            hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest()
            == manifest["function_source_sha256"][node.name]
        )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    module = ModuleType("geometry_reference")
    module.__dict__.update(np=np, jnp=np, math=math, Mesh=SimpleNamespace)
    exec(compile(tree, "reference-geometry", "exec"), module.__dict__)
    exec(
        """
def build_block_model(spec, mesh_size, dtype, **kwargs):
    return create_structured_quad_block(spec, mesh_size, dtype, **kwargs)

def build_case_model(case, config):
    refinement = None if config.fault_end_refinement_size is None else dict(
        size=config.fault_end_refinement_size, growth=1.2, x_points=[1.0], y_points=[0.0,1.0])
    moving = build_block_model(case.moving, 0.2, np.float64, refinement=refinement,
        leading_chamfer_along_fault=config.moving_leading_chamfer_along_fault,
        leading_chamfer_perpendicular=config.moving_leading_chamfer_perpendicular)
    stationary = build_block_model(case.stationary, 0.2, np.float64, refinement=refinement)
    return moving,stationary
""",
        module.__dict__,
    )
    return module


def inputs(fine):
    case = SimpleNamespace(
        moving=SimpleNamespace(origin=(0.0, 0.0), dimensions=(1.0, 1.0), name="moving-block"),
        stationary=SimpleNamespace(
            origin=(1.0, 0.0), dimensions=(0.7, 1.1), name="stationary-block"
        ),
    )
    config = SimpleNamespace(
        fault_end_refinement_size=fine,
        dimension=2,
        element_type="quad4",
        moving_leading_chamfer_along_fault=0.2,
        moving_leading_chamfer_perpendicular=0.05,
    )
    return case, config


def test_null_refinement_preserves_the_original_mesh_bitwise():
    module = geometry_module()
    case, config = inputs(None)
    expected = module.build_case_model(case, config)
    actual = TOOL["make_chamfer_refined_builder"](module)(case, config)
    for (a, _, _), (b, _, _) in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(a.coords, b.coords)
        np.testing.assert_array_equal(a.elements, b.elements)


def test_refined_chamfer_preserves_polygon_and_positive_quad_jacobians():
    module = geometry_module()
    case, config = inputs(0.025)
    before = {
        k: getattr(module, k)
        for k in [
            "build_case_model",
            "build_block_model",
            "_structured_2d_grid",
            "create_structured_quad_block",
        ]
    }
    with pytest.raises(ValueError, match="not combined with a chamfer"):
        module.build_case_model(case, config)
    moving, stationary = TOOL["make_chamfer_refined_builder"](module)(case, config)
    mesh, nodes, _ = moving
    front = mesh.coords[nodes["moving-block-front"]]
    np.testing.assert_allclose(front[:, 0], 1, rtol=0, atol=1e-14)
    assert front[-1, 1] == pytest.approx(0.8)
    assert np.diff(front[:, 1])[-1] == pytest.approx(0.025)
    smesh, snodes, _ = stationary
    back = smesh.coords[snodes["stationary-block-back"]]
    np.testing.assert_array_equal(front[:, 1], back[back[:, 1] <= 0.8 + 1e-14, 1])
    above = mesh.coords[:, 1] > 0.8
    np.testing.assert_allclose(
        np.max(mesh.coords[above, 0]), 1 - 0.05 / 0.2 * (np.min(mesh.coords[above, 1]) - 0.8)
    )
    area = 0.0
    cells = mesh.coords[mesh.elements]
    for r in [-1 / np.sqrt(3), 1 / np.sqrt(3)]:
        for s in [-1 / np.sqrt(3), 1 / np.sqrt(3)]:
            gradient = (
                np.array([[-(1 - s), 1 - s, 1 + s, -(1 + s)], [-(1 - r), -(1 + r), 1 + r, 1 - r]])
                / 4
            )
            determinants = np.linalg.det(gradient @ cells)
            assert np.all(determinants > 0)
            area += np.sum(determinants)
    assert area == pytest.approx(1 - 0.2 * 0.05 / 2, rel=1e-12)
    assert all(getattr(module, k) is v for k, v in before.items())
    with pytest.raises(ValueError, match="not combined with a chamfer"):
        module.build_case_model(case, config)


def test_adapter_rejects_other_element_families():
    module = geometry_module()
    case, config = inputs(0.025)
    config.element_type = "tri3"
    with pytest.raises(ValueError, match="2D quad4"):
        TOOL["make_chamfer_refined_builder"](module)(case, config)


def test_transient_entry_rebinding_preserves_original_module_and_keyword_defaults():
    module = geometry_module()
    exec(
        "def run_simulation_dumped(case, config, *, marker='original'):\n"
        "    return build_case_model(case, config), marker\n",
        module.__dict__,
    )
    builder, original = module.build_case_model, module.run_simulation_dumped
    case, config = inputs(0.025)
    adapted = TOOL["make_chamfer_refined_builder"](module)
    runner = TOOL["rebind"](original, {"build_case_model": adapted})
    meshes, marker = runner(case, config)
    assert marker == "original"
    assert len(meshes[0][0].coords) > 0
    assert module.build_case_model is builder
    assert module.run_simulation_dumped is original
    with pytest.raises(ValueError, match="not combined with a chamfer"):
        original(case, config)

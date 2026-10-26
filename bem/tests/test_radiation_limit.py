import runpy
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

from finiteblock_bem.material import ElasticMaterial
from finiteblock_bem.mesh import polygon_boundary
from finiteblock_bem.nodal import load_nodal_model

EXPERIMENT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/radiation_limit_experiment.py")
)


def test_constant_boundary_velocity_has_exact_plane_wave_radiation_work():
    mesh = polygon_boundary([[0, 0], [2, 0], [2, 1], [0, 1]], ["a", "b", "c", "d"], 1)
    material = ElasticMaterial(12, 0.2, 2)
    limits = EXPERIMENT["body_limits"](mesh, material)
    velocity = np.array([0.3, -0.7])
    expected = 0.0
    for length, normal, tangent in zip(mesh.lengths, mesh.normals, mesh.tangents, strict=True):
        expected += (
            length
            * material.density
            * (
                material.cp * np.dot(normal, velocity) ** 2
                + material.cs * np.dot(tangent, velocity) ** 2
            )
        )
    nodal = np.tile(velocity, mesh.count)
    np.testing.assert_allclose(nodal @ limits["mixed"] @ nodal, expected, rtol=1e-14)
    assert np.linalg.eigvalsh(limits["mixed"])[0] > 0
    # A spatially alternating P1 trace has zero P0 projection, retaining half
    # of the consistent radiation mass. It must not be replaced by lumping.
    alternating = np.repeat((-1.0) ** np.arange(mesh.count), 2)
    np.testing.assert_allclose(
        alternating @ limits["mixed"] @ alternating,
        0.5 * alternating @ limits["consistent"] @ alternating,
        rtol=1e-14,
    )


def test_condensed_radiation_preserves_constrained_minimum_energy():
    case = (
        Path(__file__).resolve().parents[1]
        / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    )
    model = load_nodal_model(case, max_panel_length=150, contact_panel_length=120)
    blocks = [
        EXPERIMENT["body_limits"](m, model.reference.material)["mixed"]
        for m in [model.reference.moving, model.reference.stationary]
    ]
    z = EXPERIMENT["condense"](model, blocks)
    p = model.constraints.prolongation
    b = np.vstack(
        (model.constraints.normal_jump, model.constraints.tangential_jump, np.eye(p.shape[1])[-1])
    )
    k = p.T @ block_diag(*blocks) @ p
    q = np.random.default_rng(46).normal(size=len(z))
    u = np.linalg.solve(k, b.T @ z @ q)
    np.testing.assert_allclose(b @ u, q, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(u @ k @ u, q @ z @ q, rtol=1e-13)

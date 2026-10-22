"""Direct interior boundary integral equation with P0 straight panels.

    H(s) u = V(s) t,    H = 1/2 I + K.

Single-layer self panels are integrated by analytic static-log subtraction.
The principal-value double-layer self integral is zero at a straight panel's
midpoint. The geometric jump 1/2 I is added explicitly, never estimated by
row sums from a dynamic matrix.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve

from .kernels import green_and_gradient, source_traction, static_green_and_gradient
from .material import ElasticMaterial
from .mesh import BoundaryMesh


@dataclass(frozen=True)
class BoundaryOperators:
    single_layer: np.ndarray
    trace_double_layer: np.ndarray

    def impedance(self):
        """Physical boundary traction from boundary displacement: V^-1 H."""
        return solve(self.single_layer, self.trace_double_layer, check_finite=False)


def _analytic_static_self(mesh, material, reference_length):
    length = mesh.lengths
    factor = 1 / (8 * np.pi * material.mu * (1 - material.poisson))
    kappa = 3 - 4 * material.poisson
    integral_log = length * (np.log(length / (2 * reference_length)) - 1)
    return factor * (
        -kappa * integral_log[:, None, None] * np.eye(2)
        + np.einsum("n,ni,nj->nij", length, mesh.tangents, mesh.tangents)
    )


def assemble_operators(mesh: BoundaryMesh, material: ElasticMaterial, s=None, *, quadrature=20):
    if quadrature < 4 or quadrature % 2:
        raise ValueError("Use an even Gauss order of at least 4")
    abscissa, weights = leggauss(quadrature)
    count = mesh.count
    dtype = float if s is None else complex
    vblocks = np.zeros((count, count, 2, 2), dtype=dtype)
    kblocks = np.zeros_like(vblocks)
    reference_length = max(np.ptp(mesh.centers, axis=0).max(), 1.0)
    for first in range(0, count, 16):
        ids = np.arange(first, min(first + 16, count))
        source = (
            mesh.centers[ids, None, :]
            + abscissa[None, :, None] * (mesh.ends[ids] - mesh.starts[ids])[:, None, :] / 2
        )
        vector = source[None, ...] - mesh.centers[:, None, None, :]
        if s is None:
            green, gradient = static_green_and_gradient(vector, material, reference_length)
        else:
            green, gradient = green_and_gradient(vector, s, material)
        double = source_traction(gradient, mesh.normals[ids][None, :, None, :], material)
        measure = mesh.lengths[ids, None] * weights[None, :] / 2
        vblocks[:, ids] = np.einsum("abqij,bq->abij", green, measure)
        kblocks[:, ids] = np.einsum("abqij,bq->abij", double, measure)
    static_self = _analytic_static_self(mesh, material, reference_length)
    for i in range(count):
        # Same quadrature nodes for the dynamic-minus-static remainder.
        vector = abscissa[:, None] * (mesh.ends[i] - mesh.starts[i])[None, :] / 2
        if s is None:
            vblocks[i, i] = static_self[i]
        else:
            dynamic, _ = green_and_gradient(vector, s, material)
            static, _ = static_green_and_gradient(vector, material, reference_length)
            vblocks[i, i] = static_self[i] + np.einsum(
                "qij,q->ij", dynamic - static, weights * mesh.lengths[i] / 2
            )
        kblocks[i, i] = 0
    v = vblocks.transpose(0, 2, 1, 3).reshape(mesh.dofs, mesh.dofs)
    h = kblocks.transpose(0, 2, 1, 3).reshape(mesh.dofs, mesh.dofs)
    h += 0.5 * np.eye(mesh.dofs)
    return BoundaryOperators(v, h)

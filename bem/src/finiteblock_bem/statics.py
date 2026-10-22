"""Stream static operators without preparing any frequency-reuse geometry.

The dynamic assembler deliberately caches O(N² Q) angular geometry. Static
initialization only needs one matrix: preparing that cache wastes time and
memory on refined meshes. This path uses the same kernels/self subtraction
and bounds quadrature work to a source batch.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

from .kernels import source_traction, static_green_and_gradient
from .operators import BoundaryOperators, _analytic_static_self


def assemble_static_operators(mesh, material, *, quadrature=20):
    if quadrature < 4 or quadrature % 2:
        raise ValueError("Use an even Gauss order of at least 4")
    abscissa, weights = leggauss(quadrature)
    centers, lengths, normals = mesh.centers, mesh.lengths, mesh.normals
    reference = max(np.ptp(centers, axis=0).max(), 1.0)
    v = np.zeros((mesh.count, mesh.count, 2, 2))
    k = np.zeros_like(v)
    for first in range(0, mesh.count, 16):
        ids = np.arange(first, min(first + 16, mesh.count))
        source = (
            centers[ids, None, :]
            + abscissa[None, :, None] * (mesh.ends[ids] - mesh.starts[ids])[:, None, :] / 2
        )
        vector = source[None, ...] - centers[:, None, None, :]
        green, gradient = static_green_and_gradient(vector, material, reference)
        measure = lengths[ids, None] * weights[None, :] / 2
        v[:, ids] = np.einsum("abqij,bq->abij", green, measure)
        k[:, ids] = np.einsum(
            "abqij,bq->abij",
            source_traction(gradient, normals[ids][None, :, None, :], material),
            measure,
        )
    # Apply exactly the same adaptive near correction for nonuniform panels.
    from .operators import _near_quadrature

    vector, normal, measure, pairs, offsets = _near_quadrature(mesh, abscissa, weights)
    if len(pairs):
        green, gradient = static_green_and_gradient(vector, material, reference)
        v[pairs[:, 0], pairs[:, 1]] = np.add.reduceat(green * measure[:, None, None], offsets)
        k[pairs[:, 0], pairs[:, 1]] = np.add.reduceat(
            source_traction(gradient, normal, material) * measure[:, None, None], offsets
        )
    diagonal = np.arange(mesh.count)
    v[diagonal, diagonal] = _analytic_static_self(mesh, material, reference)
    k[diagonal, diagonal] = 0.0
    v = v.transpose(0, 2, 1, 3).reshape(mesh.dofs, mesh.dofs)
    k = k.transpose(0, 2, 1, 3).reshape(mesh.dofs, mesh.dofs) + 0.5 * np.eye(mesh.dofs)
    return BoundaryOperators(v, k)

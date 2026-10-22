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

from .kernels import _radial, source_traction, static_green_and_gradient
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


def _angular_geometry(vector, normal):
    r = np.linalg.norm(vector, axis=-1)
    e = vector / r[..., None]
    ee = e[..., :, None] * e[..., None, :]
    en = np.sum(e * normal, axis=-1)
    return (
        r,
        ee,
        en[..., None, None] * np.eye(2) + normal[..., :, None] * e[..., None, :],
        en[..., None, None] * ee,
        e[..., :, None] * normal[..., None, :],
    )


def _green_and_traction(radial, indices, angular, material):
    a, b, da, db = (value[indices] for value in radial)
    r, ee, t1, t2, t3 = angular
    green = a[..., None, None] * np.eye(2) + b[..., None, None] * ee
    c1 = material.mu * (da + b / r)
    c2 = material.mu * (2 * db - 4 * b / r)
    c3 = 2 * material.mu * b / r + material.lam * (da + db + b / r)
    return green, c1[..., None, None] * t1 + c2[..., None, None] * t2 + c3[..., None, None] * t3


def _near_quadrature(mesh, abscissa, weights):
    """Subdivide geometrically near short targets beside long source panels."""
    centers, lengths = mesh.centers, mesh.lengths
    vectors, normals, measures, pairs, offsets = [], [], [], [], []
    total = 0
    for j, (a, b) in enumerate(zip(mesh.starts, mesh.ends, strict=True)):
        edge = b - a
        projection = np.clip((centers - a) @ edge / lengths[j] ** 2, 0, 1)
        distance = np.linalg.norm(centers - a - projection[:, None] * edge, axis=1)
        near = np.flatnonzero(
            (distance < 0.5 * lengths[j] * (1 - 1e-12)) & (np.arange(mesh.count) != j)
        )
        for i in near:
            stack, intervals = [(0.0, 1.0, 0)], []
            while stack:
                left, right, depth = stack.pop()
                closest = a + np.clip(projection[i], left, right) * edge
                if (right - left) * lengths[j] <= 2 * np.linalg.norm(closest - centers[i]):
                    intervals.append((left, right))
                elif depth >= 20:
                    raise ValueError("Near-panel quadrature exceeded its subdivision limit")
                else:
                    middle = (left + right) / 2
                    stack.extend([(left, middle, depth + 1), (middle, right, depth + 1)])
            offsets.append(total)
            pairs.append((i, j))
            for left, right in intervals:
                fraction = left + (abscissa + 1) * (right - left) / 2
                vectors.extend(a + fraction[:, None] * edge - centers[i])
                normals.extend(np.repeat(mesh.normals[j][None, :], len(weights), axis=0))
                measures.extend(weights * (right - left) * lengths[j] / 2)
                total += len(weights)
    return (
        np.asarray(vectors).reshape(-1, 2),
        np.asarray(normals).reshape(-1, 2),
        np.asarray(measures),
        np.asarray(pairs, dtype=int).reshape(-1, 2),
        np.asarray(offsets, int),
    )


class PreparedBoundaryOperators:
    """Reuse quadrature geometry and batch every self panel in one kernel call.

    Preparation contains no frequency dependence. Instances may be evaluated
    concurrently because all frequency-specific work arrays are local.
    """

    def __init__(self, mesh: BoundaryMesh, material: ElasticMaterial, *, quadrature=20):
        if quadrature < 4 or quadrature % 2:
            raise ValueError("Use an even Gauss order of at least 4")
        self.mesh, self.material = mesh, material
        abscissa, weights = leggauss(quadrature)
        centers, normals, lengths = mesh.centers, mesh.normals, mesh.lengths
        self.reference_length = max(np.ptp(centers, axis=0).max(), 1.0)
        self.batches = []
        for first in range(0, mesh.count, 16):
            ids = np.arange(first, min(first + 16, mesh.count))
            source = (
                centers[ids, None, :]
                + abscissa[None, :, None] * (mesh.ends[ids] - mesh.starts[ids])[:, None, :] / 2
            )
            self.batches.append(
                (
                    ids,
                    source[None, ...] - centers[:, None, None, :],
                    normals[ids][None, :, None, :],
                    lengths[ids, None] * weights[None, :] / 2,
                )
            )
        self.self_vectors = abscissa[None, :, None] * (mesh.ends - mesh.starts)[:, None, :] / 2
        self.self_measure = lengths[:, None] * weights[None, :] / 2
        self.static_self = _analytic_static_self(mesh, material, self.reference_length)
        self.static_self_integrand, _ = static_green_and_gradient(
            self.self_vectors, material, self.reference_length
        )
        self.near = _near_quadrature(mesh, abscissa, weights)
        # Isotropy separates the costly Bessel functions from angular geometry.
        # Deduplicate EXACT floating-point radii, without rounding distances.
        radii = [np.linalg.norm(b[1], axis=-1) for b in self.batches]
        self_r = np.linalg.norm(self.self_vectors, axis=-1)
        radii.extend([self_r, np.linalg.norm(self.near[0], axis=-1)])
        self.radii, inverse = np.unique(
            np.concatenate([r.ravel() for r in radii]), return_inverse=True
        )
        self.radial_indices = []
        self.angular = []
        first = 0
        for batch, r in zip(self.batches, radii[: len(self.batches)], strict=True):
            vector, normal = batch[1], batch[2]
            self.angular.append(_angular_geometry(vector, normal))
            self.radial_indices.append(inverse[first : first + r.size].reshape(r.shape))
            first += r.size
        self.self_indices = inverse[first : first + self_r.size].reshape(self_r.shape)
        self.near_indices = inverse[first + self_r.size :]
        self.near_angular = _angular_geometry(self.near[0], self.near[1])
        e = self.self_vectors / self_r[..., None]
        self.self_ee = e[..., :, None] * e[..., None, :]

    def evaluate(self, s=None):
        mesh, material = self.mesh, self.material
        dtype = float if s is None else complex
        vblocks = np.zeros((mesh.count, mesh.count, 2, 2), dtype=dtype)
        kblocks = np.zeros_like(vblocks)
        radial = None
        if s is not None:
            if not np.isfinite(s) or np.real(s) <= 0:
                raise ValueError("Laplace parameter must have positive real part")
            radial = _radial(self.radii, s, material)
        for batch, indices, angular in zip(
            self.batches, self.radial_indices, self.angular, strict=True
        ):
            ids, vector, normals, measure = batch
            if s is None:
                green, gradient = static_green_and_gradient(vector, material, self.reference_length)
                double = source_traction(gradient, normals, material)
            else:
                # Direct source traction of a*I+b*e*e, without a third-order gradient tensor.
                green, double = _green_and_traction(radial, indices, angular, material)
            vblocks[:, ids] = np.einsum("abqij,bq->abij", green, measure)
            kblocks[:, ids] = np.einsum("abqij,bq->abij", double, measure)
        vector, normals, measure, pairs, offsets = self.near
        if len(pairs):
            if s is None:
                green, gradient = static_green_and_gradient(vector, material, self.reference_length)
                double = source_traction(gradient, normals, material)
            else:
                green, double = _green_and_traction(
                    radial, self.near_indices, self.near_angular, material
                )
            vblocks[pairs[:, 0], pairs[:, 1]] = np.add.reduceat(
                green * measure[:, None, None], offsets
            )
            kblocks[pairs[:, 0], pairs[:, 1]] = np.add.reduceat(
                double * measure[:, None, None], offsets
            )
        diagonal = np.arange(mesh.count)
        vblocks[diagonal, diagonal] = self.static_self
        if s is not None:
            a, b = (radial[i][self.self_indices] for i in (0, 1))
            dynamic = a[..., None, None] * np.eye(2) + b[..., None, None] * self.self_ee
            vblocks[diagonal, diagonal] += np.einsum(
                "nqij,nq->nij", dynamic - self.static_self_integrand, self.self_measure
            )
        kblocks[diagonal, diagonal] = 0
        v = vblocks.transpose(0, 2, 1, 3).reshape(mesh.dofs, mesh.dofs)
        h = kblocks.transpose(0, 2, 1, 3).reshape(mesh.dofs, mesh.dofs)
        h += 0.5 * np.eye(mesh.dofs)
        return BoundaryOperators(v, h)


def assemble_operators(mesh: BoundaryMesh, material: ElasticMaterial, s=None, *, quadrature=20):
    return PreparedBoundaryOperators(mesh, material, quadrature=quadrature).evaluate(s)

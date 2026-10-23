"""Prepared, batched evaluation of the reference symmetric Galerkin integrals.

Geometry is shared read-only by frequency workers. Each worker streams bounded
pair batches. The quadrature, singularity subtraction and bilinear reciprocity
are identical to galerkin.py; no approximation is introduced here.
"""

from dataclasses import dataclass

import numpy as np

from .galerkin import ROTATION, GalerkinOperators, _basis, _pair_rule
from .kernels import _radial, _tensor_from_radial, source_traction, static_green_and_gradient


def prepared_geometry_bytes(mesh, quadrature):
    """Conservative bound including cached geometry, indices and construction copies."""
    n = mesh.count
    pairs = n * (n + 1) // 2
    points = (pairs + n) * quadrature**2
    return points * 96 + pairs * 4096


@dataclass(frozen=True)
class _PairBatch:
    i: np.ndarray
    j: np.ndarray
    vector: np.ndarray
    measure: np.ndarray
    px: np.ndarray
    py: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    angular: np.ndarray
    log: np.ndarray
    integral: np.ndarray
    log_moment: np.ndarray
    vindices: np.ndarray
    hindices: np.ndarray
    reverse_hindices: np.ndarray
    dindices: np.ndarray
    self_pair: bool


class PreparedGalerkinOperators:
    """Reuse a closed loop's P1/P0 quadrature geometry over many frequencies."""

    def __init__(self, mesh, material, *, quadrature=12, batch_pairs=32):
        if not isinstance(quadrature, int) or quadrature < 4 or quadrature % 2:
            raise ValueError("Use an even Gauss order of at least 4")
        if not isinstance(batch_pairs, int) or batch_pairs < 1:
            raise ValueError("batch_pairs must be a positive integer")
        self.mesh, self.material = mesh, material
        self.reference = max(np.ptp(mesh.starts, axis=0).max(), 1.0)
        n = mesh.count
        self.mass = np.zeros((mesh.dofs, mesh.dofs))
        for i in range(n):
            for endpoint in [i, (i + 1) % n]:
                self.mass[2 * i : 2 * i + 2, 2 * endpoint : 2 * endpoint + 2] += (
                    mesh.lengths[i] / 2 * np.eye(2)
                )
        groups = [[], [], []]  # self, adjacent, disjoint: fixed quadrature sizes
        for i in range(n):
            for j in range(i, n):
                kind = 0 if i == j else (1 if j == i + 1 or (i == 0 and j == n - 1) else 2)
                groups[kind].append((i, j))
        self.batches = []
        for pairs in groups:
            for first in range(0, len(pairs), batch_pairs):
                self.batches.append(self._prepare(pairs[first : first + batch_pairs], quadrature))
        # Prevent accidental writes when this instance is shared by workers.
        self.mass.flags.writeable = False
        for batch in self.batches:
            for value in vars(batch).values():
                if isinstance(value, np.ndarray):
                    value.flags.writeable = False

    def _prepare(self, pairs, order):
        mesh = self.mesh
        i, j = np.array(pairs).T
        rules = [_pair_rule(mesh, int(a), int(b), order) for a, b in pairs]
        x, y, measure = [np.stack([r[k] for r in rules]) for k in range(3)]
        px, py = _basis(x.ravel()).reshape(*x.shape, 2), _basis(y.ravel()).reshape(*y.shape, 2)
        vector = mesh.starts[j, None] + y[..., None] * (mesh.ends[j] - mesh.starts[j])[:, None]
        vector -= mesh.starts[i, None] + x[..., None] * (mesh.ends[i] - mesh.starts[i])[:, None]
        dx, dy = (
            np.array([-1.0, 1.0]) / mesh.lengths[i, None],
            np.array([-1.0, 1.0]) / mesh.lengths[j, None],
        )
        nx, ny, ty = mesh.normals[i], mesh.normals[j], mesh.tangents[j]
        angular = -np.sum(nx * ny, axis=1)[:, None, None] * np.eye(2)
        angular += np.sum(nx * ty, axis=1)[:, None, None] * ROTATION
        self_pair = bool(i[0] == j[0])
        log = np.zeros_like(x)
        integral = np.zeros(len(pairs))
        moments = np.zeros((len(pairs), 2, 2))
        if self_pair:
            log = np.log(np.linalg.norm(vector, axis=2) / self.reference)
            length = mesh.lengths[i]
            integral = length**2 * (np.log(length / self.reference) - 1.5)
            moments = length[:, None, None] ** 2 * (
                np.log(length / self.reference)[:, None, None] / 4
                + np.array([[-7.0, -5.0], [-5.0, -7.0]]) / 16
            )
        elif rules[0][3] is not None:
            log = np.stack([r[3] for r in rules])
            integral = -mesh.lengths[i] * mesh.lengths[j] / 2
            moments = np.stack([r[4] for r in rules])
        row, col = 2 * i[:, None] + np.arange(2), 2 * j[:, None] + np.arange(2)
        idofs = (2 * np.column_stack((i, (i + 1) % mesh.count))[:, :, None] + np.arange(2)).reshape(
            -1, 4
        )
        jdofs = (2 * np.column_stack((j, (j + 1) % mesh.count))[:, :, None] + np.arange(2)).reshape(
            -1, 4
        )

        def indices(a, b):
            return a[:, :, None] * mesh.dofs + b[:, None, :]

        return _PairBatch(
            i,
            j,
            vector,
            measure,
            px,
            py,
            dx,
            dy,
            nx,
            ny,
            angular,
            log,
            integral,
            moments,
            indices(row, col),
            indices(row, jdofs),
            indices(col, idofs),
            indices(idofs, jdofs),
            self_pair,
        )

    def assemble(self, s=None):
        if s is not None and (not np.isfinite(s) or np.real(s) <= 0):
            raise ValueError("Laplace parameter must have positive real part")
        material, n = self.material, self.mesh.dofs
        mu, lam = material.mu, material.lam
        log_green = -(lam + 3 * mu) / (4 * np.pi * mu * (lam + 2 * mu))
        log_derivative = 4 * mu**2 * (log_green + 1 / (2 * np.pi * (lam + 2 * mu)))
        effective_s = 0.0 if s is None else complex(s)
        vmat = np.zeros((n, n), dtype=float if s is None else complex)
        hmat, dmat = vmat.copy(), vmat.copy()
        for b in self.batches:
            p, q = b.measure.shape
            vector = b.vector.reshape(-1, 2)
            r = np.linalg.norm(vector, axis=1)
            if s is None:
                green, gradient = static_green_and_gradient(vector, material, self.reference)
                gs = gp = -np.log(r / self.reference) / (2 * np.pi)
                difference, gradient_y = np.zeros_like(r), np.zeros_like(vector)
            else:
                a, radial_b, da, db, gs, gp, dr = _radial(r, s, material, scalars=True)
                green, gradient = _tensor_from_radial(vector, r, a, radial_b, da, db)
                difference, gradient_y = gs - gp, vector * (dr / r)[:, None]
            green = green.reshape(p, q, 2, 2)
            gradient = gradient.reshape(p, q, 2, 2, 2)
            double = source_traction(gradient, b.ny[:, None], material)
            reverse = source_traction(-gradient, b.nx[:, None], material)
            gs, gp, difference = [a.reshape(p, q) for a in (gs, gp, difference)]
            gradient_y = gradient_y.reshape(p, q, 2)
            green = green - log_green * b.log[..., None, None] * np.eye(2)
            gs, gp = gs + b.log / (2 * np.pi), gp + b.log / (2 * np.pi)
            mass = (
                -material.density
                * effective_s**2
                * (
                    difference[..., None, None] * (b.nx[:, :, None] * b.ny[:, None, :])[:, None]
                    + gs[..., None, None] * b.angular[:, None]
                )
            )
            deriv = 4 * mu**2 * (green - gp[..., None, None] * np.eye(2) / (lam + 2 * mu))
            left = -2 * mu * b.nx[:, None, :, None] * (gradient_y @ ROTATION)[:, :, None, :]
            right = -2 * mu * (gradient_y @ ROTATION.T)[:, :, :, None] * b.ny[:, None, None, :]
            local_d = np.einsum("pq,pqab,pqi,pqj->piajb", b.measure, mass, b.px, b.py)
            local_d += np.einsum("pq,pqab,pi,pj->piajb", b.measure, deriv, b.dx, b.dy)
            local_d += np.einsum("pq,pqab,pqi,pj->piajb", b.measure, left, b.px, b.dy)
            local_d += np.einsum("pq,pqab,pi,pqj->piajb", b.measure, right, b.dx, b.py)
            local_v = np.einsum("pq,pqab->pab", b.measure, green)
            if b.self_pair:
                local_v *= 2
                local_d += local_d.transpose(0, 3, 4, 1, 2)
            local_v += log_green * b.integral[:, None, None] * np.eye(2)
            local_d += np.einsum(
                "p,pi,pj,ab->piajb", log_derivative * b.integral, b.dx, b.dy, np.eye(2)
            )
            angular = -np.broadcast_to(np.eye(2), b.angular.shape) if b.self_pair else b.angular
            local_d += np.einsum(
                "pij,pab->piajb",
                material.density * effective_s**2 / (2 * np.pi) * b.log_moment,
                angular,
            )
            local_k = np.einsum(
                "pq,pqab,pqi->paib", b.measure, double, b.py - b.px if b.self_pair else b.py
            ).reshape(p, 2, 4)
            np.add.at(vmat.ravel(), b.vindices, local_v)
            np.add.at(hmat.ravel(), b.hindices, local_k)
            np.add.at(dmat.ravel(), b.dindices, local_d.reshape(p, 4, 4))
            if not b.self_pair:
                # Transposed global matrix indices, not a post-hoc symmetrization.
                np.add.at(vmat.ravel(), (b.vindices % n) * n + b.vindices // n, local_v)
                np.add.at(
                    dmat.ravel(), (b.dindices % n) * n + b.dindices // n, local_d.reshape(p, 4, 4)
                )
                local_reverse = np.einsum("pq,pqab,pqi->paib", b.measure, reverse, b.px).reshape(
                    p, 2, 4
                )
                np.add.at(hmat.ravel(), b.reverse_hindices, local_reverse)
        hmat += 0.5 * self.mass
        return GalerkinOperators(vmat, hmat, dmat, self.mass)

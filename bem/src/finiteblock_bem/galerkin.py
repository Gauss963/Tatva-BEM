"""Reference P1-displacement/P0-density symmetric Galerkin boundary operators.

The returned stiffness maps continuous nodal displacement to work-conjugate
NODAL FORCE, not midpoint traction. V is P0/P0, H=(M/2+K) is P0/P1, and D is
the positive hypersingular form -T_x D_y. S=H.T V^-1 H+D; the transpose is
bilinear (not conjugate) at a complex Laplace frequency.

D uses the weak regularization of Bao/Xu/Yin, arXiv:1608.08759 eq. (3.6),
analytically continued with omega=i*s. This sign equals minus the W convention
of Aimi/Di Credico/Gimperlein, arXiv:2405.08566 eq. (50). No eigenvalue clipping
or post-hoc symmetrization is applied. This first assembler prioritizes
auditable quadrature; frequency/spatial acceleration is a separate step.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve
from scipy.special import kv

from .kernels import (
    _difference_derivatives_series,
    green_and_gradient,
    source_traction,
    static_green_and_gradient,
)

ROTATION = np.array([[0.0, -1.0], [1.0, 0.0]])


@dataclass(frozen=True)
class GalerkinOperators:
    single_layer: np.ndarray
    trace_double_layer: np.ndarray
    hypersingular: np.ndarray
    mixed_mass: np.ndarray

    def force_stiffness(self):
        h = self.trace_double_layer
        return h.T @ solve(self.single_layer, h, check_finite=False) + self.hypersingular


def _basis(fraction):
    return np.column_stack((1 - fraction, fraction))


def _pair_rule(mesh, i, j, order):
    """Quadrature on a panel product, with self/vertex Duffy transforms.

    Returns fractions, physical measures and analytic log-correction moments.
    A self pair covers only u<v: its mirror is added algebraically later.
    For adjacent pairs both triangles are included. Their log(t) singularity
    is removed and integrated exactly against the degree <=2 shape products.
    """
    z, w = leggauss(order)
    z, w = (z + 1) / 2, w / 2
    u, v = np.meshgrid(z, z, indexing="ij")
    ww = w[:, None] * w[None, :]
    li, lj = mesh.lengths[[i, j]]
    if i == j:
        x, y = u.ravel(), (u + (1 - u) * v).ravel()
        measure = (ww * (1 - u) * li * lj).ravel()
        return x, y, measure, None, None
    n = mesh.count
    adjacent = (i + 1) % n == j or (j + 1) % n == i
    if not adjacent:
        return u.ravel(), v.ravel(), (ww * li * lj).ravel(), None, None
    at_end_x = (i + 1) % n == j
    at_end_y = not at_end_x
    xs, ys, measures, logs = [], [], [], []
    log_moment = np.zeros((2, 2))
    for cx, cy in [(np.ones_like(z), z), (z, np.ones_like(z))]:
        distance_x, distance_y = u * cx[None, :], u * cy[None, :]
        x = 1 - distance_x if at_end_x else distance_x
        y = 1 - distance_y if at_end_y else distance_y
        xs.append(x.ravel())
        ys.append(y.ravel())
        measures.append((ww * u * li * lj).ravel())
        logs.append(np.log(u).ravel())
        ax = np.array([0.0, 1.0]) if at_end_x else np.array([1.0, 0.0])
        ay = np.array([0.0, 1.0]) if at_end_y else np.array([1.0, 0.0])
        bx = cx[:, None] * (np.array([1.0, -1.0]) if at_end_x else [-1.0, 1.0])
        by = cy[:, None] * (np.array([1.0, -1.0]) if at_end_y else [-1.0, 1.0])
        moment = -np.outer(ax, ay)[None, :, :] / 4
        moment = (
            moment - (ax[None, :, None] * by[:, None, :] + bx[:, :, None] * ay[None, None, :]) / 9
        )
        moment = moment - bx[:, :, None] * by[:, None, :] / 16
        log_moment += li * lj * np.einsum("q,qab->ab", w, moment)
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(measures),
        np.concatenate(logs),
        log_moment,
    )


def _scalar_kernels(vector, s, material):
    r = np.linalg.norm(vector, axis=1)
    ks, kp = complex(s) / material.cs, complex(s) / material.cp
    gs, gp = kv(0, ks * r) / (2 * np.pi), kv(0, kp * r) / (2 * np.pi)
    derivative = (-ks * kv(1, ks * r) + kp * kv(1, kp * r)) / (2 * np.pi)
    small = np.maximum(abs(ks * r), abs(kp * r)) < 0.2
    if np.any(small):
        derivative[small] = _difference_derivatives_series(r[small], ks, kp)[0] / (2 * np.pi)
    return gs, gp, gs - gp, vector * (derivative / r)[:, None]


def _weak_kernels(green, gs, gp, difference, gradient_y, nx, ny, ty, s, material):
    """Mass, two-derivative and cross terms of the regularized positive D."""
    mu = material.mu
    eye = np.eye(2)
    angular = -np.dot(nx, ny) * eye + ROTATION * np.dot(nx, ty)
    mass = (
        -material.density
        * s
        * s
        * (difference[:, None, None] * np.outer(nx, ny) + gs[:, None, None] * angular)
    )
    derivative = 4 * mu**2 * (green - gp[:, None, None] * eye / (material.lam + 2 * mu))
    left = -2 * mu * nx[None, :, None] * (gradient_y @ ROTATION)[:, None, :]
    right = -2 * mu * (gradient_y @ ROTATION.T)[:, :, None] * ny[None, None, :]
    return mass, derivative, left, right


def _local_weak(mass, derivative, left, right, px, py, dx, dy, measure):
    result = np.einsum("q,qab,qi,qj->iajb", measure, mass, px, py)
    result += np.einsum("q,qab,i,j->iajb", measure, derivative, dx, dy)
    result += np.einsum("q,qab,qi,j->iajb", measure, left, px, dy)
    result += np.einsum("q,qab,i,qj->iajb", measure, right, dx, py)
    return result


def assemble_galerkin_operators(mesh, material, s=None, *, quadrature=12):
    """Assemble a reference symmetric Galerkin force operator on a closed loop."""
    if not isinstance(quadrature, int) or quadrature < 4 or quadrature % 2:
        raise ValueError("Use an even Gauss order of at least 4")
    if s is not None and (not np.isfinite(s) or np.real(s) <= 0):
        raise ValueError("Laplace parameter must have positive real part")
    n = mesh.count
    dtype = float if s is None else complex
    vmat = np.zeros((2 * n, 2 * n), dtype=dtype)
    hmat = np.zeros_like(vmat)
    dmat = np.zeros_like(vmat)
    mass_matrix = np.zeros_like(vmat)
    lengths, normals, tangents = mesh.lengths, mesh.normals, mesh.tangents
    reference = max(np.ptp(mesh.starts, axis=0).max(), 1.0)
    # Coefficient of log(r/reference)*I in the plane-strain Green tensor.
    log_green = -(material.lam + 3 * material.mu) / (
        4 * np.pi * material.mu * (material.lam + 2 * material.mu)
    )
    log_derivative = (
        4 * material.mu**2 * (log_green + 1 / (2 * np.pi * (material.lam + 2 * material.mu)))
    )
    for i in range(n):
        row = np.arange(2 * i, 2 * i + 2)
        inode = [i, (i + 1) % n]
        idofs = (2 * np.array(inode)[:, None] + np.arange(2)).ravel()
        for endpoint in inode:
            mass_matrix[np.ix_(row, np.arange(2 * endpoint, 2 * endpoint + 2))] += (
                lengths[i] / 2 * np.eye(2)
            )
        for j in range(i, n):
            column = np.arange(2 * j, 2 * j + 2)
            jnode = [j, (j + 1) % n]
            jdofs = (2 * np.array(jnode)[:, None] + np.arange(2)).ravel()
            x, y, measure, log_t, log_moment = _pair_rule(mesh, i, j, quadrature)
            vector = mesh.starts[j] + y[:, None] * (mesh.ends[j] - mesh.starts[j])
            vector -= mesh.starts[i] + x[:, None] * (mesh.ends[i] - mesh.starts[i])
            px, py = _basis(x), _basis(y)
            dx, dy = np.array([-1.0, 1.0]) / lengths[i], np.array([-1.0, 1.0]) / lengths[j]
            if s is None:
                green, gradient = static_green_and_gradient(vector, material, reference)
                gs = gp = -np.log(np.linalg.norm(vector, axis=1) / reference) / (2 * np.pi)
                difference = np.zeros(len(vector))
                gradient_y = np.zeros_like(vector)
                effective_s = 0.0
            else:
                green, gradient = green_and_gradient(vector, s, material)
                gs, gp, difference, gradient_y = _scalar_kernels(vector, s, material)
                effective_s = complex(s)
            double = source_traction(gradient, normals[j], material)
            reverse_double = None if i == j else source_traction(-gradient, normals[i], material)
            correction_v = np.zeros((2, 2), dtype=dtype)
            correction_d = np.zeros((2, 2, 2, 2), dtype=dtype)
            if i == j:
                log_r = np.log(np.linalg.norm(vector, axis=1) / reference)
                green = green - log_green * log_r[:, None, None] * np.eye(2)
                gs, gp = gs + log_r / (2 * np.pi), gp + log_r / (2 * np.pi)
                integral = lengths[i] ** 2 * (np.log(lengths[i] / reference) - 1.5)
                moments = lengths[i] ** 2 * (
                    np.full((2, 2), np.log(lengths[i] / reference) / 4)
                    + np.array([[-7.0, -5.0], [-5.0, -7.0]]) / 16
                )
                correction_v = log_green * integral * np.eye(2)
                correction_d += np.einsum(
                    "ij,ab->iajb", np.outer(dx, dy) * log_derivative * integral, np.eye(2)
                )
                correction_d -= np.einsum(
                    "ij,ab->iajb",
                    material.density * effective_s**2 / (2 * np.pi) * moments,
                    np.eye(2),
                )
            elif log_t is not None:
                green = green - log_green * log_t[:, None, None] * np.eye(2)
                gs, gp = gs + log_t / (2 * np.pi), gp + log_t / (2 * np.pi)
                integral = -lengths[i] * lengths[j] / 2
                correction_v = log_green * integral * np.eye(2)
                angular = -np.dot(normals[i], normals[j]) * np.eye(2) + ROTATION * np.dot(
                    normals[i], tangents[j]
                )
                correction_d += np.einsum(
                    "ij,ab->iajb", np.outer(dx, dy) * log_derivative * integral, np.eye(2)
                )
                correction_d += np.einsum(
                    "ij,ab->iajb",
                    material.density * effective_s**2 / (2 * np.pi) * log_moment,
                    angular,
                )
            mass, deriv, cross_left, cross_right = _weak_kernels(
                green,
                gs,
                gp,
                difference,
                gradient_y,
                normals[i],
                normals[j],
                tangents[j],
                effective_s,
                material,
            )
            local_d = _local_weak(mass, deriv, cross_left, cross_right, px, py, dx, dy, measure)
            local_v = np.einsum("q,qab->ab", measure, green)
            if i == j:
                local_v *= 2
                local_d += local_d.transpose(2, 3, 0, 1)
                local_k = np.einsum("q,qab,qi->abi", measure, double, py - px)
            else:
                local_k = np.einsum("q,qab,qi->abi", measure, double, py)
                local_reverse = np.einsum("q,qab,qi->abi", measure, reverse_double, px)
                hmat[np.ix_(column, idofs)] += local_reverse.transpose(0, 2, 1).reshape(2, 4)
            local_v += correction_v
            local_d += correction_d
            vmat[np.ix_(row, column)] += local_v
            hmat[np.ix_(row, jdofs)] += local_k.transpose(0, 2, 1).reshape(2, 4)
            dmat[np.ix_(idofs, jdofs)] += local_d.reshape(4, 4)
            if i != j:
                vmat[np.ix_(column, row)] += local_v.T
                dmat[np.ix_(jdofs, idofs)] += local_d.reshape(4, 4).T
    hmat += 0.5 * mass_matrix
    return GalerkinOperators(vmat, hmat, dmat, mass_matrix)

"""Interior displacement, gradient and plane-strain stress layer potentials.

Somigliana: u = V_p lambda - K_p u_boundary. P0 lambda is a traction DENSITY,
not the symmetric Galerkin nodal force. For a P1 Galerkin trace, recover it
from V lambda = H u; the symmetric and nonsymmetric discrete DtN maps differ.
Zero-initial perturbation velocity has Laplace transform s*u. A static lift
must be reconstructed separately and has zero velocity.

This separate tool does not modify the frozen event solver or its kernels.
Representation reference: Aimi/Di Credico/Gimperlein, arXiv:2405.08566, (46).
"""

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve
from scipy.special import kv

from finiteblock_bem.galerkin import assemble_galerkin_operators
from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.kernels import (
    _difference_derivatives_series,
    _radial,
    green_and_gradient,
    source_traction,
    static_green_and_gradient,
)


def _fourth_difference_series(r, ks, kp):
    """Fourth derivative of K0(ks*r)-K0(kp*r), including the r² log(r) term."""
    result = np.zeros_like(r, dtype=complex)
    harmonic = 0.0
    for k in range(1, 10):
        harmonic += 1 / k
        n = 2 * k
        cs, cp = ((z / 2) ** n / math.factorial(k) ** 2 for z in (ks, kp))
        a = cs * (harmonic - np.euler_gamma - np.log(ks / 2))
        a -= cp * (harmonic - np.euler_gamma - np.log(kp / 2))
        b = -(cs - cp)
        polynomial = n * (n - 1) * (n - 2) * (n - 3)
        derivative = 4 * n**3 - 18 * n**2 + 22 * n - 6
        result += r ** (n - 4) * (polynomial * (a + b * np.log(r)) + derivative * b)
    return result


def green_hessian(vector, material, s=None, *, reference_length=1.0):
    """Analytic source Hessian d_l d_k G_ij, with small-argument series."""
    vector = np.asarray(vector, dtype=float)
    if vector.shape[-1:] != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError("Require finite two-dimensional separation vectors")
    r = np.linalg.norm(vector, axis=-1)
    if np.any(r <= 0):
        raise ValueError("Interior kernels require positive separation")
    flat = r.reshape(-1)
    if s is None:
        if not np.isfinite(reference_length) or reference_length <= 0:
            raise ValueError("Static reference length must be positive and finite")
        factor = 1 / (8 * np.pi * material.mu * (1 - material.poisson))
        b = np.full_like(flat, factor)
        da = -factor * (3 - 4 * material.poisson) / flat
        db = np.zeros_like(flat)
        dda, ddb = -da / flat, np.zeros_like(flat)
    else:
        if not np.isfinite(s) or np.real(s) <= 0:
            raise ValueError("Laplace frequency must have positive real part")
        ks, kp = complex(s) / material.cs, complex(s) / material.cp
        k0s, k1s, k0p, k1p = (kv(n, z * flat) for n, z in [(0, ks), (1, ks), (0, kp), (1, kp)])
        f1 = -ks * k1s + kp * k1p
        f2 = ks**2 * k0s - kp**2 * k0p - f1 / flat
        f3 = -(ks**3) * k1s + kp**3 * k1p - f2 / flat + f1 / flat**2
        f4 = ks**4 * k0s - kp**4 * k0p
        f4 += 2 * (ks**3 * k1s - kp**3 * k1p) / flat
        f4 += 3 * (ks**2 * k0s - kp**2 * k0p) / flat**2
        f4 += 6 * (ks * k1s - kp * k1p) / flat**3
        small = np.maximum(abs(ks * flat), abs(kp * flat)) < 0.2
        if np.any(small):
            f1[small], f2[small], f3[small] = _difference_derivatives_series(flat[small], ks, kp)
            f4[small] = _fourth_difference_series(flat[small], ks, kp)
        _, b, da, db = _radial(flat, s, material)
        scale = 1 / (2 * np.pi * material.mu)
        dda = ks**2 * k0s + ks * k1s / flat
        dda -= (f3 / flat - 2 * f2 / flat**2 + 2 * f1 / flat**3) / ks**2
        ddb = -(f4 - f3 / flat + 2 * f2 / flat**2 - 2 * f1 / flat**3) / ks**2
        dda, ddb = dda * scale, ddb * scale
    b, da, db, dda, ddb = (v.reshape(r.shape) for v in (b, da, db, dda, ddb))
    e, eye = vector / r[..., None], np.eye(2)
    ee = e[..., :, None] * e[..., None, :]
    de = (eye - ee) / r[..., None, None]
    hessian = np.empty((*r.shape, 2, 2, 2, 2), dtype=float if s is None else complex)
    for ell in range(2):
        dl = de[..., ell, :]
        d_ee = dl[..., :, None] * e[..., None, :] + e[..., :, None] * dl[..., None, :]
        for k in range(2):
            angular = eye[:, k, None] * e[..., None, :] + e[..., :, None] * eye[k, :]
            angular -= 2 * e[..., k, None, None] * ee
            d_angular = eye[:, k, None] * dl[..., None, :] + dl[..., :, None] * eye[k, :]
            d_angular -= 2 * de[..., ell, k, None, None] * ee
            d_angular -= 2 * e[..., k, None, None] * d_ee
            value = (dda * e[..., ell] * e[..., k] + da * de[..., ell, k])[..., None, None] * eye
            value += (ddb * e[..., ell] * e[..., k] + db * de[..., ell, k])[..., None, None] * ee
            value += (db * e[..., k])[..., None, None] * d_ee
            value += ((db / r - b / r**2) * e[..., ell])[..., None, None] * angular
            value += (b / r)[..., None, None] * d_angular
            hessian[..., ell, k, :, :] = value
    if not np.all(np.isfinite(hessian)):
        raise FloatingPointError("Nonfinite interior Green Hessian")
    return hessian


def plane_strain_stress(gradient, material):
    """Return (xx, yy, xy, zz); gradient has component, derivative axes last."""
    gradient = np.asarray(gradient)
    if gradient.shape[-2:] != (2, 2) or not np.all(np.isfinite(gradient)):
        raise ValueError("Require a finite two-dimensional displacement gradient")
    trace = np.trace(gradient, axis1=-2, axis2=-1)
    return np.stack(
        [
            material.lam * trace + 2 * material.mu * gradient[..., 0, 0],
            material.lam * trace + 2 * material.mu * gradient[..., 1, 1],
            material.mu * (gradient[..., 0, 1] + gradient[..., 1, 0]),
            material.lam * trace,
        ],
        axis=-1,
    )


class InteriorPotentials:
    """P0 density / P1 trace potentials with target-dependent panel subdivision."""

    def __init__(self, mesh, material, points, *, quadrature=16, max_depth=24):
        points = np.asarray(points, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not len(points)
            or not np.all(np.isfinite(points))
            or not isinstance(quadrature, int)
            or isinstance(quadrature, bool)
            or quadrature < 4
            or quadrature % 2
            or not isinstance(max_depth, int)
            or max_depth < 0
        ):
            raise ValueError("Require finite interior points and even quadrature >=4")
        self.mesh, self.material, self.points = mesh, material, points.copy()
        self.reference_length = max(float(np.ptp(mesh.starts, axis=0).max()), 1.0)
        abscissa, weights = leggauss(quadrature)
        self.geometry = []
        for point in points:
            edge = mesh.ends - mesh.starts
            projection = np.sum((point - mesh.starts) * edge, axis=1) / mesh.lengths**2
            nearest = mesh.starts + np.clip(projection, 0, 1)[:, None] * edge
            distance = np.linalg.norm(point - nearest, axis=1)
            a, b = mesh.starts - point, mesh.ends - point
            angle = np.sum(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], np.sum(a * b, axis=1)))
            if (
                np.min(distance) <= 128 * np.finfo(float).eps * self.reference_length
                or abs(angle - 2 * np.pi) > 1e-8
            ):
                raise ValueError("Targets must be strictly interior, away from boundary roundoff")
            vectors, measures, fractions, normals, offsets = [], [], [], [], [0]
            for j in range(mesh.count):
                stack, intervals = [(0.0, 1.0, 0)], []
                while stack:
                    left, right, depth = stack.pop()
                    closest = mesh.starts[j] + np.clip(projection[j], left, right) * edge[j]
                    if (right - left) * mesh.lengths[j] <= 2 * np.linalg.norm(point - closest):
                        intervals.append((left, right))
                    elif depth >= max_depth:
                        raise ValueError(
                            "Interior near-panel quadrature exceeded subdivision limit"
                        )
                    else:
                        middle = (left + right) / 2
                        stack.extend([(middle, right, depth + 1), (left, middle, depth + 1)])
                for left, right in intervals:
                    fraction = left + (abscissa + 1) * (right - left) / 2
                    vectors.extend(mesh.starts[j] + fraction[:, None] * edge[j] - point)
                    fractions.extend(fraction)
                    measures.extend(weights * (right - left) * mesh.lengths[j] / 2)
                    normals.extend([mesh.normals[j]] * quadrature)
                offsets.append(len(vectors))
            self.geometry.append(
                tuple(np.asarray(v) for v in (vectors, measures, fractions, normals, offsets))
            )

    def matrices(self, s=None):
        """Return Vp,Kp and target-gradient maps, flattened component first."""
        n, count = self.mesh.count, len(self.points)
        dtype = float if s is None else complex
        single, double = (np.zeros((count, 2, 2 * n), dtype=dtype) for _ in range(2))
        single_g, double_g = (np.zeros((count, 2, 2, 2 * n), dtype=dtype) for _ in range(2))
        for p, (vector, measure, fraction, normals, offsets) in enumerate(self.geometry):
            green, gradient = (
                static_green_and_gradient(vector, self.material, self.reference_length)
                if s is None
                else green_and_gradient(vector, s, self.material)
            )
            hessian = green_hessian(
                vector, self.material, s, reference_length=self.reference_length
            )
            double_kernel = source_traction(gradient, normals, self.material)
            double_gradient = source_traction(hessian, normals[:, None, :], self.material)
            for j in range(n):
                sl = slice(offsets[j], offsets[j + 1])
                single[p, :, 2 * j : 2 * j + 2] = np.einsum("q,qij->ij", measure[sl], green[sl])
                single_g[p, :, :, 2 * j : 2 * j + 2] = -np.einsum(
                    "q,qkij->ikj", measure[sl], gradient[sl]
                )
                for node, phi in [(j, 1 - fraction[sl]), ((j + 1) % n, fraction[sl])]:
                    w = measure[sl] * phi
                    double[p, :, 2 * node : 2 * node + 2] += np.einsum(
                        "q,qij->ij", w, double_kernel[sl]
                    )
                    double_g[p, :, :, 2 * node : 2 * node + 2] -= np.einsum(
                        "q,qkij->ikj", w, double_gradient[sl]
                    )
        return dict(
            single=single.reshape(2 * count, 2 * n),
            double=double.reshape(2 * count, 2 * n),
            single_gradient=single_g.reshape(4 * count, 2 * n),
            double_gradient=double_g.reshape(4 * count, 2 * n),
        )

    def evaluate(self, displacement, density, s=None):
        """Use physical P1 displacement and P0 traction density, not nodal force."""
        u, t = np.asarray(displacement), np.asarray(density)
        if (
            u.shape != (self.mesh.dofs,)
            or t.shape != u.shape
            or not np.all(np.isfinite(u))
            or not np.all(np.isfinite(t))
        ):
            raise ValueError("Require finite P1 displacement and P0 density vectors")
        maps, p = self.matrices(s), len(self.points)
        value = (maps["single"] @ t - maps["double"] @ u).reshape(p, 2)
        gradient = (maps["single_gradient"] @ t - maps["double_gradient"] @ u).reshape(p, 2, 2)
        return dict(
            displacement=value,
            gradient=gradient,
            stress=plane_strain_stress(gradient, self.material),
        )

    def from_galerkin_trace(self, displacement, s=None, *, operators=None, quadrature=12):
        """Recover the auxiliary density from V lambda=H u, never S u / weights."""
        displacement = np.asarray(displacement)
        if displacement.shape != (self.mesh.dofs,) or not np.all(np.isfinite(displacement)):
            raise ValueError("Require a finite full P1 boundary displacement")
        operators = (
            assemble_galerkin_operators(self.mesh, self.material, s, quadrature=quadrature)
            if operators is None
            else operators
        )
        density = solve(
            operators.single_layer, operators.trace_double_layer @ displacement, check_finite=False
        )
        return self.evaluate(displacement, density, s)


class InteriorRetainedTransfer:
    """Dynamic q=(normal jump,tangent jump,platen) -> interior perturbations.

    Assemble each body's original Galerkin operators once per frequency and
    reuse them for condensation, density recovery and every interior probe.
    Supports and unforced eliminated boundaries match NodalInterfaceTransfer.
    This is a frequency-domain map, not a complete time-history reconstruction.
    """

    def __init__(self, model, points_by_body, *, quadrature=12, probe_quadrature=16, prepared=True):
        if len(points_by_body) != 2:
            raise ValueError("Provide moving and stationary interior probe arrays")
        self.model = model
        self.quadrature = quadrature
        con = model.constraints
        self.jump = np.vstack((con.normal_jump, con.tangential_jump, np.eye(con.size)[-1]))
        self.blocks = []
        first = 0
        for mesh, points in zip(
            (model.reference.moving, model.reference.stationary), points_by_body, strict=True
        ):
            p = con.prolongation[first : first + mesh.dofs]
            columns = np.flatnonzero(np.any(p != 0, axis=0))
            potentials = InteriorPotentials(
                mesh, model.reference.material, points, quadrature=probe_quadrature
            )
            assembler = (
                PreparedGalerkinOperators(mesh, model.reference.material, quadrature=quadrature)
                if prepared
                else None
            )
            self.blocks.append((mesh, p[:, columns], self.jump[:, columns], potentials, assembler))
            first += mesh.dofs

    def evaluate(self, s):
        if not np.isfinite(s) or np.real(s) <= 0:
            raise ValueError(
                "Dynamic retained transfer needs Re(s)>0; reconstruct the static lift separately"
            )
        d = len(self.jump)
        compliance = np.zeros((d, d), complex)
        responses = []
        for mesh, p, b, potentials, assembler in self.blocks:
            operators = (
                assembler.assemble(s)
                if assembler is not None
                else assemble_galerkin_operators(
                    mesh, self.model.reference.material, s, quadrature=self.quadrature
                )
            )
            stiffness = operators.force_stiffness()
            reduced = solve(p.T @ stiffness @ p, b.T, check_finite=False)
            compliance += b @ reduced
            responses.append((p @ reduced, operators, potentials))
        impedance = solve(compliance, np.eye(d), check_finite=False)
        displacement, gradient, boundary = [], [], []
        for response, operators, potentials in responses:
            trace = response @ impedance
            density = solve(
                operators.single_layer, operators.trace_double_layer @ trace, check_finite=False
            )
            maps = potentials.matrices(s)
            boundary.append(trace)
            displacement.append(maps["single"] @ density - maps["double"] @ trace)
            gradient.append(maps["single_gradient"] @ density - maps["double_gradient"] @ trace)
        displacement, gradient = np.vstack(displacement), np.vstack(gradient)
        point_gradient = gradient.reshape(-1, 2, 2, d).transpose(3, 0, 1, 2)
        stress = plane_strain_stress(point_gradient, self.model.reference.material).transpose(
            1, 2, 0
        )
        return dict(
            impedance=impedance,
            boundary_displacement=np.vstack(boundary),
            displacement=displacement,
            gradient=gradient,
            stress=stress.reshape(-1, d),
            velocity=s * displacement,
        )

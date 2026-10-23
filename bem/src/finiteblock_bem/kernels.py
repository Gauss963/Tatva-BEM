"""Laplace-domain Green tensor for the 2D Navier elastodynamic equation.

For Re(s)>0, (rho*s**2 - mu*Delta - (lambda+mu)*grad div) G = I*delta.
Writing phi_k=K0(k*r)/(2*pi), k_s=s/c_s and k_p=s/c_p,

    G = (phi_s*I - Hessian(phi_s-phi_p)/k_s**2) / mu.

Source derivatives use r_vector = source - target. The double-layer kernel
is the source traction of each row of G, with the SOURCE outward normal.
Small arguments use the analytic K0 series before differentiation, avoiding
subtraction of the common 1/r terms in k_s*K1 and k_p*K1.
"""

import math

import numpy as np
from scipy.special import kv

from .material import ElasticMaterial


def _difference_derivatives_series(r, ks, kp):
    """First three radial derivatives of K0(ks*r)-K0(kp*r)."""
    f1 = np.zeros_like(r, dtype=complex)
    f2 = np.zeros_like(r, dtype=complex)
    f3 = np.zeros_like(r, dtype=complex)
    harmonic = 0.0
    log_r = np.log(r)
    for k in range(1, 10):
        harmonic += 1 / k
        n = 2 * k
        cs = (ks / 2) ** n / math.factorial(k) ** 2
        cp = (kp / 2) ** n / math.factorial(k) ** 2
        a = cs * (harmonic - np.euler_gamma - np.log(ks / 2))
        a -= cp * (harmonic - np.euler_gamma - np.log(kp / 2))
        b = -(cs - cp)
        value = a + b * log_r
        f1 += r ** (n - 1) * (n * value + b)
        f2 += r ** (n - 2) * (n * (n - 1) * value + (2 * n - 1) * b)
        f3 += r ** (n - 3) * (n * (n - 1) * (n - 2) * value + (3 * n * n - 6 * n + 2) * b)
    return f1, f2, f3


def _radial(r, s, material, *, scalars=False):
    ks, kp = complex(s) / material.cs, complex(s) / material.cp
    zs, zp = ks * r, kp * r
    k0s, k1s, k0p, k1p = kv(0, zs), kv(1, zs), kv(0, zp), kv(1, zp)
    f1 = -ks * k1s + kp * k1p
    f2 = ks * ks * k0s - kp * kp * k0p - f1 / r
    f3 = -(ks**3) * k1s + kp**3 * k1p - f2 / r + f1 / r**2
    small = np.maximum(np.abs(zs), np.abs(zp)) < 0.2
    if np.any(small):
        d1, d2, d3 = _difference_derivatives_series(r[small], ks, kp)
        f1[small], f2[small], f3[small] = d1, d2, d3
    a = k0s - f1 / (ks * ks * r)
    b = -(f2 - f1 / r) / (ks * ks)
    da = -ks * k1s - (f2 / r - f1 / r**2) / (ks * ks)
    db = -(f3 - f2 / r + f1 / r**2) / (ks * ks)
    scale = 1 / (2 * np.pi * material.mu)
    result = (a * scale, b * scale, da * scale, db * scale)
    if scalars:
        return (*result, k0s / (2 * np.pi), k0p / (2 * np.pi), f1 / (2 * np.pi))
    return result


def _tensor_from_radial(vector, r, a, b, da, db):
    e = vector / r[..., None]
    eye = np.eye(2)
    tensor = a[..., None, None] * eye + np.einsum("..., ...i,...j->...ij", b, e, e)
    gradient = np.einsum("...,...k,ij->...kij", da, e, eye)
    gradient += np.einsum("...,...k,...i,...j->...kij", db, e, e, e)
    gradient += (b / r)[..., None, None, None] * (
        np.einsum("ik,...j->...kij", eye, e)
        + np.einsum("jk,...i->...kij", eye, e)
        - 2 * np.einsum("...k,...i,...j->...kij", e, e, e)
    )
    return tensor, gradient


def green_and_gradient(vector, s, material: ElasticMaterial):
    vector = np.asarray(vector, dtype=float)
    if vector.shape[-1:] != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError("Expected finite 2D separation vectors")
    if not np.isfinite(s) or np.real(s) <= 0:
        raise ValueError("Laplace parameter must have positive real part")
    r = np.linalg.norm(vector, axis=-1)
    if np.any(r <= 0):
        raise ValueError("The fundamental solution is singular at zero separation")
    # Flattening also supports one separation vector without scalar indexing.
    shape = r.shape
    flat = r.reshape(-1)
    radial = [x.reshape(shape) for x in _radial(flat, s, material)]
    return _tensor_from_radial(vector, r, *radial)


def static_green_and_gradient(vector, material: ElasticMaterial, reference_length=1.0):
    vector = np.asarray(vector, dtype=float)
    r = np.linalg.norm(vector, axis=-1)
    if np.any(r <= 0) or reference_length <= 0:
        raise ValueError("Positive separation and reference length are required")
    factor = 1 / (8 * np.pi * material.mu * (1 - material.poisson))
    kappa = 3 - 4 * material.poisson
    a = -factor * kappa * np.log(r / reference_length)
    b = np.full_like(r, factor)
    da, db = -factor * kappa / r, np.zeros_like(r)
    return _tensor_from_radial(vector, r, a, b, da, db)


def source_traction(gradient, normals, material: ElasticMaterial):
    """Double-layer tensor D_ij = (T_y G_i)_j, not its transpose."""
    normals = np.asarray(normals)
    divergence = np.einsum("...kik->...i", gradient)
    directional = np.einsum("...k,...kij->...ij", normals, gradient)
    second = np.einsum("...k,...jik->...ij", normals, gradient)
    return material.lam * divergence[..., :, None] * normals[..., None, :] + material.mu * (
        directional + second
    )

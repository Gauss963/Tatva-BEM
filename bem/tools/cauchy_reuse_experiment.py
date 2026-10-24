"""Reuse a sampled analytic BDF2 transfer on another, guarded contour.

The circle barycentric formula evaluates its DFT polynomial, not the exact
analytic transfer. Dense summation is only for small independent tests; the
optional FMM2D backend batches Cauchy sums across matrix entries. Neither
backend certifies the interpolation error of a BEM transfer.
"""

import numpy as np


def bdf2_preimage(s, dt):
    """Inverse of delta(z)/dt on the branch connected to z=0 at s=1.5/dt."""
    s = np.asarray(s, dtype=complex)
    if not np.isfinite(dt) or dt <= 0 or not np.all(np.isfinite(s)) or np.any(s.real <= 0):
        raise ValueError("Finite right-half-plane frequencies and positive dt are required")
    # Rationalized form avoids cancellation near z=0.
    root = np.sqrt(1 + 2 * dt * s)
    return (3 - 2 * dt * s) / (2 + root)


def _direct(sources, densities, targets, derivative):
    potential = np.empty((len(densities), len(targets)), complex)
    gradient = np.empty_like(potential) if derivative else None
    for first in range(0, len(targets), 32):
        last = first + 32
        kernel = 1 / (targets[None, first:last] - sources[:, None])
        potential[:, first:last] = densities @ kernel
        if derivative:
            gradient[:, first:last] = -(densities @ kernel**2)
    return potential, gradient


def fmm2d_backend(eps=1e-13):
    if not np.isfinite(eps) or not 0 < eps < 1:
        raise ValueError("FMM tolerance must lie between zero and one")
    try:
        from fmm2dpy import cfmm2d
    except ImportError as error:
        raise ImportError("Optional fmm2dpy is required for large contour reuse") from error

    def evaluate(sources, densities, targets, derivative):
        out = cfmm2d(
            eps=eps,
            sources=np.stack((sources.real, sources.imag)),
            dipstr=densities,
            targets=np.stack((targets.real, targets.imag)),
            pgt=2 if derivative else 1,
            nd=len(densities),
        )
        if getattr(out, "ier", 0):
            raise ValueError(f"FMM2D failed with code {out.ier}")
        return out.pottarg, out.gradtarg if derivative else None

    return evaluate


def fortran_fmm2d_backend(library_path, eps=1e-13):
    """Call official default-INTEGER Fortran wrappers without a NumPy ABI.

    The library must export cfmm2d_t_d_{p,g}_vec_ and use 32-bit default
    INTEGER, as in the official makefiles. A build with integer promotion
    is incompatible. No third-party code is vendored in this wrapper.
    """
    import ctypes

    if not np.isfinite(eps) or not 0 < eps < 1:
        raise ValueError("FMM tolerance must lie between zero and one")
    library = ctypes.CDLL(str(library_path))
    functions = {}
    for derivative, name in [(False, "cfmm2d_t_d_p_vec_"), (True, "cfmm2d_t_d_g_vec_")]:
        function = getattr(library, name)
        function.argtypes = [ctypes.c_void_p] * (10 if derivative else 9)
        function.restype = None
        functions[derivative] = function

    def evaluate(sources, densities, targets, derivative):
        sources, targets = np.asarray(sources, dtype=complex), np.asarray(targets, dtype=complex)
        densities = np.asarray(densities, dtype=complex)
        if (
            sources.ndim != 1
            or targets.ndim != 1
            or densities.ndim != 2
            or densities.shape[1] != len(sources)
            or not all(np.all(np.isfinite(a)) for a in [sources, targets, densities])
        ):
            raise ValueError("Finite complex coordinates and matching density arrays are required")
        coords = np.asfortranarray(np.stack((sources.real, sources.imag)), dtype=np.float64)
        target_coords = np.asfortranarray(np.stack((targets.real, targets.imag)), dtype=np.float64)
        strengths = np.asfortranarray(densities, dtype=np.complex128)
        dimensions = [len(densities), len(sources), len(targets)]
        if not all(0 < count < 2**31 for count in dimensions):
            raise ValueError("FMM wrapper dimensions exceed its 32-bit integer interface")
        nd, ns, nt = map(ctypes.c_int, dimensions)
        tolerance, error = ctypes.c_double(eps), ctypes.c_int()
        value = np.empty((nd.value, nt.value), dtype=np.complex128, order="F")
        gradient = np.empty_like(value, order="F") if derivative else None
        arguments = [
            ctypes.byref(nd),
            ctypes.byref(tolerance),
            ctypes.byref(ns),
            coords.ctypes.data_as(ctypes.c_void_p),
            strengths.ctypes.data_as(ctypes.c_void_p),
            ctypes.byref(nt),
            target_coords.ctypes.data_as(ctypes.c_void_p),
            value.ctypes.data_as(ctypes.c_void_p),
        ]
        if derivative:
            arguments.append(gradient.ctypes.data_as(ctypes.c_void_p))
        arguments.append(ctypes.byref(error))
        functions[derivative](*arguments)
        if error.value:
            raise ValueError(f"FMM2D failed with code {error.value}")
        return value, gradient

    return evaluate


class CircleCauchy:
    def __init__(self, fft_size, radius, *, max_amplification=4.0):
        if not isinstance(fft_size, int) or fft_size < 4 or fft_size % 2:
            raise ValueError("An even source FFT size of at least four is required")
        if not np.isfinite(radius) or not 0 < radius < 1:
            raise ValueError("Source radius must lie strictly inside the unit disk")
        if not np.isfinite(max_amplification) or max_amplification < 1:
            raise ValueError("Maximum extrapolation amplification must be finite and >= 1")
        self.size, self.radius = fft_size, radius
        self.max_amplification = max_amplification
        self.sources = radius * np.exp(-2j * np.pi * np.arange(fft_size) / fft_size)

    def validate_targets(self, targets):
        targets = np.asarray(targets, dtype=complex)
        if targets.ndim != 1 or not np.all(np.isfinite(targets)):
            raise ValueError("Finite one-dimensional target coordinates are required")
        magnitude = abs(targets)
        if np.any(magnitude >= 1):
            raise ValueError("BDF2 reuse cannot leave the analytic unit disk")
        log_ratio = np.log(np.maximum(magnitude, self.radius) / self.radius)
        if np.any(self.size * log_ratio > np.log(self.max_amplification)):
            raise ValueError("Contour extrapolation exceeds the amplification guard")
        return targets

    def evaluate(self, half_values, targets, *, derivative=False, backend=None):
        """Return batched values and optionally d/dz; preserve trailing dimensions.

        Values at the real-axis endpoints use the same real projection as
        IRFFT. A single extra constant density normalizes the FMM sums, so
        any common kernel normalization cancels from values and derivatives.
        """
        targets = self.validate_targets(targets)
        half = np.asarray(half_values, dtype=complex)
        if half.ndim < 1 or len(half) != self.size // 2 + 1 or not np.all(np.isfinite(half)):
            raise ValueError("Finite half-circle samples of the matching size are required")
        shape, count = half.shape[1:], int(np.prod(half.shape[1:]))
        full = np.empty((self.size, count + 1), complex)
        full[: len(half), :-1] = half.reshape(len(half), count)
        full[0, :-1] = full[0, :-1].real
        full[self.size // 2, :-1] = full[self.size // 2, :-1].real
        full[len(half) :, :-1] = full[1 : self.size // 2, :-1][::-1].conj()
        full[:, -1] = 1
        nearest = np.rint(-np.angle(targets) * self.size / (2 * np.pi)).astype(int) % self.size
        exact = abs(targets - self.sources[nearest]) <= 4 * np.finfo(float).eps
        value = np.empty((len(targets), count), complex)
        grad = np.empty_like(value) if derivative else None
        for index in np.flatnonzero(exact):
            node = nearest[index]
            value[index] = full[node, :-1]
            if derivative:
                keep = np.arange(self.size) != node
                factor = self.sources[keep] / self.sources[node]
                factor /= self.sources[node] - self.sources[keep]
                grad[index] = factor @ (full[keep, :-1] - full[node, :-1])
        if np.any(~exact):
            if backend is None and self.size * np.count_nonzero(~exact) * count > 100_000_000:
                raise ValueError("Dense Cauchy evaluation is a small-test backend; supply FMM2D")
            densities = (full * self.sources[:, None]).T
            potential, gradient = (_direct if backend is None else backend)(
                self.sources, densities, targets[~exact], derivative
            )
            potential = np.asarray(potential)
            denominator = potential[-1]
            value[~exact] = (potential[:-1] / denominator).T
            if derivative:
                gradient = np.asarray(gradient)
                grad[~exact] = ((gradient[:-1] - value[~exact].T * gradient[-1]) / denominator).T
        if not np.all(np.isfinite(value)) or (derivative and not np.all(np.isfinite(grad))):
            raise ValueError("Nonfinite normalized Cauchy result")
        value = value.reshape((len(targets), *shape))
        return (value, grad.reshape(value.shape)) if derivative else value

"""Experimental frequency-local scaled-Bessel interpolation on a single ray.

The table interpolates exp(z)*K_0(z) and exp(z)*K_1(z), not the oscillatory
exponential. Independent off-grid checks and coefficient tails are empirical
acceptance tests, not a uniform error certificate. Rejected bins and arguments
outside the positive-real ray domain use the original SciPy implementation.
Only private copies of the original radial/assembly globals are changed.
"""

from dataclasses import dataclass
from types import FunctionType

import numpy as np
from scipy.fft import dct
from scipy.special import kv, kve

from finiteblock_bem import kernels
from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators


@dataclass(frozen=True)
class RayBesselTable:
    direction: complex
    first_exponent: int
    coefficients: np.ndarray
    accepted: np.ndarray
    checks: tuple
    check_rtol: float

    @classmethod
    def prepare(cls, s, minimum, maximum, *, degree=24, check_rtol=5e-14):
        """Freeze dyadic intervals covering the declared positive radius domain."""
        if (
            not np.isfinite(s)
            or np.real(s) <= 0
            or not np.isfinite(minimum)
            or not np.isfinite(maximum)
            or not 0 < minimum <= maximum
            or not isinstance(degree, int)
            or not 8 <= degree <= 64
            or not np.isfinite(check_rtol)
            or not 0 < check_rtol <= 1e-11
        ):
            raise ValueError("Require a finite right-half-plane ray and bounded positive domain")
        direction = complex(s) / abs(s)
        first = int(np.frexp(minimum)[1]) - 1
        last = int(np.frexp(maximum)[1]) - 1
        if first < -900 or last > 900 or last - first > 256:
            raise ValueError("Bessel table exponent range exceeds the explicit finite budget")
        count = degree + 1
        x = np.cos(np.pi * (np.arange(count) + 0.5) / count)
        # These nodes differ from every interpolation node; include bin seams.
        check_x = np.r_[
            -1.0,
            0.0,
            1.0,
            np.cos(np.pi * (np.arange(2 * count + 3) + 0.37) / (2 * count + 3)),
        ]
        coefficients = np.empty((2, last - first + 1, count), complex)
        accepted = np.zeros(last - first + 1, bool)
        checks = []
        for i, exponent in enumerate(range(first, last + 1)):
            lo = np.ldexp(1.0, exponent)
            z = direction * (lo * (1.5 + 0.5 * x))
            check_z = direction * (lo * (1.5 + 0.5 * check_x))
            record = []
            for order in [0, 1]:
                values = kve(order, z)
                c = dct(values, type=2) / count
                c[0] *= 0.5
                coefficients[order, i] = c
                reference = kve(order, check_z)
                actual = np.polynomial.chebyshev.chebval(check_x, c)
                with np.errstate(divide="ignore", invalid="ignore"):
                    error = float(np.max(abs(actual - reference) / abs(reference)))
                    tail = float(np.sum(abs(c[-4:])) / np.min(abs(reference)))
                finite = all(np.all(np.isfinite(a)) for a in [values, c, reference, actual])
                record.append((error, tail, finite and error <= check_rtol and tail <= check_rtol))
            accepted[i] = all(v[2] for v in record)
            checks.append((exponent, *[v for item in record for v in item]))
        coefficients.flags.writeable = accepted.flags.writeable = False
        return cls(direction, first, coefficients, accepted, tuple(checks), check_rtol)

    def _coordinates(self, z):
        radius = abs(z)
        mantissa, exponent = np.frexp(radius)
        bins = exponent.astype(np.int64) - 1 - self.first_exponent
        # frexp's treatment of nonfinite inputs must not index the coefficient table.
        in_range = np.isfinite(z) & (radius > 0) & (bins >= 0) & (bins < len(self.accepted))
        ids = np.clip(bins, 0, len(self.accepted) - 1)
        rotated = z * self.direction.conjugate()
        # The tolerance admits multiplication roundoff on this ray only.
        on_ray = (rotated.real > 0) & (abs(rotated.imag) <= 8 * np.finfo(float).eps * radius)
        use = in_range & on_ray & self.accepted[ids]
        return use, ids, 4 * mantissa - 3

    def __call__(self, order, z):
        if np.ndim(order) or order not in (0, 1):
            return kv(order, z)
        original = np.asarray(z)
        # Preserve SciPy's real-valued scalar/array convention as an exact fallback.
        if not np.iscomplexobj(original):
            return kv(order, z)
        array = np.asarray(z, complex)
        flat = array.reshape(-1)
        use, bins, x = self._coordinates(flat)
        result = np.empty_like(flat)
        result[~use] = kv(order, flat[~use])
        if np.any(use):
            ids, coordinate = bins[use], x[use]
            c = self.coefficients[int(order)]
            # Batched Clenshaw recurrence without a degree-by-all-points allocation.
            a = np.zeros(len(ids), complex)
            b = np.zeros_like(a)
            for k in range(c.shape[-1] - 1, 0, -1):
                a, b = c[ids, k] + 2 * coordinate * a - b, a
            result[use] = (c[ids, 0] + coordinate * a - b) * np.exp(-flat[use])
        value = result.reshape(array.shape)
        return value[()] if array.ndim == 0 else value

    def statistics(self):
        return dict(
            first_exponent=self.first_exponent,
            bins=len(self.accepted),
            accepted_bins=int(np.count_nonzero(self.accepted)),
            degree=self.coefficients.shape[-1] - 1,
            table_bytes=self.coefficients.nbytes + self.accepted.nbytes,
            offgrid_relative_tolerance=self.check_rtol,
            bin_checks=[
                dict(
                    exponent=v[0],
                    k0_relative_error=v[1] if np.isfinite(v[1]) else None,
                    k0_relative_tail=v[2] if np.isfinite(v[2]) else None,
                    k0_accepted=v[3],
                    k1_relative_error=v[4] if np.isfinite(v[4]) else None,
                    k1_relative_tail=v[5] if np.isfinite(v[5]) else None,
                    k1_accepted=v[6],
                )
                for v in self.checks
            ],
            uniform_error_certificate=False,
        )


class RayBesselGalerkin:
    """Share immutable quadrature and freeze a private Bessel table per assembly."""

    def __init__(self, prepared, *, degree=24, check_rtol=5e-14):
        if type(prepared) is not PreparedGalerkinOperators:
            raise ValueError("Require the original prepared Galerkin assembler")
        if prepared.mass.flags.writeable or any(
            v.flags.writeable
            for batch in prepared.batches
            for v in vars(batch).values()
            if isinstance(v, np.ndarray)
        ):
            raise ValueError("Shared Galerkin geometry must be immutable")
        minimum, maximum = np.inf, 0.0
        for batch in prepared.batches:
            radius = np.linalg.norm(batch.vector, axis=-1)
            minimum, maximum = min(minimum, float(radius.min())), max(maximum, float(radius.max()))
        if not np.isfinite(minimum) or minimum <= 0:
            raise ValueError("Quadrature must have finite strictly positive separations")
        self.prepared, self.minimum, self.maximum = prepared, minimum, maximum
        self.degree, self.check_rtol = degree, check_rtol

    def prepare(self, s):
        material = self.prepared.material
        # Outward padding covers rounding in ks*r versus |s|*r/cs at dyadic seams.
        padding = 32 * np.finfo(float).eps
        table = RayBesselTable.prepare(
            s,
            abs(s) * self.minimum / max(material.cs, material.cp) * (1 - padding),
            abs(s) * self.maximum / min(material.cs, material.cp) * (1 + padding),
            degree=self.degree,
            check_rtol=self.check_rtol,
        )
        original = kernels._radial
        radial = FunctionType(
            original.__code__, {**original.__globals__, "kv": table}, original.__name__
        )
        radial.__defaults__, radial.__kwdefaults__ = original.__defaults__, original.__kwdefaults__
        original = PreparedGalerkinOperators.assemble
        assemble = FunctionType(
            original.__code__, {**original.__globals__, "_radial": radial}, original.__name__
        )
        assemble.__defaults__ = original.__defaults__

        def evaluate():
            return assemble(self.prepared, s)

        return evaluate, table

    def assemble(self, s=None):
        if s is None:
            return self.prepared.assemble(None)
        evaluate, _ = self.prepare(s)
        return evaluate()

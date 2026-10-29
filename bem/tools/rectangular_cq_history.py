"""Bounded offline compact Radau observation and its complete real transpose.

First invert each output stripe into ALL physical causal lags, then FFT the
zero-padded finite weights. Never multiply contour samples directly by a
finite source FFT: that would introduce future-input response. No full time
weight file or expanded two-stage kernel is stored.
"""

import mmap
import runpy
from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft

TOOLS = Path(__file__).resolve().parent
RK = runpy.run_path(str(TOOLS / "radau_experiment.py"))
FILES = runpy.run_path(str(TOOLS / "direct_fortran_radau.py"))
DIGEST = FILES["JOURNAL"]["digest_range"]


def resources(plan, inputs, outputs, output_rows=1, *, transpose=False):
    m, half, n = plan.fft_size, plan.fft_size // 2 + 1, plan.steps + 1
    rows = min(output_rows, outputs)
    source, target = (outputs, inputs) if transpose else (inputs, outputs)
    peak = 4 * m * rows * inputs * 8 + 2 * half * rows * inputs * 16
    peak += half * 2 * (source + (target if transpose else rows)) * 16
    peak += 2 * n * (source + target) * 8 + 64 * 1024**2
    return dict(
        array_peak_bytes=peak,
        output_rows=rows,
        complete_time_weight_files=0,
        expanded_stage_weight_files=0,
        every_requested_lag_retained=True,
        caller_state_and_operator_assembly_excluded=True,
    )


def apply_spectra(
    first, second, plan, values, *, output_rows=1, max_workspace_mb=2048, transpose=False
):
    """Apply finite causal CQ or its exact transpose on the fixed-zero stage space."""
    if (
        len(first.shape) != 3
        or second.shape != first.shape
        or first.shape[0] != plan.fft_size // 2 + 1
        or min(first.shape[1:]) < 1
        or not 0 < plan.radius < 1
        or plan.steps < 1
        or not np.isfinite(plan.dt)
        or plan.dt <= 0
        or plan.fft_size % 2
        or plan.fft_size < 2 * (plan.steps + 1)
        or not isinstance(output_rows, int)
        or isinstance(output_rows, bool)
        or output_rows < 1
        or not np.isfinite(max_workspace_mb)
        or max_workspace_mb <= 0
    ):
        raise ValueError("Require complete rectangular compact spectra and a valid padded CQ plan")
    outputs, inputs = first.shape[1:]
    n, m = plan.steps + 1, plan.fft_size
    source, target = (outputs, inputs) if transpose else (inputs, outputs)
    values = np.asarray(values)
    if (
        values.shape != (n, 2, source)
        or np.iscomplexobj(values)
        or values.dtype != np.float64
        or not np.all(np.isfinite(values))
        or (not transpose and np.any(values[0] != 0))
    ):
        raise ValueError("Require real float64 stages and a fixed-zero artificial initial source")
    budget = resources(plan, inputs, outputs, output_rows, transpose=transpose)
    if budget["array_peak_bytes"] > max_workspace_mb * 1024**2:
        raise ValueError("Rectangular CQ exceeds its array workspace budget")
    if transpose:
        values = values.copy()
        values[0] = 0  # Output[0] is fixed at zero and has no physical derivative.
    source_fft = rfft(values, n=m, axis=0, workers=1)
    answer = np.zeros((n, 2, target), np.float64)
    accumulated = np.zeros((m // 2 + 1, 2, inputs), np.complex128) if transpose else None
    delta0 = np.linalg.inv(RK["A"])
    delta1 = -np.outer(delta0 @ np.ones(2), [0, 1])
    unit_shift = np.exp(-2j * np.pi * np.arange(m // 2 + 1) / m)
    amplification = plan.radius ** (-np.arange(n))
    for row in range(0, outputs, budget["output_rows"]):
        last = min(row + budget["output_rows"], outputs)
        result_fft = np.zeros((m // 2 + 1, 2, last - row), np.complex128) if not transpose else None
        stripe_source = source_fft[:, :, row:last] if transpose else source_fft
        for family, spectrum in enumerate([first, second]):
            samples = np.array(spectrum[:, row:last, :], dtype=np.complex128, copy=True)
            backing = getattr(spectrum, "_mmap", None)
            if hasattr(backing, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                backing.madvise(mmap.MADV_DONTNEED)
            if not np.all(np.isfinite(samples)):
                raise ValueError("Nonfinite compact observation spectrum")
            for endpoint in [0, len(samples) - 1]:
                if np.linalg.norm(samples[endpoint].imag) > 1e-10 * max(
                    np.linalg.norm(samples[endpoint].real), 1
                ):
                    raise ValueError("Observation spectrum violates real endpoint symmetry")
                samples[endpoint] = samples[endpoint].real
            impulse = irfft(samples, n=m, axis=0, workers=1)
            del samples
            wrap = plan.radius * impulse[-1].copy() if family else None
            impulse[:n] *= amplification[:, None, None]
            kernel_fft = rfft(impulse[:n], n=m, axis=0, workers=1)
            del impulse
            if not transpose:
                product = np.einsum("kri,ksi->ksr", kernel_fft, stripe_source)
                if family == 0:
                    result_fft += product
                else:
                    result_fft += np.einsum("ab,kbr->kar", delta0, product)
                    product *= unit_shift[:, None, None]
                    product += np.einsum("ri,ksi->ksr", wrap, stripe_source)
                    result_fft += np.einsum("ab,kbr->kar", delta1, product)
            else:
                if family == 0:
                    mixed = stripe_source
                else:
                    shifted = np.einsum("ba,kbr->kar", delta1, stripe_source)
                    mixed = np.einsum("ba,kbr->kar", delta0, stripe_source)
                    mixed += unit_shift.conj()[:, None, None] * shifted
                    accumulated += np.einsum("ri,ksr->ksi", wrap, shifted)
                accumulated += np.einsum("kri,ksr->ksi", kernel_fft.conj(), mixed)
                if family:
                    del mixed, shifted
            del kernel_fft
            if not transpose:
                del product
        if not transpose:
            answer[:, :, row:last] = irfft(result_fft, n=m, axis=0, workers=1)[:n]
    if transpose:
        answer[:] = irfft(accumulated, n=m, axis=0, workers=1)[:n]
    answer[0] = 0
    if not np.all(np.isfinite(answer)):
        raise ValueError("Nonfinite rectangular CQ result")
    return answer, budget


class FixedRectangularCQ:
    """Verified read-only file-backed fixed kernel; JAX callbacks retain no trajectory.

    Stream SHA256 once at construction, then guard inode/size/mtime before and
    after direct operations. Verify outside compiled JAX work. Files must remain
    immutable for the object's lifetime.
    Kernel/material/geometry and static-lift derivatives are outside this map.
    """

    def __init__(self, records, plan, *, output_rows=1, max_workspace_mb=2048):
        if (
            len(records) != 2
            or not isinstance(output_rows, int)
            or isinstance(output_rows, bool)
            or output_rows < 1
            or not np.isfinite(max_workspace_mb)
            or max_workspace_mb <= 0
        ):
            raise ValueError("Require both complete compact observation receipts")
        self.paths = [Path(r["path"]).resolve() for r in records]
        self.plan, self.rows, self.budget = plan, output_rows, max_workspace_mb
        self.maps = []
        try:
            initial_signature = self._stat()
            for r, p in zip(records, self.paths, strict=True):
                shape, fortran, dtype, _ = FILES["array_header"](p)
                if (
                    len(shape) != 3
                    or shape[0] != plan.fft_size // 2 + 1
                    or min(shape[1:]) < 1
                    or fortran
                    or dtype != np.complex128
                    or list(shape) != r.get("shape")
                    or p.stat().st_size != r.get("bytes")
                    or r.get("all_frequency_matrices_written") is not True
                    or DIGEST(p) != r.get("sha256")
                ):
                    raise ValueError("Require complete unmodified compact rectangular spectra")
                self.maps.append(np.load(p, mmap_mode="r", allow_pickle=False))
            if self.maps[0].shape != self.maps[1].shape or self.paths[0] == self.paths[1]:
                raise ValueError("Compact observation families must be distinct and aligned")
            if self._stat() != initial_signature:
                raise ValueError("Fixed observation spectra changed during initial verification")
            self.signature = initial_signature
        except BaseException:
            self.close()
            raise

    def _stat(self):
        return [
            (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) for s in [p.stat() for p in self.paths]
        ]

    def apply(self, values, *, transpose=False):
        self.verify()
        result, _ = apply_spectra(
            *self.maps,
            self.plan,
            values,
            output_rows=self.rows,
            max_workspace_mb=self.budget,
            transpose=transpose,
        )
        self.verify()
        return result

    def verify(self):
        """Guard immutable files outside JAX compiled regions."""
        if len(self.maps) != 2 or self._stat() != self.signature:
            raise ValueError("Fixed observation spectra changed or were closed")

    def as_jax(self):
        """Custom VJP of the immutable fixed map; verify files outside compiled work.

        Only valid finite float64 fixed-zero-stage inputs are supported. Callback
        execution may be reordered/elided; it performs no file checks or writes.
        Keep this object open and immutable. Static-lift derivatives are separate.
        """
        import jax

        self.verify()
        if not jax.config.read("jax_enable_x64"):
            raise ValueError("Observation AD requires JAX float64")
        n = self.plan.steps + 1
        outputs, inputs = self.maps[0].shape[1:]
        for transpose in [False, True]:
            if (
                resources(self.plan, inputs, outputs, self.rows, transpose=transpose)[
                    "array_peak_bytes"
                ]
                > self.budget * 1024**2
            ):
                raise ValueError("Observation AD exceeds its array workspace budget")
        output_shape = jax.ShapeDtypeStruct((n, 2, outputs), np.float64)
        input_shape = jax.ShapeDtypeStruct((n, 2, inputs), np.float64)

        def calculate(value, transpose=False):
            return apply_spectra(
                *self.maps,
                self.plan,
                value,
                output_rows=self.rows,
                max_workspace_mb=self.budget,
                transpose=transpose,
            )[0]

        @jax.custom_vjp
        def observed(value):
            return jax.pure_callback(calculate, output_shape, value, vmap_method="sequential")

        def forward(value):
            return observed(value), ()

        def backward(saved, cotangent):
            return (
                jax.pure_callback(
                    lambda y: calculate(y, True), input_shape, cotangent, vmap_method="sequential"
                ),
            )

        observed.defvjp(forward, backward)
        return observed

    def close(self):
        for m in self.maps:
            m._mmap.close()
        self.maps = []

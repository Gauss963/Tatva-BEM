"""Dense reference implementation of BDF2 convolution quadrature.

Zeta -> W(delta(zeta)/dt), delta(zeta)=3/2-2*zeta+zeta**2/2.
IFFT on a damped contour generates causal traction/displacement weights.
The complete displacement history is retained. No local radiation-damping
replacement or unreported history truncation is used.
"""

import ast
import hashlib
import math
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft
from threadpoolctl import threadpool_limits

from .material import ElasticMaterial
from .mesh import BoundaryMesh
from .operators import PreparedBoundaryOperators


class ResourceBudgetError(ValueError):
    """The dense reference solver would exceed its explicit memory budget."""


@dataclass(frozen=True)
class CQPlan:
    dt: float
    steps: int
    dofs: int
    fft_size: int
    radius: float
    estimated_peak_bytes: int
    weight_bytes: int


def plan_cq(dofs, dt, steps, *, alias_tolerance=1e-10, max_memory_mb=512, disk_spectrum=False):
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(dofs, (int, np.integer)) or dofs < 2 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Positive dt and at least two degrees of freedom are required")
    if not 1e-14 <= alias_tolerance <= 1e-4:
        raise ValueError("alias_tolerance must be between 1e-14 and 1e-4")
    if not np.isfinite(max_memory_mb) or max_memory_mb <= 0:
        raise ValueError("max_memory_mb must be finite and positive")
    fft_size = 1 << math.ceil(math.log2(2 * (steps + 1)))
    radius = alias_tolerance ** (1 / fft_size)
    weight_bytes = (steps + 1) * dofs * dofs * 8
    # Assembly and online stepping are separate phases. Only levels that can
    # reuse their kernel FFT are cached. Last-use levels stream four output
    # rows, retaining every history term. Count consecutive row FFTs plus
    # scipy's zero-padding copy conservatively.
    spectrum = (fft_size // 2 + 1) * dofs**2 * 16
    inversion = 2 * fft_size * min(dofs, 4) * dofs * 8
    if disk_spectrum:
        # One bounded file map plus copied row spectra and consecutive IFFTs.
        # The time weights and online history still reside in memory.
        spectrum = min(64, fft_size // 2 + 1) * dofs**2 * 16
        inversion = 4 * fft_size * min(dofs, 4) * dofs * 8
    levels = int(steps).bit_length()
    cached = sum((1 << k) + 1 for k in range(levels) if 3 * (1 << k) <= steps)
    cached *= dofs**2 * 16
    largest = 1 << (levels - 1)
    rows = min(dofs, 4)
    streamed = 2 * (largest + 1) * rows * dofs * 16 + 2 * largest * rows * dofs * 8
    vectors = 8 * (steps + 1) * dofs * 8
    peak = weight_bytes + max(spectrum + inversion, cached + vectors + streamed) + 64 * dofs**2
    if peak > max_memory_mb * 1024**2:
        raise ResourceBudgetError(
            f"Dense CQ needs about {peak / 1024**2:.1f} MiB peak for {steps} steps and "
            f"{dofs} boundary DOFs; budget is {max_memory_mb:g} MiB. "
            "Reduce the diagnostic horizon/mesh, or implement accelerated CQ; "
            "do not silently truncate history."
        )
    return CQPlan(dt, steps, dofs, fft_size, radius, peak, weight_bytes)


def plan_domains(dofs, dt, steps, *, max_memory_mb=512):
    """Include retained weights from previous domains in the peak estimate."""
    plans = [plan_cq(size, dt, steps, max_memory_mb=max_memory_mb) for size in dofs]
    if not plans:
        raise ValueError("At least one domain is required")
    live = 0
    peak = 0
    for plan in plans:
        peak = max(peak, live + plan.estimated_peak_bytes)
        live += plan.weight_bytes
    if peak > max_memory_mb * 1024**2:
        raise ResourceBudgetError(
            f"Multi-domain CQ peak estimate {peak / 1024**2:.1f} MiB exceeds budget"
        )
    return plans, peak


def bdf2_symbol(zeta):
    return 1.5 - 2 * zeta + 0.5 * zeta * zeta


def cq_plan_signature(plan):
    """Mathematical contour, independent of workspace estimates and budgets."""
    return plan.dt, plan.steps, plan.dofs, plan.fft_size, plan.radius


def cq_math_signature(source=None):
    """Hash weight-generating code, separately from online history storage.

    These functions contain the complete contour/sampling/inversion
    operations in this module. Operator and adaptive-sampling files are hashed
    separately by their cache builders. Historical migration checks the same AST.
    """
    source = Path(__file__).read_text() if source is None else source
    names = ["bdf2_symbol", "convolution_weights", "invert_spectrum"]
    if any(
        isinstance(node, ast.FunctionDef) and node.name == "spectrum_buffer"
        for node in ast.parse(source).body
    ):
        names.append("spectrum_buffer")
    nodes = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    if set(nodes) != set(names):
        raise ValueError("Incomplete CQ weight-generating source")
    digest = hashlib.sha256()
    for name in names:
        digest.update(ast.dump(nodes[name], include_attributes=False).encode())
    return digest.hexdigest()


def spectrum_buffer(plan, provided=None):
    """Allocate a RAM spectrum, or validate an explicitly supplied storage object."""
    shape = (plan.fft_size // 2 + 1, plan.dofs, plan.dofs)
    if provided is None:
        return np.empty(shape, dtype=complex)
    if provided.shape != shape or provided.dtype != np.dtype("complex128"):
        raise ValueError("Incorrect half-spectrum shape or dtype")
    return provided


def convolution_weights(
    transfer: Callable, plan: CQPlan, *, progress=None, workers=1, spectrum=None
):
    """Sample a real causal transfer function, using conjugate symmetry."""
    half = plan.fft_size // 2
    spectrum = spectrum_buffer(plan, spectrum)

    def sample(k):
        zeta = plan.radius * np.exp(-2j * np.pi * k / plan.fft_size)
        value = transfer(bdf2_symbol(zeta) / plan.dt)
        if np.shape(value) != (plan.dofs, plan.dofs) or not np.all(np.isfinite(value)):
            raise FloatingPointError("Nonfinite or incorrectly sized Laplace transfer matrix")
        return value

    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    # Bound submitted work: executor.map otherwise retains thousands of matrices.
    # Each independent frequency has one BLAS thread, avoiding nested parallelism.
    with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(workers) as pool:
        for first in range(0, half + 1, workers):
            ids = range(first, min(first + workers, half + 1))
            values = map(sample, ids) if workers == 1 else pool.map(sample, ids)
            for k, value in zip(ids, values, strict=True):
                spectrum[k] = value
                if progress is not None:
                    progress(k + 1, half + 1)
    return invert_spectrum(spectrum, plan)


def invert_spectrum(spectrum, plan):
    """Invert a sampled half spectrum; shared by dense and experimental sampling."""
    half = plan.fft_size // 2
    if spectrum.shape != (half + 1, plan.dofs, plan.dofs):
        raise ValueError("Incorrect CQ half-spectrum shape")
    # Positive real s at k=0 and Nyquist must produce real matrices.
    for k in (0, half):
        if np.linalg.norm(spectrum[k].imag) > 1e-10 * max(np.linalg.norm(spectrum[k].real), 1):
            raise FloatingPointError("Real-axis transfer violates conjugate symmetry")
        spectrum[k] = spectrum[k].real
    amplification = plan.radius ** (-np.arange(plan.steps + 1))
    weights = np.empty((plan.steps + 1, plan.dofs, plan.dofs))
    # Real causal kernels have a Hermitian spectrum. Invert a few output rows
    # at a time instead of materializing the full complex IFFT and real copy.
    for first in range(0, plan.dofs, 4):
        transformed = irfft(spectrum[:, first : first + 4], n=plan.fft_size, axis=0)
        weights[:, first : first + 4] = transformed[: plan.steps + 1] * amplification[:, None, None]
    return weights


@dataclass(frozen=True)
class BoundaryHistory:
    plan: CQPlan
    weights: np.ndarray

    @property
    def instantaneous(self):
        return self.weights[0]

    def past_traction(self, displacement_history, step):
        if not 0 <= step <= self.plan.steps:
            raise ValueError("Requested step is outside the generated CQ horizon")
        if step == 0:
            return np.zeros(self.plan.dofs)
        if displacement_history.shape[1:] != (self.plan.dofs,) or len(displacement_history) < step:
            raise ValueError("Incomplete or incompatible boundary history")
        return np.einsum(
            "kij,kj->i",
            self.weights[1 : step + 1],
            displacement_history[step - 1 :: -1],
            optimize=False,
        )


class OnlineConvolution:
    """Exact dyadic FFT updates for a causal matrix convolution.

    Every source/target pair belongs to one adjacent dyadic block pair.
    Completed left blocks update their future right blocks, so no future
    displacement is needed and no memory tail is truncated. Complexity is
    O(N log²(N) d²). Reused levels cache their kernel transforms; a level
    without another future use streams a bounded number of output rows.
    The small-block direct path avoids FFT overhead. W[0] stays implicit.
    fft_rows=None restores the cache-all diagnostic reference and is not
    covered by the default memory planner.
    """

    def __init__(self, history: BoundaryHistory, *, direct_block=16, fft_rows=4):
        if fft_rows is not None and (not isinstance(fft_rows, int) or fft_rows < 1):
            raise ValueError("fft_rows must be a positive integer or None")
        self.history = history
        self.pending = np.zeros((history.plan.steps + 1, history.plan.dofs))
        self.inputs = np.zeros_like(self.pending)
        self.transforms = {}
        self.next_step = 0
        self.direct_block = direct_block
        self.fft_rows = fft_rows
        self.streamed_levels = []

    def past(self, step):
        if step != self.next_step or step > self.history.plan.steps:
            raise ValueError("Online history must be consumed in order, once per accepted step")
        return self.pending[step].copy()

    def append(self, displacement):
        n = self.next_step
        if n > self.history.plan.steps:
            raise ValueError("Online history is already complete")
        value = np.asarray(displacement)
        if value.shape != (self.history.plan.dofs,) or not np.all(np.isfinite(value)):
            raise ValueError("Invalid accepted boundary displacement")
        self.inputs[n] = value
        self.next_step += 1
        # Largest dyadic block ending here whose sibling has not been processed.
        b = (n + 1) & -(n + 1)
        start = n + 1 - b
        count = min(b, len(self.pending) - n - 1)
        if count <= 0:
            return
        if b <= self.direct_block:
            for j in range(b):
                lag = b - j
                self.pending[n + 1 : n + 1 + count] += np.einsum(
                    "kij,j->ki", self.history.weights[lag : lag + count], self.inputs[start + j]
                )
        else:
            source = rfft(self.inputs[start : n + 1], n=2 * b, axis=0)
            # Same-level blocks end at n+2*b. At steps there are no future
            # targets, so only n+2*b < steps can reuse a newly built FFT.
            if (
                self.fft_rows is not None
                and b not in self.transforms
                and n + 2 * b >= self.history.plan.steps
            ):
                self.streamed_levels.append(b)
                for row in range(0, self.history.plan.dofs, self.fft_rows):
                    kernel = self.history.weights[
                        : min(2 * b, len(self.history.weights)), row : row + self.fft_rows
                    ]
                    transform = rfft(kernel, n=2 * b, axis=0)
                    product = np.einsum("kij,kj->ki", transform, source, optimize=False)
                    self.pending[n + 1 : n + 1 + count, row : row + self.fft_rows] += irfft(
                        product, n=2 * b, axis=0
                    )[b : b + count]
                return
            if b not in self.transforms:
                kernel = self.history.weights[: min(2 * b, len(self.history.weights))]
                self.transforms[b] = rfft(kernel, n=2 * b, axis=0)
            product = np.einsum("kij,kj->ki", self.transforms[b], source, optimize=False)
            # Circular aliases only affect indices < b, which are discarded.
            self.pending[n + 1 : n + 1 + count] += irfft(product, n=2 * b, axis=0)[b : b + count]


def build_history(
    mesh: BoundaryMesh,
    material: ElasticMaterial,
    dt,
    steps,
    *,
    quadrature=20,
    alias_tolerance=1e-10,
    max_memory_mb=512,
    progress=None,
    workers=1,
    cache_directory=None,
):
    plan = plan_cq(
        mesh.dofs, dt, steps, alias_tolerance=alias_tolerance, max_memory_mb=max_memory_mb
    )
    cached = None
    if cache_directory is not None:
        digest = hashlib.sha256()
        digest.update(repr((plan, material, quadrature)).encode())
        for value in (mesh.starts, mesh.ends):
            digest.update(value.tobytes())
        for filename in ("cq.py", "operators.py", "kernels.py", "material.py", "mesh.py"):
            digest.update(Path(__file__).with_name(filename).read_bytes())
        directory = Path(cache_directory)
        directory.mkdir(parents=True, exist_ok=True)
        cached = directory / f"{digest.hexdigest()}.npy"
        if cached.is_file():
            weights = np.load(cached, mmap_mode="r", allow_pickle=False)
            if weights.shape != (steps + 1, mesh.dofs, mesh.dofs) or weights.dtype != np.float64:
                raise ValueError("Invalid cached CQ weights")
            if progress is not None:
                progress(1, 1)
            return BoundaryHistory(plan, weights)
    prepared = PreparedBoundaryOperators(mesh, material, quadrature=quadrature)
    weights = convolution_weights(
        lambda s: prepared.evaluate(s).impedance(),
        plan,
        progress=progress,
        workers=workers,
    )
    if cached is not None:
        temporary = cached.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, weights, allow_pickle=False)
        temporary.replace(cached)
    return BoundaryHistory(plan, weights)

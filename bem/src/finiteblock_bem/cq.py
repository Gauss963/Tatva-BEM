"""Dense reference implementation of BDF2 convolution quadrature.

Zeta -> W(delta(zeta)/dt), delta(zeta)=3/2-2*zeta+zeta**2/2.
IFFT on a damped contour generates causal traction/displacement weights.
The complete displacement history is retained. No local radiation-damping
replacement or unreported history truncation is used.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .material import ElasticMaterial
from .mesh import BoundaryMesh
from .operators import assemble_operators


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


def plan_cq(dofs, dt, steps, *, alias_tolerance=1e-10, max_memory_mb=512):
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
    # Spectrum, complex IFFT result, retained real weights, plus kernel/LU work.
    peak = 2 * fft_size * dofs * dofs * 16 + weight_bytes + 16 * dofs * dofs * 16
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


def convolution_weights(transfer: Callable, plan: CQPlan, *, progress=None):
    """Sample a real causal transfer function, using conjugate symmetry."""
    spectrum = np.empty((plan.fft_size, plan.dofs, plan.dofs), dtype=complex)
    half = plan.fft_size // 2
    for k in range(half + 1):
        zeta = plan.radius * np.exp(-2j * np.pi * k / plan.fft_size)
        value = transfer(bdf2_symbol(zeta) / plan.dt)
        if np.shape(value) != (plan.dofs, plan.dofs) or not np.all(np.isfinite(value)):
            raise FloatingPointError("Nonfinite or incorrectly sized Laplace transfer matrix")
        spectrum[k] = value
        if k not in (0, half):
            spectrum[-k] = np.conjugate(value)
        if progress is not None:
            progress(k + 1, half + 1)
    # Positive real s at k=0 and Nyquist must produce real matrices.
    for k in (0, half):
        if np.linalg.norm(spectrum[k].imag) > 1e-10 * max(np.linalg.norm(spectrum[k].real), 1):
            raise FloatingPointError("Real-axis transfer violates conjugate symmetry")
        spectrum[k] = spectrum[k].real
    transformed = np.fft.ifft(spectrum, axis=0)[: plan.steps + 1]
    amplification = plan.radius ** (-np.arange(plan.steps + 1))
    weights = transformed.real * amplification[:, None, None]
    imaginary = np.max(np.abs(transformed.imag * amplification[:, None, None]))
    if imaginary > 1e-10 * max(np.max(np.abs(weights)), 1):
        raise FloatingPointError("CQ inversion has a significant imaginary remainder")
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
):
    plan = plan_cq(
        mesh.dofs, dt, steps, alias_tolerance=alias_tolerance, max_memory_mb=max_memory_mb
    )
    weights = convolution_weights(
        lambda s: assemble_operators(mesh, material, s, quadrature=quadrature).impedance(),
        plan,
        progress=progress,
    )
    return BoundaryHistory(plan, weights)

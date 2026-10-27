"""Fresh raw-layer sampling of the two Radau modes, one branch at a time.

Each branch keeps the original raw-family, tail and independent output
checks. Both branches are needed before forming the two compact families;
individual branch half spectra do not generally have real endpoints.
The BDF2 parameter used internally by the original sampler is only an
invertible coordinate label. No BDF2 operator value is reused here.
"""

import runpy
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
RAW = runpy.run_path(str(TOOLS / "raw_layer_sampling.py"))


def lower_half_radau_modes(zeta, dt):
    """Continuous mode labels on the lower half of a CQ circle.

    det(w I-Delta(z)) = w^2-(4+2z)w+6(1-z). Choose the square root
    with nonpositive imaginary part on this half circle. The labels need
    not extend separately to analytic functions on the complete unit disk.
    """
    zeta = np.asarray(zeta, dtype=complex)
    if (
        not np.all(np.isfinite(zeta))
        or np.any(abs(zeta) >= 1)
        or np.any(zeta.imag > 1e-13)
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("Require lower-half unit-disk nodes and a positive time step")
    discriminant = zeta * zeta + 10 * zeta - 2
    root = np.sqrt(discriminant)
    root = np.where(root.imag > 0, -root, root)
    if np.any(abs(root) < 1e-12):
        raise ValueError("Nearly coincident Radau modes require a derivative evaluator")
    fast = 2 + zeta + root
    # Product of the eigenvalues avoids cancellation in 3-sqrt(9) near z=1.
    slow = 6 * (1 - zeta) / fast
    values = np.stack((slow, fast), axis=-1) / dt
    if not np.all(np.isfinite(values)) or np.any(values.real <= 0):
        raise ValueError("Radau sampling modes must remain in the right half plane")
    return values


def node_from_sampler_parameter(parameter, plan):
    """Recover z from the original scalar sampler label delta_BDF2(z)/dt."""
    parameter = complex(parameter)
    if not np.isfinite(parameter) or parameter.real <= 0:
        raise ValueError("Invalid original sampler coordinate label")
    root = np.sqrt(1 + 2 * plan.dt * parameter)
    zeta = (3 - 2 * plan.dt * parameter) / (2 + root)
    if abs(abs(zeta) - plan.radius) > 2e-12:
        raise ValueError("Sampler parameter left its declared CQ circle")
    return zeta


def sample_radau_branch(
    raw_provider,
    transform,
    plan,
    branch,
    consume,
    *,
    options=None,
    workers=1,
    max_workspace_mb=2048,
    progress=None,
    diagnostics=None,
    sampler=None,
):
    """Sample actual BEM matrices at one physical Radau mode per grid point.

    Call sequentially for branch 0 and 1 to keep one branch's interpolation
    workspace live. Geometry may be shared read-only. The caller must retain
    both complete branch spectra before constructing compact time weights.
    """
    if not isinstance(branch, int) or isinstance(branch, bool) or branch not in [0, 1]:
        raise ValueError("Choose Radau branch 0 or 1")
    if not np.isfinite(plan.dt) or plan.dt <= 0 or not 0 < plan.radius < 1:
        raise ValueError("Invalid CQ time step or radius")

    def frequency(parameter):
        return complex(
            lower_half_radau_modes(node_from_sampler_parameter(parameter, plan), plan.dt)[branch]
        )

    def provider(parameter):
        return raw_provider(frequency(parameter))

    def output(packet, parameter):
        return transform(packet, frequency(parameter))

    def emit(index, parameter, value):
        consume(index, frequency(parameter), value)

    records = (RAW["sample_raw_layers"] if sampler is None else sampler)(
        provider,
        output,
        plan,
        emit,
        options=options,
        workers=workers,
        max_workspace_mb=max_workspace_mb,
        progress=progress,
        diagnostics=diagnostics,
    )
    return {
        **records,
        "mode_branch": branch,
        "time_method": "radau2",
        "source_frequency_values_reused": False,
        "branch_spectrum_is_not_a_real_kernel": True,
    }


def compact_coefficients(zeta, first, second):
    """Combine complete branch values into F0+Delta(zeta)*F1.

    Coefficients multiply the dimensionless RK symbol, matching the existing
    compact history/inversion API. This is not an IFFT of either branch alone.
    """
    first, second = np.asarray(first), np.asarray(second)
    values = lower_half_radau_modes(zeta, 1.0)
    if (
        first.ndim < 2
        or first.shape != second.shape
        or first.shape[-1] != first.shape[-2]
        or first.shape[:-2] != values.shape[:-1]
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError("Require aligned finite branch matrices and contour nodes")
    a, b = values[..., 0, None, None], values[..., 1, None, None]
    return (b * first - a * second) / (b - a), (second - first) / (b - a)

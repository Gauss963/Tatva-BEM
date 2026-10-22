"""Independent analytic finite-bar wave-arrival and reflection benchmark."""

from dataclasses import dataclass

import numpy as np

from .cq import build_history
from .linear import solve_mixed
from .material import ElasticMaterial
from .mesh import polygon_boundary


def smooth_step(time, rise_time):
    time = np.asarray(time)
    if rise_time <= 0:
        raise ValueError("Use a positive rise time")
    phase = np.clip(time / rise_time, 0, 1)
    return (1 - np.cos(np.pi * phase)) / 2


def integrated_smooth_step(time, rise_time):
    time = np.maximum(np.asarray(time), 0)
    phase = np.minimum(time, rise_time)
    return (phase - rise_time / np.pi * np.sin(np.pi * phase / rise_time)) / 2 + np.maximum(
        time - rise_time, 0
    )


def bar_displacement_exact(time, x, *, length=1.0, speed=1.0, impedance=1.0, rise_time=0.3):
    """Unit left-face pressure; fixed right end, lambda=0 lateral-free strip.

    The Laplace transfer is sinh(k*(L-x))/(Z*s*cosh(k*L)). Expanding the
    denominator gives delayed ramps with alternating reflected wave signs.
    """
    time = np.asarray(time)
    result = np.zeros_like(time, dtype=float)
    reflections = int(np.ceil(np.max(time, initial=0) * speed / (2 * length))) + 1
    for m in range(reflections):
        first = (x + 2 * m * length) / speed
        second = ((2 * m + 2) * length - x) / speed
        result += (
            (-1) ** m
            * (
                integrated_smooth_step(time - first, rise_time)
                - integrated_smooth_step(time - second, rise_time)
            )
            / impedance
        )
    return result


@dataclass(frozen=True)
class BarBenchmark:
    time: np.ndarray
    loading_displacement: np.ndarray
    loading_exact: np.ndarray
    probe_displacement: np.ndarray
    probe_exact: np.ndarray
    probe_x: float
    loading_relative_l2: float
    probe_relative_l2: float
    precursor_relative_max: float
    boundary_residual: float
    panels: int


def finite_bar_benchmark(
    *, h=0.1, dt=0.02, duration=3.6, quadrature=20, max_memory_mb=512, progress=None
):
    material = ElasticMaterial(1.0, 0.0, 1.0)
    mesh = polygon_boundary(
        [[0, 0], [1, 0], [1, 0.5], [0, 0.5]], ["bottom", "fixed", "top", "loaded"], h
    )
    steps = round(duration / dt)
    kernel = build_history(
        mesh,
        material,
        dt,
        steps,
        quadrature=quadrature,
        max_memory_mb=max_memory_mb,
        progress=progress,
    )
    time = np.arange(steps + 1) * dt
    applied = np.zeros((steps + 1, mesh.dofs))
    loaded = mesh.indices("loaded")
    applied[:, 2 * loaded] = smooth_step(time, 0.3)[:, None]
    prescribed = {
        int(d): np.zeros(steps + 1) for i in mesh.indices("fixed") for d in (2 * i, 2 * i + 1)
    }
    result = solve_mixed(kernel, prescribed, applied)
    displacement = np.mean(result.displacement[:, 2 * loaded], axis=1)
    exact = bar_displacement_exact(time, 0.0)
    top = mesh.indices("top")
    probe = top[np.argmin(abs(mesh.centers[top, 0] - 0.5))]
    probe_x = float(mesh.centers[probe, 0])
    probe_u = result.displacement[:, 2 * probe]
    probe_exact = bar_displacement_exact(time, probe_x)
    precursor = time < probe_x - 3 * dt
    return BarBenchmark(
        time,
        displacement,
        exact,
        probe_u,
        probe_exact,
        probe_x,
        float(np.linalg.norm(displacement - exact) / np.linalg.norm(exact)),
        float(np.linalg.norm(probe_u - probe_exact) / np.linalg.norm(probe_exact)),
        float(np.max(abs(probe_u[precursor]), initial=0) / max(abs(probe_exact))),
        result.max_boundary_residual,
        mesh.count,
    )

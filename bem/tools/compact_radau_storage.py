"""Bounded row inversion and complete causal startup for two Radau families."""

import mmap
import runpy
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.fft import irfft

TOOLS = Path(__file__).resolve().parent
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))
REPLAY = runpy.run_path(str(TOOLS / "replay_resolvent.py"))


def compact_rows(eigenvalues, values):
    """Convert sampled eigenmode rows to the two dimensionless coefficients."""
    eigenvalues, values = np.asarray(eigenvalues), np.asarray(values)
    if (
        eigenvalues.ndim != 2
        or eigenvalues.shape[1] != 2
        or values.ndim != 3
        or values.shape[:2] != eigenvalues.shape
        or not np.all(np.isfinite(eigenvalues))
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Expected finite pairs of eigenvalues and transfer rows")
    gap = eigenvalues[:, 1] - eigenvalues[:, 0]
    if np.any(abs(gap) <= 1e-12 * np.maximum(1, np.max(abs(eigenvalues), axis=1))):
        raise ValueError("Nearly coincident Radau modes require a derivative evaluator")
    a, b = eigenvalues[:, 0, None], eigenvalues[:, 1, None]
    return (b * values[:, 0] - a * values[:, 1]) / gap[:, None], (
        values[:, 1] - values[:, 0]
    ) / gap[:, None]


def create_array_file(path, shape, dtype):
    """Create an exclusive NPY allocation without mapping its entire contents."""
    with Path(path).open("xb") as stream:
        np.lib.format.write_array_header_2_0(
            stream,
            dict(descr=np.dtype(dtype).str, fortran_order=False, shape=tuple(shape)),
        )
        stream.truncate(stream.tell() + int(np.prod(shape)) * np.dtype(dtype).itemsize)


def invert_families_to_files(f0, f1, plan, folder, *, fft_rows=4, max_workspace_mb=2048):
    """Write all requested lags and return the exact finite-DFT shift wrap.

    Spectra may be bounded readers. No full real time-weight array or expanded
    stage matrix is allocated. Workspace is an array estimate, not an RSS cap.
    """
    if not isinstance(fft_rows, int) or isinstance(fft_rows, bool) or fft_rows < 1:
        raise ValueError("FFT rows must be a positive integer")
    if not np.isfinite(max_workspace_mb) or max_workspace_mb <= 0:
        raise ValueError("Workspace budget must be finite and positive")
    if not 0 < plan.radius < 1 or not 1 <= plan.steps < plan.fft_size:
        raise ValueError("Invalid damped CQ contour or requested lag count")
    shape = (plan.fft_size // 2 + 1, plan.dofs, plan.dofs)
    if f0.shape != shape or f1.shape != shape:
        raise ValueError("Compact spectra do not match the CQ plan")
    rows = min(fft_rows, plan.dofs)
    estimate = 4 * plan.fft_size * rows * plan.dofs * 8
    estimate += 2 * shape[0] * rows * plan.dofs * 16
    estimate += (plan.steps + 1) * rows * plan.dofs * 8 + 4 * plan.dofs**2 * 8
    if estimate > max_workspace_mb * 1024**2:
        raise ValueError(f"Compact IFFT array estimate is {estimate / 1024**2:.1f} MiB")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    required = 2 * (plan.steps + 1) * plan.dofs**2 * 8 + 1024**2
    if shutil.disk_usage(folder).free < required:
        raise ValueError("Insufficient disk space for both complete compact weight families")
    paths = [folder / f"weights{i}.npy" for i in range(2)]
    if any(path.exists() for path in paths):
        raise ValueError("Choose unused compact weight paths")
    for path in paths:
        create_array_file(path, (plan.steps + 1, plan.dofs, plan.dofs), np.float64)
    amplification = plan.radius ** (-np.arange(plan.steps + 1))
    wrap = np.empty((plan.dofs, plan.dofs))
    for family, spectrum in enumerate([f0, f1]):
        for row in range(0, plan.dofs, rows):
            values = np.array(spectrum[:, row : row + rows], copy=True)
            if not np.all(np.isfinite(values)):
                raise ValueError("Nonfinite compact spectrum")
            for endpoint in [0, len(values) - 1]:
                if np.linalg.norm(values[endpoint].imag) > 1e-10 * max(
                    np.linalg.norm(values[endpoint].real), 1
                ):
                    raise ValueError("Compact endpoint violates conjugate symmetry")
                values[endpoint] = values[endpoint].real
            transformed = irfft(values, n=plan.fft_size, axis=0)
            if family == 1:
                wrap[row : row + rows] = plan.radius * transformed[-1]
            target = np.load(paths[family], mmap_mode="r+", allow_pickle=False)
            try:
                target[:, row : row + rows] = (
                    transformed[: plan.steps + 1] * amplification[:, None, None]
                )
                target.flush()
                if hasattr(target._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                    target._mmap.madvise(mmap.MADV_DONTNEED)
            finally:
                target._mmap.close()
            del target, values, transformed
    return (
        paths,
        wrap,
        dict(
            estimated_ifft_array_peak_mib=estimate / 1024**2,
            fft_rows=rows,
            weight_family_bytes=required - 1024**2,
            complete_requested_lags_retained=True,
            expanded_stage_time_weights_stored=False,
        ),
    )


def causal_compact_startup(f0_column, f1_column, wrap_column, plan, stage_force):
    """Complete causal stage response, including the finite-circle wrap.

    First invert each scalar-input family for the full horizon. Directly
    multiplying the sampled frequency response by a finite forcing FFT would
    add response from future inputs. The existing causal helper avoids it.
    """
    force = np.asarray(stage_force)
    shape = (plan.fft_size // 2 + 1, plan.dofs)
    if (
        f0_column.shape != shape
        or f1_column.shape != shape
        or np.shape(wrap_column) != (plan.dofs,)
        or force.ndim != 2
        or force.shape[1] != 2
        or not 2 <= len(force) <= plan.fft_size // 2
        or not all(np.all(np.isfinite(v)) for v in [f0_column, f1_column, wrap_column, force])
    ):
        raise ValueError("Invalid complete contour columns, wrap or stage forcing")
    full_plan = replace(plan, steps=len(force) - 1)
    causal = REPLAY["causal_elastic_baseline"]
    first, second = [
        np.stack([causal(column, full_plan, force[:, stage]) for stage in range(2)], axis=1)
        for column in [f0_column, f1_column]
    ]
    delta0 = np.linalg.inv(RADAU["A"])
    delta1 = -np.outer(delta0 @ np.ones(2), [0, 1])
    shifted = np.zeros_like(second)
    shifted[1:] = second[:-1]
    # z*F1 has a lag-zero finite-DFT alias as well as its one-lag shift.
    shifted += force[:, :, None] * np.asarray(wrap_column)[None, None, :]
    return (
        first + np.einsum("ab,nbd->nad", delta0, second) + np.einsum("ab,nbd->nad", delta1, shifted)
    )

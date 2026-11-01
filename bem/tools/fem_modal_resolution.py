"""Mass-consistent constrained elastic modes and native-path spectral diagnostics.

These are frozen elastic controls and saved-field analyses. They do not prove
stability or convergence of a changing frictional active set.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def constrained_mass_map(mass, fixed, rigid_face):
    """One coordinate per free DOF, with one common rigid-platen coordinate."""
    mass = np.asarray(mass, float)
    fixed, face = (np.asarray(a, dtype=np.int64) for a in [fixed, rigid_face])
    n = len(mass)
    if (
        mass.ndim != 1
        or np.any(~np.isfinite(mass))
        or np.any(mass <= 0)
        or any(np.any((a < 0) | (a >= n)) for a in [fixed, face])
        or len(np.unique(face)) != len(face)
        or np.intersect1d(fixed, face).size
    ):
        raise ValueError("Require positive masses and disjoint valid supports/platen DOFs")
    free = np.setdiff1d(np.arange(n), np.r_[fixed, face])
    rows = np.r_[free, face]
    columns = np.r_[np.arange(len(free)), np.full(len(face), len(free))]
    size = len(free) + bool(len(face))
    transform = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(n, size))
    reduced_mass = np.asarray(transform.T @ mass).ravel()
    if size < 2 or np.any(reduced_mass <= 0):
        raise ValueError("Require at least two positive reduced masses")
    return transform, reduced_mass


def interface_jump_matrices(total, master, slave, moving_offset):
    master, slave = (np.asarray(a, dtype=np.int64) for a in [master, slave])
    if (
        master.ndim != 1
        or master.shape != slave.shape
        or np.any(master < 0)
        or np.any(slave < 0)
        or np.any(2 * master + 1 >= moving_offset)
        or np.any(moving_offset + 2 * slave + 1 >= total)
    ):
        raise ValueError("Invalid two-body contact DOF map")
    row = np.repeat(np.arange(len(master)), 2)
    result = []
    for component in [0, 1]:
        col = np.column_stack(
            (2 * master + component, moving_offset + 2 * slave + component)
        ).ravel()
        result.append(
            sparse.csr_matrix(
                (np.tile([1.0, -1.0], len(master)), (row, col)), shape=(len(master), total)
            )
        )
    return tuple(result)


def largest_elastic_modes(stiffness, mass, *, count=6, tolerance=1e-10):
    """Solve M^-1/2 K M^-1/2; independently gate symmetry and Ritz residuals."""
    mass = np.asarray(mass, float)
    stiffness = sparse.csr_matrix(stiffness)
    if (
        mass.ndim != 1
        or stiffness.shape != (len(mass), len(mass))
        or np.any(~np.isfinite(mass))
        or np.any(mass <= 0)
        or not 1 <= count < len(mass)
        or not 0 < tolerance <= 1e-6
    ):
        raise ValueError("Require a finite positive mass and a bounded eigenproblem")
    norm = np.linalg.norm(stiffness.data)
    difference = (stiffness - stiffness.T).tocsr()
    symmetry = np.linalg.norm(difference.data) / norm if norm else np.inf
    if not np.all(np.isfinite(stiffness.data)) or not np.isfinite(symmetry) or symmetry > 1e-12:
        raise ValueError("Frozen elastic control is not symmetric to its declared tolerance")
    # Exact energy Hessians are symmetric. Record the roundoff-only correction.
    symmetric = (stiffness + stiffness.T) * 0.5
    scale = sparse.diags(1 / np.sqrt(mass))
    operator = (scale @ symmetric @ scale).tocsr()
    start = np.random.default_rng(904).normal(size=len(mass))
    values, vectors = eigsh(operator, k=count, which="LA", v0=start, tol=tolerance, maxiter=10000)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    residual = np.linalg.norm(operator @ vectors - vectors * values[None], axis=0)
    relative = residual / abs(values)
    orthogonality = float(np.linalg.norm(vectors.T @ vectors - np.eye(count), np.inf))
    if (
        np.any(values <= 0)
        or np.any(~np.isfinite(relative))
        or np.max(relative) > 1e-8
        or not np.isfinite(orthogonality)
        or orthogonality > 1e-10
    ):
        raise ValueError("Positive modal frequency/Ritz residual gate failed")
    return (
        values,
        vectors / np.sqrt(mass)[:, None],
        dict(
            relative_symmetry_defect=float(symmetry),
            symmetric_roundoff_correction=True,
            relative_ritz_residuals=relative.tolist(),
            mass_orthogonality_inf_error=orthogonality,
        ),
    )


def leapfrog_phase_control(omega, dt, duration):
    """Exact dispersion of the conservative scalar leapfrog oscillator."""
    if not all(np.isfinite(v) and v > 0 for v in [omega, dt, duration]):
        raise ValueError("Require positive finite frequency, step and duration")
    argument = dt * omega
    # The repeated eigenvalue at h*omega=2 has a Jordan block: strict inequality
    # is needed for bounded evolution of arbitrary oscillator initial data.
    stable = argument < 2
    numerical = 2 * np.arcsin(argument / 2) / dt if stable else None
    return dict(
        omega_dt=float(argument),
        inside_conservative_oscillator_stability_interval=bool(stable),
        numerical_omega_rad_per_s=None if numerical is None else float(numerical),
        phase_error_over_duration_rad=None
        if numerical is None
        else float((numerical - omega) * duration),
        nonlinear_friction_stability_certificate=False,
    )


def spatial_p1_mass(y):
    y = np.asarray(y, float)
    if y.ndim != 1 or len(y) < 2 or not np.all(np.isfinite(y)) or np.any(np.diff(y) <= 0):
        raise ValueError("Require distinct ordered spatial knots")
    h = np.diff(y)
    diagonal = np.r_[h, 0] / 3 + np.r_[0, h] / 3
    return sparse.diags([h / 6, diagonal, h / 6], [-1, 0, 1], format="csr")


def native_normal_jump_spectrum(time, y, values):
    """Full spatial P1 power of a Hann-windowed, mean-removed native time path.

    This is sampled displacement power, not mechanical energy, a continuous
    norm bound, or a filtered replacement for any FEM/BEM acceptance metric.
    """
    time, values = np.asarray(time, float), np.asarray(values, float)
    if (
        time.ndim != 1
        or len(time) < 8
        or values.shape != (len(time), len(y))
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(time))
    ):
        raise ValueError("Require complete finite time-by-contact fields")
    dt = float(np.mean(np.diff(time)))
    if dt <= 0 or np.max(abs(np.diff(time) - dt)) > 1e-5 * dt:
        raise ValueError("Spectral diagnostic requires uniform native times")
    mass = spatial_p1_mass(y)
    window = np.hanning(len(time))
    mean = np.sum(window[:, None] * values, axis=0) / np.sum(window)
    sample = window[:, None] * (values - mean)
    fourier = np.fft.rfft(sample, axis=0)
    power = np.real(np.sum(fourier.conj() * (mass @ fourier.T).T, axis=1))
    if np.min(power) < -1e-12 * np.max(power):
        raise ValueError("Spatial spectral power lost positivity")
    power = np.maximum(power, 0)
    multiplicity = np.full(len(power), 2.0)
    multiplicity[0] = 1
    if len(time) % 2 == 0:
        multiplicity[-1] = 1
    power *= multiplicity / len(time) ** 2
    sampled_norm = float(np.sum(sample * (mass @ sample.T).T) / len(time))
    parseval = float(abs(np.sum(power) - sampled_norm) / max(sampled_norm, 1e-300))
    if parseval > 1e-11 or np.sum(power) <= 0:
        raise ValueError("Positive fluctuation/Parseval gate failed")
    frequency = np.fft.rfftfreq(len(time), dt)
    return (
        frequency,
        power,
        dict(
            native_dt_s=dt,
            native_samples=len(time),
            spatial_knots=len(y),
            hann_window=True,
            weighted_temporal_mean_removed=True,
            relative_parseval_error=parseval,
            frequency_resolution_hz=float(frequency[1]),
            peak_frequency_hz=float(frequency[np.argmax(power)]),
            spectrum_scope="Full-interface spatial P1 displacement power of windowed native samples; not physical energy or a continuous-time error bound.",
        ),
    )

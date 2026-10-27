"""Interpolate raw Galerkin layers before the finite-body eliminations.

Each V/H/D family gets its own empirical relative/tail check. Independent
condensed-output checks also guard amplification through the solves. This
is frequency-only Chebyshev interpolation, not 3D-ACA or an error theorem.
Every FFT grid point is transformed and emitted; unresolved intervals use
fresh raw operators. No time history, matrix rank or contact mode is removed.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.fft import dct
from scipy.interpolate import BarycentricInterpolator
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import bdf2_symbol
from finiteblock_bem.galerkin import GalerkinOperators


@dataclass(frozen=True)
class RawLayerPacket:
    families: tuple
    masses: tuple

    @classmethod
    def from_operators(cls, operators):
        operators = tuple(operators)
        return cls(
            tuple(
                a
                for op in operators
                for a in (op.single_layer, op.trace_double_layer, op.hypersingular)
            ),
            tuple(op.mixed_mass for op in operators),
        )

    def operators(self):
        return tuple(
            GalerkinOperators(*self.families[3 * i : 3 * i + 3], mass)
            for i, mass in enumerate(self.masses)
        )

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.families)

    def validate(self, reference=None):
        if not self.masses or len(self.families) != 3 * len(self.masses):
            raise ValueError("Require three raw layers and one fixed mass per body")
        for i, mass in enumerate(self.masses):
            if mass.ndim != 2 or mass.shape[0] != mass.shape[1] or not np.all(np.isfinite(mass)):
                raise ValueError("Invalid fixed mixed mass")
            for a in self.families[3 * i : 3 * i + 3]:
                if a.shape != mass.shape or not np.all(np.isfinite(a)):
                    raise ValueError("Raw layer matrices must be finite and match the body mass")
        if reference is not None:
            if len(self.masses) != len(reference.masses):
                raise ValueError("Raw body count changed")
            if any(
                not np.array_equal(a, b) for a, b in zip(self.masses, reference.masses, strict=True)
            ):
                raise ValueError("Fixed raw mixed mass changed")


@dataclass(frozen=True)
class RawSamplingOptions:
    degree: int = 16
    check_points: int = 7
    raw_rtol: float = 1e-11
    raw_atol: float = 1e-14
    output_rtol: float = 1e-9
    output_atol: float = 1e-14

    def __post_init__(self):
        if not isinstance(self.degree, int) or not 8 <= self.degree <= 64:
            raise ValueError("Chebyshev degree must be between 8 and 64")
        if not isinstance(self.check_points, int) or self.check_points < 3:
            raise ValueError("At least three independent check points required")
        if not all(np.isfinite(v) and v > 0 for v in (self.raw_rtol, self.output_rtol)):
            raise ValueError("Positive finite relative tolerances required")
        if not all(np.isfinite(v) and v >= 0 for v in (self.raw_atol, self.output_atol)):
            raise ValueError("Finite nonnegative absolute tolerances required")


def sample_raw_layers(
    raw,
    transform,
    plan,
    consume,
    *,
    options=None,
    workers=1,
    max_workspace_mb=2048,
    progress=None,
    diagnostics=None,
):
    """Emit (grid index, s, transformed matrix), using bounded piece storage."""
    options = RawSamplingOptions() if options is None else options
    if (
        not isinstance(workers, int)
        or workers < 1
        or not np.isfinite(max_workspace_mb)
        or max_workspace_mb <= 0
    ):
        raise ValueError("Positive workers and finite raw workspace required")
    half = plan.fft_size // 2
    angles = np.linspace(0, np.pi, half + 1)

    def frequency(angle):
        return bdf2_symbol(plan.radius * np.exp(-1j * angle)) / plan.dt

    reference = raw(frequency(0.0))
    reference.validate()
    estimated = (
        4 * (options.degree + 1) + 3 * options.check_points + 2 * workers + 4
    ) * reference.nbytes
    if estimated > max_workspace_mb * 1024**2:
        raise ValueError(f"Raw interpolation arrays need about {estimated / 1024**2:.1f} MiB")
    records = dict(
        method="raw_layers_before_condensation",
        source_evaluations=1,
        transformed_grid_points=0,
        direct_grid_points=0,
        interpolated_grid_points=0,
        accepted_intervals=0,
        rejected_intervals=0,
        maximum_raw_normalized_check=0.0,
        maximum_raw_normalized_tail=0.0,
        maximum_output_normalized_check=0.0,
        estimated_interpolation_array_bytes=estimated,
        provider_geometry_and_solver_workspace_excluded=True,
        certified_error_bound=False,
        all_fft_grid_points_emitted=False,
    )
    degree = options.degree
    nodes = np.cos(np.arange(degree + 1) * np.pi / degree)
    wi = (-1.0) ** np.arange(degree + 1)
    wi[[0, -1]] *= 0.5
    checks = 2 * (np.arange(options.check_points) + 0.3819660112501051) / options.check_points - 1

    def evaluate(angle):
        if angle == 0.0:
            return reference
        packet = raw(frequency(angle))
        packet.validate(reference)
        return packet

    def output(args):
        angle, packet = args
        value = np.asarray(transform(packet, frequency(angle)))
        if value.shape != (plan.dofs, plan.dofs) or not np.all(np.isfinite(value)):
            raise ValueError("Invalid condensed raw-layer output")
        return value

    with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(workers) as pool:

        def samples(theta):
            values = []
            for first in range(0, len(theta), workers):
                chunk = theta[first : first + workers]
                values.extend(pool.map(evaluate, chunk))
                records["source_evaluations"] += int(sum(t != 0.0 for t in chunk))
                if diagnostics is not None:
                    diagnostics(dict(records))
            return values

        def emit(left, theta, packets):
            values = list(pool.map(output, zip(theta, packets, strict=True)))
            for j, (angle, value) in enumerate(zip(theta, values, strict=True)):
                consume(left + j, frequency(angle), value)
            records["transformed_grid_points"] += len(values)

        stack = [(0, half + 1)]
        while stack:
            left, right = stack.pop()
            count = right - left
            if count <= 2 * (degree + 1 + options.check_points):
                for first in range(left, right, workers):
                    theta = angles[first : min(first + workers, right)]
                    emit(first, theta, samples(theta))
                records["direct_grid_points"] += count
            else:
                center = (angles[left] + angles[right - 1]) / 2
                radius = (angles[right - 1] - angles[left]) / 2
                exact_nodes = samples(center + radius * nodes)
                check_angles = center + radius * checks
                exact = samples(check_angles)
                polynomials = []
                error = 0.0
                tail = 0.0
                for family in range(len(reference.families)):
                    values = np.stack([p.families[family] for p in exact_nodes])
                    test = np.stack([p.families[family] for p in exact])
                    poly = BarycentricInterpolator(nodes, values, wi=wi, axis=0)
                    scale = max(
                        np.linalg.norm(values, axis=(-2, -1)).max(),
                        np.linalg.norm(test, axis=(-2, -1)).max(),
                    )
                    tolerance = max(
                        options.raw_atol + options.raw_rtol * scale, np.finfo(float).tiny
                    )
                    error = max(
                        error,
                        float(np.linalg.norm(poly(checks) - test, axis=(-2, -1)).max() / tolerance),
                    )
                    coefficients = dct(values, type=1, axis=0) / degree
                    coefficients[[0, -1]] *= 0.5
                    tail = max(
                        tail,
                        float(np.linalg.norm(coefficients[-4:], axis=(-2, -1)).sum() / tolerance),
                    )
                    polynomials.append(poly)
                    del coefficients, values, test
                output_error = np.inf
                if error <= 1 and tail <= 0.25:
                    predicted = [poly(checks) for poly in polynomials]
                    packets = [
                        RawLayerPacket(tuple(a[j] for a in predicted), reference.masses)
                        for j in range(len(checks))
                    ]
                    actual_values = list(pool.map(output, zip(check_angles, exact, strict=True)))
                    predicted_values = list(
                        pool.map(output, zip(check_angles, packets, strict=True))
                    )
                    output_error = max(
                        float(
                            np.linalg.norm(a - b)
                            / max(
                                options.output_atol + options.output_rtol * np.linalg.norm(a),
                                np.finfo(float).tiny,
                            )
                        )
                        for a, b in zip(actual_values, predicted_values, strict=True)
                    )
                    del packets, predicted, actual_values, predicted_values
                if error <= 1 and tail <= 0.25 and output_error <= 1:
                    for first in range(left, right, workers):
                        theta = angles[first : min(first + workers, right)]
                        coordinates = (theta - center) / radius
                        values = [p(coordinates) for p in polynomials]
                        packets = [
                            RawLayerPacket(tuple(a[j] for a in values), reference.masses)
                            for j in range(len(theta))
                        ]
                        emit(first, theta, packets)
                        del values, packets
                    records["accepted_intervals"] += 1
                    records["interpolated_grid_points"] += count
                    records["maximum_raw_normalized_check"] = max(
                        records["maximum_raw_normalized_check"], error
                    )
                    records["maximum_raw_normalized_tail"] = max(
                        records["maximum_raw_normalized_tail"], tail
                    )
                    records["maximum_output_normalized_check"] = max(
                        records["maximum_output_normalized_check"], output_error
                    )
                else:
                    records["rejected_intervals"] += 1
                    split = (left + right) // 2
                    stack.extend([(split, right), (left, split)])
                del polynomials, exact_nodes, exact
            if progress is not None:
                progress(records["transformed_grid_points"], half + 1)
            if diagnostics is not None:
                diagnostics(dict(records))
    if records["transformed_grid_points"] != half + 1:
        raise ValueError("Raw sampling did not emit the complete frequency grid")
    records["all_fft_grid_points_emitted"] = True
    records["raw_evaluation_ratio"] = records["source_evaluations"] / (half + 1)
    return records

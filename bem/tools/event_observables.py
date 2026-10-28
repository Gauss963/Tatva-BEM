"""Sampling brackets for monotone cumulative slip and fixed-gauge diagnostics.

These intervals account for saved time spacing and optional float32 storage
rounding. They do not bound PDE, spatial discretization, integration, unsaved
oscillation, or floating-point evaluation errors. No fitted time shifts.
"""

import numpy as np


def validate_field(time, values):
    time, values = np.asarray(time, dtype=float), np.asarray(values, dtype=float)
    if (
        time.ndim != 1
        or len(time) < 2
        or values.ndim != 2
        or values.shape[0] != len(time)
        or values.shape[1] < 1
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(values))
        or np.any(np.diff(time) <= 0)
        or time[0] < 0
    ):
        raise ValueError("Require finite fields on a strictly increasing nonnegative clock")
    return time, values


def storage_bounds(values):
    """Round-to-nearest cells for f32; f64 denotes the saved numerical values."""
    values = np.asarray(values)
    if values.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.all(
        np.isfinite(values)
    ):
        raise ValueError("Require finite float32 or float64 storage")
    current = values.astype(float)
    if values.dtype == np.float64:
        return current, current
    previous = np.nextafter(values, np.float32(-np.inf)).astype(float)
    following = np.nextafter(values, np.float32(np.inf)).astype(float)
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(following)):
        raise ValueError("Cannot bound unbounded storage rounding cells")
    return (previous + current) / 2, (current + following) / 2


def sample_columns(y, values, knots):
    """P1 spatial sampling, including one fixed gauge; never extrapolate."""
    y, values, knots = (np.asarray(v, dtype=float) for v in (y, values, knots))
    if (
        y.ndim != 1
        or len(y) < 2
        or np.any(np.diff(y) <= 0)
        or values.ndim != 2
        or values.shape[1] != len(y)
        or knots.ndim != 1
        or not len(knots)
        or np.any(np.diff(knots) <= 0)
        or not all(np.all(np.isfinite(v)) for v in (y, values, knots))
        or knots[0] < y[0]
        or knots[-1] > y[-1]
    ):
        raise ValueError("Require ordered physical nodes and gauges without extrapolation")
    left = np.clip(np.searchsorted(y, knots, side="right") - 1, 0, len(y) - 2)
    fraction = (knots - y[left]) / (y[left + 1] - y[left])
    return (1 - fraction) * values[:, left] + fraction * values[:, left + 1]


def crossing_brackets(time, values, threshold, *, bounds=None):
    """First strict > crossing, with explicit left/right censoring and no fit."""
    time, values = validate_field(time, values)
    if (
        not np.isfinite(threshold)
        or threshold < 0
        or np.any(values < 0)
        or np.any(np.diff(values, axis=0) < 0)
    ):
        raise ValueError("Require nonnegative monotone cumulative slip and threshold")
    lower, upper = (
        (values, values) if bounds is None else tuple(np.asarray(v, dtype=float) for v in bounds)
    )
    if (
        lower.shape != values.shape
        or upper.shape != values.shape
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower > values)
        or np.any(upper < values)
        or np.any(np.diff(lower, axis=0) < 0)
        or np.any(np.diff(upper, axis=0) < 0)
    ):
        raise ValueError("Require finite ordered monotone value bounds containing observations")
    result = []
    for a, b in zip(lower.T, upper.T, strict=True):
        possible, certain = np.flatnonzero(b > threshold), np.flatnonzero(a > threshold)
        if not len(possible):
            result.append(dict(state="unreached", lower_s=float(time[-1]), upper_s=None))
            continue
        first = int(possible[0])
        lo = None if first == 0 else float(time[first - 1])
        hi = None if not len(certain) else float(time[certain[0]])
        state = "bounded" if lo is not None and hi is not None else "censored"
        result.append(dict(state=state, lower_s=lo, upper_s=hi))
    return result


def first_event(y, brackets):
    """All locations compatible with the earliest native-node crossing."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) != len(brackets) or not np.all(np.isfinite(y)):
        raise ValueError("First-event brackets must match finite native positions")
    lo = [p["lower_s"] for p in brackets]
    hi = [p["upper_s"] for p in brackets if p["upper_s"] is not None]
    lower, upper = None if None in lo else min(lo), min(hi) if hi else None
    possible = [
        float(yy)
        for yy, p in zip(y, brackets, strict=True)
        if p["state"] != "unreached"
        and (upper is None or p["lower_s"] is None or p["lower_s"] <= upper)
    ]
    return dict(
        state="unreached"
        if not possible
        else "bounded"
        if lower is not None and upper is not None
        else "censored",
        lower_s=lower,
        upper_s=upper,
        possible_first_native_locations_mm=possible,
    )


def difference_bounds(reference, actual):
    """Bounds of actual - reference; None retains unbounded censoring."""
    ar, br = reference["lower_s"], reference["upper_s"]
    aa, ba = actual["lower_s"], actual["upper_s"]
    lower = None if aa is None or br is None else aa - br
    upper = None if ba is None or ar is None else ba - ar
    bounded = lower is not None and upper is not None
    minimum = None if not bounded else max(lower, -upper, 0.0)
    maximum = None if not bounded else max(abs(lower), abs(upper))
    return dict(
        signed_difference_interval_s=[lower, upper],
        minimum_absolute_difference_s=minimum,
        maximum_absolute_difference_s=maximum,
    )


def location_difference(reference, actual):
    """Keep ambiguity in earliest native location; do not choose an argmin fit."""
    r, a = (
        np.asarray(p["possible_first_native_locations_mm"], dtype=float)
        for p in (reference, actual)
    )
    if not len(r) or not len(a):
        return dict(minimum_possible_distance_mm=None, maximum_possible_distance_mm=None)
    distance = abs(a[:, None] - r[None, :])
    return dict(
        minimum_possible_distance_mm=float(np.min(distance)),
        maximum_possible_distance_mm=float(np.max(distance)),
    )


def relative_interval_gate(reference, actual, tolerance):
    """Pass only if every pair of times in the brackets meets the tolerance."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Require a finite nonnegative relative tolerance")
    result = dict(
        status="inconclusive",
        relative_tolerance=float(tolerance),
        minimum_possible_relative_error=None,
        maximum_possible_relative_error=None,
        **difference_bounds(reference, actual),
    )
    r, a = (
        [reference[k] for k in ("lower_s", "upper_s")],
        [actual[k] for k in ("lower_s", "upper_s")],
    )
    if None in r or None in a or r[0] <= 0 or a[0] < 0:
        return result
    maximum = max(abs(aa / rr - 1) for aa in a for rr in r)
    minimum = max(a[0] / r[1] - 1, 1 - a[1] / r[0], 0.0)
    result.update(
        status="pass"
        if maximum <= tolerance
        else "fail"
        if minimum > tolerance
        else "inconclusive",
        minimum_possible_relative_error=minimum,
        maximum_possible_relative_error=maximum,
    )
    return result


def gauge_travel(y_a, y_b, arrival_a, arrival_b):
    """Gauge-to-gauge first-passage speed, not instantaneous rupture velocity."""
    if not np.isfinite(y_a) or not np.isfinite(y_b) or y_a >= y_b:
        raise ValueError("Travel gauges must have increasing finite positions")
    difference = difference_bounds(arrival_a, arrival_b)
    lo, hi = difference["signed_difference_interval_s"]
    result = dict(
        gauge_y_mm=[float(y_a), float(y_b)],
        signed_travel_time_interval_s=[lo, hi],
        duration_interval_s=None,
        direction="unresolved",
        speed_interval_m_per_s=None,
    )
    if lo is None or hi is None or lo <= 0 <= hi:
        return result
    duration = [min(abs(lo), abs(hi)), max(abs(lo), abs(hi))]
    distance = (y_b - y_a) / 1000
    result.update(
        duration_interval_s=duration,
        direction="increasing_y" if lo > 0 else "decreasing_y",
        speed_interval_m_per_s=[distance / duration[1], distance / duration[0]],
    )
    return result


def waveform_error(reference_time, reference, actual_time, actual):
    """Exact time L2 of saved P1 signals, with a separate incremental diagnostic."""
    rt, rr = validate_field(reference_time, np.asarray(reference)[:, None])
    at, aa = validate_field(actual_time, np.asarray(actual)[:, None])
    start, stop = max(rt[0], at[0]), min(rt[-1], at[-1])
    if stop <= start:
        raise ValueError("Waveforms need a shared saved time interval")
    union = np.union1d(rt, at)
    knots = np.union1d([start, stop], union[(union > start) & (union < stop)])
    r, a = np.interp(knots, rt, rr[:, 0]), np.interp(knots, at, aa[:, 0])

    def norm2(v):
        return float(np.sum(np.diff(knots) / 3 * (v[:-1] ** 2 + v[:-1] * v[1:] + v[1:] ** 2)))

    def report(target, candidate):
        error2, reference2 = norm2(candidate - target), norm2(target)
        return dict(
            absolute_time_p1_l2=float(np.sqrt(error2)),
            relative_time_p1_l2=None if reference2 == 0 else float(np.sqrt(error2 / reference2)),
            maximum_saved_interpolant_absolute_difference=float(np.max(abs(candidate - target))),
            reference_squared_time_integral=reference2,
        )

    return dict(
        absolute_signal=report(r, a),
        increment_from_first_shared_time=report(r - r[0], a - a[0]),
        common_time_interval_s=[float(start), float(stop)],
        temporal_union_knots=len(knots),
        fitted_time_shift_s=0.0,
        initial_absolute_difference=float(a[0] - r[0]),
        scope="Saved P1 signals only; incremental diagnostic does not replace absolute agreement",
    )

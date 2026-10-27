"""Advisory initial intervals for the unchanged raw-layer acceptance checks.

A cheap scalar Green screen suggests intervals. Every actual matrix interval
still passes the original raw-family, tail and condensed-output checks, or is
subdivided/directly evaluated. This is an experiment, not a new error bound.
The running source sampler is left untouched; only its initial stack is cloned.
"""

import ast
import inspect
import numbers
import textwrap
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.cq import CQPlan
from finiteblock_bem.galerkin import _scalar_kernels

DEFAULT_DISTANCES = (0.015625, 0.125, 1.0, 10.0, 100.0, 500.0)


def validate_partition(intervals, count):
    result = []
    first = 0
    for pair in intervals:
        if (
            len(pair) != 2
            or any(isinstance(v, bool) or not isinstance(v, numbers.Integral) for v in pair)
            or pair[0] != first
            or not first < pair[1] <= count
        ):
            raise ValueError("Initial intervals must cover every FFT index once in order")
        result.append(tuple(int(v) for v in pair))
        first = pair[1]
    if not result or first != count:
        raise ValueError("Initial intervals omit part of the complete FFT grid")
    return tuple(result)


def sampler_with_partition(original, intervals, count):
    """Replace exactly the initial stack value in a private function copy."""
    intervals = validate_partition(intervals, count)
    expected_path = Path(__file__).with_name("raw_layer_sampling.py").resolve()
    if Path(original.__code__.co_filename).resolve() != expected_path:
        raise ValueError("Require the original raw_layer_sampling.py function")
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    before = ast.dump(tree, include_attributes=False)
    matches = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "stack"
    ]
    if len(matches) != 1:
        raise ValueError("Original sampler must have exactly one initial stack assignment")
    target = matches[0]
    expected = ast.parse("[(0, half + 1)]", mode="eval").body
    if ast.dump(target.value, include_attributes=False) != ast.dump(
        expected, include_attributes=False
    ):
        raise ValueError("Original global-bisection stack changed")
    prior = target.value
    replacement = ast.Call(
        func=ast.Name(id="list", ctx=ast.Load()),
        args=[ast.Name(id="_raw_initial_stack", ctx=ast.Load())],
        keywords=[],
    )
    target.value = replacement
    target.value = prior
    if ast.dump(tree, include_attributes=False) != before:
        raise RuntimeError("Partition adapter changed another original statement")
    target.value = replacement
    namespace = {**original.__globals__, "_raw_initial_stack": intervals[::-1]}
    exec(
        compile(ast.fix_missing_locations(tree), str(expected_path) + "|initial-partition", "exec"),
        namespace,
    )
    result = namespace[original.__name__]
    if inspect.signature(result) != inspect.signature(original):
        raise RuntimeError("Partition adapter changed the sampler API")
    return result


def primitive_partition(original, plan, material, *, options=None, distances=DEFAULT_DISTANCES):
    """Learn only a starting partition; never reuse these values as body data."""
    distance = np.asarray(distances, dtype=float)
    if (
        distance.ndim != 1
        or not 1 <= len(distance) <= 64
        or not np.all(np.isfinite(distance))
        or np.any(distance <= 0)
        or not np.isfinite(plan.dt)
        or plan.dt <= 0
        or not np.isfinite(plan.radius)
        or not 0 < plan.radius < 1
        or not isinstance(plan.fft_size, int)
        or plan.fft_size < 2
        or plan.fft_size & (plan.fft_size - 1)
    ):
        raise ValueError("Require a valid FFT contour and 1-64 positive proxy distances")
    namespace = original.__globals__
    options = namespace["RawSamplingOptions"]() if options is None else options
    packet = namespace["RawLayerPacket"]
    n = len(distance)
    mass = np.eye(n)
    vectors = np.column_stack([distance, np.zeros(n)])
    cheap_plan = CQPlan(plan.dt, 1, n, plan.fft_size, plan.radius, 0, 0)

    def raw(s):
        gs, gp, _, gradient = _scalar_kernels(vectors, s, material)
        return packet((np.diag(gs), np.diag(gp), np.diag(gradient[:, 0])), (mass,))

    intervals = []
    emitted = [0]

    def consume(index, *args):
        if index != emitted[0]:
            raise ValueError("Proxy sampler did not emit consecutive FFT indices")
        emitted[0] += 1

    def progress(done, total):
        first = 0 if not intervals else intervals[-1][1]
        if done != emitted[0] or done < first or total != plan.fft_size // 2 + 1:
            raise ValueError("Proxy partition progress differs from its complete emissions")
        if done > first:
            intervals.append((first, done))

    begun = perf_counter()
    record = original(
        raw,
        lambda p, s: mass,
        cheap_plan,
        consume,
        options=options,
        workers=1,
        max_workspace_mb=32,
        progress=progress,
    )
    intervals = validate_partition(intervals, plan.fft_size // 2 + 1)
    return intervals, dict(
        distances_mm=distance.tolist(),
        sampling_options=asdict(options),
        records=record,
        intervals=[list(v) for v in intervals],
        diagnostic_seconds=perf_counter() - begun,
        body_operator_values_reused=False,
        partition_certified_for_body_matrices=False,
        scope="Advisory scalar Green partition; actual raw/output acceptance checks remain mandatory",
    )


def sample_with_partition(original, raw, transform, plan, consume, *, initial_intervals, **kwargs):
    count = plan.fft_size // 2 + 1
    intervals = validate_partition(initial_intervals, count)
    sample = sampler_with_partition(original, intervals, count)
    record = sample(raw, transform, plan, consume, **kwargs)
    record.update(
        initial_intervals=[list(v) for v in intervals],
        initial_partition_only_changed=True,
        original_acceptance_checks_retained=True,
    )
    return record


def sample_with_primitive_partition(
    original, raw, transform, plan, consume, *, material, distances=DEFAULT_DISTANCES, **kwargs
):
    intervals, proxy = primitive_partition(
        original, plan, material, options=kwargs.get("options"), distances=distances
    )
    record = sample_with_partition(
        original, raw, transform, plan, consume, initial_intervals=intervals, **kwargs
    )
    record["primitive_partition_screen"] = proxy
    return record

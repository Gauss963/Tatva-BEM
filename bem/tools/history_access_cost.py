"""Forecast coefficient requests by the existing exact RK history adapters.

This counts logical source-array slices and kernel FFT calls, not physical
disk traffic or runtime. Every source and future target of the original
implementation remains present. Cached FFTs are initially empty, as after
restoring a checkpoint; source-channel FFTs and initialization are excluded.
"""

import runpy
from numbers import Integral
from pathlib import Path

BOUNDED = runpy.run_path(str(Path(__file__).with_name("bounded_history.py")))


def forecast_history_accesses(
    plan, first, last, *, prefix_lags=0, fft_cache_mb=0, fft_rows=4, families=2
):
    """Count inclusive local append indices using the original full CQ plan.

    ``fft_cache_mb`` is shared equally among families, matching
    CompactRadauConvolution. A request crossing the RAM-prefix boundary
    falls back to the complete source slice, matching _PrefixWeights.
    The source arrays may be memory maps; these counts do not model their
    page cache, read amplification, or filesystem readahead.
    """
    values = [plan.steps, plan.dofs, first, last, prefix_lags, fft_rows, families]
    if (
        any(not isinstance(v, Integral) or isinstance(v, bool) for v in values)
        or min(plan.steps, plan.dofs, fft_rows, families) < 1
        or not 0 <= first <= last <= plan.steps
        or prefix_lags < 0
    ):
        raise ValueError("Require a valid full plan, local append interval and cache dimensions")
    levels, _ = BOUNDED["cache_layout"](plan, fft_cache_mb / families)
    prefix = min(prefix_lags, plan.steps + 1)
    retained, updates = set(), {}
    reads = builds = 0
    for index in range(first, last + 1):
        block = (index + 1) & -(index + 1)
        count = min(block, plan.steps - index)
        if count <= 0:
            continue
        if block <= 16:
            for offset in range(block):
                if block - offset + count > prefix:
                    reads += count * plan.dofs**2 * 8
            continue
        updates[block] = updates.get(block, 0) + 1
        if block in retained:
            continue
        length = min(2 * block, plan.steps + 1)
        if length > prefix:
            reads += length * plan.dofs**2 * 8
        if block in levels:
            retained.add(block)
            builds += 1
        else:
            builds += (plan.dofs + fft_rows - 1) // fft_rows
    return dict(
        families=families,
        effective_prefix_lags=prefix,
        logical_source_coefficient_bytes_requested=reads * families,
        kernel_fft_builds=builds * families,
        fft_updates_per_family=updates,
        retained_levels_built=sorted(retained),
        retained_fft_bytes_at_end=sum(levels[b] for b in retained) * families,
        counts_unqueried_future_targets_of_original_implementation=True,
        physical_disk_read_or_runtime_forecast=False,
    )

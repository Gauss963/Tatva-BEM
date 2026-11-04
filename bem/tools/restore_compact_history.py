"""Restore exact online CQ history from an accepted source prefix.

Only completed dyadic blocks whose right siblings reach the resume point
need replaying. Their contributions use the original append implementation,
in its original chronological order. Past, already-consumed pending rows are
not reconstructed. Every future lag and the complete accepted input remain.
This helper does not reconstruct contact states or authorize a PDE restart.
"""

import runpy
from pathlib import Path

import numpy as np

HISTORY = runpy.run_path(str(Path(__file__).with_name("compact_radau_history.py")))


def _prefix(stream, accepted):
    accepted = np.asarray(accepted)
    expected = (stream.history.plan.dofs, stream.channels)
    if (
        accepted.ndim != 3
        or accepted.shape[1:] != expected
        or len(accepted) > stream.history.plan.steps + 1
        or accepted.dtype != np.dtype("float64")
        or not np.all(np.isfinite(accepted))
    ):
        raise ValueError("Require a finite float64 accepted prefix aligned with the CQ plan")
    if stream.next_step or np.any(stream.inputs) or np.any(stream.pending) or stream.transforms:
        raise ValueError("Restore requires a fresh, empty history")
    return accepted


def restore_batched_prefix(stream, accepted):
    """Restore input and future pending rows; consumed rows are unspecified.

    For P accepted inputs, the chronological binary cover of [0,P) contains
    exactly the completed blocks whose right siblings can intersect [P,N].
    All other completed updates ended before P. Replaying these blocks with
    the unchanged append preserves both their arithmetic and addition order.
    Kernel transforms may be rebuilt lazily; accounting begins at restoration.
    """
    accepted = _prefix(stream, accepted)
    count = len(accepted)
    stream.inputs[:count] = accepted
    end, blocks = 0, []
    for bit in reversed(range(count.bit_length())):
        size = 1 << bit
        if not count & size:
            continue
        end += size
        stream.next_step = end - 1
        stream.append(accepted[end - 1])
        blocks.append(dict(size=size, end=end))
    if stream.next_step != count:
        raise AssertionError("Binary frontier failed to cover the accepted prefix")
    return dict(
        accepted_inputs=count,
        replayed_frontier_blocks=blocks,
        consumed_pending_rows_restored=False,
        complete_future_history_retained=True,
        inherited_compute_cost_excluded_from_local_statistics=True,
    )


def restore_compact_prefix(memory, accepted):
    """Restore both families and the delayed second-family endpoint value."""
    accepted = np.asarray(accepted)
    d = memory.plan.dofs
    if (
        accepted.ndim != 2
        or accepted.shape[1] != 2 * d
        or accepted.dtype != np.dtype("float64")
        or len(accepted) > memory.plan.steps + 1
        or not np.all(np.isfinite(accepted))
        or memory.next_step
        or np.any(memory.previous_f1)
    ):
        raise ValueError("Require a fresh compact history and an aligned float64 prefix")
    stages = accepted.reshape(len(accepted), 2, d).transpose(0, 2, 1)
    # Reconstruct pending at P-1, then append the last accepted source using
    # the original compact method. It computes the exact delayed F1 value.
    before_last = stages[:-1] if len(stages) else stages
    records = [restore_batched_prefix(stream, before_last) for stream in memory.streams]
    if len(accepted):
        memory.append(accepted[-1])
    return dict(
        accepted_inputs=len(accepted),
        family_frontiers=records,
        delayed_family_restored_by_original_append=bool(len(accepted)),
        complete_future_history_retained=True,
    )


def restore_correction_prefix(memory, sources, active):
    """Restore the original zero-source rebase and its active compact history."""
    sources, active = np.asarray(sources), np.asarray(active)
    d = memory.families[0].plan.dofs
    if (
        memory.next_index
        or memory.start is not None
        or memory.online is not None
        or sources.ndim != 2
        or sources.shape[1] != 2 * d
        or sources.dtype != np.dtype("float64")
        or len(sources) > memory.families[0].plan.steps + 1
        or active.shape != (len(sources),)
        or active.dtype != np.dtype("bool")
        or not np.all(np.isfinite(sources))
    ):
        raise ValueError("Require an empty correction history and original accepted source flags")
    nonzero = np.flatnonzero(active)
    start = int(nonzero[0]) if len(nonzero) else len(sources)
    # These sources were deliberately discarded by the original inactive
    # branch. Preserve its exact tolerance and its maximum-source diagnostic.
    np.testing.assert_allclose(sources[:start], 0, rtol=0, atol=1e-7)
    if start:
        memory.maximum_inactive_source_force = float(np.max(abs(sources[:start])))
    record = None
    if len(nonzero):
        memory.start = start
        memory.online = HISTORY["CompactRadauConvolution"](
            *memory.families,
            memory.wrap,
            cache_mb=memory.cache_mb,
            fft_rows=memory.fft_rows,
        )
        record = restore_compact_prefix(memory.online, sources[start:])
    memory.next_index = len(sources)
    return dict(
        accepted_inputs=len(sources),
        first_active_index=memory.start,
        active_history=record,
        complete_future_history_retained=True,
    )

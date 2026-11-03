"""The optional CLI reserves extra memory and binds unchanged solver inputs."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from finiteblock_bem.cq import BoundaryHistory, plan_cq

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CLI = runpy.run_path(str(TOOLS / "simulate_prefix_cached_radau.py"))


def test_cache_reservation_is_subtracted_before_original_planning():
    capacity, remaining = CLI["reserved_budget"](14336, 144)
    assert capacity == 144 * 1024**2
    assert remaining == 14192
    for total, prefix in [(10, 10), (10, -1), (np.inf, 1), (10, np.nan)]:
        with pytest.raises(ValueError, match="memory budget"):
            CLI["reserved_budget"](total, prefix)


def test_original_solver_inputs_and_history_values_are_preserved():
    plan = plan_cq(2, 0.01, 8, max_memory_mb=32)
    weights = np.arange((plan.steps + 1) * 4, dtype=float).reshape(-1, 2, 2)
    weights.flags.writeable = False
    first, second = [BoundaryHistory(plan, weights) for _ in range(2)]
    inputs, policies = [], []
    marker = object()

    def original(reference, first, second, *, state, tolerance):
        inputs.append((reference, state, tolerance))
        for h in [first, second]:
            np.testing.assert_array_equal(h.weights[:], weights)
        return SimpleNamespace(diagnostics={}), marker

    wrapped = CLI["prefix_simulator"](original, capacity_bytes=4096, on_cache=policies.append)
    reference, state = object(), object()
    result, stages = wrapped(reference, first, second, state=state, tolerance=1e-7)
    assert inputs == [(reference, state, 1e-7)]
    assert stages is marker
    assert result.diagnostics["history_prefix_cache"] is policies[0]
    assert policies[0]["all_history_terms_retained"]
    assert policies[0]["cached_bytes"] <= 4096
    assert first.weights is second.weights is weights

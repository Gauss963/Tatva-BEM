"""Curve following can cross a forcing fold, with an independent target gate."""

import runpy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
TOOL = runpy.run_path(
    str(Path(__file__).parents[1] / "tools/diagnose_radau_branch_continuation.py")
)


def test_reverse_orientation_crosses_fold_before_reaching_target():
    def fun(x):
        return x**3 - x + 0.5

    def jac(x):
        return np.diag(3 * x * x - 1)

    def accept(x):
        return float(np.max(abs(fun(x)))) < 1e-9

    answer, trace = TOOL["trace_forcing_curve"](fun, jac, np.array([-0.3]), accept, max_steps=100)
    assert answer is not None and accept(answer)
    assert len(trace["paths"]) == 2
    path = np.asarray(trace["paths"][-1]["points"])
    assert path[:, -1].min() < -0.1
    assert path[-1, -1] > 1
    assert all(
        c["residual_mpa"] < 1e-7 for p in trace["paths"] for c in p["correctors"] if c["accepted"]
    )


def test_external_gate_can_reject_a_reduced_root():
    def fun(x):
        return x - 0.2

    def jac(x):
        return np.eye(1)

    answer, trace = TOOL["trace_forcing_curve"](
        fun, jac, np.zeros(1), lambda x: False, max_steps=20
    )
    assert answer is None
    assert trace["target_trials"]
    assert all(not row["accepted"] for row in trace["target_trials"])

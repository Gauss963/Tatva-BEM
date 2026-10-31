"""Prevent raw tolerances/nonfinite metadata from bypassing independent maps."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BENCH = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/benchmark_galerkin_throughput.py")
)


def sample(single):
    operator = SimpleNamespace(
        single_layer=np.array(single),
        trace_double_layer=np.eye(2),
        hypersingular=2 * np.eye(2),
    )
    return dict(operators=[operator], impedance=np.eye(2), work_compliance=np.eye(2))


class InverseMap:
    def evaluate_operators(self, operators, *, reconstruction):
        assert reconstruction
        return np.eye(2), np.linalg.inv(operators[0].single_layer)


def test_valid_independent_reference_and_reused_full_reference_pass():
    exact = sample(np.diag([2.0, 3.0]))
    actual = sample(np.diag([2.0, 3.0]))
    checks, passed, full = BENCH["audit"]([actual], [exact], InverseMap())
    assert passed
    assert all(v == 0 for c in checks for k, v in c.items() if k.endswith("normalized"))
    _, passed, _ = BENCH["audit"]([actual], [exact], InverseMap(), full_reference=full)
    assert passed


def test_small_raw_error_cannot_replace_independent_ill_conditioned_map_check():
    exact = sample(np.diag([1.0, 1e-12]))
    actual = sample(np.diag([1.0, 1e-12 + 1e-14]))
    checks, passed, _ = BENCH["audit"]([actual], [exact], InverseMap())
    assert all(c["raw_normalized"] < 1 for c in checks if "raw_normalized" in c)
    reconstruction = next(c for c in checks if c["field"] == "full_boundary_reconstruction")
    assert reconstruction["relative_frobenius"] > 0.009
    assert not passed


@pytest.mark.parametrize("field", BENCH["FIELDS"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_each_nonfinite_raw_family_fails_with_json_safe_metadata(field, value):
    class FixedMap:
        def evaluate_operators(self, operators, *, reconstruction):
            return np.eye(2), np.eye(2)

    exact, actual = sample(np.eye(2)), sample(np.eye(2))
    getattr(actual["operators"][0], field)[0, 0] = value
    checks, passed, _ = BENCH["audit"]([actual], [exact], FixedMap())
    assert not passed
    assert next(c for c in checks if c["field"] == field)["raw_normalized"] is None
    json.dumps(checks, allow_nan=False)

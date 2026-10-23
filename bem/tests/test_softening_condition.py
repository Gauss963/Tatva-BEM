import runpy
from pathlib import Path

import numpy as np

AUDIT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/audit_weakening_resolution.py")
)


def test_updated_map_diagnostic_only_flags_weakening_sliding_denominators():
    pressure = np.array([[100.0, 600.0], [500.0, 600.0]])
    weakening = np.array([[True, True], [False, True]])
    sliding = np.array([[True, True], [True, False]])
    record, bad, eta = AUDIT["updated_local_condition"](
        pressure, weakening, sliding, 7662.0, 0.8, 0.6, 0.0125959
    )
    np.testing.assert_array_equal(bad, [[False, True], [False, False]])
    assert record["nonpositive_denominator_weakening_sliding_entries"] == 1
    assert record["nonpositive_denominator_nodes"] == 1
    assert eta[0, 1] > 1
    assert record["minimum_denominator_while_weakening_and_sliding"] < 0


def test_constant_friction_has_no_softening_condition_failure():
    pressure = np.array([[1e8, 1e8]])
    mask = np.ones_like(pressure, dtype=bool)
    record, bad, eta = AUDIT["updated_local_condition"](pressure, mask, mask, 100.0, 0.6, 0.6, 0.1)
    assert not np.any(bad)
    np.testing.assert_array_equal(eta, 0)
    assert record["critical_pressure_min_mpa"] is None


def test_nonpositive_denominator_does_not_alone_prove_multiple_endpoint_roots():
    p = np.array([[600.0, 600.0]])
    trial = np.array([[470.0, 500.0]])
    old_c = np.zeros_like(p)
    count, flat = AUDIT["updated_endpoint_candidates"](p, trial, old_c, 7662.0, 0.8, 0.6, 0.0125959)
    # At 470 MPa a sticking state and two positive slip states satisfy the
    # endpoint graph. At 500 MPa only the fully weakened state is admissible,
    # despite the same nonpositive soft-branch denominator.
    np.testing.assert_array_equal(count, [[3, 1]])
    assert not np.any(flat)

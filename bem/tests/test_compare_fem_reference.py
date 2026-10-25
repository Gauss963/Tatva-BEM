import runpy
from pathlib import Path

import numpy as np
import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/compare_fem_reference.py"))


def test_shear_alignment_excludes_different_normal_output_cadence():
    reference = dict(phase_id=np.array([0, 1, 2, 2]), step_id=np.array([0, 10, 20, 30]))
    actual = dict(phase_id=np.array([0, 1, 1, 2, 2]), step_id=np.array([0, 5, 10, 20, 30]))
    a, b = TOOL["aligned_shear_rows"](reference, actual)
    np.testing.assert_array_equal(a, [2, 3])
    np.testing.assert_array_equal(b, [3, 4])
    actual["step_id"][-1] += 1
    with pytest.raises(ValueError, match="exactly matching step"):
        TOOL["aligned_shear_rows"](reference, actual)


@pytest.mark.parametrize("steps", [[2, 2], [3, 2], []])
def test_shear_alignment_rejects_duplicates_reordering_or_missing_phase(steps):
    reference = dict(phase_id=np.full(len(steps), 2), step_id=np.asarray(steps))
    with pytest.raises(ValueError, match="strictly increasing"):
        TOOL["aligned_shear_rows"](reference, reference)


def test_array_comparison_retains_mismatch_and_zero_reference_diagnostics():
    compare = TOOL["differences"]
    record = compare(np.zeros(2), np.array([0, 0.01]), rtol=1e-6, atol=1e-9)
    assert not record["within_tolerance"] and not record["bitwise_equal"]
    assert record["relative_l2"] is None
    assert record["maximum_absolute_difference"] == 0.01
    with pytest.raises(ValueError, match="finite"):
        compare(np.zeros(2), np.array([0, np.nan]), rtol=1e-6, atol=1e-9)

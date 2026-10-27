import runpy
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
TOOL = runpy.run_path(str(TOOLS / "compare_refined_primal.py"))


def test_complete_field_gate_detects_an_interior_history_error():
    fields = {k: np.zeros((17, 5)) for k in TOOL["TOLERANCES"]}
    fields.update(
        time=np.arange(17) * 0.01,
        contact_y=np.arange(5),
        contact_weights=np.ones(5),
        contact_interpolation=np.eye(5),
        retained_contact_y=np.arange(5),
    )
    candidate = {k: v.copy() for k, v in fields.items()}
    assert all(v["passed"] for v in TOOL["compare_fields"](fields, candidate).values())
    candidate["pressure"][7, 2] = 0.01
    result = TOOL["compare_fields"](fields, candidate)
    assert not result["pressure"]["passed"]
    assert result["pressure"]["maximum_absolute_difference"] == 0.01
    candidate["time"][3] += 0.001
    with pytest.raises(AssertionError):
        TOOL["compare_fields"](fields, candidate)

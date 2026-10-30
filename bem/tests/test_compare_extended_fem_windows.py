import runpy
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
COMPARE = runpy.run_path(str(TOOLS / "compare_extended_fem_windows.py"))


def data(factor):
    steps = np.array([12, 14])
    return dict(
        step_id=steps,
        time=steps * 0.01,
        contact_y=np.array([0.0, 1.0, 2.0]),
        all_scalar_history=np.zeros((4 * factor, 17)),
        **{k: np.ones((2, 3)) for k in COMPARE["BASE"]["BASE"]["FIELDS"]},
    )


@pytest.mark.parametrize("factor", [4, 8, 16])
def test_complete_original_grid_and_all_fine_rows_are_preserved(tmp_path, factor):
    payload = data(factor)
    np.savez(tmp_path / f"factor{factor}.npz", **payload)
    result = COMPARE["load_branch"](tmp_path, dict(factor=factor, dt_s=0.01 / factor), 10, 14)
    assert result["scalar"].shape == (factor * 4, 17)
    np.testing.assert_array_equal(result["step_id"], [12, 14])
    np.testing.assert_array_equal(result["y"], payload["contact_y"])


@pytest.mark.parametrize("bad", ["missing_scalar", "clock", "nonfinite", "missing_endpoint"])
def test_incomplete_or_misaligned_data_is_rejected(tmp_path, bad):
    payload = data(8)
    expected_y = payload["contact_y"].copy()
    if bad == "missing_scalar":
        payload["all_scalar_history"] = payload["all_scalar_history"][:-1]
    elif bad == "clock":
        payload["time"] = payload["time"] + 1e-9
    elif bad == "nonfinite":
        payload["pressure"][0, 1] = np.nan
    else:
        payload["contact_y"] = payload["contact_y"][:-1]
    np.savez(tmp_path / "factor8.npz", **payload)
    with pytest.raises(ValueError):
        COMPARE["load_branch"](
            tmp_path, dict(factor=8, dt_s=0.01 / 8), 10, 14, expected_y=expected_y
        )

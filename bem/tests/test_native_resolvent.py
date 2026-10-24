import runpy
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def test_disk_physical_conversion_matches_original_and_retains_all_lags(tmp_path):
    original = runpy.run_path(str(TOOLS / "build_resolvent_history.py"))
    native = runpy.run_path(str(TOOLS / "prepare_native_resolvent.py"))
    rng = np.random.default_rng(20261024)
    expected = rng.normal(size=(131, 3, 3))
    path = tmp_path / "weights.npy"
    np.save(path, expected, allow_pickle=False)
    line_weights = np.array([0.5, 3.7, 80.0])
    original["physical_weights_inplace"](expected, line_weights)
    actual = native["physical_weights_file_inplace"](path, line_weights)
    assert isinstance(actual, np.memmap) and not actual.flags.writeable
    np.testing.assert_array_equal(actual, expected)

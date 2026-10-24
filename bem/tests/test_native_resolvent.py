import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from finiteblock_bem.cq import plan_cq

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


def test_time_storage_preflight_covers_actual_serialized_outputs(tmp_path, monkeypatch):
    native = runpy.run_path(str(TOOLS / "prepare_native_resolvent.py"))
    plan = plan_cq(3, 0.01, 5, max_memory_mb=32)
    np.save(tmp_path / "weights.npy", np.zeros((6, 3, 3)), allow_pickle=False)
    np.save(tmp_path / "initial_impedance.npy", np.eye(3), allow_pickle=False)
    np.savez(
        tmp_path / "elastic.npz",
        time=np.arange(13),
        elastic=np.zeros((13, 3)),
        uninterrupted_actuator=np.zeros(13),
    )
    required = native["preflight_time_storage"](tmp_path, plan, 12)
    assert required >= sum(path.stat().st_size for path in tmp_path.iterdir())
    monkeypatch.setattr(
        native["shutil"], "disk_usage", lambda _: SimpleNamespace(free=required - 1)
    )
    with pytest.raises(ValueError, match="complete time weights"):
        native["preflight_time_storage"](tmp_path, plan, 12)

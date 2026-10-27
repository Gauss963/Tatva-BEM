import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
TOOL = runpy.run_path(str(TOOLS / "fortran_weight_storage.py"))


@pytest.mark.parametrize("rows", [1, 3])
def test_fortran_file_keeps_every_float_bit_and_logical_coordinate(tmp_path, rows):
    values = np.random.default_rng(44).normal(size=(131, 7, 7))
    values[3, 2, 1] = -0.0
    values[101, 5, 4] = np.nextafter(0.0, 1.0)
    source, target = tmp_path / "original.npy", tmp_path / "fortran.npy"
    np.save(source, values)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    record = TOOL["convert_weights_fortran"](
        source, target, max_workspace_mb=65, max_rows=rows, time_tile=17
    )
    actual = np.load(target, mmap_mode="r", allow_pickle=False)
    assert actual.shape == values.shape and actual.flags.f_contiguous
    np.testing.assert_array_equal(actual.view(np.uint64), values.view(np.uint64))
    assert record["bitwise_equal_to_source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="fresh distinct"):
        TOOL["convert_weights_fortran"](source, target)

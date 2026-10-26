import mmap
import runpy
from pathlib import Path

import numpy as np
import pytest

TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/bounded_weight_conversion.py")
)


@pytest.mark.parametrize("d,lags,block", [(3, 141, 64), (17, 77, 13), (275, 3, 1)])
def test_bounded_writable_maps_preserve_original_division_bitwise(
    tmp_path, monkeypatch, d, lags, block
):
    rng = np.random.default_rng(180)
    original = rng.normal(size=(lags, d, d))
    line = np.geomspace(0.125, 200, d)
    root = np.sqrt(line)
    expected = original.copy()
    for first in range(0, lags, 64):
        expected[first : first + 64] /= root[:, None] * root[None, :]
    path = tmp_path / "weights.npy"
    np.save(path, original)
    before_size = path.stat().st_size
    native_mmap = mmap.mmap
    writable_lengths = []

    def record_mapping(fd, length, *args, **kwargs):
        if kwargs.get("access") == mmap.ACCESS_WRITE:
            writable_lengths.append(length)
        return native_mmap(fd, length, *args, **kwargs)

    monkeypatch.setattr(mmap, "mmap", record_mapping)
    actual = TOOL["convert_physical_weights"](path, line, block_lags=block)
    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable
    assert path.stat().st_size == before_size
    assert len(writable_lengths) == (lags + block - 1) // block
    assert max(writable_lengths) <= block * d * d * 8 + mmap.ALLOCATIONGRANULARITY
    actual._mmap.close()


def test_invalid_storage_is_rejected_before_modifying_weights(tmp_path):
    path = tmp_path / "weights.npy"
    np.save(path, np.ones((4, 3, 2)))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="square-matrix"):
        TOOL["convert_physical_weights"](path, np.ones(3))
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="positive finite"):
        TOOL["convert_physical_weights"](path, [1, 0, 1])

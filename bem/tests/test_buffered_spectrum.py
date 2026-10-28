import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import plan_cq
from finiteblock_bem.cq_storage import DiskSpectrum

WRITER = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/buffered_spectrum.py"))[
    "BufferedSpectrumWriter"
]


def test_buffered_storage_is_byte_identical_to_immediate_writes_and_owns_its_values(tmp_path):
    plan = plan_cq(3, 0.01, 64, disk_spectrum=True, max_memory_mb=64)
    reference = DiskSpectrum(tmp_path / "direct.npy", plan)
    target = DiskSpectrum(tmp_path / "buffered.npy", plan)
    writer = WRITER(target, matrices=7)
    rng = np.random.default_rng(6)
    for i in range(reference.shape[0]):
        value = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        reference[i] = value
        writer.append(i, value)
        value[:] = 0
    assert not np.all(target.written)
    writer.finish()
    reference.sync()
    assert np.all(target.written)
    assert writer.committed == reference.shape[0]
    assert (
        hashlib.sha256(reference.path.read_bytes()).digest()
        == hashlib.sha256(target.path.read_bytes()).digest()
    )


def test_unflushed_rows_are_not_reported_as_committed_and_incomplete_finish_fails(tmp_path):
    plan = plan_cq(2, 0.01, 8, disk_spectrum=True, max_memory_mb=64)
    target = DiskSpectrum(tmp_path / "partial.npy", plan)
    writer = WRITER(target, matrices=4)
    writer.append(0, np.eye(2))
    assert writer.emitted == 1 and writer.committed == 0 and not target.written[0]
    with pytest.raises(ValueError, match="Incomplete"):
        writer.finish()
    assert target.written[0] and writer.committed == 1
    with pytest.raises(ValueError):
        target.record()
    with pytest.raises(ValueError):
        writer.append(0, np.eye(2))

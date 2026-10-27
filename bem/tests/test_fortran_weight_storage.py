import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import BoundaryHistory, plan_cq

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


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_existing_scalar_and_batched_histories_read_converted_files(tmp_path, transpose, batched):
    rng = np.random.default_rng(91)
    n, d = 70, 7
    weights = rng.normal(size=(n + 1, d, d))
    source, target = tmp_path / "c.npy", tmp_path / "f.npy"
    np.save(source, weights)
    TOOL["convert_weights_fortran"](source, target, max_workspace_mb=65, max_rows=3)
    mapped = np.load(target, mmap_mode="r", allow_pickle=False)
    if transpose:
        weights, mapped = (a.transpose(0, 2, 1) for a in (weights, mapped))
    plan = plan_cq(d, 0.01, n, max_memory_mb=32)
    if batched:
        cls = runpy.run_path(str(TOOLS / "batched_history.py"))["BatchedBoundedConvolution"]
        extra, shape = dict(right_hand_sides=5), (d, 5)
    else:
        cls = runpy.run_path(str(TOOLS / "bounded_history.py"))["BoundedOnlineConvolution"]
        extra, shape = {}, (d,)
    memory = cls(BoundaryHistory(plan, mapped), cache_mb=0, fft_rows=3, direct_block=2, **extra)
    inputs = rng.normal(size=(n + 1, *shape))
    for i in range(n + 1):
        expected = sum((weights[i - j] @ inputs[j] for j in range(i)), start=np.zeros(shape))
        np.testing.assert_allclose(memory.past(i), expected, rtol=1e-12, atol=1e-12)
        memory.append(inputs[i])
    assert memory.streamed_updates > 0
    assert memory.statistics()["all_history_terms_retained"]

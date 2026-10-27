"""Bit-preserving out-of-core conversion to time-contiguous NPY weights.

Logical shape stays (lag,row,column); NumPy's standard Fortran-order header
lets existing readers load it unchanged. Larger contiguous time histories
reduce file seeks during repeated FFTs. The original input is never modified.
"""

import mmap
import os
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np


def convert_weights_fortran(source, target, *, max_workspace_mb=2048, max_rows=32, time_tile=128):
    source, target = Path(source), Path(target)
    if target.exists() or source.resolve() == target.resolve():
        raise ValueError("Choose a fresh distinct Fortran weight file")
    if (
        not np.isfinite(max_workspace_mb)
        or max_workspace_mb <= 0
        or not isinstance(max_rows, int)
        or max_rows < 1
        or not isinstance(time_tile, int)
        or time_tile < 1
    ):
        raise ValueError("Require positive conversion workspace and tile sizes")
    with source.open("rb") as f:
        version = np.lib.format.read_magic(f)
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }
        if version not in readers:
            raise ValueError("Unsupported source NPY format")
        shape, fortran, dtype = readers[version](f)
        header = f.tell()
    if (
        len(shape) != 3
        or shape[0] < 1
        or shape[1] != shape[2]
        or fortran
        or dtype != np.float64
        or source.stat().st_size != header + int(np.prod(shape)) * 8
    ):
        raise ValueError("Require complete C-order float64 square-matrix weights")
    n, d, _ = shape
    reserve = 64 * 1024**2
    rows = min(d, max_rows, int((max_workspace_mb * 1024**2 - reserve) // (2 * n * d * 8)))
    if rows < 1:
        raise ValueError("Conversion budget cannot hold one bounded row tile")
    if shutil.disk_usage(target.parent).free < int(np.prod(shape)) * 8 + 8192:
        raise ValueError("Insufficient disk for verified Fortran conversion")
    before = source.stat()
    begun = perf_counter()
    original = np.load(source, mmap_mode="r", allow_pickle=False)
    with target.open("xb") as output:
        np.lib.format.write_array_header_2_0(
            output, dict(descr=dtype.str, fortran_order=True, shape=shape)
        )
        offset = output.tell()
        output.truncate(offset + int(np.prod(shape)) * 8)

    def mapped_column(stream, first, count, column, write, operation):
        start = offset + (column * d + first) * n * 8
        aligned = start // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
        length = start - aligned + n * count * 8
        mapping = mmap.mmap(
            stream.fileno(),
            length,
            offset=aligned,
            access=mmap.ACCESS_WRITE if write else mmap.ACCESS_READ,
        )
        array = np.ndarray(
            (n, count), dtype=dtype, buffer=mapping, offset=start - aligned, order="F"
        )
        try:
            result = operation(array)
            if write:
                mapping.flush()
            return result
        finally:
            del array
            mapping.close()

    with target.open("r+b") as output:
        for first in range(0, d, rows):
            count = min(rows, d - first)
            old = np.array(original[:, first : first + count, :], order="C", copy=True)
            tile = np.empty_like(old, order="F")
            for t in range(0, n, time_tile):
                tile[t : t + time_tile] = old[t : t + time_tile]
            del old
            for column in range(d):
                value = tile[:, :, column]
                mapped_column(
                    output,
                    first,
                    count,
                    column,
                    True,
                    lambda a, v=value: a.__setitem__(slice(None), v),
                )
            os.fsync(output.fileno())
            for column in range(d):
                expected = tile[:, :, column]
                equal = mapped_column(
                    output,
                    first,
                    count,
                    column,
                    False,
                    lambda a, v=expected: np.array_equal(a.view(np.uint64), v.view(np.uint64)),
                )
                if not equal:
                    raise ValueError("Fortran storage changed weight bits")
            del tile
        os.fsync(output.fileno())
    original._mmap.close()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source weights changed during storage conversion")
    return dict(
        layout="fortran_time_row_column",
        logical_shape=list(shape),
        bitwise_equal_to_source=True,
        source_unchanged=True,
        rows_per_tile=rows,
        time_tile=time_tile,
        estimated_conversion_array_peak_bytes=2 * n * rows * d * 8 + reserve,
        full_conversion_seconds=perf_counter() - begun,
        full_time_history_retained=True,
        scope="Storage conversion only; runtime and trajectory equivalence require separate validation",
    )

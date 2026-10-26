"""Convert work-coordinate weights with bounded writable file mappings.

The old converter flushes and advises the whole weight mapping per time
block. Here each writable mapping covers just that same block, including the
necessary aligned prefix. Arithmetic and coefficient order are unchanged.
Only a read-only memmap of the completed file is returned to the caller.
"""

import mmap
import os
from pathlib import Path

import numpy as np


def convert_physical_weights(path, line_weights, *, block_lags=64):
    path = Path(path)
    line = np.asarray(line_weights, dtype=float)
    if (
        line.ndim != 1
        or not np.all(np.isfinite(line))
        or np.any(line <= 0)
        or not isinstance(block_lags, int)
        or isinstance(block_lags, bool)
        or block_lags < 1
    ):
        raise ValueError("Require positive finite work weights and a positive time block")
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }
        if version not in readers:
            raise ValueError("Unsupported NPY weight format")
        shape, fortran, dtype = readers[version](stream)
        header = stream.tell()
    d = len(line)
    if (
        len(shape) != 3
        or shape[1:] != (d, d)
        or shape[0] < 1
        or fortran
        or dtype != np.float64
        or path.stat().st_size != header + int(np.prod(shape)) * dtype.itemsize
    ):
        raise ValueError("Require a complete C-order float64 square-matrix weight file")
    root = np.sqrt(line)
    scale = root[:, None] * root[None, :]
    matrix_bytes = d * d * dtype.itemsize
    with path.open("r+b") as stream:
        for first in range(0, shape[0], block_lags):
            count = min(block_lags, shape[0] - first)
            begin = header + first * matrix_bytes
            aligned = begin // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
            length = begin - aligned + count * matrix_bytes
            mapping = mmap.mmap(stream.fileno(), length, offset=aligned, access=mmap.ACCESS_WRITE)
            array = np.ndarray((count, d, d), dtype=dtype, buffer=mapping, offset=begin - aligned)
            try:
                array /= scale
                mapping.flush()
            finally:
                del array
                mapping.close()
        os.fsync(stream.fileno())
    return np.load(path, mmap_mode="r", allow_pickle=False)

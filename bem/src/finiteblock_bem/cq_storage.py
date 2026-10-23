"""Exact disk storage for CQ half spectra, with bounded public mmap mappings.

Only frequency matrices are stored on disk. Inversion still returns dense
time weights, and the complete online history is retained. Array views never
escape a mapping, so closing it does not leave live dangling NumPy views.
"""

import hashlib
import mmap
import os
from pathlib import Path

import numpy as np

MAP_MATRICES = 64


def implementation_signature():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def spectrum_bytes(plan):
    return (plan.fft_size // 2 + 1) * plan.dofs**2 * 16


class DiskSpectrum:
    """Half-spectrum array supporting complete-matrix writes and copied reads."""

    def __init__(self, path, plan, *, overwrite=False):
        self.path = Path(path)
        self.shape = (plan.fft_size // 2 + 1, plan.dofs, plan.dofs)
        self.dtype = np.dtype("complex128")
        self.written = np.zeros(self.shape[0], dtype=bool)
        self.pending_writes = 0
        with self.path.open("w+b" if overwrite else "xb") as stream:
            np.lib.format.write_array_header_2_0(
                stream, dict(descr=self.dtype.str, fortran_order=False, shape=self.shape)
            )
            self.offset = stream.tell()
            stream.truncate(self.offset + spectrum_bytes(plan))

    def sync(self):
        if self.pending_writes:
            with self.path.open("r+b") as stream:
                os.fsync(stream.fileno())
            self.pending_writes = 0

    def _mapped(self, operation, first, last, *, write=False):
        matrix_bytes = self.shape[1] * self.shape[2] * self.dtype.itemsize
        offset = self.offset + first * matrix_bytes
        aligned = offset // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
        length = offset - aligned + (last - first) * matrix_bytes
        with self.path.open("r+b" if write else "rb") as stream:
            mapping = mmap.mmap(
                stream.fileno(),
                length,
                offset=aligned,
                access=mmap.ACCESS_WRITE if write else mmap.ACCESS_READ,
            )
            array = np.ndarray(
                (last - first, *self.shape[1:]), self.dtype, buffer=mapping, offset=offset - aligned
            )
            try:
                result = operation(array)
            finally:
                del array
                mapping.close()
        return result

    def _index(self, key):
        if not -self.shape[0] <= key < self.shape[0]:
            raise IndexError("Frequency index outside spectrum")
        return key % self.shape[0]

    def __setitem__(self, key, value):
        integer = isinstance(key, (int, np.integer))
        if integer:
            first = self._index(key)
            last = first + 1
        elif isinstance(key, slice) and key.step in [None, 1]:
            first, last, _ = key.indices(self.shape[0])
        else:
            raise ValueError("Spectrum writes require complete matrices at unit stride")
        count = max(last - first, 0)
        values = np.asarray(value, dtype=self.dtype)
        expected = self.shape[1:] if integer else (count, *self.shape[1:])
        if values.shape != expected:
            raise ValueError("Incorrect frequency matrix shape")
        if integer:
            values = values[None]
        for begin in range(first, last, MAP_MATRICES):
            end = min(begin + MAP_MATRICES, last)
            block = values[begin - first : end - first]
            self._mapped(
                lambda array: array.__setitem__(slice(None), block), begin, end, write=True
            )
            self.written[begin:end] = True
            self.pending_writes += end - begin
            if self.pending_writes >= MAP_MATRICES:
                self.sync()

    def __getitem__(self, key):
        if not np.all(self.written):
            raise ValueError("Complete the entire half spectrum before inversion")
        self.sync()
        key = key if isinstance(key, tuple) else (key,)
        first_axis, tail = key[0], key[1:]
        if isinstance(first_axis, (int, np.integer)):
            first = self._index(first_axis)
            return self._mapped(
                lambda array: np.array(array[(0, *tail)], copy=True), first, first + 1
            )
        if not isinstance(first_axis, slice):
            raise ValueError("Unsupported frequency index")
        first, last, stride = first_axis.indices(self.shape[0])
        if stride != 1:
            raise ValueError("Only unit-stride frequency reads supported")
        shape = np.broadcast_to(np.empty((), self.dtype), self.shape[1:])[tail].shape
        result = np.empty((max(last - first, 0), *shape), self.dtype)
        for begin in range(first, last, MAP_MATRICES):
            end = min(begin + MAP_MATRICES, last)
            result[begin - first : end - first] = self._mapped(
                lambda array: np.array(array[(slice(None), *tail)], copy=True), begin, end
            )
        return result

    def record(self):
        if not np.all(self.written):
            raise ValueError("Incomplete spectrum cannot be recorded as complete")
        self.sync()
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return dict(
            backend="disk_half_spectrum",
            path=str(self.path.resolve()),
            shape=list(self.shape),
            bytes=self.path.stat().st_size,
            sha256=digest.hexdigest(),
            implementation_sha256=implementation_signature(),
            all_frequency_matrices_written=True,
        )

"""Bounded ordered writes to the unchanged disk half-spectrum format."""

import numpy as np


class BufferedSpectrumWriter:
    """Only flushed matrices count as committed; numerical values are copied."""

    def __init__(self, spectrum, *, matrices=64):
        if not isinstance(matrices, int) or isinstance(matrices, bool) or matrices < 1:
            raise ValueError("A positive integer matrix buffer is required")
        self.spectrum = spectrum
        self.buffer = np.empty(
            (min(matrices, spectrum.shape[0]), *spectrum.shape[1:]), dtype=spectrum.dtype
        )
        self.emitted = self.committed = self.pending = 0

    def append(self, index, value):
        if (
            not isinstance(index, (int, np.integer))
            or isinstance(index, (bool, np.bool_))
            or index != self.emitted
            or index >= self.spectrum.shape[0]
        ):
            raise ValueError("Spectrum matrices must arrive once each in consecutive order")
        value = np.asarray(value, dtype=self.spectrum.dtype)
        if value.shape != self.spectrum.shape[1:]:
            raise ValueError("Incorrect buffered frequency matrix shape")
        self.buffer[self.pending] = value
        self.pending += 1
        self.emitted += 1
        if self.pending == len(self.buffer):
            self.flush()

    def flush(self):
        if self.pending:
            self.spectrum[self.committed : self.emitted] = self.buffer[: self.pending]
            self.committed = self.emitted
            self.pending = 0

    def finish(self):
        self.flush()
        if self.emitted != self.spectrum.shape[0] or not np.all(self.spectrum.written):
            raise ValueError("Incomplete spectrum cannot finish its buffered writer")
        self.spectrum.sync()

"""Rectangular storage extension of the verified raw sampling journal.

Reuse its sampler stack, consecutive chunk commits and bitwise prefix writer.
Only shape/header/byte-stride handling changes. Legacy square receipts are not
accepted; this component owns a distinct rectangular preflight directory.
"""

import json
import runpy
from dataclasses import asdict
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
BASE = runpy.run_path(str(TOOLS / "radau_sampling_checkpoint.py"))
DATA = runpy.run_path(str(TOOLS / "reconstruct_interior_reference.py"))
TX = runpy.run_path(str(TOOLS / "inplace_radau_compaction.py"))


def header(path, shape):
    with Path(path).open("rb") as stream:
        if np.lib.format.read_magic(stream) != (2, 0):
            raise ValueError("Require version-2 rectangular NPY")
        actual, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        offset = stream.tell()
    if (
        actual != tuple(shape)
        or fortran
        or dtype != np.complex128
        or Path(path).stat().st_size != offset + int(np.prod(shape)) * 16
    ):
        raise ValueError("Rectangular shape, dtype, layout or length changed")
    return dict(offset=offset, shape=list(shape), dtype=dtype.str)


class RectangularJournal(BASE["SamplingJournal"]):
    def __init__(self, folder, plan, outputs, inputs, *, resume=False):
        self.shape = (plan.fft_size // 2 + 1, outputs, inputs)
        self.matrix_bytes = outputs * inputs * 16
        config = json.loads((Path(folder) / "preflight.json").read_text())
        if (
            config.get("kind") != "rectangular_interior_raw_sampling"
            or config.get("plan") != asdict(plan)
            or config.get("shape") != list(self.shape)
        ):
            raise ValueError("Require this rectangular producer preflight and exact plan")
        # No foreign legacy receipt migration. A stop before journal creation has
        # no committed prefix to recover; the owned files are recomputed.
        existing = (Path(folder) / "sampling-journal.json").exists()
        super().__init__(folder, plan, resume=resume and existing)
        if resume and not existing:
            self.data["prior_attempt_before_journal_creation_unaccounted"] = True

    def verify_channel(self, name, entry):
        path = self._path(name)
        actual = header(path, self.shape)
        if any(entry[k] != v for k, v in actual.items()):
            raise ValueError("Rectangular channel header changed")
        first = 0
        for chunk in entry["chunks"]:
            left, right = chunk["first"], chunk["last"]
            if (
                not isinstance(left, int)
                or not isinstance(right, int)
                or isinstance(left, bool)
                or isinstance(right, bool)
                or left != first
                or not left < right <= self.shape[0]
            ):
                raise ValueError("Rectangular chunks omit or overlap matrices")
            if (
                BASE["digest_range"](
                    path,
                    actual["offset"] + left * self.matrix_bytes,
                    (right - left) * self.matrix_bytes,
                )
                != chunk["sha256"]
            ):
                raise ValueError("Committed rectangular spectrum changed")
            first = right

    def spectrum(self, name):
        path = self._path(name)
        if path.exists():
            info = header(path, self.shape)
            result = DATA["RectangularSpectrum"].__new__(DATA["RectangularSpectrum"])
            result.path, result.shape, result.dtype = path, self.shape, np.dtype("complex128")
            result.offset = info["offset"]
            result.written = np.zeros(self.shape[0], dtype=bool)
            result.pending_writes = 0
        else:
            result = DATA["RectangularSpectrum"](path, self.shape)
            info = header(path, self.shape)
        if name not in self.data["channels"]:
            self.data["channels"][name] = {**info, "chunks": []}
            self.publish()
        result.written[: self.prefix(name)] = True
        return result

    def publish(self):
        super().publish()
        TX["sync_directory"](self.folder)


Writer = BASE["JournaledSpectrumWriter"]

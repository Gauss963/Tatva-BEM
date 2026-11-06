"""Verified sampling checkpoints for reciprocal Radau spectrum tiles."""

import hashlib
import runpy
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
CHECKPOINT = runpy.run_path(str(TOOLS / "radau_sampling_checkpoint.py"))
STORAGE = runpy.run_path(str(TOOLS / "reciprocal_radau_storage.py"))


class ReciprocalSamplingJournal(CHECKPOINT["SamplingJournal"]):
    """Reuse sampler boundaries and ownership; verify the packed bytes on resume."""

    def __init__(self, folder, plan, block, *, resume=False):
        self.count = plan.fft_size // 2 + 1
        self.block = block
        self.tiles = tuple(STORAGE["spectral_tiles"](plan.dofs, self.count, block))
        self.layout = dict(
            format="complex128_tile_major_upper_blocks_v1",
            dtype=np.dtype(np.complex128).str,
            dofs=plan.dofs,
            frequencies=self.count,
            block=block,
            bytes=STORAGE["reciprocal_size"](plan, block),
        )
        super().__init__(folder, plan, resume=resume)

    def _path(self, name):
        path = self.folder / name
        if name not in ("branch0.bin", "branch1.bin") or path.parent != self.folder:
            raise ValueError("Reciprocal channel is outside this owned build")
        return path

    def verify_channel(self, name, entry):
        if entry["layout"] != self.layout:
            raise ValueError("Reciprocal channel layout changed")
        final = self._path(name)
        partial = final.with_name(name + ".part")
        path = final if final.exists() else partial
        if partial.is_symlink() or final.is_symlink():
            raise ValueError("Reciprocal channel cannot be a symlink")
        if path.exists() and path.stat().st_size != self.layout["bytes"]:
            raise ValueError("Reciprocal channel length changed")
        if final.exists() and partial.exists() and not final.samefile(partial):
            raise ValueError("Reciprocal final channel and partial both exist")
        prefix = 0
        for chunk in entry["chunks"]:
            first, last = chunk["first"], chunk["last"]
            if (
                type(first) is not int
                or type(last) is not int
                or first != prefix
                or not first < last <= self.count
                or not path.is_file()
                or STORAGE["saved_tile_digest"](path, self.tiles, first, last) != chunk["sha256"]
            ):
                raise ValueError("Committed reciprocal spectrum bytes or intervals changed")
            prefix = last
        if final.exists() and prefix != self.count:
            raise ValueError("Published reciprocal channel has an incomplete checkpoint")

    def open_branch(self, name):
        """Open only an owned fresh or previously verified partial branch."""
        final = self._path(name)
        partial = final.with_name(name + ".part")
        channels = self.data["channels"]
        if name not in channels:
            if final.exists() or partial.exists():
                raise ValueError("Reciprocal branch lacks its checkpoint receipt")
            channels[name] = dict(layout=self.layout, chunks=[])
            self.publish()
        if partial.is_symlink() or final.is_symlink():
            raise ValueError("Reciprocal channel cannot be a symlink")
        return STORAGE["ReciprocalBranch"](
            final,
            self.plan,
            block=self.block,
            verified_prefix=self.prefix(name) if partial.exists() else None,
        )


class JournaledReciprocalWriter:
    """Commit ordered batches, comparing any recomputed preserved prefix bitwise."""

    def __init__(self, journal, branch, start, *, matrices=16):
        if not isinstance(matrices, int) or isinstance(matrices, bool) or matrices < 1:
            raise ValueError("Require a positive reciprocal writer batch size")
        self.journal, self.branch, self.name = journal, branch, branch.path.name
        self.emitted, self.committed, self.pending = start, journal.prefix(self.name), 0
        if not 0 <= start <= self.committed <= journal.count or self.committed != branch.next:
            raise ValueError("Reciprocal start disagrees with its verified prefix")
        self.buffer = np.empty(
            (min(matrices, journal.count), branch.dofs, branch.dofs), np.complex128
        )

    def accept(self, index, value):
        if (
            not isinstance(index, (int, np.integer))
            or isinstance(index, (bool, np.bool_))
            or index != self.emitted
            or index >= self.journal.count
        ):
            raise ValueError("Reciprocal frequency indices must be consecutive")
        if np.shape(value) != self.buffer.shape[1:]:
            raise ValueError("Incorrect reciprocal frequency matrix shape")
        self.buffer[self.pending] = value
        self.emitted += 1
        self.pending += 1
        if self.pending == len(self.buffer):
            self.flush()

    def flush(self):
        if not self.pending:
            return
        first, last = self.emitted - self.pending, self.emitted
        values = self.buffer[: self.pending]
        replayed = max(0, min(last, self.committed) - first)
        if replayed:
            self.branch.check_batch(values[:replayed])
            digest = hashlib.sha256()
            for tile in STORAGE["matrix_tile_bytes"](values[:replayed], self.journal.tiles):
                digest.update(memoryview(tile).cast("B"))
            if digest.hexdigest() != STORAGE["saved_tile_digest"](
                self.branch.partial, self.journal.tiles, first, first + replayed
            ):
                raise ValueError("Recomputed reciprocal prefix is not bitwise identical")
        if replayed < self.pending:
            if first + replayed != self.committed:
                raise ValueError("Reciprocal frequency batch skips a committed index")
            checksum = self.branch.append(values[replayed:])
            self.branch.sync()
            self.journal.commit(self.name, self.committed, last, checksum)
            self.committed = last
        self.pending = 0

    def finish(self):
        self.flush()
        if self.emitted != self.journal.count or self.committed != self.emitted:
            raise ValueError("Incomplete reciprocal frequency writer")

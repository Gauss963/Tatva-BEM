"""Append-only, atomic accepted-state chunks for conditional FEM windows.

A chunk contains every fine scalar row and all twelve accepted carry arrays.
Only an atomic, fsynced rename commits a chunk. Resume validates the complete
contiguous hash chain and rejects changed context or malformed state.
"""

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def payload_digest(arrays):
    value = hashlib.sha256()
    for key in sorted(k for k in arrays if k != "metadata"):
        a = np.asarray(arrays[key])
        value.update(json.dumps([key, a.dtype.str, a.shape]).encode())
        value.update(np.ascontiguousarray(a).tobytes())
    return value.hexdigest()


def atomic_npz(path, data):
    path = Path(path)
    if path.exists():
        raise ValueError("Refuse to overwrite a committed FEM chunk")
    temporary = path.with_name(".pending-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


class WindowChunks:
    def __init__(self, folder, context, factor, first, stops, observed, carry, fields):
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=True)
        self.context, self.factor, self.first = context, factor, first
        self.stops, self.observed = list(map(int, stops)), set(map(int, observed))
        self.templates = [(np.asarray(a).shape, np.asarray(a).dtype.str) for a in carry]
        self.fields = {name: np.asarray(a).shape for name, a in fields.items()}
        if (
            factor < 1
            or factor & (factor - 1)
            or len(self.templates) != 12
            or not self.stops
            or self.stops != sorted(set(self.stops))
            or self.stops[0] <= first
            or not self.observed.issubset(self.stops)
        ):
            raise ValueError("Require a power-of-two factor and a complete ordered window")
        self.paths, self.hashes = [], []

    def metadata(self, index):
        return dict(
            schema=1,
            context_sha256=self.context,
            factor=self.factor,
            index=index,
            first=self.first if index == 0 else self.stops[index - 1],
            last=self.stops[index],
            previous_sha256=None if index == 0 else self.hashes[index - 1],
        )

    def validate(self, arrays, metadata):
        index, first, last = metadata["index"], metadata["first"], metadata["last"]
        if index != len(self.paths) or not 0 <= index < len(self.stops):
            raise ValueError("FEM chunk index differs from the next planned commit")
        keys = {"metadata", "scalar", *[f"carry_{i}" for i in range(12)]}
        if last in self.observed:
            keys.update(self.fields)
        expected = self.metadata(index)
        actual = {k: v for k, v in metadata.items() if k != "payload_sha256"}
        if (
            set(arrays) != keys
            or actual != expected
            or metadata.get("payload_sha256") != payload_digest(arrays)
        ):
            raise ValueError("FEM chunk context, sequence or array schema differs")
        values = arrays["scalar"]
        if values.shape != ((last - first) * self.factor, 17) or values.dtype != np.float64:
            raise ValueError("FEM chunk does not contain every fine float64 scalar row")
        for i, (shape, dtype) in enumerate(self.templates):
            a = arrays[f"carry_{i}"]
            if a.shape != shape or a.dtype.str != dtype:
                raise ValueError("FEM accepted carry shape/dtype differs")
        for name, shape in self.fields.items():
            if last in self.observed and (
                arrays[name].shape != shape or arrays[name].dtype != np.float64
            ):
                raise ValueError("FEM observation shape/dtype differs")
        if not all(np.all(np.isfinite(a)) for k, a in arrays.items() if k != "metadata"):
            raise ValueError("Nonfinite FEM state or observations")

    def restore(self):
        """Validate every committed file; ignore only uncommitted pending files."""
        if self.paths:
            raise ValueError("Restore a chunk store only once")
        expected = [self.folder / f"chunk-{i:05}.npz" for i in range(len(self.stops))]
        present = sorted(self.folder.glob("chunk-*.npz"))
        if present != expected[: len(present)]:
            raise ValueError("FEM chunks are noncontiguous or outside the planned window")
        state = None
        for path in present:
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            metadata = json.loads(str(arrays["metadata"]))
            self.validate(arrays, metadata)
            self.paths.append(path)
            self.hashes.append(digest(path))
            state = tuple(arrays[f"carry_{i}"] for i in range(12))
        return state

    def append(self, carry, scalar, fields):
        index = len(self.paths)
        if index >= len(self.stops):
            raise ValueError("FEM planned window is already complete")
        metadata = self.metadata(index)
        data = dict(
            scalar=np.asarray(scalar),
            **{f"carry_{i}": np.asarray(a) for i, a in enumerate(carry)},
            **fields,
        )
        metadata["payload_sha256"] = payload_digest(data)
        data["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
        self.validate(data, metadata)
        path = self.folder / f"chunk-{index:05}.npz"
        atomic_npz(path, data)
        self.paths.append(path)
        self.hashes.append(digest(path))

    def combine(self, dt, y):
        if len(self.paths) != len(self.stops):
            raise ValueError("Do not export an incomplete FEM window as completed")
        arrays, observations, rows, steps = [], {k: [] for k in self.fields}, [], []
        final = None
        for path, fingerprint in zip(self.paths, self.hashes, strict=True):
            if digest(path) != fingerprint:
                raise ValueError("A committed FEM chunk changed")
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"]))
                scalar = data["scalar"]
                arrays.append(scalar)
                if metadata["last"] in self.observed:
                    steps.append(metadata["last"])
                    rows.append(scalar[-1])
                    for name in observations:
                        observations[name].append(data[name])
                final = {f"final_carry_{i}": data[f"carry_{i}"] for i in range(12)}
        return dict(
            time=np.asarray(steps) * dt,
            step_id=np.asarray(steps),
            contact_y=np.asarray(y),
            scalar_history=np.asarray(rows),
            all_scalar_history=np.concatenate(arrays),
            **{k: np.asarray(a) for k, a in observations.items()},
            **final,
        )

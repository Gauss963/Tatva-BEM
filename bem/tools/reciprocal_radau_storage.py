"""Experimental blocked reciprocal storage for two physical Radau branches.

Each off-diagonal block is stored once. The omitted block is the bilinear
transpose. Both original half-circle branches remain available after inversion.
"""

import hashlib
import os
import runpy
import shutil
from pathlib import Path

import numpy as np
from scipy.fft import irfft

TOOLS = Path(__file__).resolve().parent
RAW = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))
STORAGE = runpy.run_path(str(TOOLS / "compact_radau_storage.py"))


def spectral_tiles(dofs, half, block):
    if not isinstance(block, int) or isinstance(block, bool) or block < 1:
        raise ValueError("Require a positive reciprocal block width")
    offset = 0
    for i in range(0, dofs, block):
        for j in range(i, dofs, block):
            rows = slice(i, min(i + block, dofs))
            cols = slice(j, min(j + block, dofs))
            shape = (rows.stop - i, cols.stop - j)
            yield rows, cols, shape, offset
            offset += half * int(np.prod(shape)) * np.dtype(np.complex128).itemsize


def reciprocal_size(plan, block):
    half = plan.fft_size // 2 + 1
    return sum(
        half * shape[0] * shape[1] * 16 for _, _, shape, _ in spectral_tiles(plan.dofs, half, block)
    )


def matrix_tile_bytes(matrices, tiles):
    for rows, cols, _, _ in tiles:
        yield np.ascontiguousarray(matrices[:, rows, cols])


def saved_tile_digest(path, tiles, first, last):
    """Hash exactly the ordered tile payload of one consecutive frequency interval."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for _, _, shape, offset in tiles:
            stride = shape[0] * shape[1] * 16
            stream.seek(offset + first * stride)
            remaining = (last - first) * stride
            while remaining:
                block = stream.read(min(remaining, 4 * 1024**2))
                if not block:
                    raise ValueError("Truncated reciprocal spectrum tile")
                digest.update(block)
                remaining -= len(block)
    return digest.hexdigest()


class ReciprocalBranch:
    """Append increasing frequency batches to a tile-major file."""

    def __init__(self, path, plan, *, block=32, reciprocity_rtol=1e-12, verified_prefix=None):
        self.path = Path(path)
        self.dofs = plan.dofs
        self.half = plan.fft_size // 2 + 1
        self.tiles = tuple(spectral_tiles(plan.dofs, self.half, block))
        self.nbytes = reciprocal_size(plan, block)
        self.rtol = reciprocity_rtol
        self.next = 0
        self.maximum_relative_skew = 0.0
        if not 0 < reciprocity_rtol <= 1e-12:
            raise ValueError("Require a reciprocal relative tolerance in (0, 1e-12]")
        if self.path.resolve().is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
            raise ValueError("Choose owned output outside protected Tatva")
        if self.path.exists():
            raise FileExistsError(self.path)
        self.partial = self.path.with_name(self.path.name + ".part")
        if verified_prefix is None:
            if shutil.disk_usage(self.path.parent).free < self.nbytes:
                raise ValueError("Insufficient storage for the complete reciprocal branch")
            self.stream = self.partial.open("xb")
            try:
                self.stream.truncate(self.nbytes)
            except BaseException:
                self.stream.close()
                raise
        else:
            if (
                not isinstance(verified_prefix, int)
                or isinstance(verified_prefix, bool)
                or not 0 <= verified_prefix <= self.half
                or self.partial.stat().st_size != self.nbytes
            ):
                raise ValueError("Invalid verified reciprocal prefix")
            self.next = verified_prefix
            self.stream = self.partial.open("r+b")

    def __enter__(self):
        return self

    def __exit__(self, error_type, *_):
        try:
            if error_type is None:
                if self.next != self.half:
                    raise ValueError("Incomplete reciprocal branch frequency grid")
                self.sync()
                os.link(self.partial, self.path)
                self.partial.unlink()
        finally:
            self.stream.close()

    def sync(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def check_batch(self, matrices):
        if (
            matrices.ndim != 3
            or matrices.shape[1:] != (self.dofs, self.dofs)
            or matrices.dtype != np.complex128
            or not len(matrices)
            or not np.all(np.isfinite(matrices))
        ):
            raise ValueError("Require complete finite complex128 frequency matrices")
        norm = np.linalg.norm(matrices, axis=(1, 2))
        skew = np.linalg.norm(matrices - matrices.transpose(0, 2, 1), axis=(1, 2))
        if not np.all(np.isfinite(norm)) or not np.all(np.isfinite(skew)):
            raise ValueError("Nonfinite reciprocal norm")
        error = skew / np.maximum(norm, np.finfo(np.float64).tiny)
        if np.any(error > self.rtol):
            raise ValueError("Frequency matrix fails bilinear reciprocity")
        return float(np.max(error))

    def append(self, matrices):
        matrices = np.asarray(matrices)
        if len(matrices) > self.half - self.next:
            raise ValueError("Reciprocal batch exceeds the frequency grid")
        error = self.check_batch(matrices)
        digest = hashlib.sha256()
        for tile, (_, _, shape, offset) in zip(
            matrix_tile_bytes(matrices, self.tiles), self.tiles, strict=True
        ):
            self.stream.seek(offset + self.next * shape[0] * shape[1] * 16)
            payload = memoryview(tile).cast("B")
            self.stream.write(payload)
            digest.update(payload)
        self.next += len(matrices)
        self.maximum_relative_skew = max(self.maximum_relative_skew, error)
        return digest.hexdigest()


def read_frequency_matrix(path, plan, index, *, block=32):
    """Reconstruct one work-coordinate matrix for a fresh operator audit."""
    count = plan.fft_size // 2 + 1
    if not 0 <= index < count or Path(path).stat().st_size != reciprocal_size(plan, block):
        raise ValueError("Invalid reciprocal channel size or frequency index")
    matrix = np.empty((plan.dofs, plan.dofs), np.complex128)
    with Path(path).open("rb") as stream:
        for rows, cols, shape, offset in spectral_tiles(plan.dofs, count, block):
            entries = shape[0] * shape[1]
            stream.seek(offset + index * entries * 16)
            value = np.fromfile(stream, dtype=np.complex128, count=entries).reshape(shape)
            matrix[rows, cols] = value
            if rows != cols:
                matrix[cols, rows] = value.T
    return matrix


def inversion_workspace_bytes(plan, block):
    half = plan.fft_size // 2 + 1
    tile_entries = min(plan.dofs, block) ** 2
    return (
        12 * half * 16 + 3 * plan.fft_size * 8 + 2 * (plan.steps + 1) * 8
    ) * tile_entries + plan.dofs**2 * 8


def invert_reciprocal_branches(
    paths, plan, line_weights, output, *, block=32, max_workspace_mb=1024, on_create=None
):
    """Invert all requested lags directly to both dense physical weights."""
    paths = tuple(Path(path).resolve() for path in paths)
    output = Path(output).resolve()
    if np.iscomplexobj(line_weights):
        raise ValueError("Line weights must be real")
    line = np.asarray(line_weights, dtype=np.float64)
    if (
        len(paths) != 2
        or paths[0] == paths[1]
        or any(path.name.endswith(".part") for path in paths)
        or line.shape != (plan.dofs,)
        or not np.all(np.isfinite(line))
        or np.any(line <= 0)
        or not 1 <= plan.steps < plan.fft_size // 2
        or not 0 < plan.radius < 1
    ):
        raise ValueError("Require both reciprocal spectra, valid contour and positive line weights")
    if output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose owned output outside protected Tatva")
    if any(path.stat().st_size != reciprocal_size(plan, block) for path in paths):
        raise ValueError("Reciprocal branch storage differs from the declared tile layout")
    half = plan.fft_size // 2 + 1
    shape = (plan.steps + 1, plan.dofs, plan.dofs)
    workspace = inversion_workspace_bytes(plan, block)
    if not np.isfinite(max_workspace_mb) or workspace > max_workspace_mb * 1024**2:
        raise ValueError("Reciprocal inversion exceeds its array workspace budget")
    weight_bytes = 2 * int(np.prod(shape)) * 8
    if shutil.disk_usage(output.parent).free < weight_bytes + 1024**2:
        raise ValueError("Insufficient storage for both dense physical weight families")
    output.mkdir()
    if on_create is not None:
        on_create(output)
    arrays = []
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(half) / plan.fft_size)
    eigenvalues = RAW["lower_half_radau_modes"](zeta, 1.0)
    amplification = plan.radius ** (-np.arange(plan.steps + 1))[:, None, None]
    root = np.sqrt(line)
    wrap = np.empty(shape[1:])
    try:
        for i in range(2):
            arrays.append(
                np.lib.format.open_memmap(
                    output / f"weights{i}.npy",
                    mode="w+",
                    dtype=np.float64,
                    shape=shape,
                    fortran_order=True,
                )
            )
        with paths[0].open("rb") as first, paths[1].open("rb") as second:
            for rows, cols, tile_shape, offset in spectral_tiles(plan.dofs, half, block):
                values = np.empty((half, 2, tile_shape[0] * tile_shape[1]), np.complex128)
                for mode, stream in enumerate((first, second)):
                    stream.seek(offset)
                    values[:, mode] = np.fromfile(
                        stream, dtype=np.complex128, count=half * values.shape[-1]
                    ).reshape(half, -1)
                for index, family in enumerate(STORAGE["compact_rows"](eigenvalues, values)):
                    family = family.reshape(half, *tile_shape)
                    endpoints = family[[0, -1]]
                    if np.any(
                        np.linalg.norm(endpoints.imag, axis=(1, 2))
                        > 1e-10 * np.maximum(np.linalg.norm(endpoints.real, axis=(1, 2)), 1e-300)
                    ):
                        raise ValueError(
                            "Reciprocal compact spectrum has a complex real-axis endpoint"
                        )
                    family[[0, -1]] = endpoints.real
                    transformed = irfft(family, n=plan.fft_size, axis=0)
                    physical = transformed[: shape[0]] * amplification
                    physical /= root[rows, None] * root[None, cols]
                    arrays[index][:, rows, cols] = physical
                    if rows != cols:
                        arrays[index][:, cols, rows] = physical.transpose(0, 2, 1)
                    if index == 1:
                        wrap[rows, cols] = (
                            plan.radius
                            * transformed[-1]
                            / np.sqrt(line[rows, None] * line[None, cols])
                        )
                        if rows != cols:
                            wrap[cols, rows] = wrap[rows, cols].T
        for array in arrays:
            array.flush()
        np.save(output / "shift_wrap.npy", wrap, allow_pickle=False)
    finally:
        for array in arrays:
            array._mmap.close()
    return wrap

"""Experimental, grid-verified polynomial storage of a completed CQ spectrum.

This compresses stored data, not BEM evaluations or causal time history.
Every represented grid matrix is compared to its source; intervals that
fail are bisected or saved directly. No continuous-frequency bound is claimed.
"""

import argparse
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.interpolate import BarycentricInterpolator

from finiteblock_bem.io import fingerprint, write_json


def clenshaw(coefficients, x):
    x = np.asarray(x).reshape((-1,) + (1,) * (coefficients.ndim - 1))
    shape = (len(x), *coefficients.shape[1:])
    first, second = np.zeros(shape, complex), np.zeros(shape, complex)
    for coefficient in coefficients[:0:-1]:
        first, second = 2 * x * first - second + coefficient, first
    return x * first - second + coefficients[0]


def source_block(source, first, last):
    result = np.array(source[first:last], dtype=complex, copy=True)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Source spectrum contains nonfinite data")
    if first == 0 and len(result):
        result[0] = result[0].real
    if last == source.shape[0] and len(result):
        result[-1] = result[-1].real
    return result


def compress_spectrum(
    source,
    output,
    *,
    degree=16,
    rtol=1e-13,
    atol=1e-18,
    max_storage_mb=4096,
    provenance=None,
    source_check=None,
):
    implementation_sha = fingerprint(Path(__file__))
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new empty output directory")
    if (
        len(source.shape) != 3
        or source.shape[1] != source.shape[2]
        or source.shape[0] < 3
        or source.shape[1] < 2
    ):
        raise ValueError("A square matrix half spectrum is required")
    if (
        not isinstance(degree, int)
        or degree < 2
        or not np.isfinite(rtol + atol)
        or rtol <= 0
        or atol < 0
        or not np.isfinite(max_storage_mb)
        or max_storage_mb <= 0
    ):
        raise ValueError("Invalid compression settings")
    output.mkdir(parents=True, exist_ok=True)
    count, d, _ = source.shape
    endpoint_reserve = 2 * d * d * 16 + 256
    for index in [0, count - 1]:
        matrix = source[index]
        if np.linalg.norm(matrix.imag) > 1e-10 * max(np.linalg.norm(matrix.real), 1):
            raise ValueError("Real-axis endpoint violates CQ conjugate symmetry")
    nodes = np.cos(np.pi * np.arange(degree + 1) / degree)
    stack, pieces, stored = [(0, count)], [], 0
    maximum_error, interpolated, direct = 0.0, 0, 0
    while stack:
        left, right = stack.pop()
        length = right - left
        error = 0.0
        if length <= 2 * (degree + 1):
            values, kind = source_block(source, left, right), "direct"
        else:
            indices = np.rint(left + (nodes + 1) * (length - 1) / 2).astype(int)
            if len(np.unique(indices)) != degree + 1:
                middle = (left + right) // 2
                stack.extend([(middle, right), (left, middle)])
                continue
            xnodes = 2 * (indices - left) / (length - 1) - 1
            samples = np.stack([source_block(source, int(k), int(k) + 1)[0] for k in indices])
            values = np.linalg.solve(
                chebvander(xnodes, degree), samples.reshape(degree + 1, -1)
            ).reshape(degree + 1, d, d)
            kind = "polynomial"
            for first in range(left, right, 16):
                last = min(first + 16, right)
                exact = source_block(source, first, last)
                predicted = clenshaw(values, 2 * (np.arange(first, last) - left) / (length - 1) - 1)
                norm = np.linalg.norm(exact, axis=(1, 2))
                if not np.all(np.isfinite(norm)):
                    error = np.inf
                    break
                difference = np.linalg.norm(predicted - exact, axis=(1, 2))
                tolerance = atol + rtol * norm
                normalized = np.divide(
                    difference,
                    tolerance,
                    out=np.where(difference == 0, 0.0, np.inf),
                    where=tolerance > 0,
                )
                if not np.all(np.isfinite(normalized)):
                    error = np.inf
                    break
                error = max(error, float(np.max(normalized)))
                if not np.isfinite(error) or error > 1:
                    break
            if not np.isfinite(error) or error > 1:
                middle = (left + right) // 2
                stack.extend([(middle, right), (left, middle)])
                continue
        needed = values.nbytes + 1024
        if (
            stored + needed + endpoint_reserve > max_storage_mb * 1024**2
            or shutil.disk_usage(output).free < needed + endpoint_reserve
        ):
            raise ValueError("Compressed representation exceeds the explicit storage budget")
        filename = f"piece-{len(pieces):06d}.npy"
        np.save(output / filename, values, allow_pickle=False)
        stored += (output / filename).stat().st_size
        pieces.append(
            dict(
                left=left,
                right=right,
                kind=kind,
                filename=filename,
                sha256=fingerprint(output / filename),
                maximum_normalized_grid_error=error,
            )
        )
        maximum_error = max(maximum_error, error)
        if kind == "direct":
            direct += length
        else:
            interpolated += length
    np.save(
        output / "endpoints.npy",
        np.stack([source_block(source, k, k + 1)[0] for k in [0, count - 1]]),
        allow_pickle=False,
    )
    pieces.sort(key=lambda piece: piece["left"])
    record = dict(
        format_version=1,
        status="completed_grid_verified_spectrum",
        shape=list(source.shape),
        dtype="complex128",
        degree=degree,
        relative_tolerance=rtol,
        absolute_tolerance=atol,
        pieces=pieces,
        endpoints_sha256=fingerprint(output / "endpoints.npy"),
        direct_grid_points=direct,
        polynomial_grid_points=interpolated,
        maximum_normalized_grid_error=maximum_error,
        all_grid_points_checked=True,
        certified_continuous_frequency_bound=False,
        raw_array_bytes=count * d * d * 16,
        stored_array_bytes=stored + (output / "endpoints.npy").stat().st_size,
        provenance=provenance,
        implementation_sha256=fingerprint(Path(__file__)),
        endpoint_projection="Standard CQ real-axis endpoint projection",
        scope="Storage of a completed discrete frequency grid; no history truncation or reduction in source BEM evaluations",
    )
    if source_check is not None:
        source_check()
    if implementation_sha != fingerprint(Path(__file__)):
        raise ValueError("Compression implementation changed during construction")
    write_json(output / "summary.json", record)
    return PolynomialSpectrum(output)


class PolynomialSpectrum:
    """Read complete matrices or row/column slices without a dense spectrum."""

    def __init__(self, folder, *, verify=True, _allow_pending=False):
        self.folder = Path(folder)
        self.metadata = json.loads((self.folder / "summary.json").read_text())
        meta = self.metadata
        accepted_status = meta["status"] == "completed_grid_verified_spectrum" or (
            _allow_pending and meta["status"] == "pending_native_source_validation"
        )
        if meta["format_version"] not in [1, 2] or not accepted_status:
            raise ValueError("A completed supported representation is required")
        self.shape, self.dtype = tuple(meta["shape"]), np.dtype(meta["dtype"])
        self.pieces = meta["pieces"]
        covered = 0
        for piece in self.pieces:
            if piece["left"] != covered or not covered < piece["right"] <= self.shape[0]:
                raise ValueError("Spectrum pieces must cover the full grid exactly once")
            if piece["kind"] not in (
                ["direct", "polynomial", "barycentric"]
                if meta["format_version"] == 2
                else ["direct", "polynomial"]
            ):
                raise ValueError("Unknown spectrum piece")
            if verify and fingerprint(self.folder / piece["filename"]) != piece["sha256"]:
                raise ValueError("Spectrum piece digest differs")
            covered = piece["right"]
        if covered != self.shape[0] or not meta["all_grid_points_checked"]:
            raise ValueError("Incomplete spectrum coverage")
        if verify and fingerprint(self.folder / "endpoints.npy") != meta["endpoints_sha256"]:
            raise ValueError("Spectrum endpoint digest differs")
        self.endpoints = np.load(self.folder / "endpoints.npy", allow_pickle=False)
        self.endpoints.flags.writeable = False
        if meta["format_version"] == 2:
            for filename in ["grid_x.npy", "nodes.npy", "barycentric_weights.npy"]:
                if (
                    verify
                    and fingerprint(self.folder / filename) != meta["auxiliary_sha256"][filename]
                ):
                    raise ValueError("Native spectrum coordinate digest differs")
            self.grid_x = np.load(self.folder / "grid_x.npy", allow_pickle=False)
            self.nodes = np.load(self.folder / "nodes.npy", allow_pickle=False)
            self.barycentric_weights = np.load(
                self.folder / "barycentric_weights.npy", allow_pickle=False
            )
            for array in [self.grid_x, self.nodes, self.barycentric_weights]:
                array.flags.writeable = False

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        axis, tail = key[0], key[1:]
        integer = isinstance(axis, (int, np.integer))
        if integer:
            if not -self.shape[0] <= axis < self.shape[0]:
                raise IndexError("Frequency outside the represented grid")
            first, last = int(axis) % self.shape[0], int(axis) % self.shape[0] + 1
        elif isinstance(axis, slice):
            first, last, stride = axis.indices(self.shape[0])
            if stride != 1:
                raise ValueError("Only unit-stride frequency reads supported")
        else:
            raise ValueError("Unsupported frequency index")
        shape = np.broadcast_to(np.empty((), complex), self.shape[1:])[tail].shape
        result = np.empty((max(last - first, 0), *shape), complex)
        for piece in self.pieces:
            left, right = max(first, piece["left"]), min(last, piece["right"])
            if right <= left:
                continue
            mapped = np.load(self.folder / piece["filename"], mmap_mode="r", allow_pickle=False)
            try:
                if piece["kind"] == "direct":
                    result[left - first : right - first] = mapped[
                        (slice(left - piece["left"], right - piece["left"]), *tail)
                    ]
                elif piece["kind"] == "polynomial":
                    coefficients = np.array(mapped[(slice(None), *tail)], copy=True)
                    for start in range(left, right, 16):
                        stop = min(start + 16, right)
                        x = (
                            2
                            * (np.arange(start, stop) - piece["left"])
                            / (piece["right"] - piece["left"] - 1)
                            - 1
                        )
                        result[start - first : stop - first] = clenshaw(coefficients, x)
                else:
                    values = np.array(mapped[(slice(None), *tail)], copy=True)
                    interpolant = BarycentricInterpolator(
                        self.nodes, values, wi=self.barycentric_weights, axis=0
                    )
                    for start in range(left, right, 16):
                        stop = min(start + 16, right)
                        result[start - first : stop - first] = interpolant(self.grid_x[start:stop])
            finally:
                mapped._mmap.close()
        if first == 0 and last > first:
            result[0] = self.endpoints[(0, *tail)]
        if last == self.shape[0] and last > first:
            result[-1] = self.endpoints[(1, *tail)]
        return result[0] if integer else result

    def __setitem__(self, key, value):
        if isinstance(key, (int, np.integer)) and key in [0, self.shape[0] - 1]:
            if np.array_equal(self[key], value):
                return
        raise ValueError("Completed polynomial spectrum is read-only")

    def record(self):
        return dict(
            backend="polynomial_half_spectrum",
            path=str(self.folder.resolve()),
            shape=list(self.shape),
            sha256=fingerprint(self.folder / "summary.json"),
            bytes=sum(path.stat().st_size for path in self.folder.glob("*.npy"))
            + (self.folder / "summary.json").stat().st_size,
            all_frequencies_represented=True,
            implementation_sha256=fingerprint(Path(__file__)),
        )


def main():
    from finiteblock_bem.cq import CQPlan

    helper = runpy.run_path(str(Path(__file__).with_name("replay_resolvent.py")))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path, help="Completed direct-resolvent kernel directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-13)
    parser.add_argument("--atol", type=float, default=1e-18)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--max-storage-mb", type=float, default=4096)
    args = parser.parse_args()
    original = json.loads((args.kernel / "summary.json").read_text())
    if original["status"] != "completed_resolvent_kernel_only":
        raise ValueError("A completed kernel is required")
    storage = original["spectrum_storage"]
    if (
        not storage["all_frequency_matrices_written"]
        or fingerprint(Path(storage["path"])) != storage["sha256"]
    ):
        raise ValueError("Incomplete or changed source spectrum")
    source = helper["SavedDiskSpectrum"](storage["path"], CQPlan(**original["plan"]))

    def unchanged_source():
        if fingerprint(Path(storage["path"])) != storage["sha256"]:
            raise ValueError("Source spectrum changed during compression")

    result = compress_spectrum(
        source,
        args.output,
        degree=args.degree,
        rtol=args.rtol,
        atol=args.atol,
        max_storage_mb=args.max_storage_mb,
        source_check=unchanged_source,
        provenance=dict(
            source_kernel=str(args.kernel.resolve()),
            source_summary_sha256=fingerprint(args.kernel / "summary.json"),
            source_spectrum=storage,
            plan=original["plan"],
        ),
    )
    print(json.dumps({k: v for k, v in result.metadata.items() if k != "pieces"}, indent=2))


if __name__ == "__main__":
    main()

"""Store the unchanged adaptive sampler's interpolants as they are accepted.

The sampler is cloned with two private bindings. Its matrices, error tests,
point choices and call count remain unchanged; no module globals are replaced.
Plain arrays preserve all original arithmetic. A weak registry associates an
interpolant's returned array with its exact nodes and normalized coordinates.
"""

import itertools
import mmap
import runpy
import shutil
import weakref
from dataclasses import asdict
from pathlib import Path
from types import FunctionType

import numpy as np
from scipy.fft import irfft
from scipy.interpolate import BarycentricInterpolator

from finiteblock_bem.cq import invert_spectrum
from finiteblock_bem.io import fingerprint, write_json
from finiteblock_bem.sampling import SamplingOptions, parsimonious_weights

READER_PATH = Path(__file__).with_name("polynomial_spectrum.py")


def invert_spectrum_to_file(spectrum, plan, path, *, fft_rows=4, max_workspace_mb=2048):
    """Use the original row IFFTs, writing time weights into an exclusive file.

    The matrix prefix is never allocated as an ordinary ndarray. On systems
    supporting madvise, written mapping pages are released after each row block.
    The workspace bound estimates arrays, not total process RSS or OS cache.
    Every requested lag is retained, including the full damped-contour scaling.
    """
    if not isinstance(fft_rows, int) or fft_rows < 1:
        raise ValueError("fft_rows must be a positive integer")
    if not np.isfinite(max_workspace_mb) or max_workspace_mb <= 0:
        raise ValueError("A finite positive IFFT workspace budget is required")
    if spectrum.shape != (plan.fft_size // 2 + 1, plan.dofs, plan.dofs):
        raise ValueError("Incorrect CQ half-spectrum shape")
    rows = min(fft_rows, plan.dofs)
    estimated = 4 * plan.fft_size * rows * plan.dofs * 8
    estimated += (plan.steps + 1) * rows * plan.dofs * 8
    estimated += 64 * plan.dofs**2 * 16
    if estimated > max_workspace_mb * 1024**2:
        raise ValueError(f"IFFT array workspace needs about {estimated / 1024**2:.1f} MiB")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = (plan.steps + 1) * plan.dofs**2 * 8
    if shutil.disk_usage(path.parent).free < size + 8192:
        raise ValueError("Insufficient disk space for all requested time weights")
    for k in [0, plan.fft_size // 2]:
        value = spectrum[k]
        if np.linalg.norm(value.imag) > 1e-10 * max(np.linalg.norm(value.real), 1):
            raise ValueError("Real-axis transfer violates conjugate symmetry")
        spectrum[k] = value.real
    # Exclusive creation protects previous kernels, even if two callers race.
    with path.open("xb") as stream:
        np.lib.format.write_array_header_2_0(
            stream,
            dict(
                descr=np.dtype("float64").str,
                fortran_order=False,
                shape=(plan.steps + 1, plan.dofs, plan.dofs),
            ),
        )
        stream.truncate(stream.tell() + size)
    weights = np.load(path, mmap_mode="r+", allow_pickle=False)
    amplification = plan.radius ** (-np.arange(plan.steps + 1))
    for first in range(0, plan.dofs, rows):
        transformed = irfft(spectrum[:, first : first + rows], n=plan.fft_size, axis=0)
        weights[:, first : first + rows] = (
            transformed[: plan.steps + 1] * amplification[:, None, None]
        )
        del transformed
        weights.flush()
        if hasattr(weights._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
            weights._mmap.madvise(mmap.MADV_DONTNEED)
    weights._mmap.close()
    return np.load(path, mmap_mode="r", allow_pickle=False)


class TrackedInterpolator:
    def __init__(self, registry, serial, nodes, values, *, wi, axis):
        self.registry, self.serial = registry, serial
        self.nodes, self.values, self.wi = (np.array(a, copy=True) for a in [nodes, values, wi])
        self.interpolant = BarycentricInterpolator(nodes, values, wi=wi, axis=axis)

    def __call__(self, x):
        result = self.interpolant(x)
        key = id(result)
        registry = self.registry

        def remove(reference):
            if registry.get(key, (None,))[0] is reference:
                registry.pop(key, None)

        self.registry[key] = (
            weakref.ref(result, remove),
            weakref.ref(self),
            np.array(x, copy=True),
        )
        return result


class NativeSpectrumSink:
    def __init__(self, folder, plan, registry, reader_class, *, max_storage_mb, rtol, atol):
        if not np.isfinite(max_storage_mb) or max_storage_mb <= 0:
            raise ValueError("A finite positive native storage budget is required")
        self.folder, self.plan, self.registry = Path(folder), plan, registry
        if self.folder.exists() and any(self.folder.iterdir()):
            raise ValueError("Choose a new empty spectrum directory")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.shape, self.dtype = (
            (plan.fft_size // 2 + 1, plan.dofs, plan.dofs),
            np.dtype("complex128"),
        )
        self.written = np.zeros(self.shape[0], dtype=bool)
        self.grid_x = np.full(self.shape[0], np.nan)
        self.pieces, self.stored, self.maximum_error = [], 0, 0.0
        self.nodes, self.wi, self.active_serial, self.decoder = None, None, None, None
        self.reader_class, self.reader = reader_class, None
        self.max_bytes, self.rtol, self.atol = max_storage_mb * 1024**2, rtol, atol

    def save(self, values):
        reserve = self.grid_x.nbytes + 2 * self.plan.dofs**2 * 16 + 8192
        needed = values.nbytes + 1024
        if (
            self.stored + needed + reserve > self.max_bytes
            or shutil.disk_usage(self.folder).free < needed + reserve
        ):
            raise ValueError("Native spectrum exceeds its explicit storage budget")
        filename = f"piece-{len(self.pieces):06d}.npy"
        np.save(self.folder / filename, values, allow_pickle=False)
        self.stored += (self.folder / filename).stat().st_size
        return filename, fingerprint(self.folder / filename)

    def __setitem__(self, key, values):
        if self.reader is not None:
            self.reader[key] = values
            return
        if not isinstance(key, slice):
            raise ValueError("Sampling requires complete frequency intervals")
        first, last, stride = key.indices(self.shape[0])
        if stride != 1 or last <= first or np.any(self.written[first:last]):
            raise ValueError("Frequency intervals must be written exactly once")
        association = self.registry.get(id(values))
        proxy = association[1]() if association is not None and association[0]() is values else None
        array = np.asarray(values, dtype=complex)
        if array.shape != (last - first, self.plan.dofs, self.plan.dofs) or not np.all(
            np.isfinite(array)
        ):
            raise ValueError("Invalid native frequency matrices")
        if proxy is None:
            filename, digest = self.save(array)
            mapped = np.load(self.folder / filename, mmap_mode="r", allow_pickle=False)
            try:
                for start in range(0, len(array), 16):
                    if not np.array_equal(mapped[start : start + 16], array[start : start + 16]):
                        raise ValueError("Direct spectrum roundtrip differs")
            finally:
                mapped._mmap.close()
            self.pieces.append(
                dict(left=first, right=last, kind="direct", filename=filename, sha256=digest)
            )
            self.active_serial, self.decoder = None, None
        else:
            x = association[2]
            if self.active_serial != proxy.serial:
                filename, digest = self.save(proxy.values)
                saved = np.load(self.folder / filename, allow_pickle=False)
                self.decoder = BarycentricInterpolator(proxy.nodes, saved, wi=proxy.wi, axis=0)
                if self.nodes is not None and (
                    not np.array_equal(self.nodes, proxy.nodes)
                    or not np.array_equal(self.wi, proxy.wi)
                ):
                    raise ValueError("Adaptive interpolation basis changed")
                self.nodes, self.wi = proxy.nodes.copy(), proxy.wi.copy()
                self.active_serial = proxy.serial
                self.pieces.append(
                    dict(
                        left=first,
                        right=first,
                        kind="barycentric",
                        filename=filename,
                        sha256=digest,
                    )
                )
            if self.pieces[-1]["right"] != first or x.shape != (last - first,):
                raise ValueError("Interpolant grid coordinates are not contiguous")
            decoded = self.decoder(x)
            tolerance = self.atol + self.rtol * np.linalg.norm(array, axis=(1, 2))
            difference = np.linalg.norm(decoded - array, axis=(1, 2))
            normalized = np.divide(
                difference,
                tolerance,
                out=np.where(difference == 0, 0.0, np.inf),
                where=tolerance > 0,
            )
            if not np.all(np.isfinite(normalized)) or np.max(normalized) > 1:
                raise ValueError("Serialized interpolant differs from baseline grid values")
            self.maximum_error = max(self.maximum_error, float(np.max(normalized)))
            self.grid_x[first:last] = x
            self.pieces[-1]["right"] = last
        self.written[first:last] = True

    def __getitem__(self, key):
        if self.reader is None:
            raise ValueError("Native spectrum is not sealed")
        return self.reader[key]

    def seal(self, provenance):
        if not np.all(self.written):
            raise ValueError("Cannot seal an incomplete native frequency grid")
        self.pieces.sort(key=lambda piece: piece["left"])
        endpoints = []
        for index in [0, self.shape[0] - 1]:
            piece = next(p for p in self.pieces if p["left"] <= index < p["right"])
            values = np.load(self.folder / piece["filename"], allow_pickle=False)
            if piece["kind"] == "direct":
                matrix = values[index - piece["left"]]
            else:
                matrix = BarycentricInterpolator(self.nodes, values, wi=self.wi, axis=0)(
                    self.grid_x[index]
                )
            if np.linalg.norm(matrix.imag) > 1e-10 * max(np.linalg.norm(matrix.real), 1):
                raise ValueError("Native real-axis endpoint violates conjugate symmetry")
            endpoints.append(matrix.real.astype(complex))
        auxiliary = dict(
            grid_x=self.grid_x,
            nodes=np.array([]) if self.nodes is None else self.nodes,
            barycentric_weights=np.array([]) if self.wi is None else self.wi,
        )
        hashes = {}
        for name, values in auxiliary.items():
            filename = name + ".npy"
            np.save(self.folder / filename, values, allow_pickle=False)
            hashes[filename] = fingerprint(self.folder / filename)
        np.save(self.folder / "endpoints.npy", np.stack(endpoints), allow_pickle=False)
        metadata = dict(
            format_version=2,
            status="pending_native_source_validation",
            shape=list(self.shape),
            dtype="complex128",
            pieces=self.pieces,
            auxiliary_sha256=hashes,
            endpoints_sha256=fingerprint(self.folder / "endpoints.npy"),
            all_grid_points_checked=True,
            validation_target="Original adaptive interpolant grid values; not direct BEM values at every frequency",
            maximum_normalized_grid_error=self.maximum_error,
            relative_tolerance=self.rtol,
            absolute_tolerance=self.atol,
            raw_array_bytes=np.prod(self.shape).item() * 16,
            stored_array_bytes=sum(p.stat().st_size for p in self.folder.glob("*.npy")),
            certified_continuous_frequency_bound=False,
            provenance=provenance,
            scope="Native storage of original adaptive interpolants and direct samples; no complete dense spectrum is allocated",
        )
        write_json(self.folder / "summary.json", metadata)
        self.reader = self.reader_class(self.folder, _allow_pending=True)


def native_parsimonious_weights(
    transfer,
    plan,
    folder,
    *,
    options=None,
    row_scale=None,
    workers=1,
    progress=None,
    max_storage_mb=4096,
    provenance=None,
    source_check=None,
    weight_path=None,
    fft_rows=4,
    max_workspace_mb=2048,
):
    options = SamplingOptions() if options is None else options
    own_sha, reader_sha = fingerprint(Path(__file__)), fingerprint(READER_PATH)
    sampler_sha = fingerprint(Path(parsimonious_weights.__code__.co_filename))
    reader_class = runpy.run_path(str(READER_PATH))["PolynomialSpectrum"]
    registry, serials = {}, itertools.count()
    sink = NativeSpectrumSink(
        folder, plan, registry, reader_class, max_storage_mb=max_storage_mb, rtol=1e-13, atol=1e-18
    )

    def interpolator(*args, **kwargs):
        return TrackedInterpolator(registry, next(serials), *args, **kwargs)

    def finish(spectrum, candidate):
        if spectrum is not sink or candidate != plan:
            raise ValueError("Unexpected native sampling target")
        sink.seal(provenance)
        if weight_path is None:
            return invert_spectrum(sink, plan)
        return invert_spectrum_to_file(
            sink,
            plan,
            weight_path,
            fft_rows=fft_rows,
            max_workspace_mb=max_workspace_mb,
        )

    clone = FunctionType(
        parsimonious_weights.__code__,
        {
            **parsimonious_weights.__globals__,
            "BarycentricInterpolator": interpolator,
            "invert_spectrum": finish,
        },
        parsimonious_weights.__name__,
        parsimonious_weights.__defaults__,
    )
    clone.__kwdefaults__ = dict(parsimonious_weights.__kwdefaults__)
    weights, records = clone(
        transfer,
        plan,
        options=options,
        row_scale=row_scale,
        workers=workers,
        progress=progress,
        spectrum=sink,
    )
    if source_check is not None:
        source_check()
    if (
        own_sha != fingerprint(Path(__file__))
        or reader_sha != fingerprint(READER_PATH)
        or sampler_sha != fingerprint(Path(parsimonious_weights.__code__.co_filename))
    ):
        raise ValueError("Native sampling implementation changed")
    metadata = dict(
        sink.reader.metadata,
        status="completed_grid_verified_spectrum",
        source_sampling=records,
        sampling_options=asdict(options),
        implementation_sha256=own_sha,
        reader_sha256=reader_sha,
        sampler_sha256=sampler_sha,
        time_weights_storage="ram" if weight_path is None else "disk_mmap",
    )
    write_json(sink.folder / "summary.json", metadata)
    return weights, reader_class(sink.folder), records

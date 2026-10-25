"""Reuse exactly equal radii in the unchanged prepared Galerkin integrals.

This optional experiment reduces repeated Bessel evaluations on straight,
repeated panel patterns. No radius is rounded or interpolated. Angular factors,
singular corrections, integration weights and matrix scatter remain unchanged.
Each frequency owns its radial values, so shared prepared geometry is read-only.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import FunctionType

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.kernels import _radial
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model


class UniqueRadiusPrepared(PreparedGalerkinOperators):
    def __init__(self, *args, radial_chunk_size=4096, max_radial_workspace_mb=512, **kwargs):
        if not isinstance(radial_chunk_size, int) or radial_chunk_size < 1:
            raise ValueError("A positive integer radial chunk size is required")
        if not np.isfinite(max_radial_workspace_mb) or max_radial_workspace_mb <= 0:
            raise ValueError("A finite positive radial workspace budget is required")
        super().__init__(*args, **kwargs)
        radii = [np.linalg.norm(b.vector.reshape(-1, 2), axis=1) for b in self.batches]
        if any(np.any(r <= 0) or not np.all(np.isfinite(r)) for r in radii):
            raise ValueError("Quadrature requires finite positive radii")
        self.unique_radii, inverse = np.unique(np.concatenate(radii), return_inverse=True)
        counts = [len(r) for r in radii]
        index_dtype = np.uint32 if len(self.unique_radii) <= 2**32 else np.uint64
        self.radius_maps = tuple(
            a.astype(index_dtype) for a in np.split(inverse, np.cumsum(counts)[:-1])
        )
        self.radial_chunk_size = min(radial_chunk_size, len(self.unique_radii))
        # Seven complex radial families, one returned batch and conservative
        # bounded special-function temporaries; excludes original assembler arrays.
        self.radial_workspace_bytes = (
            7 * np.dtype(complex).itemsize * (len(self.unique_radii) + max(counts))
            + 1024 * self.radial_chunk_size
        )
        if self.radial_workspace_bytes > max_radial_workspace_mb * 1024**2:
            raise MemoryError("Unique-radius workspace exceeds its requested budget")
        for array in (self.unique_radii, *self.radius_maps):
            array.flags.writeable = False

    def statistics(self):
        count = sum(len(a) for a in self.radius_maps)
        return dict(
            quadrature_radii=count,
            exactly_unique_radii=len(self.unique_radii),
            radial_evaluation_count_ratio=count / len(self.unique_radii),
            radius_index_bytes=self.unique_radii.nbytes + sum(a.nbytes for a in self.radius_maps),
            radial_workspace_estimate_bytes=self.radial_workspace_bytes,
            radius_rounding_or_interpolation=False,
        )

    def assemble(self, s=None):
        if s is None:
            return super().assemble(s)
        if not np.isfinite(s) or np.real(s) <= 0:
            raise ValueError("Laplace parameter must have positive real part")
        values = np.empty((7, len(self.unique_radii)), complex)
        for start in range(0, len(self.unique_radii), self.radial_chunk_size):
            stop = start + self.radial_chunk_size
            values[:, start:stop] = _radial(
                self.unique_radii[start:stop], s, self.material, scalars=True
            )
        maps = iter(self.radius_maps)

        def lookup(r, parameter, material, *, scalars=False):
            indices = next(maps)
            if parameter != s or material is not self.material or not scalars:
                raise ValueError("Radial assembler call differs from its prepared contract")
            if not np.array_equal(r, self.unique_radii[indices]):
                raise ValueError("Quadrature geometry changed after radius preparation")
            return tuple(a[indices] for a in values)

        original = PreparedGalerkinOperators.assemble
        clone = FunctionType(
            original.__code__,
            {**original.__globals__, "_radial": lookup},
            original.__name__,
            original.__defaults__,
        )
        answer = clone(self, s)
        if next(maps, None) is not None:
            raise ValueError("Prepared assembler did not consume every radius map")
        return answer


def unique_radius_transfer(model, *, quadrature=12, **options):
    transfer = NodalInterfaceTransfer(model, quadrature=quadrature, prepared=False)
    transfer.prepared = [
        UniqueRadiusPrepared(mesh, model.reference.material, quadrature=quadrature, **options)
        for mesh in (model.reference.moving, model.reference.stationary)
    ]
    return transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 0.1, 0.5, 2, np.pi])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--radial-chunk-size", type=int, default=4096)
    parser.add_argument("--max-radial-workspace-mb", type=float, default=512)
    args = parser.parse_args()
    if args.repeats < 1 or any(not np.isfinite(a) or not 0 <= a <= np.pi for a in args.angles):
        raise ValueError("Require positive repeats and finite contour angles in [0,pi]")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new empty output directory")
    begun, tool_sha = perf_counter(), fingerprint(Path(__file__))
    manifest_path, summary_path = args.run / "run_manifest.json", args.run / "summary.json"
    options = json.loads(manifest_path.read_text())["arguments"]
    summary = json.loads(summary_path.read_text())
    if "prepared_kernel" not in summary:
        raise ValueError("Require a resolvent trajectory identifying its prepared kernel")
    kernel_path = Path(summary["prepared_kernel"]) / "summary.json"
    kernel = json.loads(kernel_path.read_text())
    frozen = {str(p.resolve()): fingerprint(p) for p in [manifest_path, summary_path, kernel_path]}
    if frozen[str(kernel_path.resolve())] != summary["kernel_summary_sha256"]:
        raise ValueError("Source kernel metadata differs from the recorded trajectory")
    environment = environment_record(options["case"])
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    fft_size, radius = kernel["plan"]["fft_size"], kernel["plan"]["radius"]
    if fft_size != summary["cq_fft_size"] or kernel["plan"]["dt"] != summary["dt_s"]:
        raise ValueError("Kernel and trajectory time-grid metadata differ")
    z = radius * np.exp(-1j * np.asarray(args.angles))
    frequencies = (1.5 - 2 * z + 0.5 * z * z) / summary["dt_s"]
    original = NodalInterfaceTransfer(model, quadrature=options["quadrature"])
    variant = unique_radius_transfer(
        model,
        quadrature=options["quadrature"],
        radial_chunk_size=args.radial_chunk_size,
        max_radial_workspace_mb=args.max_radial_workspace_mb,
    )
    prepared_seconds, samples = perf_counter() - begun, []
    with threadpool_limits(limits=1, user_api="blas"):
        for s in frequencies:
            timing, matrices = {"original": [], "unique_radius": []}, {}
            for repeat in range(args.repeats):
                order = [("original", original), ("unique_radius", variant)]
                if repeat % 2:
                    order.reverse()
                for name, transfer in order:
                    start = perf_counter()
                    matrices[name] = transfer.evaluate(s)
                    timing[name].append(perf_counter() - start)
            expected, actual = matrices["original"], matrices["unique_radius"]
            np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-6)
            samples.append(
                dict(
                    s_real=float(s.real),
                    s_imag=float(s.imag),
                    relative_frobenius_difference=float(
                        np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                    ),
                    timing_seconds=timing,
                )
            )
            print(json.dumps(samples[-1]), flush=True)
    if (
        tool_sha != fingerprint(Path(__file__))
        or environment["source_sha256"] != (environment_record(options["case"])["source_sha256"])
        or any(fingerprint(path) != checksum for path, checksum in frozen.items())
    ):
        raise ValueError("Numerical source changed during the experiment")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "summary.json",
        dict(
            samples=samples,
            body_statistics=[body.statistics() for body in variant.prepared],
            preparation_seconds=prepared_seconds,
            environment=environment,
            experiment_sha256=tool_sha,
            source_run=str(args.run.resolve()),
            frozen_input_sha256=frozen,
            scope="Sampled exact-radius reuse only; no full trajectory, material/geometry AD, or same-accuracy FEM speed claim",
        ),
    )


if __name__ == "__main__":
    main()

"""Compare raw interpolation degrees on scalar elastodynamic Green primitives.

This isolates the raw check/tail criterion. An identity emitter excludes body
condensation and nonlinear dynamics, so evaluation counts are a tuning screen,
not a full-matrix accuracy bound or a solver speed ratio.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.cq import CQPlan
from finiteblock_bem.galerkin import _scalar_kernels
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="Optional kernel/preflight JSON with a CQ plan")
    parser.add_argument("--dt", type=float)
    parser.add_argument("--fft-size", type=int)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--degrees", type=int, nargs="+", default=[8, 12, 16, 24])
    parser.add_argument(
        "--distances", type=float, nargs="+", default=[0.015625, 0.125, 1, 10, 100, 500]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a fresh diagnostic output outside protected Tatva")
    if args.plan is not None and any(v is not None for v in [args.dt, args.fft_size, args.radius]):
        raise ValueError("Choose a source plan or explicit frequency parameters")
    frequency = (
        json.loads(args.plan.read_text())["plan"]
        if args.plan is not None
        else dict(
            dt=2e-6 if args.dt is None else args.dt,
            fft_size=65536 if args.fft_size is None else args.fft_size,
            radius=0.999648715112665 if args.radius is None else args.radius,
        )
    )
    dt, fft_size, radius = (frequency[k] for k in ["dt", "fft_size", "radius"])
    distance = np.asarray(args.distances)
    if (
        not np.isfinite(dt)
        or dt <= 0
        or not np.isfinite(radius)
        or not 0 < radius < 1
        or not isinstance(fft_size, int)
        or fft_size < 2
        or fft_size & (fft_size - 1)
        or not 1 <= len(distance) <= 64
        or np.any(~np.isfinite(distance))
        or np.any(distance <= 0)
        or len(set(args.degrees)) != len(args.degrees)
    ):
        raise ValueError(
            "Require a valid CQ frequency grid, distinct degrees and 1-64 positive distances"
        )
    raw_tool = runpy.run_path(str(TOOLS / "raw_layer_sampling.py"))
    options = [raw_tool["RawSamplingOptions"](degree=d) for d in args.degrees]
    paths = [Path(__file__), TOOLS / "raw_layer_sampling.py", args.case]
    if args.plan is not None:
        paths.append(args.plan)
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    environment = environment_record(args.case)
    model = load_nodal_model(args.case, max_panel_length=160, contact_panel_length=160)
    n = len(distance)
    # Only dt/FFT circle/dofs are used; this diagnostic constructs no time weights.
    plan = CQPlan(dt, 1, n, fft_size, radius, 0, 0)
    mass = np.eye(n)
    vectors = np.column_stack([distance, np.zeros(n)])

    def raw(s):
        gs, gp, _, gradient = _scalar_kernels(vectors, s, model.reference.material)
        return raw_tool["RawLayerPacket"](
            (np.diag(gs), np.diag(gp), np.diag(gradient[:, 0])), (mass,)
        )

    output.mkdir(parents=True)
    rows = []
    for setting in options:
        start = perf_counter()
        row = raw_tool["sample_raw_layers"](
            raw,
            lambda packet, s: mass,
            plan,
            lambda *args: None,
            options=setting,
            workers=1,
            max_workspace_mb=32,
        )
        row.update(degree=setting.degree, elapsed_seconds_diagnostic_only=perf_counter() - start)
        rows.append(row)
        write_json(output / "progress.json", dict(records=rows))
        print(
            json.dumps(dict(degree=setting.degree, evaluations=row["source_evaluations"])),
            flush=True,
        )
    if any(fingerprint(p) != value for p, value in frozen.items()):
        raise ValueError("Primitive screen source/input changed")
    write_json(
        output / "summary.json",
        dict(
            status="completed_primitive_kernel_degree_screen",
            records=rows,
            frequency_grid=dict(dt_s=dt, fft_size=fft_size, radius=radius),
            distances_mm=distance.tolist(),
            frozen_input_sha256=frozen,
            environment=environment,
            output_transform="Constant identity; body elimination/amplification is excluded",
            full_bem_accuracy_or_speed_verified=False,
            timing_scope="Diagnostic timings; CPU contention is not controlled. Compare evaluation counts, not solver speed.",
        ),
    )


if __name__ == "__main__":
    main()

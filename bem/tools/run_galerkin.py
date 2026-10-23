"""Run the Galerkin finite-block backend with original or explicit diagnostic loading."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from finiteblock_bem.dynamics import LoadingSchedule
from finiteblock_bem.io import environment_record, save_dynamic, write_json
from finiteblock_bem.nodal import load_nodal_model
from finiteblock_bem.nodal_dynamics import simulate_nodal
from finiteblock_bem.sampling import SamplingOptions

ROOT = Path(__file__).resolve().parents[1]
LAST_PROGRESS = 0.0


def progress(done, total):
    global LAST_PROGRESS
    now = perf_counter()
    if done == total or now - LAST_PROGRESS >= 30:
        print(f"progress: {done}/{total}", flush=True)
        LAST_PROGRESS = now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", type=Path, default=ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    )
    parser.add_argument("--h", type=float, default=40)
    parser.add_argument("--contact-h", type=float)
    parser.add_argument("--contact-end-h", type=float)
    parser.add_argument("--corner-h", type=float)
    parser.add_argument(
        "--restart-knot-window",
        type=float,
        help="Retain seed knots within this many mm of each contact end",
    )
    parser.add_argument("--restart-knot-subdivisions", type=int, default=1)
    parser.add_argument("--dt", type=float, default=8e-6)
    span = parser.add_mutually_exclusive_group()
    span.add_argument("--steps", type=int, default=None)
    span.add_argument("--duration", type=float, default=0.045)
    parser.add_argument("--quadrature", type=int, default=12)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("FDBEM_CORES", "1")))
    parser.add_argument("--max-memory-mb", type=float, default=2048)
    parser.add_argument("--backend", choices=["numpy", "jax"], default="jax")
    parser.add_argument("--sampling", choices=["dense", "adaptive"], default="adaptive")
    parser.add_argument("--sampling-rtol", type=float, default=1e-12)
    parser.add_argument(
        "--diagnostic-actuator-increment",
        type=float,
        help="Explicit accelerated sliding diagnostic; not original event-16 loading",
    )
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/galerkin-cache")
    parser.add_argument(
        "--spectrum-directory",
        type=Path,
        help="Store the complete half spectrum on disk; time weights remain in RAM",
    )
    parser.add_argument("--name", default="galerkin-event16")
    args = parser.parse_args()
    args.case, args.cache = args.case.resolve(), args.cache.resolve()
    if args.spectrum_directory is not None:
        args.spectrum_directory = args.spectrum_directory.resolve()
    if (
        any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in args.name
        )
        or not args.name
    ):
        raise ValueError("Output name must contain only letters, digits, underscores and hyphens")
    output = ROOT / "outputs" / args.name
    output.mkdir(parents=True, exist_ok=True)
    steps = args.steps if args.steps is not None else round(args.duration / args.dt)
    model = load_nodal_model(
        args.case,
        max_panel_length=args.h,
        contact_panel_length=args.contact_h,
        contact_end_length=args.contact_end_h,
        corner_panel_length=args.corner_h,
        restart_knot_window=args.restart_knot_window,
        restart_knot_subdivisions=args.restart_knot_subdivisions,
    )
    environment = environment_record(args.case)
    write_json(
        output / "run_manifest.json",
        {
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "environment": environment,
        },
    )
    sampling = (
        SamplingOptions(relative_tolerance=args.sampling_rtol)
        if args.sampling == "adaptive"
        else None
    )
    schedule = None
    if args.diagnostic_actuator_increment is not None:
        schedule = LoadingSchedule(
            lambda _: model.reference.normal_pressure,
            lambda t: (
                model.reference.restart_actuator
                + args.diagnostic_actuator_increment * t / (steps * args.dt)
            ),
            "accelerated_nodal_sliding_diagnostic",
        )
    result = simulate_nodal(
        model,
        args.dt,
        steps,
        workers=args.workers,
        quadrature=args.quadrature,
        max_memory_mb=args.max_memory_mb,
        backend=args.backend,
        sampling=sampling,
        cache_directory=args.cache,
        spectrum_directory=args.spectrum_directory,
        schedule=schedule,
        progress=progress,
    )
    summary = save_dynamic(
        output,
        model.contact_case(),
        result,
        args.case,
        start_environment=environment,
        contact_y=model.reference.moving.starts[model.moving_contact, 1],
        contact_weights=model.constraints.contact_weights,
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "completed_requested_horizon":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

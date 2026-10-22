"""Command-line reference benchmarks and finite-block diagnostic runs."""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

from .benchmarks import finite_bar_benchmark
from .case import load_case
from .cq import ResourceBudgetError, plan_domains
from .dynamics import (
    RestartProjectionError,
    cold_smoke_schedule,
    event16_schedule,
    initialize_restart,
    simulate,
)
from .io import environment_record, save_bar, save_dynamic, write_json

_CHECKOUT = Path(__file__).resolve().parents[2]
PROJECT = (
    _CHECKOUT
    if (_CHECKOUT / "pyproject.toml").is_file()
    else Path.cwd() / "finiteblock-bem-results"
)
DEFAULT_CASE = Path(__file__).parent / "resources/cases/pmma_event16.toml"


def _output(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Output name must contain only letters, digits, underscores and hyphens")
    path = PROJECT / "outputs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _progress(stage, domain, current, total):
    stride = max(1, total // 10)
    if current == 1 or current == total or current % stride == 0:
        label = f"CQ domain {domain + 1}" if stage == "kernel" else "time steps"
        print(f"{label}: {current}/{total}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command", choices=["benchmark", "preflight", "smoke", "restart-check", "event16"]
    )
    p.add_argument("--case", type=Path, default=DEFAULT_CASE)
    p.add_argument("--h", type=float, help="Maximum panel length, in the case unit system")
    p.add_argument("--contact-h", type=float, help="Separate maximum contact panel length")
    p.add_argument(
        "--contact-end-h", type=float, help="Symmetric geometric grading at contact ends"
    )
    p.add_argument("--formulation", choices=["full", "interface"], default="full")
    p.add_argument("--sampling", choices=["dense", "adaptive"], default="dense")
    p.add_argument("--sampling-rtol", type=float, default=1e-12)
    p.add_argument("--sampling-degree", type=int, default=16)
    p.add_argument("--corner-h", type=float, help="Grade all other polygon corners as well")
    p.add_argument("--dt", type=float)
    span = p.add_mutually_exclusive_group()
    span.add_argument("--steps", type=int)
    span.add_argument("--duration", type=float)
    p.add_argument("--quadrature", type=int, default=20)
    p.add_argument("--max-memory-mb", type=float, default=512)
    p.add_argument("--name", help="Local output subdirectory name")
    p.add_argument("--allow-restart-relaxation", action="store_true")
    p.add_argument("--workers", type=int, default=int(os.environ.get("FDBEM_CORES", "1")))
    p.add_argument("--backend", choices=["numpy", "jax"], default="numpy")
    p.add_argument("--history", choices=["direct", "fft"], default="fft")
    p.add_argument("--cache", type=Path, help="Directory for reusable, fingerprinted CQ weights")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    output = _output(args.name or args.command)
    try:
        if args.command == "benchmark":
            h = 0.1 if args.h is None else args.h
            dt = 0.02 if args.dt is None else args.dt
            duration = 3.6 if args.duration is None else args.duration
            if args.steps is not None:
                duration = args.steps * dt
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Positive finite duration is required")
            result = finite_bar_benchmark(
                h=h,
                dt=dt,
                duration=duration,
                quadrature=args.quadrature,
                max_memory_mb=args.max_memory_mb,
                progress=lambda i, n: _progress("kernel", 0, i, n),
            )
            summary = save_bar(
                output,
                result,
                {"h": h, "dt": dt, "duration": duration, "quadrature": args.quadrature},
            )
        else:
            start_environment = environment_record(args.case)
            write_json(
                output / "run_manifest.json",
                {
                    "environment": start_environment,
                    "arguments": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                },
            )
            case = load_case(
                args.case,
                max_panel_length=args.h,
                contact_panel_length=args.contact_h,
                contact_end_length=args.contact_end_h,
                corner_panel_length=args.corner_h,
            )
            d = case.payload["diagnostic"]
            dt = d["dt"] if args.dt is None else args.dt
            steps = d["steps"] if args.steps is None else args.steps
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("Positive finite dt is required")
            if args.duration is not None:
                if not math.isfinite(args.duration) or args.duration <= 0:
                    raise ValueError("Positive finite duration is required")
                steps = math.ceil(args.duration / dt)
            if args.command == "restart-check":
                lift = initialize_restart(
                    case, quadrature=args.quadrature, allow_relaxation=args.allow_restart_relaxation
                )
                summary = dict(
                    lift.diagnostics,
                    environment=environment_record(args.case),
                    full_original_event_reproduced=False,
                )
                write_json(output / "summary.json", summary)
            else:
                # Preflight even before static initialization or any kernel allocation.
                sampling = None
                if args.sampling == "adaptive":
                    from .sampling import SamplingOptions

                    if args.formulation != "interface":
                        raise ValueError(
                            "Adaptive frequency sampling requires --formulation interface"
                        )
                    sampling = SamplingOptions(
                        relative_tolerance=args.sampling_rtol, degree=args.sampling_degree
                    )
                if args.formulation == "interface":
                    from .condensation import plan_interface

                    plans, peak = plan_interface(
                        case,
                        dt,
                        steps,
                        quadrature=args.quadrature,
                        workers=args.workers,
                        max_memory_mb=args.max_memory_mb,
                        sampling=sampling,
                    )
                else:
                    plans, peak = plan_domains(
                        [mesh.dofs for mesh in (case.moving, case.stationary)],
                        dt,
                        steps,
                        max_memory_mb=args.max_memory_mb,
                    )
                if args.command == "preflight":
                    summary = {
                        "status": "preflight_only",
                        "plans": [asdict(x) for x in plans],
                        "estimated_peak_mb": peak / 1024**2,
                        "duration_s": dt * steps,
                        "contact_panels": len(case.moving_contact),
                        "original_event_duration_s": case.payload["source"][
                            "original_shear_duration"
                        ],
                        "full_original_event_reproduced": False,
                    }
                    write_json(output / "summary.json", summary)
                else:
                    lift = None
                    if args.command == "event16":
                        lift = initialize_restart(
                            case,
                            quadrature=args.quadrature,
                            allow_relaxation=args.allow_restart_relaxation,
                        )
                        write_json(output / "initialization.json", lift.diagnostics)
                        schedule = event16_schedule(case)
                    else:
                        schedule = cold_smoke_schedule(case)
                    result = simulate(
                        case,
                        dt,
                        steps,
                        schedule=schedule,
                        lift=lift,
                        quadrature=args.quadrature,
                        max_memory_mb=args.max_memory_mb,
                        progress=_progress,
                        workers=args.workers,
                        backend=args.backend,
                        history_method=args.history,
                        cache_directory=args.cache,
                        formulation=args.formulation,
                        sampling=sampling,
                    )
                    summary = save_dynamic(
                        output, case, result, args.case, start_environment=start_environment
                    )
        print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
        print(f"Artifacts: {output}", flush=True)
        return 0 if summary["status"] != "nonlinear_step_failed" else 3
    except RestartProjectionError as exc:
        summary = dict(exc.diagnostics, message=str(exc), full_original_event_reproduced=False)
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2), file=sys.stderr)
        return 2
    except (ResourceBudgetError, ValueError, FloatingPointError) as exc:
        summary = {
            "status": type(exc).__name__,
            "message": str(exc),
            "full_original_event_reproduced": False,
        }
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Alternate full original and ray-table Galerkin assemblies at actual CQ modes.

All three raw operators and both complete Schur maps are audited. Table
construction is charged on every approximate assembly. This is a sampled
assembly benchmark, not complete nonlinear event timing or an FEM advantage.
"""

import argparse
import json
import runpy
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import plan_cq
from finiteblock_bem.galerkin_fast import prepared_geometry_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
RAY = runpy.run_path(str(TOOLS / "ray_bessel_galerkin.py"))
SCHUR = runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"]
MODES = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))["lower_half_radau_modes"]
ERRORS = runpy.run_path(str(TOOLS / "benchmark_factored_galerkin.py"))["errors"]
FIELDS = ["single_layer", "trace_double_layer", "hypersingular"]


def main(*, assembler=None, extra_sources=(), backend_name="ray_bessel"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--h", type=float, default=10)
    p.add_argument("--contact-h", type=float, default=2)
    p.add_argument("--contact-end-h", type=float, default=0.015625)
    p.add_argument("--corner-h", type=float, default=0.015625)
    p.add_argument("--quadrature", type=int, default=12)
    p.add_argument("--dt", type=float, default=2e-6)
    p.add_argument("--duration", type=float, default=0.045)
    p.add_argument("--indices", nargs="+", type=int, default=[0, 655])
    p.add_argument("--branches", nargs="+", type=int, default=[0])
    p.add_argument("--include-fast-zero", action="store_true")
    p.add_argument("--degree", type=int, default=24)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-memory-mb", type=float, default=12288)
    args = p.parse_args()
    begun, output = perf_counter(), args.output.resolve()
    if (
        output.exists()
        or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve())
        or args.repeats < 2
        or args.branches != sorted(set(args.branches))
        or any(b not in [0, 1] for b in args.branches)
        or args.indices != sorted(set(args.indices))
        or not np.isfinite(args.max_memory_mb)
        or args.max_memory_mb <= 0
    ):
        raise ValueError("Require fresh output, distinct physical modes, repeats and finite budget")
    model = load_nodal_model(
        args.case,
        max_panel_length=args.h,
        contact_panel_length=args.contact_h,
        contact_end_length=args.contact_end_h,
        corner_panel_length=args.corner_h,
    )
    plan = plan_cq(
        2 * len(model.moving_contact) + 1,
        args.dt,
        round(args.duration / args.dt),
        max_memory_mb=1e9,
        disk_spectrum=True,
    )
    if any(k < 0 or k > plan.fft_size // 2 for k in args.indices):
        raise ValueError("Frequency index is outside the original Radau plan")
    meshes = [model.reference.moving, model.reference.stationary]
    peak = sum(prepared_geometry_bytes(m, args.quadrature) for m in meshes)
    peak += 32 * (model.constraints.size**2 + sum(m.dofs**2 for m in meshes)) * 16
    peak += 512 * 1024**2
    if peak > args.max_memory_mb * 1024**2:
        raise ValueError("Shared geometry/solver estimate exceeds explicit array budget")
    paths = [
        Path(__file__),
        TOOLS / "ray_bessel_galerkin.py",
        TOOLS / "trace_schur_experiment.py",
        TOOLS / "schur_condensation_experiment.py",
        TOOLS / "radau_raw_sampling.py",
        TOOLS / "benchmark_factored_galerkin.py",
        TOOLS / "factored_galerkin.py",
        args.case,
        *extra_sources,
    ]
    frozen = {str(v.resolve()): fingerprint(v) for v in paths}
    environment = environment_record(args.case)
    output.mkdir(parents=True)
    write_json(
        output / "preflight.json",
        dict(
            plan=asdict(plan),
            estimated_array_bytes=peak,
            arguments=vars(args) | {"case": str(args.case.resolve()), "output": str(output)},
            frozen_inputs_sha256=frozen,
            environment=environment,
            raw_rtol=1e-11,
            raw_atol=1e-14,
            output_relative_tolerance=1e-8,
            backend=backend_name,
            timing_variant_key="ray",
        ),
    )
    samples = []
    with threadpool_limits(limits=1, user_api="blas"):
        start = perf_counter()
        reference = SCHUR(model, quadrature=args.quadrature)
        original_preparation = perf_counter() - start
        start = perf_counter()
        factory = RAY["RayBesselGalerkin"] if assembler is None else assembler
        variants = [factory(a, degree=args.degree) for a in reference.prepared]
        variant_preparation = perf_counter() - start
        modes = [(i, b) for i in args.indices for b in args.branches]
        if args.include_fast_zero and (0, 1) not in modes:
            modes.append((0, 1))
        for index, branch in modes:
            s = MODES(plan.radius * np.exp(-2j * np.pi * index / plan.fft_size), plan.dt)[branch]
            times = {
                key: []
                for key in [
                    "original_assembly",
                    "ray_table_preparation",
                    "ray_assembly",
                    "ray_preparation_and_assembly",
                    "original_condensation",
                    "ray_condensation",
                ]
            }
            attempts = []
            for repeat in range(args.repeats):
                operators, maps, tables = {}, {}, []
                order = ["original", "ray"] if repeat % 2 == 0 else ["ray", "original"]
                for name in order:
                    write_json(
                        output / "live.json",
                        dict(index=index, branch=branch, repeat=repeat, method=name),
                    )
                    if name == "ray":
                        start = perf_counter()
                        prepared = [a.prepare(s) for a in variants]
                        preparation = perf_counter() - start
                        times["ray_table_preparation"].append(preparation)
                        tables = [table.statistics() for _, table in prepared]
                    start = perf_counter()
                    operators[name] = (
                        [a.assemble(s) for a in reference.prepared]
                        if name == "original"
                        else [evaluate() for evaluate, _ in prepared]
                    )
                    elapsed = perf_counter() - start
                    times[name + "_assembly"].append(elapsed)
                    if name == "ray":
                        times["ray_preparation_and_assembly"].append(elapsed + preparation)
                    start = perf_counter()
                    maps[name] = reference.evaluate_operators(operators[name], reconstruction=True)
                    times[name + "_condensation"].append(perf_counter() - start)
                checks, passed = [], True
                for body in range(2):
                    for field in FIELDS:
                        a, b = (
                            getattr(operators[name][body], field) for name in ["ray", "original"]
                        )
                        report = ERRORS(a, b)
                        normalized = float(
                            np.linalg.norm(a - b) / (1e-14 + 1e-11 * np.linalg.norm(b))
                        )
                        checks.append(
                            dict(
                                body=body,
                                field=field,
                                mixed_tolerance_normalized=normalized,
                                **report,
                            )
                        )
                        passed &= normalized <= 1
                for field, a, b in zip(
                    ["impedance", "full_boundary_reconstruction"],
                    maps["ray"],
                    maps["original"],
                    strict=True,
                ):
                    report = ERRORS(a, b)
                    checks.append(dict(field=field, **report))
                    passed &= max(report["relative_frobenius"], report["relative_inf"]) <= 1e-8
                attempts.append(
                    dict(repeat=repeat, checks=checks, tables=tables, all_gates_passed=bool(passed))
                )
                write_json(
                    output / "attempt-progress.json",
                    dict(index=index, branch=branch, times=times, attempts=attempts),
                )
                if not passed:
                    np.savez_compressed(
                        output / "failed-operators.npz",
                        **{
                            f"{name}_{body}_{field}": getattr(operators[name][body], field)
                            for name in operators
                            for body in range(2)
                            for field in FIELDS
                        },
                    )
                    raise ValueError(
                        "Original raw/Schur gate failed; failed operators and costs saved"
                    )
                del operators, maps, prepared
            medians = {k: median(v) for k, v in times.items()}
            samples.append(
                dict(
                    index=index,
                    branch=branch,
                    s_real=float(s.real),
                    s_imag=float(s.imag),
                    times=times,
                    medians=medians,
                    assembly_speed_ratio_including_tables=medians["original_assembly"]
                    / medians["ray_preparation_and_assembly"],
                    attempts=attempts,
                )
            )
            write_json(output / "progress.json", dict(samples=samples))
            print(json.dumps(dict(index=index, branch=branch, medians=medians)), flush=True)
    if (
        any(fingerprint(f) != v for f, v in frozen.items())
        or environment_record(args.case) != environment
    ):
        raise ValueError("Source or runtime changed during benchmark")
    write_json(
        output / "summary.json",
        dict(
            status="passed_sampled_original_raw_and_schur_audits",
            backend=backend_name,
            timing_variant_key="ray",
            plan=asdict(plan),
            environment=environment,
            frozen_inputs_sha256=frozen,
            original_preparation_seconds=original_preparation,
            variant_geometry_scan_seconds=variant_preparation,
            total_seconds_including_checks=perf_counter() - begun,
            samples=samples,
            blas_threads=1,
            original_geometry_shared=True,
            full_kernel_or_event_timing=False,
            same_accuracy_fem_advantage_verified=False,
        ),
    )


if __name__ == "__main__":
    main()

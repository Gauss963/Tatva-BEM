"""Compare complete original/factored V/H/D and Schur maps on one shared mesh.

This measures a small number of actual Radau frequencies, not a complete
kernel, trajectory or FEM advantage. Preparation, assembly and condensation
are timed separately; both orders alternate, with one BLAS thread.
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
FACTORED = runpy.run_path(str(TOOLS / "factored_galerkin.py"))
SCHUR = runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"]
MODES = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))["lower_half_radau_modes"]


def errors(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or any(
        a.ndim != 2 or not np.all(np.isfinite(a)) for a in [actual, expected]
    ):
        raise ValueError("Require aligned finite body/transfer matrices")
    difference = actual - expected
    result = dict(
        relative_frobenius=float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-30)),
        relative_inf=float(
            np.linalg.norm(difference, ord=np.inf)
            / max(np.linalg.norm(expected, ord=np.inf), 1e-30)
        ),
        maximum_absolute=float(np.max(abs(difference))),
    )
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite operator audit norm")
    return result


def main():
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
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-memory-mb", type=float, default=12288)
    args = p.parse_args()
    begun = perf_counter()
    output = args.output.resolve()
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
        raise ValueError(
            "Choose fresh output, alternating repeats and distinct valid modes/indices"
        )
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
        raise ValueError("Frequency index is outside the complete Radau plan")
    meshes = [model.reference.moving, model.reference.stationary]
    peak = sum(prepared_geometry_bytes(m, args.quadrature) for m in meshes)
    peak += 32 * (model.constraints.size**2 + sum(m.dofs**2 for m in meshes)) * 16
    peak += 512 * 1024**2
    if peak > args.max_memory_mb * 1024**2:
        raise ValueError("Shared geometry/solver estimate exceeds explicit array budget")
    paths = [
        Path(__file__),
        TOOLS / "factored_galerkin.py",
        TOOLS / "trace_schur_experiment.py",
        TOOLS / "schur_condensation_experiment.py",
        TOOLS / "radau_raw_sampling.py",
        args.case,
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
        ),
    )
    prepared_at = perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        transfer = SCHUR(model, quadrature=args.quadrature)
        variant = [FACTORED["FactoredGalerkinOperators"].share(a) for a in transfer.prepared]
        preparation_seconds = perf_counter() - prepared_at
        samples = []
        for index in args.indices:
            z = plan.radius * np.exp(-2j * np.pi * index / plan.fft_size)
            for branch in args.branches:
                s = MODES(z, plan.dt)[branch]
                times = {
                    k: []
                    for k in [
                        "original_assembly",
                        "factored_assembly",
                        "original_condensation",
                        "factored_condensation",
                    ]
                }
                for repeat in range(args.repeats):
                    order = [("original", transfer.prepared), ("factored", variant)]
                    if repeat % 2:
                        order.reverse()
                    maps, operators = {}, {}
                    for name, providers in order:
                        start = perf_counter()
                        operators[name] = [a.assemble(s) for a in providers]
                        times[name + "_assembly"].append(perf_counter() - start)
                        start = perf_counter()
                        maps[name] = transfer.evaluate_operators(
                            operators[name], reconstruction=True
                        )
                        times[name + "_condensation"].append(perf_counter() - start)
                    checks = []
                    for body in range(2):
                        for field in ["single_layer", "trace_double_layer", "hypersingular"]:
                            error = errors(
                                getattr(operators["factored"][body], field),
                                getattr(operators["original"][body], field),
                            )
                            if max(error["relative_frobenius"], error["relative_inf"]) > 5e-12:
                                raise ValueError("Factored complete body operator differs")
                            checks.append(dict(body=body, field=field, **error))
                    for name, a, b in zip(
                        ["impedance", "boundary_reconstruction"],
                        maps["factored"],
                        maps["original"],
                        strict=True,
                    ):
                        error = errors(a, b)
                        if max(error["relative_frobenius"], error["relative_inf"]) > 1e-8:
                            raise ValueError("Factored original Schur solution differs")
                        checks.append(dict(field=name, **error))
                    del operators, maps
                medians = {k: median(v) for k, v in times.items()}
                samples.append(
                    dict(
                        index=index,
                        branch=branch,
                        s_real=float(s.real),
                        s_imag=float(s.imag),
                        times=times,
                        medians=medians,
                        checks=checks,
                        assembly_speed_ratio=medians["original_assembly"]
                        / medians["factored_assembly"],
                    )
                )
                write_json(output / "progress.json", dict(samples=samples))
                print(json.dumps(dict(index=index, branch=branch, medians=medians)), flush=True)
    if (
        any(fingerprint(p) != v for p, v in frozen.items())
        or environment_record(args.case) != environment
    ):
        raise ValueError("Benchmark source or numerical environment changed")
    write_json(
        output / "summary.json",
        dict(
            status="completed_alternating_single_frequency_assembly_control",
            plan=asdict(plan),
            environment=environment,
            frozen_inputs_sha256=frozen,
            preparation_seconds=preparation_seconds,
            total_seconds_including_checks=perf_counter() - begun,
            samples=samples,
            variant_resources=[v.statistics() for v in variant],
            original_geometry_shared=True,
            blas_threads=1,
            estimated_array_bytes=peak,
            full_kernel_or_event_timing=False,
            same_accuracy_fem_advantage_verified=False,
        ),
    )


if __name__ == "__main__":
    main()

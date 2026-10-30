"""Audit and time complete hierarchical operators and unchanged Schur solves.

Near-field geometry, failed ACA attempts, full fallbacks, preparation and all
independent reference checks are retained in the cost record. This samples
physical frequencies; it is not a complete CQ/event or FEM advantage proof.
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
METHOD = runpy.run_path(str(TOOLS / "hierarchical_galerkin.py"))
SCHUR = runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"]
MODES = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))["lower_half_radau_modes"]
ERROR = runpy.run_path(str(TOOLS / "benchmark_factored_galerkin.py"))["errors"]
FIELDS = ["single_layer", "trace_double_layer", "hypersingular"]


def summary_record(record):
    blocks = record["blocks"]
    return {k: v for k, v in record.items() if k != "blocks"} | dict(
        block_counts={
            kind: sum(b["kind"] == kind for b in blocks)
            for kind in ["exact_near", "exact_fallback", "empirical_aca"]
        },
        far_blocks=[b for b in blocks if b["kind"] != "exact_near"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", type=Path)
    p.add_argument("--output", type=Path, required=True)
    for flag, value in [("h", 10), ("contact-h", 2), ("end-h", 0.015625)]:
        p.add_argument("--" + flag, type=float, default=value)
    p.add_argument("--quadrature", type=int, default=12)
    p.add_argument("--leaf-panels", type=int, default=24)
    p.add_argument("--eta", type=float, default=1)
    p.add_argument("--raw-rtol", type=float, default=1e-11)
    p.add_argument("--raw-atol", type=float, default=1e-14)
    p.add_argument("--indices", type=int, nargs="+", default=[0, 655])
    p.add_argument("--branches", type=int, nargs="+", default=[0])
    p.add_argument("--include-fast-zero", action="store_true")
    p.add_argument("--packed-pairs", action="store_true")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-memory-mb", type=float, default=12288)
    args = p.parse_args()
    begun = perf_counter()
    output = args.output.resolve()
    if (
        output.exists()
        or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva"))
        or args.repeats < 2
        or not np.isfinite(args.max_memory_mb)
        or args.max_memory_mb <= 0
        or args.indices != sorted(set(args.indices))
        or any(b not in [0, 1] for b in args.branches)
    ):
        raise ValueError("Require fresh output, valid modes, budget and alternating repeats")
    model = load_nodal_model(
        args.case,
        max_panel_length=args.h,
        contact_panel_length=args.contact_h,
        contact_end_length=args.end_h,
        corner_panel_length=args.end_h,
    )
    meshes = [model.reference.moving, model.reference.stationary]
    plan = plan_cq(
        2 * len(model.moving_contact) + 1, 2e-6, 22500, max_memory_mb=1e9, disk_spectrum=True
    )
    if any(i < 0 or i > plan.fft_size // 2 for i in args.indices):
        raise ValueError("Invalid physical frequency index")
    peak = sum(prepared_geometry_bytes(m, args.quadrature) for m in meshes)
    peak += (
        32 * (model.constraints.size**2 + sum(m.dofs**2 for m in meshes)) * 16
        + 2 * 512 * 1024**2
        + 256 * 1024**2
    )
    if peak > args.max_memory_mb * 1024**2:
        raise ValueError("Explicit original/hierarchical/solver array budget exceeded")
    factory = METHOD["HierarchicalGalerkin"]
    if args.packed_pairs:
        factory = runpy.run_path(str(TOOLS / "packed_galerkin_blocks.py"))[
            "PackedHierarchicalGalerkin"
        ]
    files = [
        Path(__file__),
        *[
            TOOLS / n
            for n in [
                "hierarchical_galerkin.py",
                "packed_galerkin_blocks.py",
                "galerkin_blocks.py",
                "trace_schur_experiment.py",
                "schur_condensation_experiment.py",
                "radau_raw_sampling.py",
                "benchmark_factored_galerkin.py",
            ]
        ],
        args.case,
    ]
    frozen = {str(f.resolve()): fingerprint(f) for f in files}
    env = environment_record(args.case)
    output.mkdir(parents=True)
    write_json(
        output / "preflight.json",
        dict(
            frozen_inputs_sha256=frozen,
            environment=env,
            plan=asdict(plan),
            estimated_array_bytes=peak,
            arguments=vars(args) | dict(case=str(args.case.resolve()), output=str(output)),
        ),
    )
    samples = []
    with threadpool_limits(limits=1, user_api="blas"):
        start = perf_counter()
        reference = SCHUR(model, quadrature=args.quadrature)
        original_preparation = perf_counter() - start
        start = perf_counter()
        variants = [
            factory(
                m,
                model.reference.material,
                quadrature=args.quadrature,
                leaf_panels=args.leaf_panels,
                eta=args.eta,
                rtol=args.raw_rtol,
                atol=args.raw_atol,
            )
            for m in meshes
        ]
        variant_preparation = perf_counter() - start
        modes = [(i, b) for i in args.indices for b in args.branches]
        if args.include_fast_zero and (0, 1) not in modes:
            modes.append((0, 1))
        write_json(
            output / "preparation.json",
            dict(
                original_seconds=original_preparation,
                hierarchical_seconds=variant_preparation,
                partitions=[
                    dict(
                        near=sum(not far for _, _, far in v.blocks),
                        far=sum(far for _, _, far in v.blocks),
                        pinned_bytes=v.pinned_geometry_bytes,
                    )
                    for v in variants
                ],
            ),
        )
        for index, branch in modes:
            s = MODES(plan.radius * np.exp(-2j * np.pi * index / plan.fft_size), plan.dt)[branch]
            times = {
                key: []
                for key in [
                    "original_assembly",
                    "hierarchical_assembly",
                    "original_condensation",
                    "hierarchical_condensation",
                ]
            }
            attempts = []
            for repeat in range(args.repeats):
                operators, maps, details = {}, {}, None
                order = (
                    ["original", "hierarchical"]
                    if repeat % 2 == 0
                    else ["hierarchical", "original"]
                )
                for name in order:
                    start = perf_counter()
                    operators[name] = []
                    records = []
                    for body in range(2):
                        write_json(
                            output / "live.json",
                            dict(
                                index=index,
                                branch=branch,
                                repeat=repeat,
                                method=name,
                                body=body,
                                elapsed_seconds=perf_counter() - begun,
                            ),
                        )
                        if name == "original":
                            operator = reference.prepared[body].assemble(s)
                        else:
                            operator, record = variants[body].evaluate(s)
                            records.append(summary_record(record))
                        operators[name].append(operator)
                    times[name + "_assembly"].append(perf_counter() - start)
                    if name == "hierarchical":
                        details = records
                    start = perf_counter()
                    maps[name] = reference.evaluate_operators(operators[name], reconstruction=True)
                    times[name + "_condensation"].append(perf_counter() - start)
                checks = []
                passed = True
                for body in range(2):
                    for field in FIELDS:
                        a, b = (
                            getattr(operators[name][body], field)
                            for name in ["hierarchical", "original"]
                        )
                        report = ERROR(a, b)
                        mixed = float(
                            np.linalg.norm(a - b)
                            / (args.raw_atol + args.raw_rtol * np.linalg.norm(b))
                        )
                        checks.append(
                            dict(body=body, field=field, mixed_tolerance_normalized=mixed, **report)
                        )
                        passed &= mixed <= 1
                for field, a, b in zip(
                    ["impedance", "full_boundary_reconstruction"],
                    maps["hierarchical"],
                    maps["original"],
                    strict=True,
                ):
                    report = ERROR(a, b)
                    checks.append(dict(field=field, **report))
                    passed &= max(report["relative_frobenius"], report["relative_inf"]) <= 1e-8
                hermitian = {name: (z / s + (z / s).conj().T) / 2 for name, (z, _) in maps.items()}
                passivity = dict(
                    original_minimum_eigenvalue=float(np.linalg.eigvalsh(hermitian["original"])[0]),
                    hierarchical_minimum_eigenvalue=float(
                        np.linalg.eigvalsh(hermitian["hierarchical"])[0]
                    ),
                    hermitian_difference_frobenius_bound=float(
                        np.linalg.norm(hermitian["hierarchical"] - hermitian["original"])
                    ),
                )
                attempts.append(
                    dict(
                        repeat=repeat,
                        checks=checks,
                        compression=details,
                        sampled_velocity_impedance_hermitian=passivity,
                        all_gates_passed=bool(passed),
                    )
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
                        "Complete original raw/Schur gate failed; retained failed operators and costs"
                    )
                del operators, maps
            samples.append(
                dict(
                    index=index,
                    branch=branch,
                    s_real=float(s.real),
                    s_imag=float(s.imag),
                    times=times,
                    medians={k: median(v) for k, v in times.items()},
                    attempts=attempts,
                )
            )
            write_json(output / "progress.json", dict(samples=samples))
            print(
                json.dumps(dict(index=index, branch=branch, medians=samples[-1]["medians"])),
                flush=True,
            )
    if any(fingerprint(f) != v for f, v in frozen.items()) or environment_record(args.case) != env:
        raise ValueError("Source/runtime changed")
    write_json(
        output / "summary.json",
        dict(
            status="passed_sampled_original_raw_and_schur_audits",
            samples=samples,
            original_preparation_seconds=original_preparation,
            hierarchical_preparation_seconds=variant_preparation,
            total_seconds_including_checks=perf_counter() - begun,
            frozen_inputs_sha256=frozen,
            environment=env,
            empirical_spatial_compression=True,
            frequency_compression=False,
            full_cq_or_event_control=False,
            same_accuracy_fem_advantage_verified=False,
        ),
    )


if __name__ == "__main__":
    main()

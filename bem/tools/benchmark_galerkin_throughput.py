"""Compare actual two-worker reference batches with one/two-worker JIT batches.

Two nearby physical contour modes mimic one provider batch in the existing raw
sampler. Raw assembly and Schur/work-coordinate transforms each use the selected
thread pool, as in that sampler. All results retain independent original raw,
impedance, full-boundary reconstruction and work-compliance gates. This is not a
complete interpolation/kernel/event runtime or FEM performance comparison.
"""

import argparse
import json
import resource
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor
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
FIELDS = ("single_layer", "trace_double_layer", "hypersingular")


def relative_errors(actual, exact):
    result = {}
    for name, order in [("relative_frobenius", None), ("relative_inf", np.inf)]:
        denominator = np.linalg.norm(exact, ord=order)
        numerator = np.linalg.norm(actual - exact, ord=order)
        value = numerator / denominator if denominator > 0 else np.nan
        result[name] = float(value) if np.isfinite(value) else None
    return result


def audit(actual, exact, transfer, *, full_reference=None):
    """Compare each physical field/map; retain per-mode checks and failure data."""
    checks, full = [], []
    for index, (a, b) in enumerate(zip(actual, exact, strict=True)):
        for body, (x, y) in enumerate(zip(a["operators"], b["operators"], strict=True)):
            for field in FIELDS:
                first, second = getattr(x, field), getattr(y, field)
                value = float(
                    np.linalg.norm(first - second) / (1e-14 + 1e-11 * np.linalg.norm(second))
                )
                checks.append(
                    dict(
                        mode=index,
                        body=body,
                        field=field,
                        raw_normalized=value if np.isfinite(value) else None,
                    )
                )
        recovered = transfer.evaluate_operators(a["operators"], reconstruction=True)[1]
        full.append(recovered)
        wanted = (
            transfer.evaluate_operators(b["operators"], reconstruction=True)[1]
            if full_reference is None
            else full_reference[index]
        )
        for field, x, y in [
            ("impedance", a["impedance"], b["impedance"]),
            ("work_compliance", a["work_compliance"], b["work_compliance"]),
            ("full_boundary_reconstruction", recovered, wanted),
        ]:
            checks.append(dict(mode=index, field=field, **relative_errors(x, y)))
    passed = all(
        all(v is not None and np.isfinite(v) and v <= limit for v in values)
        for c in checks
        for values, limit in [
            ([c["raw_normalized"]], 1)
            if "raw_normalized" in c
            else ([c["relative_frobenius"], c["relative_inf"]], 1e-8)
        ]
    )
    return checks, bool(passed), full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    for flag, default in [
        ("h", 10),
        ("contact-h", 2),
        ("contact-end-h", 0.015625),
        ("corner-h", 0.015625),
    ]:
        parser.add_argument("--" + flag, type=float, default=default)
    parser.add_argument("--quadrature", type=int, default=12)
    parser.add_argument("--indices", type=int, nargs=2, default=[655, 656])
    parser.add_argument("--branch", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dt", type=float, default=2e-6)
    parser.add_argument("--duration", type=float, default=0.045)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-memory-mb", type=float, default=12288)
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        output.exists()
        or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve())
        or args.indices != sorted(set(args.indices))
        or args.repeats < 2
        or not np.isfinite(args.max_memory_mb)
        or args.max_memory_mb <= 0
    ):
        raise ValueError("Require fresh owned output, two distinct ordered modes and finite budget")
    output.mkdir(parents=True)
    begun = perf_counter()
    attempts = []
    try:
        jit = runpy.run_path(str(TOOLS / "jax_ray_galerkin.py"))
        schur = runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"]
        modes = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))["lower_half_radau_modes"]
        work = runpy.run_path(str(TOOLS / "build_resolvent_history.py"))[
            "compliance_in_work_coordinates"
        ]
        imports_seconds = perf_counter() - begun
        frozen = {str(p.resolve()): fingerprint(p) for p in [*TOOLS.glob("*.py"), args.case]}
        environment = environment_record(args.case)
        model = load_nodal_model(
            args.case,
            max_panel_length=args.h,
            contact_panel_length=args.contact_h,
            contact_end_length=args.contact_end_h,
            corner_panel_length=args.corner_h,
        )
        steps = round(args.duration / args.dt)
        if steps < 1 or abs(steps * args.dt - args.duration) > 1e-12 * max(args.duration, args.dt):
            raise ValueError("Require an integer number of positive time steps")
        plan = plan_cq(
            2 * len(model.moving_contact) + 1, args.dt, steps, max_memory_mb=1e9, disk_spectrum=True
        )
        if min(args.indices) < 0 or max(args.indices) > plan.fft_size // 2:
            raise ValueError("Physical contour index outside the current plan")
        estimated = sum(
            prepared_geometry_bytes(m, args.quadrature)
            for m in [model.reference.moving, model.reference.stationary]
        )
        estimated += (
            64
            * (
                model.constraints.size**2
                + sum(m.dofs**2 for m in [model.reference.moving, model.reference.stationary])
            )
            * 16
        )
        estimated += 1024**3
        if estimated > args.max_memory_mb * 1024**2:
            raise ValueError("Shared geometry/two-worker estimate exceeds explicit budget")
        frequencies = [
            complex(
                modes(plan.radius * np.exp(-2j * np.pi * i / plan.fft_size), plan.dt)[args.branch]
            )
            for i in args.indices
        ]
        setup = perf_counter()
        with threadpool_limits(limits=1, user_api="blas"):
            transfer = schur(model, quadrature=args.quadrature)
            variants = [jit["JaxRayGalerkin"](a) for a in transfer.prepared]
            prepared_seconds = perf_counter() - setup
            providers = {"original": transfer.prepared, "jax": variants}
            con, ref, law = model.constraints, model.reference, model.reference.contact
            line = np.r_[con.contact_weights, con.contact_weights, con.platen_width]
            tangent = np.diag(
                np.r_[
                    con.contact_weights * law.normal_penalty,
                    con.contact_weights * law.tangential_penalty,
                    ref.spring_stiffness * con.platen_width,
                ]
            )
            configurations = [("original", 2), ("jax", 1), ("jax", 2)]
            reference = full_reference = None
            for repeat in range(args.repeats):
                order = configurations if repeat % 2 == 0 else configurations[::-1]
                schedule = [(b, w, False) for b, w in order]
                if repeat == 0:
                    # Charge compilation separately, then compare both JIT
                    # worker counts with the same already-compiled kernels.
                    schedule.insert(1, ("jax", 1, True))
                for backend, workers, warmup in schedule:
                    write_json(
                        output / "live.json",
                        dict(repeat=repeat, backend=backend, workers=workers, warmup=warmup),
                    )

                    def assemble(s):
                        if backend == "original":
                            return dict(
                                operators=[a.assemble(s) for a in providers[backend]], tables=[]
                            )
                        private = [a.prepare(s) for a in providers[backend]]
                        return dict(
                            operators=[evaluate() for evaluate, _ in private],
                            tables=[table.statistics() for _, table in private],
                        )

                    def transform(value):
                        impedance = transfer.evaluate_operators(value["operators"])
                        return {
                            **value,
                            "impedance": impedance,
                            "work_compliance": work(impedance, tangent, line),
                        }

                    signatures_before = len(jit["COMPILED_SIGNATURES"])
                    start = perf_counter()
                    with ThreadPoolExecutor(workers) as pool:
                        raw = list(pool.map(assemble, frequencies))
                        assembly_seconds = perf_counter() - start
                        transformed = list(pool.map(transform, raw))
                    elapsed = perf_counter() - start
                    del raw
                    if reference is None:
                        if (backend, workers) != ("original", 2):
                            raise RuntimeError("The independent original batch must run first")
                        reference = transformed
                    validation = perf_counter()
                    checks, passed, full = audit(
                        transformed, reference, transfer, full_reference=full_reference
                    )
                    if full_reference is None:
                        full_reference = full
                    record = dict(
                        repeat=repeat,
                        backend=backend,
                        workers=workers,
                        warmup=warmup,
                        compiled_signatures_before=signatures_before,
                        compiled_signatures_after=len(jit["COMPILED_SIGNATURES"]),
                        assembly_seconds=assembly_seconds,
                        assembly_and_transform_seconds=elapsed,
                        validation_seconds=perf_counter() - validation,
                        checks=checks,
                        all_gates_passed=passed,
                        tables=[v["tables"] for v in transformed],
                    )
                    attempts.append(record)
                    write_json(output / "progress.json", dict(attempts=attempts))
                    if not passed:
                        np.savez_compressed(
                            output / "failed-operators.npz",
                            **{
                                f"{name}_{i}_{body}_{field}": getattr(op, field)
                                for name, values in [
                                    ("actual", transformed),
                                    ("reference", reference),
                                ]
                                for i, v in enumerate(values)
                                for body, op in enumerate(v["operators"])
                                for field in FIELDS
                            },
                        )
                        raise ValueError("Original raw/Schur/work gate failed; outputs retained")
                    print(
                        json.dumps(
                            {
                                k: record[k]
                                for k in [
                                    "repeat",
                                    "backend",
                                    "workers",
                                    "warmup",
                                    "assembly_and_transform_seconds",
                                    "all_gates_passed",
                                ]
                            }
                        ),
                        flush=True,
                    )
                    del transformed, full
        if (
            any(fingerprint(p) != v for p, v in frozen.items())
            or environment_record(args.case) != environment
        ):
            raise ValueError("Numerical source or environment changed")
        timings = [
            {
                "backend": b,
                "workers": w,
                "median_assembly_and_transform_seconds": median(
                    a["assembly_and_transform_seconds"]
                    for a in attempts
                    if not a["warmup"] and (a["backend"], a["workers"]) == (b, w)
                ),
            }
            for b, w in configurations
        ]
        write_json(
            output / "summary.json",
            dict(
                status="passed_sampled_two_frequency_worker_throughput_control",
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                import_and_jax_startup_seconds=imports_seconds,
                shared_geometry_preparation_seconds=prepared_seconds,
                estimated_array_bytes=estimated,
                attempts=attempts,
                timings=timings,
                compiled_signatures=jit["COMPILED_SIGNATURES"],
                frequencies=[dict(real=s.real, imag=s.imag) for s in frequencies],
                frozen_inputs_sha256=frozen,
                environment=environment,
                peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                total_seconds_including_all_controls=perf_counter() - begun,
                per_attempt_tables_synchronization_included=True,
                compilation_charged_in_separate_audited_warmup=True,
                compilation_warmup_in_total_control_cost=True,
                all_original_raw_and_schur_gates_retained=True,
                full_interpolation_or_event_runtime=False,
                same_accuracy_fem_advantage_verified=False,
            ),
        )
    except BaseException as error:
        write_json(
            output / "failure.json",
            dict(error=repr(error), attempts=attempts, seconds=perf_counter() - begun),
        )
        raise


if __name__ == "__main__":
    main()

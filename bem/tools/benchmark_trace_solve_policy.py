"""Measure actual fine-mesh solve policies on identical preserved raw operators.

Assembly is original Q12 and paid once per physical frequency. The two-worker
timings cover impedance plus work compliance, matching raw-sampler output.
Full reconstruction is independently audited outside these timings. This is
not full-kernel/event/FEM runtime or nonlinear stability evidence.
"""

import argparse
import hashlib
import resource
import runpy
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
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
POLICY = runpy.run_path(str(TOOLS / "trace_solve_policy.py"))
MODES = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))["lower_half_radau_modes"]
ERRORS = runpy.run_path(str(TOOLS / "benchmark_galerkin_throughput.py"))["relative_errors"]
VARIANTS = {
    "auto": (False, False),
    "general": (True, False),
    "lazy": (False, True),
    "lazy_general": (True, True),
}


def array_digest(operators):
    value = hashlib.sha256()
    for operator in operators:
        for name in ["single_layer", "trace_double_layer", "hypersingular", "mixed_mass"]:
            array = np.asarray(getattr(operator, name))
            value.update(str((name, array.shape, array.dtype.str)).encode())
            value.update(np.ascontiguousarray(array).view(np.uint8))
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-memory-mb", type=float, default=12288)
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        output.exists()
        or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve())
        or args.repeats < 3
        or not np.isfinite(args.max_memory_mb)
        or args.max_memory_mb <= 0
    ):
        raise ValueError("Require fresh owned output and three or more finite-budget repeats")
    begun = perf_counter()
    output.mkdir(parents=True)
    frozen = {str(p.resolve()): fingerprint(p) for p in [*TOOLS.glob("*.py"), args.case]}
    environment = environment_record(args.case)
    results = []
    try:
        model = load_nodal_model(
            args.case,
            max_panel_length=10,
            contact_panel_length=2,
            contact_end_length=0.015625,
            corner_panel_length=0.015625,
        )
        meshes = [model.reference.moving, model.reference.stationary]
        estimate = sum(prepared_geometry_bytes(m, 12) for m in meshes)
        estimate += 64 * (model.constraints.size**2 + sum(m.dofs**2 for m in meshes)) * 16 + 1024**3
        raw_bytes = sum(3 * m.dofs**2 * 16 for m in meshes)
        if (
            estimate > args.max_memory_mb * 1024**2
            or shutil.disk_usage(output).free < 4 * raw_bytes + 1024**3
        ):
            raise ValueError("Two-worker geometry/solve or preserved raw-data budget exceeded")
        plan = plan_cq(
            2 * len(model.moving_contact) + 1, 2e-6, 22500, max_memory_mb=1e9, disk_spectrum=True
        )
        con, ref, law = model.constraints, model.reference, model.reference.contact
        line = np.r_[con.contact_weights, con.contact_weights, con.platen_width]
        tangent = np.diag(
            np.r_[
                con.contact_weights * law.normal_penalty,
                con.contact_weights * law.tangential_penalty,
                ref.spring_stiffness * con.platen_width,
            ]
        )
        write_json(
            output / "manifest.json",
            dict(
                frozen_inputs_sha256=frozen,
                environment=environment,
                plan=asdict(plan),
                workers=2,
                repeats=args.repeats,
                mesh=dict(
                    h=10, contact_h=2, contact_end_h=0.015625, corner_h=0.015625, quadrature=12
                ),
                estimated_array_mib=estimate / 1024**2,
                variants=VARIANTS,
                original_case_and_galerkin_equations_unchanged=True,
            ),
        )
        with threadpool_limits(limits=1, user_api="blas"):
            transfer = POLICY["TRACE"](model, quadrature=12)
            policies = {
                name: POLICY["transfer_policy"](transfer, general=general, lazy=lazy)
                for name, (general, lazy) in VARIANTS.items()
            }
            for branch, indices in [(0, [0, 655]), (1, [11000, 11001])]:
                frequencies = [
                    complex(
                        MODES(plan.radius * np.exp(-2j * np.pi * i / plan.fft_size), plan.dt)[
                            branch
                        ]
                    )
                    for i in indices
                ]
                write_json(
                    output / "progress.json",
                    dict(phase="original_raw_assembly", branch=branch, indices=indices),
                )
                assembly = perf_counter()
                with ThreadPoolExecutor(2) as pool:
                    raw = list(
                        pool.map(lambda s: [p.assemble(s) for p in transfer.prepared], frequencies)
                    )
                assembly = perf_counter() - assembly
                hashes = [array_digest(group) for group in raw]
                raw_records = []
                for index, s, group in zip(indices, frequencies, raw, strict=True):
                    path = output / f"raw-branch{branch}-mode{index}.npz"
                    payload = {
                        f"body{body}_{field}": getattr(op, field)
                        for body, op in enumerate(group)
                        for field in [
                            "single_layer",
                            "trace_double_layer",
                            "hypersingular",
                            "mixed_mass",
                        ]
                    }
                    np.savez(path, frequency=s, **payload)
                    frozen[str(path)] = fingerprint(path)
                    raw_records.append(
                        dict(
                            path=str(path),
                            sha256=frozen[str(path)],
                            frequency=dict(real=s.real, imag=s.imag),
                        )
                    )
                attempts, reference = [], None
                for repeat in range(args.repeats):
                    names = list(VARIANTS) if repeat % 2 == 0 else list(VARIANTS)[::-1]
                    for name in names:
                        general, lazy = VARIANTS[name]
                        write_json(
                            output / "progress.json",
                            dict(
                                phase="paired_solve_timing",
                                branch=branch,
                                repeat=repeat,
                                variant=name,
                            ),
                        )

                        def evaluate(group):
                            impedance = policies[name].evaluate_operators(
                                POLICY["layers"](group, general)
                            )
                            return impedance, POLICY["work_policy"](general)(
                                impedance, tangent, line
                            )

                        start = perf_counter()
                        with ThreadPoolExecutor(2) as pool:
                            actual = list(pool.map(evaluate, raw))
                        seconds = perf_counter() - start
                        if reference is None:
                            if name != "auto":
                                raise RuntimeError("Reference must be evaluated first")
                            reference = actual
                        checks = [
                            {"mode": index, "field": field, **ERRORS(a, b)}
                            for index, values, expected in zip(
                                indices, actual, reference, strict=True
                            )
                            for field, a, b in zip(
                                ["impedance", "work_compliance"], values, expected, strict=True
                            )
                        ]
                        bitwise = all(
                            np.array_equal(a[0], b[0])
                            for a, b in zip(actual, reference, strict=True)
                        )
                        passed = all(
                            v is not None and np.isfinite(v) and v <= 1e-8
                            for c in checks
                            for k, v in c.items()
                            if k.startswith("relative_")
                        )
                        if lazy and not general:
                            passed = passed and bitwise
                        attempts.append(
                            dict(
                                variant=name,
                                repeat=repeat,
                                seconds=seconds,
                                checks=checks,
                                impedance_bitwise=bitwise,
                                passed=bool(passed),
                            )
                        )
                        write_json(
                            output / f"branch{branch}-attempts.json", dict(attempts=attempts)
                        )
                write_json(
                    output / "progress.json",
                    dict(phase="independent_full_reconstruction", branch=branch),
                )
                full_reference = [
                    transfer.evaluate_operators(group, reconstruction=True)[1] for group in raw
                ]
                reconstruction = {}
                for name, (general, _) in VARIANTS.items():
                    checks = []
                    for index, group, expected in zip(indices, raw, full_reference, strict=True):
                        actual = policies[name].evaluate_operators(
                            POLICY["layers"](group, general), reconstruction=True
                        )[1]
                        checks.append(dict(mode=index, **ERRORS(actual, expected)))
                    reconstruction[name] = checks
                if [array_digest(group) for group in raw] != hashes:
                    raise ValueError("A solve policy modified shared original raw matrices")
                times = {
                    name: median(a["seconds"] for a in attempts if a["variant"] == name)
                    for name in VARIANTS
                }
                passed = {
                    name: all(a["passed"] for a in attempts if a["variant"] == name)
                    and all(
                        v is not None and np.isfinite(v) and v <= 1e-8
                        for c in reconstruction[name]
                        for k, v in c.items()
                        if k.startswith("relative_")
                    )
                    for name in VARIANTS
                }
                results.append(
                    dict(
                        branch=branch,
                        indices=indices,
                        raw_assembly_seconds=assembly,
                        raw_inputs=raw_records,
                        raw_arrays_unchanged=True,
                        paired_two_worker_median_seconds=times,
                        reference_to_variant_time_ratio={
                            k: times["auto"] / v for k, v in times.items()
                        },
                        attempts=attempts,
                        full_reconstruction_checks=reconstruction,
                        all_operator_gates_passed=passed,
                    )
                )
                write_json(output / "completed_pairs.json", dict(results=results))
                del raw, reference, actual, full_reference
        if (
            any(fingerprint(p) != value for p, value in frozen.items())
            or environment_record(args.case) != environment
        ):
            raise ValueError("Frozen physical source, saved raw inputs or runtime changed")
        write_json(
            output / "summary.json",
            dict(
                status="completed_identical_raw_solve_policy_diagnostic",
                results=results,
                seconds_including_original_assembly=perf_counter() - begun,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                all_source_and_raw_inputs_unchanged=True,
                full_event_or_fem_speed_claim=False,
                scope="Two pairs of actual fine-mesh physical modes; original assembly cost charged separately. Solve-only paired timings, full reconstruction gates. No full kernel, causal weights, nonlinear/AD or FEM performance proof.",
            ),
        )
    except BaseException as error:
        write_json(
            output / "failure.json",
            dict(error=repr(error), seconds=perf_counter() - begun, completed_pairs=results),
        )
        raise


if __name__ == "__main__":
    main()

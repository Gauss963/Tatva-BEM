"""Compare selected raw Galerkin blocks/fibers to independent full assembly.

This times an entry-access primitive for future compression. It does not build
an ACA kernel or claim a complete-event/FEM speed advantage.
"""

import argparse
import json
import runpy
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import plan_cq
from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators, prepared_geometry_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
BLOCK = runpy.run_path(str(TOOLS / "galerkin_blocks.py"))
MODES = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))["lower_half_radau_modes"]
FIELDS = ["single_layer", "trace_double_layer", "hypersingular", "mixed_mass"]


def check(actual, expected):
    if actual.shape != expected.shape or not all(
        np.all(np.isfinite(a)) for a in [actual, expected]
    ):
        raise ValueError("Invalid finite aligned block")
    error = np.linalg.norm(actual - expected)
    normalized = float(error / (1e-12 + 1e-11 * np.linalg.norm(expected)))
    if not np.isfinite(normalized) or normalized > 1:
        raise ValueError("Selected block differs from the complete original operator")
    return dict(
        relative_frobenius=float(error / max(np.linalg.norm(expected), 1e-30)),
        maximum_absolute=float(np.max(abs(actual - expected))),
        mixed_tolerance_normalized=normalized,
        bitwise_equal=bool(np.array_equal(actual, expected)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--body", choices=["moving", "stationary"], default="moving")
    p.add_argument("--h", type=float, default=10)
    p.add_argument("--contact-h", type=float, default=2)
    p.add_argument("--end-h", type=float, default=0.015625)
    p.add_argument("--quadrature", type=int, default=12)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-memory-mb", type=float, default=8192)
    args = p.parse_args()
    begun = perf_counter()
    output = args.output.resolve()
    if (
        output.exists()
        or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva"))
        or args.repeats < 2
    ):
        raise ValueError("Require a fresh output and alternating repetitions")
    model = load_nodal_model(
        args.case,
        max_panel_length=args.h,
        contact_panel_length=args.contact_h,
        contact_end_length=args.end_h,
        corner_panel_length=args.end_h,
    )
    mesh = getattr(model.reference, args.body)
    n = mesh.dofs
    estimate = prepared_geometry_bytes(mesh, args.quadrature) + 32 * n * n * 16 + 512 * 1024**2
    if not np.isfinite(args.max_memory_mb) or estimate > args.max_memory_mb * 1024**2:
        raise ValueError("Original single-body reference exceeds the explicit array budget")
    plan = plan_cq(
        2 * len(model.moving_contact) + 1, 2e-6, 22500, max_memory_mb=1e9, disk_spectrum=True
    )
    environment = environment_record(args.case)
    frozen = {
        str(q.resolve()): fingerprint(q)
        for q in [
            Path(__file__),
            TOOLS / "galerkin_blocks.py",
            TOOLS / "radau_raw_sampling.py",
            args.case,
        ]
    }
    output.mkdir(parents=True)
    write_json(
        output / "preflight.json",
        dict(
            environment=environment,
            frozen_inputs_sha256=frozen,
            mesh_dofs=n,
            body=args.body,
            quadrature=args.quadrature,
            estimated_array_bytes=estimate,
            original_tatva_readonly=True,
            array_budget_is_not_measured_rss=True,
        ),
    )
    with threadpool_limits(limits=1, user_api="blas"):
        start = perf_counter()
        provider = BLOCK["GalerkinBlockProvider"](
            mesh, model.reference.material, quadrature=args.quadrature
        )
        specs = {
            "corner": (np.arange(4), np.arange(4)),
            "separated": (np.arange(n // 3, n // 3 + 4), np.arange(2 * n // 3, 2 * n // 3 + 4)),
            "column_fiber": (np.arange(n), np.array([1])),
        }
        requests = {key: provider.request(*spec) for key, spec in specs.items()}
        block_preparation = perf_counter() - start
        start = perf_counter()
        reference = PreparedGalerkinOperators(
            mesh, model.reference.material, quadrature=args.quadrature
        )
        full_preparation = perf_counter() - start
        samples = []
        for index, branch in [(0, 0), (655, 0), (0, 1)]:
            z = plan.radius * np.exp(-2j * np.pi * index / plan.fft_size)
            s = MODES(z, plan.dt)[branch]
            times = {"full": [], "blocks": []}
            records = []
            for repeat in range(args.repeats):
                values = {}
                for name in ["full", "blocks"] if repeat % 2 == 0 else ["blocks", "full"]:
                    start = perf_counter()
                    values[name] = (
                        reference.assemble(s)
                        if name == "full"
                        else {k: r.evaluate(s) for k, r in requests.items()}
                    )
                    times[name].append(perf_counter() - start)
                errors = {
                    key: {
                        field: check(
                            getattr(values["blocks"][key], field),
                            getattr(values["full"], field)[np.ix_(*specs[key])],
                        )
                        for field in FIELDS
                    }
                    for key in specs
                }
                records.append(dict(repeat=repeat, errors=errors))
                del values
            samples.append(
                dict(
                    index=index,
                    branch=branch,
                    s_real=float(s.real),
                    s_imag=float(s.imag),
                    times=times,
                    median_seconds={key: median(v) for key, v in times.items()},
                    checks=records,
                )
            )
            write_json(output / "progress.json", dict(samples=samples))
            print(
                json.dumps({k: samples[-1][k] for k in ["index", "branch", "median_seconds"]}),
                flush=True,
            )
    if (
        any(fingerprint(q) != v for q, v in frozen.items())
        or environment_record(args.case) != environment
    ):
        raise ValueError("Numerical source/environment changed")
    write_json(
        output / "summary.json",
        dict(
            status="completed_exact_selected_raw_operator_control",
            environment=environment,
            frozen_inputs_sha256=frozen,
            body=args.body,
            mesh_dofs=n,
            block_preparation_seconds=block_preparation,
            original_full_preparation_seconds=full_preparation,
            requests={key: r.statistics() for key, r in requests.items()},
            samples=samples,
            estimated_array_bytes=estimate,
            total_seconds_including_checks=perf_counter() - begun,
            blas_threads=1,
            low_rank_or_frequency_approximation=False,
            complete_aca_or_cq_kernel_built=False,
            same_accuracy_fem_advantage_verified=False,
        ),
    )


if __name__ == "__main__":
    main()

"""Audit fixed geometric quadrature on a saved finite-block CQ contour.

Saves every sampled Z and work-resolvent matrix for later checks. Timings may
be recorded under contention; sampled differences and point counts do not
establish a whole-solver/FEM speed advantage or a global frequency error bound.
"""

import argparse
import gc
import json
import runpy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import bdf2_symbol, plan_cq
from finiteblock_bem.galerkin_fast import prepared_geometry_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--near-order", type=int, default=12)
    parser.add_argument("--far-orders", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--reference-order", type=int, default=18)
    parser.add_argument("--separation", type=float, default=2)
    parser.add_argument(
        "--angles", type=float, nargs="+", default=[0, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, np.pi]
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-memory-mb", type=float, default=8192)
    parser.add_argument("--sampled-rtol", type=float, default=1e-6)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new empty audit directory")
    if not isinstance(args.workers, int) or args.workers < 1:
        raise ValueError("Positive worker count required")
    if not np.isfinite(args.max_memory_mb) or args.max_memory_mb <= 0:
        raise ValueError("Finite positive memory budget required")
    if not np.isfinite(args.sampled_rtol) or args.sampled_rtol <= 0:
        raise ValueError("Finite positive sampled tolerance required")
    for order in [args.reference_order, args.near_order, *args.far_orders]:
        if order < 4 or order % 2:
            raise ValueError("Use even orders of at least 4")
    if args.reference_order < args.near_order or max(args.far_orders) > args.near_order:
        raise ValueError("Require far <= near <= reference orders")
    if args.far_orders != sorted(set(args.far_orders)):
        raise ValueError("Far orders must be strictly increasing and unique")
    if not np.isfinite(args.separation) or args.separation <= 0:
        raise ValueError("Finite positive separation required")
    angles = np.asarray(args.angles)
    if (
        not len(angles)
        or not np.all(np.isfinite(angles))
        or np.any((angles < 0) | (angles > np.pi))
    ):
        raise ValueError("Finite contour angles in [0, pi] required")
    helper_path = Path(__file__).with_name("mixed_quadrature_experiment.py")
    frozen = {
        str(path): fingerprint(path)
        for path in [
            Path(__file__),
            helper_path,
            args.source / "summary.json",
            args.source / "run_manifest.json",
        ]
    }
    source = json.loads((args.source / "summary.json").read_text())
    options = json.loads((args.source / "run_manifest.json").read_text())["arguments"]
    environment = environment_record(options["case"])
    keys = ["source_sha256", "case_sha256", "restart_data_sha256"]
    if (
        source["source_changed_during_run"]
        or source["status"] != "completed_requested_horizon"
        or any(source["environment"][name] != environment[name] for name in keys)
    ):
        raise ValueError("Require a complete source-frozen run")
    dt = source["dt_s"] if args.dt is None else args.dt
    duration = source["duration_s"] if args.duration is None else args.duration
    if not all(np.isfinite(x) and x > 0 for x in [dt, duration]):
        raise ValueError("Positive finite dt and duration required")
    steps = round(duration / dt)
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    d = 2 * len(model.moving_contact) + 1
    plan = plan_cq(d, dt, steps, disk_spectrum=True, max_memory_mb=1e9)
    # Only individual frequency matrices/geometry are allocated, no CQ weights.
    bodies = [model.reference.moving, model.reference.stationary]
    peak = sum(prepared_geometry_bytes(mesh, args.reference_order) for mesh in bodies)
    peak += args.workers * (
        14 * max(mesh.dofs**2 for mesh in bodies) * 16 + 32 * args.reference_order**2 * 1024
    )
    peak += 8 * len(angles) * d**2 * 16 + 128 * 1024**2
    if peak > args.max_memory_mb * 1024**2:
        raise ValueError(f"Audit geometry/array estimate needs {peak / 1024**2:.1f} MiB")
    args.output.mkdir(parents=True, exist_ok=True)
    frequencies = bdf2_symbol(plan.radius * np.exp(-1j * angles)) / dt
    line = np.r_[
        model.constraints.contact_weights,
        model.constraints.contact_weights,
        model.constraints.platen_width,
    ]
    scale = np.sqrt(line)
    n, law = len(model.moving_contact), model.reference.contact
    tangent = np.diag(
        np.r_[line[:n] * law.normal_penalty, line[n : 2 * n] * law.tangential_penalty, 0]
    )
    tangent[-1, -1] += model.reference.spring_stiffness * model.constraints.platen_width
    helper = runpy.run_path(str(helper_path))
    records, values = [], {}
    configurations = [
        ("reference", args.reference_order, None),
        ("uniform_near", args.near_order, None),
    ]
    configurations += [(f"mixed_far{order}", args.near_order, order) for order in args.far_orders]
    with threadpool_limits(limits=1, user_api="blas"):
        for name, order, far in configurations:
            tick = perf_counter()
            transfer = (
                NodalInterfaceTransfer(model, quadrature=order)
                if far is None
                else helper["mixed_transfer"](
                    model, near_order=order, far_order=far, separation=args.separation
                )
            )
            preparation = perf_counter() - tick
            geometry = [] if far is None else [body.statistics() for body in transfer.prepared]
            print(
                json.dumps(
                    dict(stage="prepared", name=name, seconds=preparation, geometry=geometry)
                ),
                flush=True,
            )

            def evaluate(item, operator=transfer):
                index, s = item
                begin = perf_counter()
                z = operator.evaluate(s)
                h = scale[:, None] * solve(z + tangent, np.diag(scale), check_finite=False)
                if not np.all(np.isfinite(z)) or not np.all(np.isfinite(h)):
                    raise ValueError("Nonfinite sampled operator")
                elapsed = perf_counter() - begin
                print(
                    json.dumps(dict(stage="frequency", name=name, index=index, seconds=elapsed)),
                    flush=True,
                )
                return z, h, elapsed

            with ThreadPoolExecutor(args.workers) as pool:
                samples = list(pool.map(evaluate, enumerate(frequencies)))
            z, h = (np.stack([sample[index] for sample in samples]) for index in [0, 1])
            values[name] = (z, h)
            np.savez(
                args.output / (name + ".npz"),
                impedance=z,
                work_resolvent=h,
                frequencies=frequencies,
            )
            records.append(
                dict(
                    name=name,
                    preparation_seconds=preparation,
                    frequency_seconds=[sample[2] for sample in samples],
                    geometry=geometry,
                    matrices_sha256=fingerprint(args.output / (name + ".npz")),
                )
            )
            del samples, evaluate, transfer
            gc.collect()

    def differences(reference):
        errors = {}
        rz, rh = values[reference]
        for name, (z, h) in values.items():
            ez = (
                np.linalg.norm((z - rz) / line[None, :, None], axis=(1, 2))
                / np.linalg.norm(rz / line[None, :, None], axis=(1, 2))
            ).tolist()
            eh = (np.linalg.norm(h - rh, axis=(1, 2)) / np.linalg.norm(rh, axis=(1, 2))).tolist()
            errors[name] = dict(
                traction_scaled_impedance_relative=ez,
                work_resolvent_relative=eh,
                sampled_tolerance_satisfied=max(*ez, *eh) <= args.sampled_rtol,
            )
        return errors

    errors, added_errors = differences("reference"), differences("uniform_near")
    current = environment_record(options["case"])
    if any(current[name] != environment[name] for name in keys) or any(
        fingerprint(path) != value for path, value in frozen.items()
    ):
        raise ValueError("Audit implementation, package or source changed")
    record = dict(
        status="completed_sampled_quadrature_audit",
        source_run=str(args.source.resolve()),
        source_mesh_arguments=options,
        plan=asdict(plan),
        angles=angles.tolist(),
        actual_s_real=frequencies.real.tolist(),
        actual_s_imag=frequencies.imag.tolist(),
        records=records,
        errors_relative_to_reference=errors,
        errors_relative_to_uniform_near=added_errors,
        sampled_relative_tolerance=args.sampled_rtol,
        environment=environment,
        frozen_inputs_sha256=frozen,
        estimated_array_peak_mib=peak / 1024**2,
        timing_scope="May overlap other jobs; no controlled speed claim",
        scope="Sampled geometric quadrature differences only, not full-event/FEM accuracy or a continuous-frequency error bound",
    )
    write_json(args.output / "summary.json", record)
    print(json.dumps(errors, indent=2), flush=True)


if __name__ == "__main__":
    main()

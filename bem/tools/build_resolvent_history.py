"""Build elastic-resolvent CQ directly, with a bounded correction horizon.

The contour covers the requested full time interval. Only the correction
prefix has matrix-valued time weights; the elastic startup is a full-horizon
vector. No nonlinear simulation, rational fit or tail deletion is performed.
This prototype stays outside the production package until trajectory and
adjoint equivalence have been checked.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import runpy
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import convolution_weights, plan_cq
from finiteblock_bem.cq_storage import DiskSpectrum
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model
from finiteblock_bem.nodal_dynamics import plan_nodal_history
from finiteblock_bem.sampling import SamplingOptions, parsimonious_weights

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path(__file__).with_name("replay_resolvent.py")


def compliance_in_work_coordinates(impedance, tangent, line_weights):
    scale = np.sqrt(line_weights)
    return scale[:, None] * solve(impedance + tangent, np.diag(scale), check_finite=False)


def physical_weights_inplace(weights, line_weights):
    scale = np.sqrt(line_weights)
    factor = scale[:, None] * scale[None, :]
    for first in range(0, len(weights), 64):
        weights[first : first + 64] /= factor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Completed run specifying geometry/case")
    parser.add_argument("--dt", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--correction-duration", type=float)
    parser.add_argument("--sampling", choices=["dense", "adaptive"], default="adaptive")
    parser.add_argument("--sampling-rtol", type=float, default=1e-8)
    parser.add_argument(
        "--sampling-atol",
        type=float,
        default=1e-14,
        help="Absolute tolerance in work-compliance coordinates",
    )
    parser.add_argument("--workers", type=int, default=int(os.environ.get("FDBEM_CORES", "1")))
    parser.add_argument("--max-memory-mb", type=float, default=4096)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/direct-resolvent-history")
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        raise ValueError("Choose a new output folder; completed kernels are never overwritten")
    started = perf_counter()
    source = json.loads((args.source / "summary.json").read_text())
    options = json.loads((args.source / "run_manifest.json").read_text())["arguments"]
    environment = environment_record(options["case"])
    for name in ["source_sha256", "case_sha256", "restart_data_sha256"]:
        if environment[name] != source["environment"][name]:
            raise ValueError(f"Source case differs in {name}")
    if source["status"] != "completed_requested_horizon" or source["source_changed_during_run"]:
        raise ValueError("A complete frozen source run is required")
    helper_digest = hashlib.sha256(HELPER.read_bytes()).hexdigest()
    helper_source = HELPER.read_text()
    helper_node = next(
        node
        for node in ast.parse(helper_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "causal_elastic_baseline"
    )
    helper_math_digest = hashlib.sha256(
        ast.dump(helper_node, include_attributes=False).encode()
    ).hexdigest()
    helper = runpy.run_path(str(HELPER))
    dt = options["dt"] if args.dt is None else args.dt
    duration = source["duration_s"] if args.duration is None else args.duration
    correction = duration if args.correction_duration is None else args.correction_duration
    if (
        not all(np.isfinite(x) and x > 0 for x in [dt, duration, correction])
        or correction > duration
    ):
        raise ValueError("Require finite positive dt and 0 < correction duration <= full duration")
    full_steps, correction_steps = round(duration / dt), math.ceil(correction / dt)
    if full_steps < 1 or correction_steps > full_steps:
        raise ValueError("Full/correction horizons are incompatible with the timestep")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    sampling = (
        SamplingOptions(
            relative_tolerance=args.sampling_rtol, absolute_tolerance=args.sampling_atol
        )
        if args.sampling == "adaptive"
        else None
    )
    d = 2 * len(model.moving_contact) + 1
    short = plan_cq(d, dt, correction_steps, max_memory_mb=args.max_memory_mb, disk_spectrum=True)
    # These contour parameters depend on the FULL physical horizon, while
    # matrix time storage depends on the correction horizon.
    fft_size = 1 << math.ceil(math.log2(2 * (full_steps + 1)))
    radius = 1e-10 ** (1 / fft_size)
    plan = replace(short, fft_size=fft_size, radius=radius)
    _, base_peak = plan_nodal_history(
        model,
        dt,
        correction_steps,
        workers=args.workers,
        quadrature=options["quadrature"],
        sampling=sampling,
        max_memory_mb=args.max_memory_mb,
        spectrum_directory=args.output,
    )
    extra = 4 * (fft_size - short.fft_size) * min(d, 4) * d * 8
    extra += 8 * fft_size * d * 16 + 128 * 1024**2
    peak = base_peak + extra
    if peak > args.max_memory_mb * 1024**2:
        raise ValueError(f"Direct resolvent workspace needs about {peak / 1024**2:.1f} MiB")
    args.output.mkdir(parents=True, exist_ok=True)
    required_disk = (fft_size // 2 + 1) * d * d * 16 + plan.weight_bytes + extra
    if shutil.disk_usage(args.output).free < required_disk:
        raise ValueError("Insufficient disk space for complete spectrum and matrix prefix")
    line_weights = np.r_[
        model.constraints.contact_weights,
        model.constraints.contact_weights,
        model.constraints.platen_width,
    ]
    n = len(model.moving_contact)
    contact = model.reference.contact
    cdiag = np.r_[
        line_weights[:n] * contact.normal_penalty,
        line_weights[n : 2 * n] * contact.tangential_penalty,
        0,
    ]
    spring = model.reference.spring_stiffness * model.constraints.platen_width
    tangent = np.diag(cdiag)
    tangent[-1, -1] += spring
    spectrum = DiskSpectrum(args.output / "work-resolvent.npy", plan)
    transfer = NodalInterfaceTransfer(model, quadrature=options["quadrature"])
    last_progress = [0.0]

    def progress(done, total):
        now = perf_counter()
        if done == total or now - last_progress[0] >= 30:
            print(f"progress: {done}/{total}", flush=True)
            last_progress[0] = now

    def evaluate(s, operator=transfer):
        return compliance_in_work_coordinates(operator.evaluate(s), tangent, line_weights)

    with threadpool_limits(limits=1, user_api="blas"):
        if sampling is None:
            weights = convolution_weights(
                evaluate, plan, workers=args.workers, spectrum=spectrum, progress=progress
            )
            metadata = {"method": "dense", "transfer_evaluations": fft_size // 2 + 1}
        else:
            weights, metadata = parsimonious_weights(
                evaluate,
                plan,
                options=sampling,
                workers=args.workers,
                spectrum=spectrum,
                progress=progress,
            )
        # The generating-function constant coefficient is Z(delta(0)/dt).
        impedance0 = transfer.evaluate(1.5 / dt)
    if np.max(abs(impedance0.imag)) > 1e-10:
        raise ValueError("Positive real s must have a real instantaneous impedance")
    del evaluate, transfer
    physical_weights_inplace(weights, line_weights)
    np.save(args.output / "weights.npy", weights, allow_pickle=False)
    np.save(args.output / "initial_impedance.npy", impedance0.real, allow_pickle=False)
    composition = float(np.linalg.norm(weights[0] @ (impedance0 + tangent) - np.eye(d), ord=np.inf))
    if composition > 1e-6:
        raise ValueError(f"CQ inverse composition failed: {composition}")
    column = spectrum[:, :, -1] / np.sqrt(line_weights * line_weights[-1])
    full_plan = replace(plan, steps=full_steps)
    time = np.arange(full_steps + 1) * dt
    increment = options.get("diagnostic_actuator_increment")
    rate = model.reference.actuator_rate if increment is None else increment / time[-1]
    force = spring * rate * time
    elastic = helper["causal_elastic_baseline"](column, full_plan, force)
    np.savez(
        args.output / "elastic.npz",
        time=time,
        elastic=elastic,
        uninterrupted_actuator=model.reference.restart_actuator + rate * time,
    )
    final_environment = environment_record(options["case"])
    if (
        environment["source_sha256"] != final_environment["source_sha256"]
        or helper_digest != hashlib.sha256(HELPER.read_bytes()).hexdigest()
    ):
        raise ValueError("Numerical source changed during kernel construction")
    record = dict(
        status="completed_resolvent_kernel_only",
        source_run=str(args.source.resolve()),
        source_mesh_arguments=options,
        dt_s=dt,
        full_steps=full_steps,
        correction_steps=correction_steps,
        plan=asdict(plan),
        history_dofs=d,
        frequency_sampling=metadata,
        sampling_options=None if sampling is None else asdict(sampling),
        spectrum_storage={**spectrum.record(), "coordinates": "diag(sqrt(w)) * L * diag(sqrt(w))"},
        initial_composition_inf_error=composition,
        estimated_array_peak_mib=peak / 1024**2,
        total_seconds=perf_counter() - started,
        environment=final_environment,
        helper_sha256=helper_digest,
        helper_math_sha256=helper_math_digest,
        weights_sha256=helper["digest_file"](args.output / "weights.npy"),
        elastic_sha256=helper["digest_file"](args.output / "elastic.npz"),
        prototype_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        full_original_event_reproduced=False,
        scope="Direct BEM elastic resolvent kernel and full elastic startup; matrix prefix only; nonlinear trajectory and adjoint still require validation",
    )
    write_json(args.output / "summary.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

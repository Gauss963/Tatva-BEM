"""Isolate loading-feedback error using frozen BEM kernels and FEM actuator data.

This is a prescribed-loading diagnostic, not the original contact-coverage
hold policy. Materials, contact law, static lift, mesh and CQ horizon stay
fixed. No new CQ kernels are allowed to be built by this tool.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from finiteblock_bem.cq import plan_cq
from finiteblock_bem.dynamics import LoadingSchedule
from finiteblock_bem.io import environment_record, fingerprint, save_dynamic, write_json
from finiteblock_bem.nodal import load_nodal_model
from finiteblock_bem.nodal_dynamics import nodal_cache_key, simulate_nodal
from finiteblock_bem.sampling import SamplingOptions

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "validation/static_bem2d/reference/fem_history.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--history", type=Path, default=REFERENCE)
    parser.add_argument("--name", default="galerkin-fem-loading-replay")
    args = parser.parse_args()
    if not args.name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.name
    ):
        raise ValueError("Name must contain only letters, digits, underscores and hyphens")
    manifest = json.loads((args.run / "run_manifest.json").read_text())
    original = json.loads((args.run / "summary.json").read_text())
    if original["status"] != "completed_requested_horizon" or original["source_changed_during_run"]:
        raise ValueError("Provide a completed source-frozen reference BEM run")
    options = manifest["arguments"]
    case = Path(options["case"])
    if fingerprint(case) != manifest["environment"]["case_sha256"]:
        raise ValueError("The original case is required")
    model = load_nodal_model(
        case,
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    ref = model.reference
    dt, steps = original["dt_s"], original["completed_steps"]
    plan = plan_cq(2 * len(model.moving_contact) + 1, dt, steps, max_memory_mb=65536)
    sampling = (
        SamplingOptions(relative_tolerance=options["sampling_rtol"])
        if options["sampling"] == "adaptive"
        else None
    )
    key = nodal_cache_key(model, plan, options["quadrature"], sampling)
    cached = Path(options["cache"]) / f"galerkin-{key}.npy"
    if not cached.is_file() or not cached.with_suffix(".json").is_file():
        raise ValueError("Matching verified CQ cache required; kernel rebuilding is disabled")
    data = np.genfromtxt(args.history, delimiter=",", names=True)
    keep = data["time"] >= 0.002
    time = np.r_[0.0, data["time"][keep] - 0.002]
    actuator = np.r_[ref.restart_actuator, data["applied_shear_displacement"][keep]]
    if np.any(np.diff(time) <= 0) or time[-1] < steps * dt:
        raise ValueError("FEM history must cover the complete BEM interval")
    stop = np.flatnonzero(data["shear_loading_stopped"][keep] > 0.5)
    prescribed_hold = None if not len(stop) else float(time[stop[0] + 1])
    schedule = LoadingSchedule(
        lambda _: ref.normal_pressure,
        lambda t: float(np.interp(t, time, actuator)),
        "recorded_fem_actuator_diagnostic",
    )
    output = ROOT / "outputs" / args.name
    output.mkdir(parents=True, exist_ok=True)
    environment = environment_record(case)
    if environment["restart_data_sha256"] != manifest["environment"]["restart_data_sha256"]:
        raise ValueError("The original restart seed is required")
    loading = dict(
        mode=schedule.name,
        history_path=str(args.history.resolve()),
        history_sha256=fingerprint(args.history),
        reference_bem_manifest_sha256=fingerprint(args.run / "run_manifest.json"),
        runner_sha256=fingerprint(__file__),
        time_origin_s=0.002,
        initial_anchor="At t=0 use exact static-lift actuator, before the first positive-time FEM sample",
        prescribed_hold_time_s=prescribed_hold,
        contact_coverage_feedback_enabled=False,
    )
    write_json(
        output / "run_manifest.json",
        dict(
            arguments=dict(options, name=args.name, workers=1),
            environment=environment,
            loading=loading,
        ),
    )
    result = simulate_nodal(
        model,
        dt,
        steps,
        schedule=schedule,
        backend="jax",
        workers=1,
        quadrature=options["quadrature"],
        sampling=sampling,
        max_memory_mb=options["max_memory_mb"],
        cache_directory=options["cache"],
    )
    result.diagnostics.update(loading_diagnostic=loading, actuator_hold_time_s=prescribed_hold)
    summary = save_dynamic(
        output,
        model.contact_case(),
        result,
        case,
        start_environment=environment,
        contact_y=ref.moving.starts[model.moving_contact, 1],
        contact_weights=model.constraints.contact_weights,
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "status",
                    "schedule",
                    "max_cumulative_slip_mm",
                    "actuator_hold_time_s",
                    "frequency_sampling",
                ]
            },
            indent=2,
        )
    )
    if summary["status"] != "completed_requested_horizon":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Check full contact-parameter gradients including the healed static lift.

Requires a completed, source-frozen fem-active-set primal. Reverse VJP and
batched forward JVP use complete CQ history. The physical bulk, original
restart input, normal load, selected branches and recorded actuator stay fixed.
"""

import argparse
import gc
import json
import os
import resource
import runpy
import sys
from functools import partial
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.fft import set_workers

from finiteblock_bem.cq import BoundaryHistory, CQPlan
from finiteblock_bem.friction import ContactState
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model, nodal_body_stiffness

TOOLS = Path(__file__).resolve().parent
READ = runpy.run_path(str(TOOLS / "replay_resolvent.py"))


def load_primal(run, *, cache_mb, fft_rows, max_memory_mb):
    record = json.loads((run / "summary.json").read_text())
    kernel = Path(record["prepared_kernel"])
    metadata = json.loads((kernel / "summary.json").read_text())
    plan = CQPlan(**metadata["plan"])
    if (
        record["status"] != "completed_requested_horizon"
        or record["history_representation"] != "elastic_resolvent_refined_contact"
        or record["correction_start"] != 1
        or record.get("static_initialization") != "fem-active-set"
        or record["duration_s"] != 0.045
        or not record["all_history_retained"]
        or record["max_boundary_contact_residual_mpa"] >= 1e-7
        or record["completed_steps"] != plan.steps
        or fingerprint(kernel / "summary.json") != record["kernel_summary_sha256"]
    ):
        raise ValueError("Require a complete converged healed-static refined-contact primal")
    if any(fingerprint(p) != v for p, v in record["frozen_input_sha256"].items()):
        raise ValueError("Primal numerical source or lift input changed")
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    for key in ("source_sha256", "case_sha256", "restart_data_sha256"):
        if (
            environment[key] != record["environment"][key]
            or environment[key] != metadata["environment"][key]
        ):
            raise ValueError("Physical bulk package or inputs changed")
    if READ["digest_file"](kernel / "weights.npy") != metadata["weights_sha256"]:
        raise ValueError("Complete physical history weights changed")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    seed_path = Path(record["restart_seed_override"]["path"])
    model, _ = runpy.run_path(str(TOOLS / "restart_seed_experiment.py"))["apply_restart_seed"](
        model, seed_path
    )
    refined = runpy.run_path(str(TOOLS / "refined_contact.py"))
    space = refined["ContactQuadrature"].build(
        model.reference.moving.starts[model.moving_contact, 1], record["subdivisions"]
    )
    bounded = runpy.run_path(str(TOOLS / "bounded_history.py"))
    estimate = bounded["estimate_online_arrays"](plan, plan.steps, cache_mb, fft_rows=fft_rows)
    estimate += (plan.steps + 1) * (4 * space.points + space.dofs) * 8
    if estimate > max_memory_mb * 1024**2:
        raise ValueError("Complete initialization-aware AD arrays exceed the budget")
    if record["seed_interpolation"] == "coarse-p1":
        seed = space.seed_from_bulk_p1(model.restart)
    else:
        with np.load(seed_path, allow_pickle=False) as data:
            seed = ContactState(
                np.interp(space.y, data["y"], data["plastic_slip"]),
                np.interp(space.y, data["y"], data["cumulative_slip"]),
            )
    reduction = refined["reduce_static_trace"](
        model, nodal_body_stiffness(model, quadrature=options["quadrature"])
    )
    static = refined["RefinedContactEquilibrium"](
        model, space, reduction.impedance, base_force=-reduction.effective_load
    )
    lift = runpy.run_path(str(TOOLS / "fem_static_contact.py"))["initialize_fem_static_contact"](
        model, space, reduction, seed
    )
    trajectory_sha = READ["digest_file"](run / "trajectory.npz")
    with np.load(run / "trajectory.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["contact_y"], space.y)
        np.testing.assert_array_equal(data["contact_weights"], space.weights)
        np.testing.assert_allclose(
            data["initial_retained_displacement"], lift.retained_displacement, rtol=0, atol=1e-9
        )
        np.testing.assert_allclose(
            data["initial_retained_bulk_force"], lift.retained_bulk_force, rtol=0, atol=1e-7
        )
        result = SimpleNamespace(
            **{k: data[k] for k in ("time", "plastic_slip", "cumulative_slip")},
            bulk_displacement=data["retained_displacement"],
        )
    np.testing.assert_allclose(result.plastic_slip[0], lift.state.plastic_slip, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(result.cumulative_slip[0], lift.state.cumulative_slip)
    reference = refined["dynamic_refined_reference"](
        model, space, lift, np.load(kernel / "initial_impedance.npy", allow_pickle=False)
    )
    jacobian = runpy.run_path(str(TOOLS / "fem_static_contact_ad.py"))[
        "healed_static_parameter_jacobian"
    ](
        static,
        model.reference.normal_pressure,
        model.reference.restart_actuator,
        seed,
        lift.retained_displacement,
    )
    history = BoundaryHistory(
        plan, np.load(kernel / "weights.npy", mmap_mode="r", allow_pickle=False)
    )
    del reduction
    return (
        record,
        environment,
        static,
        jacobian,
        reference,
        history,
        result,
        bounded,
        trajectory_sha,
        estimate,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fft-cache-mb", type=float, default=512)
    parser.add_argument("--fft-rows", type=int, default=4)
    parser.add_argument("--max-memory-mb", type=float, default=8192)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a new output outside protected Tatva")
    if args.fft_rows < 1 or not all(
        np.isfinite(v) and v > 0 for v in (args.fft_cache_mb, args.max_memory_mb)
    ):
        raise ValueError("Require positive finite history budgets")
    names = [
        "initialized_refined_adjoint.py",
        "fem_static_contact_ad.py",
        "fem_static_contact.py",
        "batched_history.py",
        "batched_refined_adjoint.py",
        "refined_contact_adjoint.py",
        "refined_contact.py",
        "bounded_history.py",
        "replay_resolvent.py",
        "substepped_contact_experiment.py",
        "restart_seed_experiment.py",
    ]
    paths = [Path(__file__), args.run / "summary.json", *[TOOLS / n for n in names]]
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    begun = perf_counter()
    (
        record,
        environment,
        static,
        jacobian,
        reference,
        history,
        result,
        bounded,
        trajectory_sha,
        estimate,
    ) = load_primal(
        args.run,
        cache_mb=args.fft_cache_mb,
        fft_rows=args.fft_rows,
        max_memory_mb=args.max_memory_mb,
    )
    frozen.update(record["frozen_input_sha256"])
    weights_path = Path(record["prepared_kernel"]) / "weights.npy"
    weights_stat = (weights_path.stat().st_size, weights_path.stat().st_mtime_ns)
    output.mkdir(parents=True)
    write_json(
        output / "run_manifest.json",
        dict(
            status="prepared_not_completed",
            frozen_input_sha256=frozen,
            primal_trajectory_sha256=trajectory_sha,
            estimated_arrays_mib=estimate / 1024**2,
            original_tatva_readonly=True,
        ),
    )
    initialized = runpy.run_path(str(TOOLS / "initialized_refined_adjoint.py"))[
        "initialized_gradient"
    ]
    keyword = dict(constitutive_substeps=record["constitutive_substeps"])
    with set_workers(int(os.environ.get("FDBEM_CORES", "1"))):
        reverse = initialized(
            "reverse_gradient",
            static,
            jacobian,
            reference,
            history,
            result,
            convolution_factory=partial(
                bounded["BoundedOnlineConvolution"],
                cache_mb=args.fft_cache_mb,
                fft_rows=args.fft_rows,
            ),
            **keyword,
        )
        gc.collect()
        forward = initialized(
            "batched_forward_gradient",
            static,
            jacobian,
            reference,
            history,
            result,
            cache_mb=args.fft_cache_mb,
            fft_rows=args.fft_rows,
            **keyword,
        )
    error = np.asarray(reverse["gradient"]) - np.asarray(forward["gradient"])
    passed = bool(np.allclose(reverse["gradient"], forward["gradient"], rtol=1e-7, atol=1e-7))
    if (
        any(fingerprint(p) != v for p, v in frozen.items())
        or READ["digest_file"](args.run / "trajectory.npz") != trajectory_sha
    ):
        raise ValueError("Initialization-aware AD source or primal changed")
    metadata = json.loads((Path(record["prepared_kernel"]) / "summary.json").read_text())
    final_environment = environment_record(metadata["source_mesh_arguments"]["case"])
    if any(
        final_environment[k] != environment[k]
        for k in ("source_sha256", "case_sha256", "restart_data_sha256")
    ):
        raise ValueError("Bulk environment changed")
    if (weights_path.stat().st_size, weights_path.stat().st_mtime_ns) != weights_stat:
        raise ValueError("Complete physical history weights changed during AD")
    np.savez(
        output / "gradients.npz",
        reverse=reverse["gradient"],
        forward=forward["gradient"],
        actuator_gradient=reverse["actuator_gradient"],
        initial_displacement_jacobian=jacobian["displacement_jacobian"],
        initial_plastic_jacobian=jacobian["plastic_jacobian"],
    )
    write_json(
        output / "summary.json",
        dict(
            status="passed_initialized_full_history_ad_check"
            if passed
            else "failed_initialized_full_history_ad_check",
            primal_run=str(args.run.resolve()),
            frozen_input_sha256=frozen,
            primal_trajectory_sha256=trajectory_sha,
            reverse={
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in reverse.items()
                if k != "actuator_gradient"
            },
            forward={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in forward.items()},
            gradient_maximum_absolute_difference=float(np.max(abs(error))),
            rtol=1e-7,
            atol=1e-7,
            total_seconds_including_load_and_jit=perf_counter() - begun,
            peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2 if sys.platform == "darwin" else 1024),
            environment=environment,
            original_tatva_readonly=True,
            same_accuracy_fem_advantage_verified=False,
            scope="Full event discrete parameter gradients; fixed bulk/restart input/recorded actuator, selected static and transient branches",
        ),
    )
    if not passed:
        raise ValueError("Full initialized reverse/JVP gradient check failed at original tolerance")
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

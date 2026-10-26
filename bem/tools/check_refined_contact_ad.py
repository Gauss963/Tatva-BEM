"""Check full-history refined-contact reverse AD against five forward JVPs.

Outputs are separate from the frozen primal run. The objective is the final
contact-weighted mean cumulative slip, conditional on the saved lift, bulk,
initial state, actuator schedule and selected discrete contact branches.
"""

import argparse
import gc
import json
import resource
import runpy
import sys
from functools import partial
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from finiteblock_bem.cq import BoundaryHistory, CQPlan, OnlineConvolution
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
READ = runpy.run_path(str(TOOLS / "replay_resolvent.py"))


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
        raise ValueError("Choose a new AD output outside protected Tatva")
    if args.fft_rows < 1 or not all(
        np.isfinite(v) and v > 0 for v in (args.fft_cache_mb, args.max_memory_mb)
    ):
        raise ValueError("Require finite positive FFT budgets")
    begun = perf_counter()
    summary_path = args.run / "summary.json"
    summary = json.loads(summary_path.read_text())
    kernel = Path(summary["prepared_kernel"])
    metadata_path = kernel / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    plan = CQPlan(**metadata["plan"])
    if (
        summary["status"] != "completed_requested_horizon"
        or summary["history_representation"] != "elastic_resolvent_refined_contact"
        or not summary["all_history_retained"]
        or summary["correction_start"] != 1
        or summary["completed_steps"] != metadata["full_steps"]
        or summary["max_boundary_contact_residual_mpa"] >= 1e-7
        or plan.steps < metadata["full_steps"]
        or fingerprint(metadata_path) != summary["kernel_summary_sha256"]
    ):
        raise ValueError("Require a completed converged full refined-contact primal")
    # The short regression suite and a full event check are separate evidence.
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    for key in ("source_sha256", "case_sha256", "restart_data_sha256"):
        if (
            environment[key] != summary["environment"][key]
            or environment[key] != metadata["environment"][key]
        ):
            raise ValueError("Bulk package or physical inputs changed")
    if any(fingerprint(path) != value for path, value in summary["frozen_input_sha256"].items()):
        raise ValueError("Primal solver, seed or prepared lifting inputs changed")
    selected = [
        Path(__file__),
        summary_path,
        TOOLS / "refined_contact_adjoint.py",
        TOOLS / "refined_contact.py",
        TOOLS / "substepped_contact_experiment.py",
        TOOLS / "bounded_history.py",
        TOOLS / "replay_resolvent.py",
        kernel / "initial_impedance.npy",
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in selected}
    trajectory_path = args.run / "trajectory.npz"
    trajectory_sha = READ["digest_file"](trajectory_path)
    if READ["digest_file"](kernel / "weights.npy") != metadata["weights_sha256"]:
        raise ValueError("Complete physical history weights changed")
    tool = runpy.run_path(str(TOOLS / "refined_contact.py"))
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    space = tool["ContactQuadrature"].build(
        model.reference.moving.starts[model.moving_contact, 1], summary["subdivisions"]
    )
    bounded = runpy.run_path(str(TOOLS / "bounded_history.py"))
    estimated_arrays = (
        bounded["estimate_online_arrays"](
            plan, metadata["full_steps"], args.fft_cache_mb, fft_rows=args.fft_rows
        )
        + (metadata["full_steps"] + 1) * (4 * space.points + space.dofs) * 8
    )
    if estimated_arrays > args.max_memory_mb * 1024**2:
        raise ValueError("Estimated five-direction full-history AD arrays exceed the budget")
    with np.load(trajectory_path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["contact_y"], space.y)
        np.testing.assert_array_equal(saved["retained_contact_y"], space.bulk_y)
        np.testing.assert_array_equal(saved["contact_weights"], space.weights)
        q0 = saved["initial_retained_displacement"]
        base_normal = space.interpolate(q0[: space.bulk_nodes])
        base_tangent = space.interpolate(q0[space.bulk_nodes : -1])
        jumps = saved["contact_jumps"][0]
        np.testing.assert_array_equal(jumps, np.r_[base_normal, base_tangent])
        reference = tool["RefinedContactEquilibrium"](
            model,
            space,
            np.load(kernel / "initial_impedance.npy", allow_pickle=False),
            base_normal=base_normal,
            base_tangent=base_tangent,
            base_platen=q0[-1],
            base_force=saved["initial_retained_bulk_force"],
        )
        result = SimpleNamespace(
            **{name: saved[name] for name in ("time", "plastic_slip", "cumulative_slip")},
            bulk_displacement=saved["retained_displacement"],
        )
    factory = partial(
        bounded["BoundedOnlineConvolution"],
        cache_mb=args.fft_cache_mb,
        fft_rows=args.fft_rows,
    )
    # Keep the ordinary and bounded history API aligned explicitly.
    if not issubclass(bounded["BoundedOnlineConvolution"], OnlineConvolution):
        raise ValueError("Re-audit the bounded transpose/JVP history adapter")
    history = BoundaryHistory(
        plan, np.load(kernel / "weights.npy", mmap_mode="r", allow_pickle=False)
    )
    output.mkdir(parents=True)
    write_json(
        output / "run_manifest.json",
        dict(
            status="prepared_not_completed",
            primal_run=str(args.run.resolve()),
            primal_trajectory_sha256=trajectory_sha,
            frozen_input_sha256=frozen,
        ),
    )
    ad = runpy.run_path(str(TOOLS / "refined_contact_adjoint.py"))
    keywords = dict(
        constitutive_substeps=summary.get("constitutive_substeps", 1), convolution_factory=factory
    )
    reverse = ad["reverse_gradient"](reference, history, result, **keywords)
    gc.collect()
    forward = ad["forward_gradient"](reference, history, result, **keywords)
    error = reverse["gradient"] - forward["gradient"]
    passed = bool(np.allclose(reverse["gradient"], forward["gradient"], rtol=1e-8, atol=1e-8))
    if (
        any(fingerprint(path) != value for path, value in frozen.items())
        or READ["digest_file"](trajectory_path) != trajectory_sha
    ):
        raise ValueError("Source or primal trajectory changed during AD")
    final_environment = environment_record(options["case"])
    if any(
        final_environment[key] != environment[key]
        for key in ("source_sha256", "case_sha256", "restart_data_sha256")
    ):
        raise ValueError("Bulk package or physical inputs changed during AD")
    np.savez(
        output / "gradients.npz",
        reverse_gradient=reverse["gradient"],
        forward_gradient=forward["gradient"],
        actuator_gradient=reverse["actuator_gradient"],
    )
    report = dict(
        status="passed_discrete_reverse_jvp_check" if passed else "failed_reverse_jvp_check",
        primal_run=str(args.run.resolve()),
        primal_trajectory_sha256=trajectory_sha,
        frozen_input_sha256=frozen,
        environment=environment,
        parameter_order=ad["PARAMETERS"],
        objective=reverse["objective"],
        value_mm=reverse["value_mm"],
        reverse_gradient=reverse["gradient"].tolist(),
        forward_gradient=forward["gradient"].tolist(),
        maximum_absolute_difference=float(np.max(abs(error))),
        reverse_seconds_including_jit=reverse["seconds_including_jit"],
        forward_seconds_including_jit=forward["seconds_including_jit"],
        total_including_input_hashes_seconds=perf_counter() - begun,
        peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        estimated_array_peak_mib=estimated_arrays / 1024**2,
        scope=reverse["scope"],
        every_history_source_included=True,
        spatial_temporal_gradient_convergence_verified=False,
        same_accuracy_fem_advantage_verified=False,
        original_tatva_readonly=True,
    )
    write_json(output / "summary.json", report)
    print(output / "summary.json", flush=True)
    if not passed:
        raise ValueError("Saved full-event reverse gradient differs from independent forward JVPs")


if __name__ == "__main__":
    main()

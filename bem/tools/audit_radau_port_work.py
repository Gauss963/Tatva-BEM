"""Measure incremental RK bulk work under a verified original-equation bound.

Uses existing saved stages and a completed independent causal operator audit.
No full frequency matrices or interior body fields are reconstructed here.
The result is one selected trajectory diagnostic, not a stability proof.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
WORK = runpy.run_path(str(TOOLS / "radau_port_work.py"))
DIGEST = runpy.run_path(str(TOOLS / "replay_resolvent.py"))["digest_file"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("operator_audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-prefix", action="store_true")
    parser.add_argument("--max-memory-mb", type=float, default=4096)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose fresh work evidence outside protected Tatva")
    if not np.isfinite(args.max_memory_mb) or args.max_memory_mb <= 0:
        raise ValueError("A positive finite memory budget is required")
    begun = perf_counter()
    run_path, audit_path = args.run / "summary.json", args.operator_audit / "summary.json"
    run, audit, manifest = (
        json.loads(p.read_text())
        for p in [run_path, audit_path, args.operator_audit / "run_manifest.json"]
    )
    kernel = Path(run["prepared_kernel"])
    metadata_path = kernel / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    if (
        run["status"] != "completed_requested_horizon"
        or run["time_method"] != "radau2"
        or not run["all_history_retained"]
        or run["source_changed_during_run"]
        or audit["status"] != "passed_original_causal_radau_equation_audit"
        or Path(audit["kernel"]).resolve() != kernel.resolve()
        or (not audit["complete_prepared_horizon_checked"] and not args.allow_prefix)
        or audit["plan"]["steps"] != run["completed_steps"]
        or audit["plan"]["dt"] != run["dt_s"]
        or audit["plan"]["fft_size"] != run["cq_fft_size"]
        or metadata["status"] != "completed_compact_radau_resolvent_kernel_only"
        or run["kernel_summary_sha256"] != fingerprint(metadata_path)
    ):
        raise ValueError("Require matching complete RK stages and a passed original causal audit")
    candidates = [r for r in audit["results"] if Path(r["run"]).resolve() == args.run.resolve()]
    if len(candidates) != 1:
        raise ValueError("The causal audit does not identify this trajectory uniquely")
    result = candidates[0]
    residual = result["maximum_original_equation_residual_mpa"]
    if (
        not result["all_saved_stages_checked"]
        or not np.isfinite(residual)
        or not 0 <= residual <= audit["original_equation_tolerance_mpa"]
    ):
        raise ValueError("Require a finite passed bound for every saved stage")
    for p in [run_path, metadata_path]:
        if audit["frozen_inputs_sha256"][str(p.resolve())] != fingerprint(p):
            raise ValueError("Original causal-audit run/kernel input changed")
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    for record in [run, metadata, audit]:
        if any(
            record["environment"][k] != environment[k]
            for k in ["source_sha256", "case_sha256", "restart_data_sha256"]
        ):
            raise ValueError("Physical case or bulk numerical package changed")
    d, steps = audit["plan"]["dofs"], audit["plan"]["steps"]
    n = (d - 1) // 2
    estimate = 8 * (steps + 1) * (12 * d + 14 * n + 16) + 512 * 1024**2
    if estimate > args.max_memory_mb * 1024**2:
        raise ValueError("Incremental work arrays exceed their explicit memory budget")
    selected = [
        Path(__file__),
        TOOLS / "radau_port_work.py",
        TOOLS / "radau_experiment.py",
        TOOLS / "replay_resolvent.py",
        run_path,
        audit_path,
        args.operator_audit / "run_manifest.json",
        metadata_path,
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in selected}
    output.mkdir(parents=True)
    write_json(
        output / "preflight.json",
        dict(
            status="preflight_only_not_a_work_result",
            estimated_array_mib=estimate / 1024**2,
            steps=steps,
            stage_dofs=2 * d,
            complete_prepared_horizon_checked=audit["complete_prepared_horizon_checked"],
            prefix_explicitly_allowed=args.allow_prefix,
            frozen_input_sha256=frozen,
        ),
    )
    if args.preflight_only:
        print(output / "preflight.json", flush=True)
        return
    stages_path = args.run / "stages.npz"
    stage_sha = DIGEST(stages_path)
    if manifest["primal_stage_sha256"][str(stages_path.resolve())] != stage_sha:
        raise ValueError("Stage values differ from those checked by the original causal audit")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    wc, width = model.constraints.contact_weights, model.constraints.platen_width
    line = np.r_[wc, wc, width]
    if line.shape != (d,):
        raise ValueError("Work observer requires the saved identity nodal contact grid")
    with np.load(stages_path, allow_pickle=False) as stages:
        q, response, actuator = (stages[k] for k in ["displacement", "response", "actuator"])
        if q.shape != (steps + 1, 2, d):
            raise ValueError("Incomplete stage trace")
        expected_time = np.zeros((steps + 1, 2))
        expected_time[1:] = (np.arange(steps)[:, None] + WORK["RADAU"]["C"]) * run["dt_s"]
        np.testing.assert_allclose(stages["stage_time"], expected_time, rtol=1e-12, atol=1e-15)
        force = WORK["equilibrium_bulk_force"](
            q,
            response,
            actuator,
            stages["initial_residual_force"],
            stages["initial_contact_force"],
            line,
            model.reference.spring_stiffness * width,
        )
        values, report = WORK["bulk_port_work"](
            force,
            q,
            run["dt_s"],
            line,
            residual_bound_mpa=residual,
            radius=audit["plan"]["radius"],
        )
    if DIGEST(stages_path) != stage_sha or any(fingerprint(p) != v for p, v in frozen.items()):
        raise ValueError("Frozen work-observer inputs changed")
    np.savez(output / "work_history.npz", **values, endpoint_time=expected_time[:, 1])
    write_json(
        output / "summary.json",
        dict(
            status="completed_incremental_radau_bulk_port_work_diagnostic",
            report=report,
            run=str(args.run.resolve()),
            original_equation_audit=str(args.operator_audit.resolve()),
            complete_prepared_horizon_checked=audit["complete_prepared_horizon_checked"],
            reconstruction="R=0 equilibrium force, with the saved original-RK residual propagated as a cumulative work-error bound",
            work_scope="Incremental elastic-body port work about the static lift; excludes static preload cross terms, contact spring storage and frictional dissipation",
            physical_energy_balance_reconstructed=False,
            global_passivity_proved=False,
            slip_weakening_stability_proved=False,
            full_original_event_reproduced=False,
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
            primal_stage_sha256=stage_sha,
            frozen_input_sha256=frozen,
            environment=environment,
            history_sha256=DIGEST(output / "work_history.npz"),
            saved_micro_friction_work_mpa_mm2=run["micro_friction_work_mpa_mm2"],
            saved_friction_work_is_not_added_to_incremental_bulk_work=True,
            estimated_array_mib=estimate / 1024**2,
            total_seconds=perf_counter() - begun,
        ),
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

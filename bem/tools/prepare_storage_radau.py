"""Build fresh audited Radau kernels with bounded durable storage stages.

Original raw sampling -> atomic ownership handoff -> in-place compaction ->
direct physical Fortran IFFT -> original stage/inverse/startup validation.
No legacy producer folder or foreign frequency spectrum is consumed.
"""

import argparse
import json
import os
import runpy
import shutil
import signal
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from finiteblock_bem.cq import CQPlan
from finiteblock_bem.cq_storage import spectrum_bytes
from finiteblock_bem.io import environment_record

TOOLS = Path(__file__).resolve().parent
TX = runpy.run_path(str(TOOLS / "inplace_radau_compaction.py"))
HANDOFF = runpy.run_path(str(TOOLS / "radau_stage_handoff.py"))
FINISH = runpy.run_path(str(TOOLS / "finish_storage_radau.py"))
DIGEST = TX["CHECKPOINT"]["digest_range"]


def parse_arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dt", type=float, default=2e-6)
    p.add_argument("--duration", type=float, default=0.045)
    for flag in ["h", "contact-h", "contact-end-h", "corner-h", "restart-knot-window"]:
        p.add_argument("--" + flag, type=float)
    p.add_argument("--restart-knot-subdivisions", type=int)
    p.add_argument("--clear-restart-knots", action="store_true")
    p.add_argument("--quadrature", type=int)
    p.add_argument("--degree", type=int, default=16)
    p.add_argument("--raw-rtol", type=float, default=1e-11)
    p.add_argument("--output-rtol", type=float, default=1e-9)
    p.add_argument("--workers", type=int, default=int(os.environ.get("FDBEM_CORES", "1")))
    p.add_argument("--fft-rows", type=int, default=2)
    p.add_argument("--combine-matrices", type=int, default=8)
    p.add_argument("--spectrum-buffer-matrices", type=int, default=64)
    p.add_argument("--startup-rows", type=int, default=8)
    p.add_argument("--primitive-partition", action="store_true")
    p.add_argument("--max-workspace-mb", type=float, default=10240)
    p.add_argument("--max-memory-mb", type=float, default=18432)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args()


def raw_command(args, output, *, resume=False, preflight=False):
    command = [
        sys.executable,
        "-B",
        str(TOOLS / "raw_radau_stage.py"),
        str(args.source.resolve()),
        "--output",
        str(output),
        "--durable-build",
    ]
    for key, value in vars(args).items():
        if key in {"source", "output", "resume", "preflight_only"} or value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append("--" + key.replace("_", "-"))
        else:
            command += ["--" + key.replace("_", "-"), str(value)]
    if resume:
        command.append("--resume")
    if preflight:
        command.append("--preflight-only")
    return command


def child_live(child):
    if child is None:
        return False
    r = subprocess.run(
        ["ps", "-p", str(child["pid"]), "-o", "lstart=,state=,command="],
        capture_output=True,
        text=True,
    )
    if r.returncode == 1 and not r.stdout.strip():
        return False
    if r.returncode:
        raise RuntimeError("Child observation failed; inspect before resuming")
    value = r.stdout.strip()
    return (
        value.startswith(child["start"])
        and "raw_radau_stage.py" in value
        and not value[len(child["start"]) :].strip().startswith("Z")
    )


def main():
    args = parse_arguments()
    begun = perf_counter()
    output = args.output.resolve()
    if output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()) or (
        args.resume and args.preflight_only
    ):
        raise ValueError("Use owned output outside protected Tatva; preflight is fresh-only")
    if not args.source.is_file():
        raise ValueError("Require case TOML or small case/mesh metadata")
    case = (
        args.source.resolve()
        if args.source.suffix.lower() == ".toml"
        else Path(json.loads(args.source.read_text())["source_mesh_arguments"]["case"])
    )
    environment = environment_record(case)
    tools = [
        TOOLS / n
        for n in [
            "prepare_storage_radau.py",
            "raw_radau_stage.py",
            "prepare_raw_radau_resolvent.py",
            "radau_stage_handoff.py",
            "finish_storage_radau.py",
            "inplace_radau_compaction.py",
            "direct_fortran_radau.py",
            "radau_sampling_checkpoint.py",
            "radau_raw_sampling.py",
            "raw_layer_sampling.py",
            "seeded_raw_sampling.py",
            "trace_schur_experiment.py",
            "schur_condensation_experiment.py",
            "radau_experiment.py",
            "compact_radau_storage.py",
            "prepare_native_resolvent.py",
            "replay_resolvent.py",
            "build_resolvent_history.py",
            "fortran_weight_storage.py",
            "bounded_weight_conversion.py",
            "buffered_spectrum.py",
        ]
    ]
    context = dict(
        arguments={
            k: str(v.resolve()) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k != "resume"
        },
        environment={
            k: environment[k]
            for k in ["source_sha256", "case_sha256", "restart_data_sha256", "numpy", "scipy"]
        },
        source_sha256={str(p): DIGEST(p) for p in [args.source.resolve(), *tools]},
    )
    journal_path = output / "storage-pipeline.json"
    if args.resume:
        if not journal_path.is_file():
            raise ValueError("Resume needs this storage producer's journal")
    else:
        if output.exists():
            raise ValueError("Choose a fresh storage producer output")
        output.mkdir(parents=True)
    lease = TX["CHECKPOINT"]["exclusive_lease"](output)
    state, attempt, process = None, None, None
    prior = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Storage producer received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)

    def publish():
        TX["publish"](journal_path, state)

    def verify():
        if any(DIGEST(p) != v for p, v in context["source_sha256"].items()):
            raise ValueError("Storage producer source/input changed")
        current = environment_record(case)
        if any(current[k] != v for k, v in context["environment"].items()):
            raise ValueError("Storage producer numerical environment changed")

    def execute(command, name):
        nonlocal process
        verify()
        log = output / f"{name}-attempt{len(state['attempts'])}.log"
        started = perf_counter()
        with log.open("x") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            anchor = subprocess.check_output(
                ["ps", "-p", str(process.pid), "-o", "lstart="], text=True
            ).strip()
            state["child"] = dict(pid=process.pid, start=anchor, command=command)
            publish()
            code = process.wait()
        process = None
        state["stages"].append(
            dict(name=name, seconds=perf_counter() - started, exit_code=code, log=str(log))
        )
        state["child"] = None
        publish()
        if code:
            raise RuntimeError(f"Raw stage failed ({code}); inspect {log}")

    try:
        state = (
            json.loads(journal_path.read_text())
            if args.resume
            else dict(
                schema_version=1,
                context=context,
                phase="preflight",
                attempts=[],
                stages=[],
                child=None,
            )
        )
        if state["schema_version"] != 1 or state["context"] != context:
            raise ValueError("Storage producer resume changes configuration/source/runtime")
        if child_live(state.get("child")):
            raise RuntimeError("The same raw child is still live; do not start another producer")
        if state.get("child") is not None:
            state["inherited_child_completion_unaccounted"] = True
        if state["phase"] not in {
            "preflight",
            "raw",
            "handoff",
            "compact",
            "weights",
            "completed",
            "preflight_only",
        }:
            raise ValueError("Unknown storage producer phase")
        if state["phase"] in {"completed", "preflight_only"}:
            raise ValueError("This completed/preflight-only producer needs no resume")
        attempt = dict(started_utc=datetime.now(UTC).isoformat(), status="running")
        state["attempts"].append(attempt)
        publish()
        if state["phase"] == "preflight":
            stage = output / "raw-preflight"
            # A successful saved preflight is reused; failed preflight files are preserved.
            if not (stage / "preflight.json").exists():
                if stage.exists():
                    raise ValueError("Inspect incomplete raw preflight before resuming")
                execute(raw_command(args, stage, preflight=True), "preflight")
            preflight = json.loads((stage / "preflight.json").read_text())
            plan = CQPlan(**preflight["plan"])
            compact_resources = TX["compaction_resources"](plan, args.combine_matrices)
            inversion = FINISH["DIRECT"]["inversion_resources"](plan, args.fft_rows)
            extra = compact_resources["maximum_extra_disk_bytes"] + 32 * plan.dofs**2 * 16
            extra += 8 * (plan.steps + 1) * plan.dofs * 8 + 128 * 1024**2
            required = 2 * spectrum_bytes(plan) + 2 * plan.weight_bytes + extra
            peak = max(
                preflight["estimated_array_mib"]["peak"] * 1024**2,
                compact_resources["estimated_workspace_bytes"],
                inversion["array_peak_bytes"] + extra,
            )
            if (
                required > shutil.disk_usage(output).free
                or peak > args.max_memory_mb * 1024**2
                or max(
                    compact_resources["estimated_workspace_bytes"], inversion["array_peak_bytes"]
                )
                > args.max_workspace_mb * 1024**2
            ):
                raise ValueError(
                    "Complete storage pipeline exceeds available disk or workspace budgets"
                )
            state["preflight"] = dict(
                plan=asdict(plan),
                required_disk_bytes=required,
                estimated_array_peak_bytes=peak,
                compaction=compact_resources,
                inversion=inversion,
            )
            state["raw_preflight_sha256"] = DIGEST(stage / "preflight.json")
            state["phase"] = "preflight_only" if args.preflight_only else "raw"
            publish()
        if state["phase"] == "raw":
            raw_folder = output / "raw-stage"
            if not (raw_folder / "summary.json").exists():
                execute(raw_command(args, raw_folder, resume=raw_folder.exists()), "raw")
            raw = json.loads((raw_folder / "summary.json").read_text())
            if (
                raw["status"] != "completed_independently_audited_raw_radau_stage"
                or raw["plan"] != state["preflight"]["plan"]
            ):
                raise ValueError("Complete independently audited raw stage is missing")
            for p, v in raw["frozen_inputs_sha256"].items():
                if DIGEST(p) != v:
                    raise ValueError("Audited raw-stage source/preflight changed")
            state["raw_summary_sha256"] = DIGEST(raw_folder / "summary.json")
            state["phase"] = "handoff"
            publish()
        if state["phase"] in {"handoff", "compact", "weights"}:
            raw_folder = output / "raw-stage"
            if DIGEST(raw_folder / "summary.json") != state["raw_summary_sha256"]:
                raise ValueError("Audited raw-stage receipt changed")
            raw = json.loads((raw_folder / "summary.json").read_text())
            plan = CQPlan(**raw["plan"])
        if state["phase"] == "handoff":
            handoff_path = output / "ownership-handoff.json"
            records, _ = HANDOFF["handoff_raw"](
                raw_folder,
                output / "compact-stage",
                raw["raw_records"],
                plan,
                handoff_path,
                resume=handoff_path.exists(),
            )
            state["staging_raw_records"] = records
            state["phase"] = "compact"
            publish()
        if state["phase"] == "compact":
            stage = output / "compact-stage"
            records, _ = TX["compact_inplace"](
                state["staging_raw_records"],
                plan,
                stage,
                batch_matrices=args.combine_matrices,
                max_workspace_mb=args.max_workspace_mb,
                resume=(stage / "inplace-compaction-journal.json").exists(),
            )
            state["compact_records"] = records
            state["phase"] = "weights"
            publish()
        if state["phase"] == "weights":
            result = FINISH["finish"](
                raw,
                state["compact_records"],
                output,
                fft_rows=args.fft_rows,
                startup_rows=args.startup_rows,
                max_workspace_mb=args.max_workspace_mb,
            )
            verify()
            elapsed = perf_counter() - begun
            known = sum(a.get("seconds", 0) for a in state["attempts"][:-1]) + elapsed
            accounted = all("seconds" in a for a in state["attempts"][:-1]) and not state.get(
                "inherited_child_completion_unaccounted", False
            )
            report = dict(
                status="completed_compact_radau_resolvent_kernel_only",
                time_method="radau2",
                assembly="fresh_raw_radau_before_trace_schur",
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                source_mesh_arguments=raw["source_mesh_arguments"],
                environment=raw["environment"],
                plan=raw["plan"],
                dt_s=plan.dt,
                full_steps=plan.steps,
                correction_steps=plan.steps,
                history_dofs=plan.dofs,
                stage_dofs=2 * plan.dofs,
                family_spectra=state["compact_records"],
                family_spectra_coordinates="diag(sqrt(w)) * L * diag(sqrt(w))",
                frequency_sampling=raw["frequency_sampling"],
                sampling_options=raw["sampling_options"],
                sampled_fresh_bem_mode_checks=raw["sampled_fresh_bem_mode_checks"],
                pre_combination_branch_digests=[
                    dict(
                        branch=i,
                        pre_combination_sha256=r["sha256"],
                        retained_separately_after_combination=False,
                    )
                    for i, r in enumerate(raw["raw_records"])
                ],
                timing_seconds=dict(
                    total_before_reporting=known if accounted else None,
                    known_main_interval_seconds=known,
                ),
                all_attempt_timings_accounted=accounted,
                timing_scope="Main intervals after parsing/module imports include nested stages. External command intervals are required for total costs; unfinished/orphaned attempts remain unaccounted",
                storage_pipeline=str(journal_path),
                original_tatva_readonly=True,
                full_original_event_reproduced=False,
                same_accuracy_fem_advantage_verified=False,
                **result,
            )
            TX["publish"](output / "summary.json", report)
            state["phase"] = "completed"
        attempt.update(status="completed", seconds=perf_counter() - begun)
        publish()
        print(
            output / ("summary.json" if state["phase"] == "completed" else "storage-pipeline.json"),
            flush=True,
        )
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
        if attempt is not None:
            attempt.update(
                status="interrupted_or_failed",
                seconds=perf_counter() - begun,
                error=f"{type(error).__name__}: {error}",
            )
            publish()
        raise
    finally:
        signal.signal(signal.SIGTERM, prior)
        lease.close()


if __name__ == "__main__":
    main()

"""Run pinned FEM with composed contact and float64 interior output observers.

Only an output callback is added through the existing AST-checked adapter.
Keep the original source manifest/summary intact and record this explicit
composition separately. Resume requires existing probe data through checkpoint;
missing past observations are never backfilled.
"""

import argparse
import hashlib
import json
import runpy
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time
from types import SimpleNamespace

TOOLS = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--probe-plan", type=Path, default=TOOLS.parent / "examples/event16_interior_probes.json"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("runner_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = args.output.resolve()
    plan_path = args.probe_plan.resolve()
    arguments = args.runner_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()) or not plan_path.is_file():
        raise ValueError("Choose a local output outside protected Tatva and an existing probe plan")
    if "--float64-contact-observations" not in arguments or any(
        a in ["--output", "--resume", "--prepare-only"] or a.startswith("--output=")
        for a in arguments
    ):
        raise ValueError(
            "Pass separate high-rate float64 contact output arguments; output/resume belong to this wrapper"
        )
    paths = [
        Path(__file__),
        TOOLS / "fem_interior_observations.py",
        TOOLS / "fem_contact_observations.py",
        TOOLS / "run_fem_reference.py",
        TOOLS / "run_checkpointed_fem_reference.py",
        plan_path,
    ]
    frozen = {str(p.resolve()): digest(p) for p in paths}
    manifest = dict(
        schema_version=1,
        output=str(output),
        probe_plan=json.loads(plan_path.read_text()),
        probe_plan_file_sha256=digest(plan_path),
        frozen_input_sha256=frozen,
        original_runner_arguments=arguments,
        integration_equations_changed=False,
        only_high_rate_output_observer_composed=True,
    )
    manifest_path = output / "interior_manifest.json"
    if args.resume:
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Interior resume source, plan or original arguments changed")
    else:
        if output.exists():
            raise ValueError("Choose a fresh FEM interior output directory")
        output.mkdir(parents=True)
        write_json(manifest_path, manifest)
    started = perf_counter()
    cpu_started = process_time()
    invocation = uuid.uuid4().hex
    record = dict(
        status="running",
        utc_start=datetime.now(UTC).isoformat(),
        resume=args.resume,
        interior_manifest_sha256=digest(manifest_path),
        full_original_event_reproduced=False,
        same_accuracy_fem_advantage_verified=False,
    )
    status = output / "interior_wrapper_status.json"
    ledger = output / "interior_invocations.jsonl"
    with ledger.open("a") as stream:
        stream.write(
            json.dumps(
                dict(event="started", id=invocation, resume=args.resume, utc=record["utc_start"])
            )
            + "\n"
        )
    original_run_path = runpy.run_path
    runner = original_run_path(str(TOOLS / "run_fem_reference.py"))
    adapters = {}

    def adapted(path, *positional, **kwargs):
        records = original_run_path(path, *positional, **kwargs)
        if Path(path).resolve() == (TOOLS / "fem_contact_observations.py").resolve():
            # This executes after the original runner configures CPU/JAX/NumPy.
            interior = original_run_path(str(TOOLS / "fem_interior_observations.py"))
            plan, plan_sha = interior["canonical_plan"](manifest["probe_plan"])
            adapters.update(interior=interior, plan_sha256=plan_sha)
            adapters["validate_original_clocks"] = original_run_path(
                str(TOOLS / "run_checkpointed_fem_reference.py")
            )["validate_clocks"]
            if args.resume:
                import h5py
                import numpy as np

                with np.load(output / "checkpoint.npz", allow_pickle=False) as saved:
                    checkpoint = json.loads(str(saved["metadata"]))
                with h5py.File(output / "simulation.h5", "r") as saved:
                    adapters["validate_original_clocks"](saved, checkpoint)
                    interior["validate_probe_group"](
                        saved, count=checkpoint["interface_frame_count"], plan_sha256=plan_sha
                    )
            baseline = records
            records = {
                **records,
                "make_observed_runner": lambda function: interior["make_observed_runner"](
                    function, plan, baseline
                ),
            }
        return records

    runner["main"].__globals__["runpy"] = SimpleNamespace(run_path=adapted)
    previous = sys.argv
    try:
        write_json(status, record)
        sys.argv = [str(TOOLS / "run_fem_reference.py"), "--output", str(output), *arguments]
        if args.resume:
            sys.argv.append("--resume")
        runner["main"]()
        import h5py

        base = json.loads((output / "summary.json").read_text())
        with h5py.File(output / "simulation.h5", "r") as saved:
            adapters["validate_original_clocks"](
                saved,
                dict(
                    dt=base["summary"]["dt"],
                    pressure_steps=base["summary"]["pressure_steps"],
                    frame_count=int(saved.attrs["saved_frames"]),
                    interface_frame_count=int(saved.attrs["saved_interface_frames"]),
                ),
            )
            record["probe_validation"] = adapters["interior"]["validate_probe_group"](
                saved, plan_sha256=adapters["plan_sha256"]
            )
        if any(digest(path) != value for path, value in frozen.items()):
            raise ValueError("Interior observer or source inputs changed")
        record.update(
            status="completed_original_fem_with_float64_interior_observations",
            original_source_and_input_unchanged=base["source_and_protected_input_unchanged"],
            original_full_horizon=base["full_original_horizon"],
            original_summary_sha256=digest(output / "summary.json"),
            actual_original_clock_and_contact_fields_read=True,
        )
    except BaseException as error:
        record.update(status="interrupted_or_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        sys.argv = previous
        record.update(
            wrapper_seconds_including_original_runner=perf_counter() - started,
            utc_update=datetime.now(UTC).isoformat(),
        )
        with ledger.open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        event="ended",
                        id=invocation,
                        status=record["status"],
                        seconds=record["wrapper_seconds_including_original_runner"],
                        cpu_seconds=process_time() - cpu_started,
                        utc=record["utc_update"],
                    )
                )
                + "\n"
            )
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        starts = [row["id"] for row in rows if row["event"] == "started"]
        ends = [row for row in rows if row["event"] == "ended"]
        accounted = len(starts) == len(set(starts)) == len(ends) and set(starts) == {
            row["id"] for row in ends
        }
        record["wrapper_invocation_timing"] = dict(
            all_invocations_accounted=accounted,
            invocations=len(starts),
            seconds=sum(row["seconds"] for row in ends) if accounted else None,
            cpu_seconds=sum(row["cpu_seconds"] for row in ends) if accounted else None,
            scope="Original runner intervals are nested inside these wrapper intervals; do not add both",
        )
        write_json(status, record)


if __name__ == "__main__":
    main()

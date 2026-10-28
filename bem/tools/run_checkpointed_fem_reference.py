"""Replay a damaged FEM prefix, then continue with paired output/checkpoint snapshots.

The pinned FEM runner and integration equations are unchanged. An explicit
output adapter adds an observer and a one-time checkpoint request, and wraps
the checkpoint writer. Capture occurs after both output writers and handoff.
The replay must reproduce all twelve saved state arrays bitwise before the
complete original horizon is allowed to continue.
"""

import argparse
import ast
import hashlib
import inspect
import json
import os
import runpy
import shutil
import sys
import textwrap
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

TOOLS = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_clocks(stream, metadata):
    """Read actual compressed data, including the time columns that were damaged."""
    dt, normal = metadata["dt"], metadata["pressure_steps"]
    for group, count in [
        (stream, metadata["frame_count"]),
        (stream["interface_high_rate"], metadata["interface_frame_count"]),
    ]:
        phases, steps = group["phase_id"][:count], group["step_id"][:count]
        absolute = steps.astype(np.int64) + np.where(phases == 2, normal, 0)
        history = group["history"][:count]
        if not np.all(np.isfinite(history)) or not np.array_equal(
            history[:, 0], (absolute * dt).astype(history.dtype)
        ):
            raise ValueError("Snapshot contains damaged or inconsistent scalar history")
    group = stream["contact_observations"]
    count = metadata["interface_frame_count"]
    for name in [
        "pressure",
        "elastic_shear",
        "plastic_slip",
        "cumulative_slip",
        "friction_coefficient",
        "friction_strength",
    ]:
        if not np.all(np.isfinite(group[name][:count])):
            raise ValueError("Snapshot contains invalid float64 contact observations")
    if not np.array_equal(
        group["time_s"][:count], group["absolute_step"][:count].astype(float) * dt
    ):
        raise ValueError("Snapshot observation clock changed")


def paired_snapshot(stream, checkpoint, root, metadata):
    """Copy while the sole HDF5 writer is synchronously inside its output callback."""
    root.mkdir(parents=True, exist_ok=True)
    stream.flush()
    temporary = root / (".pending-" + uuid.uuid4().hex)
    temporary.mkdir()
    shutil.copy2(stream.filename, temporary / "simulation.h5")
    shutil.copy2(checkpoint, temporary / "checkpoint.npz")
    with h5py.File(temporary / "simulation.h5", "r") as saved:
        validate_clocks(saved, metadata)
    with np.load(temporary / "checkpoint.npz", allow_pickle=False) as saved:
        if json.loads(str(saved["metadata"])) != metadata:
            raise ValueError("Snapshot checkpoint/output metadata differ")
    files = {}
    for name in ["simulation.h5", "checkpoint.npz"]:
        path = temporary / name
        with path.open("rb") as source:
            os.fsync(source.fileno())
        files[name] = dict(bytes=path.stat().st_size, sha256=digest(path))
    generation = "snapshot-" + uuid.uuid4().hex
    target = root / generation
    write_json(temporary / "record.json", dict(metadata=metadata, files=files))
    temporary.replace(target)
    previous = (
        json.loads((root / "latest.json").read_text()) if (root / "latest.json").is_file() else {}
    )
    generations = [*previous.get("generations", []), generation]
    write_json(root / "latest.json", dict(generations=generations[-2:], current=generation))
    for name in generations[:-2]:
        path = root / name
        if (
            name.startswith("snapshot-")
            and path.parent == root
            and (path / "record.json").is_file()
        ):
            shutil.rmtree(path)
    return dict(directory=str(target), files=files)


def compare_state(checkpoint, target):
    with (
        np.load(checkpoint, allow_pickle=False) as actual,
        np.load(target, allow_pickle=False) as expected,
    ):
        keys = sorted(k for k in expected.files if k.startswith("carry_"))
        if len(keys) != 12 or keys != sorted(k for k in actual.files if k.startswith("carry_")):
            raise ValueError("Require all twelve original FEM state arrays")
        checks = {
            key: actual[key].dtype == expected[key].dtype
            and actual[key].shape == expected[key].shape
            and np.array_equal(actual[key], expected[key])
            for key in keys
        }
    if not all(checks.values()):
        raise ValueError(f"Replayed checkpoint state differs: {checks}")
    return checks


def instrument_checkpoint_writer(function, observer, writer, force):
    """Change two output statements, restoring both to prove all other AST nodes match."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    original = ast.dump(tree, include_attributes=False)
    high = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "save_high_rate_interface"
    ]
    guard = ast.parse("checkpoint_path is not None", mode="eval").body
    matches = [
        n.test.values[1]
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.BoolOp)
        and isinstance(n.test.op, ast.And)
        and len(n.test.values) == 2
        and ast.dump(n.test.values[0], include_attributes=False)
        == ast.dump(guard, include_attributes=False)
        and isinstance(n.test.values[1], ast.BoolOp)
        and isinstance(n.test.values[1].op, ast.Or)
        and {v.id for v in n.test.values[1].values if isinstance(v, ast.Name)}
        == {"interval_due", "deadline_reached", "signal_due", "phase_complete"}
    ]
    if len(high) != 1 or len(matches) != 1:
        raise ValueError("Require the original single high-rate writer and checkpoint output guard")
    high[0].body.append(
        ast.parse(
            "_tatva_bem_contact_observer(h5, frame_idx, carry, phase_id, step_id, model, interface_plot_mask)"
        ).body[0]
    )
    matches[0].values.append(
        ast.parse("_tatva_bem_force_checkpoint(phase_id, stop)", mode="eval").body
    )
    request = matches[0].values.pop()
    observation = high[0].body.pop()
    if ast.dump(tree, include_attributes=False) != original:
        raise RuntimeError("Checkpoint adapter changed an integration statement")
    high[0].body.append(observation)
    matches[0].values.append(request)
    tree.body.insert(
        0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    )
    namespace = {
        **function.__globals__,
        "_tatva_bem_contact_observer": observer,
        "_write_simulation_checkpoint": writer,
        "_tatva_bem_force_checkpoint": force,
    }
    exec(
        compile(
            ast.fix_missing_locations(tree),
            function.__code__.co_filename + "|paired-checkpoint-output",
            "exec",
        ),
        namespace,
    )
    return namespace[function.__name__]


class CheckpointCapture:
    def __init__(self, output, target):
        self.output, self.target = output, target
        with np.load(target, allow_pickle=False) as saved:
            self.metadata = json.loads(str(saved["metadata"]))
        self.captured = False
        self.stream = None
        self.snapshots = []

    def bind(self, function, observer_records):
        observed = observer_records["make_observed_runner"](function)
        self.base_observer = observed.__globals__["_tatva_bem_contact_observer"]
        self.base_writer = function.__globals__["_write_simulation_checkpoint"]
        self.checkpoint_exception = function.__globals__["SimulationCheckpointed"]
        return instrument_checkpoint_writer(
            function, self.observe, self.write_checkpoint, self.force
        )

    def force(self, phase, step):
        return not self.captured and (phase, step) == (
            self.metadata["phase_id"],
            self.metadata["step_id"],
        )

    def write_checkpoint(self, path, carry, metadata):
        self.base_writer(path, carry, metadata)
        if self.stream is None:
            raise ValueError("No observed HDF5 stream for checkpoint snapshot")
        self.snapshots.append(
            paired_snapshot(self.stream, path, self.output / "paired-snapshots", metadata)
        )
        if self.force(metadata["phase_id"], metadata["step_id"]):
            if any(
                metadata.get(key) != value
                for key, value in self.metadata.items()
                if key != "saved_utc"
            ):
                raise ValueError("Replay discretization/output schedule differs from the target")
            self.captured = True
            raise self.checkpoint_exception(
                "Captured original checkpoint after all output and handoff"
            )

    def observe(self, h5, index, carry, phase, step, model, mask):
        self.base_observer(h5, index, carry, phase, step, model, mask)
        self.stream = h5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-checkpoint", type=Path, required=True)
    parser.add_argument("runner_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output, target = args.output.resolve(), args.replay_checkpoint.resolve()
    arguments = args.runner_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if (
        output.exists()
        or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve())
        or not target.is_file()
        or any(flag in arguments for flag in ["--output", "--resume", "--prepare-only"])
        or "--float64-contact-observations" not in arguments
        or "--checkpoint-interval" not in arguments
    ):
        raise ValueError(
            "Require a fresh output, original checkpoint and checkpointed f64 runner arguments"
        )
    output.mkdir(parents=True)
    paths = [
        Path(__file__),
        TOOLS / "run_fem_reference.py",
        TOOLS / "fem_contact_observations.py",
        target,
    ]
    sources = {str(path): digest(path) for path in paths}
    capture = CheckpointCapture(output, target)
    original_run_path = runpy.run_path
    runner = original_run_path(str(TOOLS / "run_fem_reference.py"))

    def adapted(path, *positional, **kwargs):
        records = original_run_path(path, *positional, **kwargs)
        if Path(path).resolve() == (TOOLS / "fem_contact_observations.py").resolve():
            baseline = records
            records = {
                **records,
                "make_observed_runner": lambda function: capture.bind(function, baseline),
            }
        return records

    # Scope the adapter to this runner's namespace; do not patch global runpy or Tatva modules.
    runner["main"].__globals__["runpy"] = SimpleNamespace(run_path=adapted)
    record = dict(
        phase="replaying_checkpoint_prefix",
        utc_start=datetime.now(UTC).isoformat(),
        source_sha256=sources,
        original_runner_arguments=arguments,
        integration_equations_changed=False,
        output_callbacks_composed=True,
        full_original_event_reproduced=False,
        same_accuracy_fem_advantage_verified=False,
    )
    previous_argv = sys.argv
    try:
        write_json(output / "checkpoint_wrapper.json", record)
        sys.argv = [str(TOOLS / "run_fem_reference.py"), "--output", str(output), *arguments]
        try:
            runner["main"]()
        except BaseException as error:
            if not capture.captured or not isinstance(error, capture.checkpoint_exception):
                raise
        if not capture.captured:
            raise ValueError("Full replay unexpectedly missed the target checkpoint")
        record["state_bitwise_checks"] = compare_state(output / "checkpoint.npz", target)
        record["phase"] = "continuing_verified_checkpoint"
        record["prefix_snapshot"] = capture.snapshots[-1]
        write_json(output / "checkpoint_wrapper.json", record)
        if any(digest(path) != expected for path, expected in sources.items()):
            raise ValueError("Checkpoint replay sources changed")
        sys.argv.append("--resume")
        runner["main"]()
        record["phase"] = "completed_original_fem_horizon_with_verified_prefix"
    except BaseException as error:
        record.update(phase="attention_required", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        sys.argv = previous_argv
        record["utc_update"] = datetime.now(UTC).isoformat()
        write_json(output / "checkpoint_wrapper.json", record)


if __name__ == "__main__":
    main()

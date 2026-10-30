"""Extend a verified FEM window to dt/8 and dt/16 with accepted-state resume.

Reuse the complete dt/4 result as the comparison, after a fresh bitwise prefix
replay through the same driver. Setup and step/scan come from the original
unchanged tools. The physical window, original stop rule and every fine scalar
step are retained; this is conditional time refinement, not full convergence.
"""

import argparse
import ast
import fcntl
import inspect
import json
import os
import runpy
import shutil
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

TOOLS = Path(__file__).resolve().parent
WINDOW = runpy.run_path(str(TOOLS / "fem_timestep_window.py"))
CHUNKS = runpy.run_path(str(TOOLS / "fem_refinement_chunks.py"))
DIGEST, WRITE = WINDOW["DIGEST"], WINDOW["CAPTURE"]["write_json"]
SETUP_SHA256 = "1f625c120c31c084677188088906adefa2d3994d9b249e1df4ebc641b7b67caa"


@contextmanager
def record_attempt(output, begun):
    path = output / ("attempt-" + uuid.uuid4().hex + ".json")
    record = dict(status="running", pid=os.getpid(), utc_start=datetime.now(UTC).isoformat())
    WRITE(path, record)
    try:
        yield
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    else:
        record["status"] = "completed"
    finally:
        record.update(
            utc_end=datetime.now(UTC).isoformat(), seconds_this_invocation=perf_counter() - begun
        )
        WRITE(path, record)


def setup_prefix(function, arguments):
    """Return the exact original driver's locals before its factor loop."""
    tree = ast.parse(inspect.getsource(function))
    target = tree.body[0]
    markers = [
        i
        for i, node in enumerate(target.body)
        if isinstance(node, ast.Assign)
        and ast.dump(node, include_attributes=False)
        == ast.dump(ast.parse("records = []").body[0], include_attributes=False)
    ]
    if len(markers) != 1:
        raise ValueError("Original setup/branch boundary changed")
    prefix = target.body[: markers[0]]
    target.body = prefix + ast.parse("return locals()").body

    class Parser:
        def __init__(self, **kwargs):
            pass

        def add_argument(self, *args, **kwargs):
            pass

        def parse_args(self):
            return arguments

    namespace = {
        **function.__globals__,
        "argparse": SimpleNamespace(ArgumentParser=Parser),
    }
    exec(compile(ast.fix_missing_locations(tree), function.__code__.co_filename, "exec"), namespace)
    return namespace[function.__name__]()


def schedule(model, first, last, factor):
    if not isinstance(factor, int) or factor < 1 or factor & (factor - 1) or not 1 <= first < last:
        raise ValueError("Require a positive power-of-two factor and ordered interval")
    columns = []
    for name in WINDOW["SCHEDULES"]:
        values = np.asarray(model[name][first - 1 : last])
        if values.ndim != 1 or len(values) != last - first + 1 or not np.all(np.isfinite(values)):
            raise ValueError("Original endpoint loading is incomplete or nonfinite")
        left = np.repeat(values[:-1], factor)
        slope = np.repeat(np.diff(values), factor)
        fraction = np.tile(np.arange(1, factor + 1) / factor, last - first)
        refined = left + slope * fraction
        refined[factor - 1 :: factor] = values[1:]
        columns.append(refined)
    return np.column_stack(columns)


def replay_control(context, parent, steps):
    """Check every f64 scalar and every shared contact field of a dt/4 prefix."""
    import jax
    import jax.numpy as jnp

    first, model, base = (context[k] for k in ["first", "model", "base"])
    last = min(first + steps, context["last"])
    observed = [int(v) for v in context["saved_steps"] if v <= last]
    stops = sorted(set([*observed, last]))
    local = WINDOW["integrator_prefix"](
        context["dynamics"].run_simulation_dumped, lambda *a: {**model, "dt": context["old_dt"] / 4}
    )(context["case"], context["config"], context["output"] / "unused.h5")
    state = WINDOW["rephase_half_velocity"](
        context["common"], context["kick"], context["old_dt"], context["old_dt"] / 4
    )
    previous, rows, fields = first, [], {k: [] for k in WINDOW["FIELDS"]}
    for stop in stops:
        state, values = local["advance_chunk"](
            state,
            jnp.asarray(schedule(model, previous, stop, 4)),
            base["shear_phase_zero_dofs"],
            base["prescribed_dofs"],
            "shear",
        )
        jax.block_until_ready(state)
        rows.append(np.asarray(values))
        if stop in observed:
            actual = WINDOW["contact_fields"](state, model)
            for name in fields:
                fields[name].append(actual[name])
        previous = stop
    with np.load(parent / "factor4.npz", allow_pickle=False) as data:
        scalar_equal = np.array_equal(
            np.concatenate(rows), data["all_scalar_history"][: 4 * (last - first)]
        )
        ids = np.flatnonzero(np.isin(data["step_id"], observed))
        field_equal = {k: np.array_equal(np.asarray(v), data[k][ids]) for k, v in fields.items()}
    if not scalar_equal or not all(field_equal.values()) or len(observed) < 2:
        raise ValueError("Fresh dt/4 prefix differs from the verified original window")
    return dict(
        original_steps=last - first,
        every_float64_scalar_row_bitwise=scalar_equal,
        contact_fields_bitwise=field_equal,
        shared_observations=len(observed),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factors", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--original-steps", type=int)
    parser.add_argument("--control-steps", type=int, default=6000)
    parser.add_argument("--max-output-mb", type=float, default=4096)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    begun, output, parent = perf_counter(), args.output.resolve(), args.parent.resolve()
    if (
        output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve())
        or output.is_relative_to(parent)
        or (output.exists() and not args.resume)
        or (not output.exists() and args.resume)
        or args.factors != sorted(set(args.factors))
        or not args.factors
        or any(f not in (8, 16) for f in args.factors)
        or args.control_steps < 6000
        or not np.isfinite(args.max_output_mb)
        or args.max_output_mb <= 0
        or DIGEST(TOOLS / "fem_timestep_window.py") != SETUP_SHA256
    ):
        raise ValueError("Require fresh/resumable owned output and the pinned original setup")
    original = json.loads((parent / "summary.json").read_text())
    manifest = json.loads((parent / "manifest.json").read_text())
    if (
        original["status"] != "completed_local_fem_timestep_window"
        or [r["factor"] for r in original["records"]] != [1, 2, 4]
        or not original["all_source_inputs_unchanged"]
        or not all(original["records"][0]["original_contact_bitwise_checks"].values())
        or any(DIGEST(p) != v for p, v in manifest["frozen_inputs_sha256"].items())
    ):
        raise ValueError("Require unchanged complete parent and source identity")
    count = original["last_shear_step"] - original["first_shear_step"]
    count = count if args.original_steps is None else args.original_steps
    if not 6000 <= count <= original["last_shear_step"] - original["first_shear_step"]:
        raise ValueError("Keep the original physical window, or choose an explicit shorter pilot")
    frozen = dict(manifest["frozen_inputs_sha256"])
    for path in [
        Path(__file__),
        TOOLS / "fem_refinement_chunks.py",
        parent / "summary.json",
        parent / "manifest.json",
        parent / "factor4.npz",
    ]:
        frozen[str(path.resolve())] = DIGEST(path)
    output.mkdir(exist_ok=args.resume)
    with (output / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with record_attempt(output, begun):
            run_extension(args, manifest, count, frozen, output, parent, begun)


def run_extension(args, manifest, count, frozen, output, parent, begun):
    setup_output = output / ("setup-" + uuid.uuid4().hex)
    files = manifest["frozen_inputs_sha256"]
    fem = next(Path(p).parent for p in files if p.endswith("reference_manifest.json"))
    checkpoint = next(Path(p) for p in files if p.endswith("checkpoint.npz"))
    context = setup_prefix(
        WINDOW["main"],
        SimpleNamespace(
            fem=fem,
            checkpoint=checkpoint,
            output=setup_output,
            original_steps=count,
            factors=[1, 2, 4],
        ),
    )
    context_sha = DIGEST(setup_output / "manifest.json")
    import jax
    import jax.numpy as jnp

    from finiteblock_bem.io import environment_record

    runtime = environment_record()
    specification = dict(
        schema=1,
        context_sha256=context_sha,
        frozen_inputs_sha256=frozen,
        runtime=runtime,
        factors=args.factors,
        original_steps=count,
        control_steps=args.control_steps,
        max_output_mb=args.max_output_mb,
        first_step=context["first"],
        last_step=context["last"],
    )
    if args.resume:
        if json.loads((output / "extension_manifest.json").read_text()) != specification:
            raise ValueError("FEM extension numerical source, runtime or window changed")
    else:
        WRITE(output / "extension_manifest.json", specification)
    carry_bytes = sum(np.asarray(a).nbytes for a in context["common"])
    logical_bytes = sum(
        2 * count * factor * 17 * 8 + (len(context["stops"]) + 3) * carry_bytes
        for factor in args.factors
    )
    logical_bytes += args.control_steps * 4 * 17 * 8 + 64 * 1024**2
    if (
        logical_bytes > args.max_output_mb * 1024**2
        or shutil.disk_usage(output).free < logical_bytes + 1024**3
    ):
        raise ValueError("FEM extension exceeds storage budget or free-space reserve")
    WRITE(output / "resource-preflight.json", dict(logical_array_bytes=logical_bytes))
    control = replay_control(context, parent, args.control_steps)
    WRITE(output / ("replay-control-" + uuid.uuid4().hex + ".json"), control)
    records = []
    for factor in args.factors:
        start = perf_counter()
        model = context["model"]
        state = WINDOW["rephase_half_velocity"](
            context["common"], context["kick"], context["old_dt"], context["old_dt"] / factor
        )
        store = CHUNKS["WindowChunks"](
            output / f"factor{factor}-chunks",
            DIGEST(output / "extension_manifest.json"),
            factor,
            context["first"],
            context["stops"],
            context["saved_steps"],
            state,
            WINDOW["contact_fields"](state, model),
        )
        restored = store.restore()
        if restored is not None:
            state = tuple(jnp.asarray(a) for a in restored)
        resumed_chunks = len(store.paths)
        local = WINDOW["integrator_prefix"](
            context["dynamics"].run_simulation_dumped,
            lambda *a: {**model, "dt": context["old_dt"] / factor},
        )(context["case"], context["config"], setup_output / "unused.h5")
        previous = context["first"] if not store.paths else store.stops[len(store.paths) - 1]
        for stop in store.stops[len(store.paths) :]:
            if any(DIGEST(p) != v for p, v in frozen.items()):
                raise ValueError("Frozen FEM extension input changed")
            state, values = local["advance_chunk"](
                state,
                jnp.asarray(schedule(model, previous, stop, factor)),
                local["shear_phase_zero_dofs"],
                local["prescribed_dofs"],
                "shear",
            )
            jax.block_until_ready(state)
            fields = WINDOW["contact_fields"](state, model) if stop in store.observed else {}
            store.append(state, np.asarray(values), fields)
            previous = stop
            WRITE(
                output / "progress.json",
                dict(
                    factor=factor,
                    committed_chunks=len(store.paths),
                    total_chunks=len(store.stops),
                    original_step=stop,
                    requested_end_step=context["last"],
                    elapsed_seconds_this_invocation=perf_counter() - begun,
                ),
            )
        final = output / f"factor{factor}.npz"
        receipt = output / f"factor{factor}-receipt.json"
        if not final.exists():
            CHUNKS["atomic_npz"](final, store.combine(context["old_dt"], context["y"]))
        elif not receipt.exists():
            expected = store.combine(context["old_dt"], context["y"])
            with np.load(final, allow_pickle=False) as data:
                if set(data.files) != set(expected) or any(
                    not np.array_equal(data[k], v) for k, v in expected.items()
                ):
                    raise ValueError("Existing final export differs from committed chunks")
        final_proof = dict(output_sha256=DIGEST(final), chunk_sha256=store.hashes)
        if receipt.exists():
            if json.loads(receipt.read_text()) != final_proof:
                raise ValueError("Existing final export or committed chunk chain changed")
        else:
            WRITE(receipt, final_proof)
        records.append(
            dict(
                factor=factor,
                dt_s=context["old_dt"] / factor,
                seconds_this_invocation=perf_counter() - start,
                resumed_chunks=resumed_chunks,
                chunks=len(store.paths),
                output_sha256=DIGEST(final),
                maximum_cumulative_slip_mm=float(np.max(np.asarray(state[3]))),
            )
        )
        WRITE(output / "branches.json", dict(records=records))
    if any(DIGEST(p) != v for p, v in frozen.items()) or environment_record() != runtime:
        raise ValueError("FEM extension input or runtime changed")
    WRITE(
        output / "summary.json",
        dict(
            status="completed_conditional_fem_window_extension",
            records=records,
            first_shear_step=context["first"],
            last_shear_step=context["last"],
            physical_shear_interval_s=[
                context["first"] * context["old_dt"],
                context["last"] * context["old_dt"],
            ],
            replay_control=control,
            every_fine_scalar_and_accepted_state_retained=True,
            all_source_inputs_unchanged=True,
            full_event_or_spatial_convergence_verified=False,
            seconds_this_invocation=perf_counter() - begun,
        ),
    )


if __name__ == "__main__":
    main()

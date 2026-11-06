"""Original full Radau simulation with verified initialization and root recovery.

An optional validated checkpoint restores every future history term before
the original time loop. This CLI is primal-only; full initialized derivatives
must use the original residual and full prefix/history sensitivities.
"""

import argparse
import json
import runpy
import sys
from functools import wraps
from pathlib import Path
from time import perf_counter
from types import FunctionType, SimpleNamespace

import numpy as np

from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
STREAM = runpy.run_path(str(TOOLS / "stream_initial_residual.py"))
RESUME = runpy.run_path(str(TOOLS / "radau_prefix_resume.py"))
RECOVER = runpy.run_path(str(TOOLS / "recovered_radau_equilibrium.py"))
PREFIX = runpy.run_path(str(TOOLS / "simulate_prefix_cached_radau.py"))


def rebind(function, values):
    clone = FunctionType(
        function.__code__,
        {**function.__globals__, **values},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--affine-cache", type=Path, required=True)
    parser.add_argument("--initial-lift", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--recovered-root", type=Path)
    parser.add_argument("--root-key", default="random_hybrid")
    parser.add_argument(
        "--recovery-random-starts",
        type=int,
        default=32,
        help="Maximum random condensed recovery guesses, after three deterministic guesses",
    )
    parser.add_argument("--restart-seed", type=Path)
    parser.add_argument("--segment-substeps", type=int, nargs=2, default=[64, 128])
    parser.add_argument(
        "--static-initialization", choices=["fem-body-active-set"], default="fem-body-active-set"
    )
    parser.add_argument("--max-memory-mb", type=float, default=14336)
    parser.add_argument("--history-prefix-cache-mb", type=float, default=144)
    parser.add_argument("--history-prefix-lags", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    if args.recovery_random_starts < 0:
        raise ValueError("Require a nonnegative random recovery budget")
    output, kernel, cache = (
        args.output.resolve(),
        args.kernel.resolve(),
        args.affine_cache.resolve(),
    )
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh owned simulation output")
    if min(args.segment_substeps) < 1 or args.history_prefix_lags < 1:
        raise ValueError("Require positive stage substep and cached-lag counts")
    metadata_path = kernel / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    cache_meta = json.loads((cache / "summary.json").read_text())
    if cache_meta["status"] != "completed_original_formula_column_streamed_primal_affine_cache":
        raise ValueError("Require the complete original affine initialization cache")
    if cache_meta["source_sha256"].get(str(metadata_path)) != fingerprint(metadata_path):
        raise ValueError("Affine cache belongs to a different complete kernel")
    if any(fingerprint(p) != h for p, h in cache_meta["source_sha256"].items()):
        raise ValueError("Affine preparation source or original physical input changed")
    if fingerprint(args.initial_lift) != cache_meta["initial_lift_sha256"]:
        raise ValueError("Saved lift differs from the affine preparation")
    seed = (
        Path(cache_meta["seed"]["path"])
        if args.restart_seed is None
        else args.restart_seed.resolve()
    )
    if fingerprint(seed) != cache_meta["seed"]["sha256"]:
        raise ValueError("Restart seed differs from the prepared initial state")
    if (
        STREAM["digest_file"](cache / "affine.npy") != cache_meta["affine_sha256"]
        or fingerprint(cache / "initial_residual_force.npy")
        != cache_meta["initial_residual_force_sha256"]
    ):
        raise ValueError("Prepared complete initialization history changed")
    affine = np.load(cache / "affine.npy", mmap_mode="r", allow_pickle=False)
    initial_force = np.load(cache / "initial_residual_force.npy", allow_pickle=False)
    snapshot = None
    capture_reserve = 0
    if args.checkpoint is not None:
        snapshot = RESUME["PrefixSnapshot"].load(
            args.checkpoint, root_path=args.recovered_root, root_key=args.root_key
        )
        if (
            snapshot.record["kernel"] != str(kernel)
            or snapshot.record["affine_sha256"] != cache_meta["affine_sha256"]
        ):
            raise ValueError("Checkpoint, kernel and affine initialization do not align")
        failure = json.loads(Path(snapshot.record["capture"]).with_name("failure.json").read_text())
        if failure["segment_substeps"] != args.segment_substeps:
            raise ValueError("Checkpoint constitutive substeps cannot be changed on resume")
        capture_reserve = sum(v.nbytes for v in snapshot.capture.values()) + sum(
            v.nbytes for v in snapshot.payload.values()
        )
    elif args.recovered_root is not None:
        raise ValueError("A recovered boundary root requires its validated checkpoint")
    capacity, remaining = PREFIX["reserved_budget"](
        args.max_memory_mb, args.history_prefix_cache_mb
    )
    remaining -= capture_reserve / 1024**2
    if remaining <= 0:
        raise ValueError("No array budget remains after checkpoint and prefix reservations")
    sources = {
        str(p.resolve()): fingerprint(p)
        for p in [
            Path(__file__),
            TOOLS / "radau_prefix_resume.py",
            TOOLS / "recovered_radau_equilibrium.py",
            TOOLS / "identity_stage_tangent.py",
            TOOLS / "diagnose_condensed_radau_root.py",
            TOOLS / "diagnose_radau_root.py",
            TOOLS / "stream_initial_residual.py",
            TOOLS / "simulate_prefix_cached_radau.py",
            TOOLS / "history_prefix_cache.py",
            cache / "summary.json",
            args.initial_lift.resolve(),
        ]
    }
    record = dict(
        source_sha256=sources,
        kernel=str(kernel),
        affine_cache=str(cache),
        checkpoint=None if args.checkpoint is None else str(args.checkpoint.resolve()),
        root_seed=None
        if args.recovered_root is None
        else dict(path=str(args.recovered_root.resolve()), sha256=fingerprint(args.recovered_root)),
        total_array_budget_mb=args.max_memory_mb,
        prefix_capacity_bytes=capacity,
        checkpoint_reserved_bytes=capture_reserve,
        original_planner_memory_mb=remaining,
        recovery_random_starts=args.recovery_random_starts,
        primal_only=True,
        full_original_goal_achieved=False,
        recoveries=[],
    )
    begun = perf_counter()
    original = runpy.run_path(str(TOOLS / "simulate_compact_radau.py"))
    initialization = runpy.run_path(str(TOOLS / "radau_initialization.py"))

    def initialize(model, *, quadrature):
        if quadrature != metadata["source_mesh_arguments"]["quadrature"]:
            raise ValueError("Initialization quadrature differs from the saved body lift")
        _loaded, lift, reference, force, _seed = STREAM["load_lifted_reference"](
            metadata, args.initial_lift, seed
        )
        np.testing.assert_array_equal(model.restart.plastic_slip, lift.state.plastic_slip)
        np.testing.assert_array_equal(model.restart.cumulative_slip, lift.state.cumulative_slip)
        np.testing.assert_array_equal(force, initial_force)
        lift.diagnostics = dict(
            status="reused_verified_original_body_lift",
            original_lift_sha256=cache_meta["initial_lift_sha256"],
            original_static_reassembly_performed=False,
        )
        return lift, reference

    def initial_history(first, second, force, steps, *, rows=8):
        np.testing.assert_array_equal(force, initial_force)
        if (
            first.shape != second.shape
            or first.shape[1:] != (len(force), len(force))
            or not 1 <= steps < len(affine)
        ):
            raise ValueError("Cached initialization does not cover the exact original grid")
        return affine[: steps + 1]

    def run_path(path, *a, **k):
        if Path(path).resolve() == (TOOLS / "radau_initialization.py").resolve():
            return {
                **initialization,
                "initialize_radau_body": initialize,
                "initial_residual_past": initial_history,
            }
        return runpy.run_path(path, *a, **k)

    def recovery_report(detail):
        record["recoveries"].append(detail)
        write_json(output / "recovered-radau-policy.json", record)

    simulate = rebind(
        original["simulate_stages"],
        dict(
            runpy=SimpleNamespace(run_path=run_path),
            CONTACT={
                **original["CONTACT"],
                "RadauContactEquilibrium": lambda *a, **k: RECOVER["recovering_stage_equilibrium"](
                    *a, **k, max_starts=args.recovery_random_starts, on_recovery=recovery_report
                ),
            },
        ),
    )
    if snapshot is not None:
        simulate = RESUME["resumable_simulator"](simulate, snapshot)
    numerical = simulate

    @wraps(numerical)
    def simulate_and_record(*a, **k):
        result, stages = numerical(*a, **k)
        result.diagnostics["recovered_radau_policy"] = dict(
            checkpoint_applied=snapshot is not None,
            prefix_rows=0 if snapshot is None else snapshot.record["next_step"],
            unavailable_historical_newton_counts=-1,
            original_physical_step_loop_retained=True,
            full_initialization_derivatives_not_validated_by_primal_cache=True,
        )
        return result, stages

    def on_cache(cache_record):
        record["history_prefix_cache"] = cache_record
        write_json(output / "recovered-radau-policy.json", record)

    simulate = PREFIX["prefix_simulator"](
        simulate_and_record,
        capacity_bytes=capacity,
        max_lags=args.history_prefix_lags,
        on_cache=on_cache,
    )
    execute = rebind(
        original["main"], dict(runpy=SimpleNamespace(run_path=run_path), simulate_stages=simulate)
    )
    previous = sys.argv
    try:
        sys.argv = [
            previous[0],
            str(kernel),
            *forwarded,
            "--output",
            str(output),
            "--max-memory-mb",
            str(remaining),
            "--restart-seed",
            str(seed),
            "--static-initialization",
            args.static_initialization,
            "--segment-substeps",
            *[str(v) for v in args.segment_substeps],
        ]
        execute()
        if snapshot is not None and not snapshot.applied:
            raise ValueError(
                "Original physical simulation did not consume the validated checkpoint"
            )
        if any(fingerprint(p) != h for p, h in sources.items()):
            raise ValueError("Recovered simulation source changed")
        record.update(
            status="completed_original_physical_loop_with_recovered_roots",
            seconds=perf_counter() - begun,
            checkpoint_applied=snapshot is not None,
            boundary_forcing_error_mpa=None
            if snapshot is None
            else snapshot.boundary_forcing_error_mpa,
        )
        write_json(output / "recovered-radau-policy.json", record)
    except BaseException as error:
        if output.exists():
            write_json(
                output / "recovered-radau-failure.json",
                dict(error=repr(error), seconds=perf_counter() - begun, policy=record),
            )
        raise
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()

"""Optional exact prefix cache around the original compact Radau simulation.

All original CLI physics/initialization options are forwarded unchanged.
The prefix budget is reserved from --max-memory-mb before the original planner
runs. Existing kernels and production simulations are never rewritten.
"""

import argparse
import inspect
import math
import runpy
import sys
from functools import wraps
from pathlib import Path
from time import perf_counter
from types import FunctionType

from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
CACHE = runpy.run_path(str(TOOLS / "history_prefix_cache.py"))


def reserved_budget(total_mb, prefix_mb):
    if not all(math.isfinite(v) for v in [total_mb, prefix_mb]) or not 0 <= prefix_mb < total_mb:
        raise ValueError("Require finite 0 <= prefix cache < total memory budget")
    capacity = int(prefix_mb * 1024**2)
    return capacity, total_mb - capacity / 1024**2


def prefix_simulator(original, *, capacity_bytes, max_lags=32, on_cache=None):
    """Bind only the two history arguments; keep the original numerical code."""
    signature = inspect.signature(original)
    if not {"first", "second"}.issubset(signature.parameters):
        raise ValueError("Original simulation history signature changed")

    @wraps(original)
    def simulate(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        begun = perf_counter()
        histories, record = CACHE["cache_history_prefixes"](
            (bound.arguments["first"], bound.arguments["second"]),
            capacity_bytes=capacity_bytes,
            max_lags=max_lags,
        )
        record["cache_setup_seconds"] = perf_counter() - begun
        bound.arguments.update(first=histories[0], second=histories[1])
        if on_cache is not None:
            on_cache(record)
        result, stages = original(*bound.args, **bound.kwargs)
        result.diagnostics["history_prefix_cache"] = record
        return result, stages

    return simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--history-prefix-cache-mb", type=float, default=144)
    parser.add_argument("--history-prefix-lags", type=int, default=32)
    parser.add_argument("--max-memory-mb", type=float, default=8192)
    parser.add_argument("--output", type=Path)
    args, forwarded = parser.parse_known_args()
    original = runpy.run_path(str(TOOLS / "simulate_compact_radau.py"))
    if "--help" in forwarded or "-h" in forwarded:
        parser.print_help()
        original["main"]()
        return
    if args.output is None or args.history_prefix_lags < 1:
        raise ValueError("Require --output and a positive history-prefix-lags count")
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh output outside the original Tatva directory")
    capacity, remaining_mb = reserved_budget(args.max_memory_mb, args.history_prefix_cache_mb)
    sources = {
        str(p.resolve()): fingerprint(p)
        for p in [Path(__file__), TOOLS / "history_prefix_cache.py"]
    }
    record = dict(
        requested_total_memory_mb=args.max_memory_mb,
        reserved_prefix_capacity_bytes=capacity,
        original_planner_memory_mb=remaining_mb,
        source_sha256=sources,
        original_command=list(sys.argv),
        original_kernel_not_modified=True,
        original_numerical_functions_unchanged=True,
        full_original_goal_achieved=False,
    )

    def verify():
        if any(fingerprint(p) != digest for p, digest in sources.items()):
            raise ValueError("Prefix-cache policy source changed during simulation")

    def on_cache(policy):
        if "cache" in record:
            raise ValueError("Expected one complete simulation per CLI invocation")
        verify()
        record["cache"] = policy
        write_json(output / "history-prefix-policy.json", record)

    original_main = original["main"]
    simulate = prefix_simulator(
        original["simulate_stages"],
        capacity_bytes=capacity,
        max_lags=args.history_prefix_lags,
        on_cache=on_cache,
    )
    execute = FunctionType(
        original_main.__code__,
        {**original_main.__globals__, "simulate_stages": simulate},
        original_main.__name__,
        original_main.__defaults__,
        original_main.__closure__,
    )
    execute.__kwdefaults__ = original_main.__kwdefaults__
    previous = sys.argv
    begun = perf_counter()
    try:
        sys.argv = [
            previous[0],
            *forwarded,
            "--output",
            str(output),
            "--max-memory-mb",
            str(remaining_mb),
        ]
        execute()
        verify()
        if "cache" not in record:
            raise ValueError("Original simulation did not consume the prefix policy")
        record.update(
            status="completed_original_simulation_with_prefix_cache", seconds=perf_counter() - begun
        )
        write_json(output / "history-prefix-policy.json", record)
    except BaseException as error:
        if output.is_dir():
            write_json(
                output / "history-prefix-failure.json",
                dict(error=repr(error), seconds=perf_counter() - begun, policy=record),
            )
        raise
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()

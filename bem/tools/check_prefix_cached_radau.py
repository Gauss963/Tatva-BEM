"""Use one saved coarse kernel for original/cached full sliding and initialized AD.

Reuses the existing complete 128-step accelerated diagnostic. Both variants use
the same saved weights and original static/root/derivative functions. No new
contour assembly, full event equivalence, or runtime advantage is asserted.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter
from types import FunctionType

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import BoundaryHistory
from finiteblock_bem.io import environment_record, fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
MIXED = runpy.run_path(str(TOOLS / "check_mixed_radau_nonlinear.py"))
CACHE = runpy.run_path(str(TOOLS / "history_prefix_cache.py"))
CLI = runpy.run_path(str(TOOLS / "simulate_prefix_cached_radau.py"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-prefix-cache-mb", type=float, default=144)
    args = parser.parse_args()
    kernel, output = args.kernel.resolve(), args.output.resolve()
    capacity, _ = CLI["reserved_budget"](8192, args.history_prefix_cache_mb)
    metadata = json.loads((kernel / "summary.json").read_text())
    mesh = metadata["source_mesh_arguments"]
    if (
        metadata["full_steps"] != 128
        or metadata["correction_steps"] != 128
        or metadata["plan"]["dt"] != 2e-6
        or metadata["plan"]["dofs"] > 64
        or any(mesh[k] != 160 for k in ["h", "contact_h", "contact_end_h", "corner_h"])
        or mesh["diagnostic_actuator_increment"] != 0.5
    ):
        raise ValueError("Use only the saved complete 128-step coarse diagnostic kernel")
    if (
        output.exists()
        or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve())
        or output.is_relative_to(kernel)
    ):
        raise ValueError("Require a fresh owned output outside the read-only kernel")
    import jax

    jax.config.update("jax_enable_x64", True)
    begun = perf_counter()
    case, seed = MIXED["CONTROL"]["CASE"], MIXED["CONTROL"]["SEED"]
    environment = environment_record(case)
    frozen = {
        str(p.resolve()): fingerprint(p)
        for p in [
            *TOOLS.glob("*.py"),
            case,
            seed,
            *[p for p in kernel.iterdir() if p.is_file()],
        ]
    }
    output.mkdir(parents=True)
    write_json(
        output / "manifest.json",
        dict(
            kernel=str(kernel),
            frozen_inputs_sha256=frozen,
            environment=environment,
            same_saved_kernel_for_both_variants=True,
            labels=dict(original="original layout", mixed="exact prefix cache"),
            prefix_capacity_bytes=capacity,
            diagnostic_actuator_rate_mm_per_s=7812.5,
            original_actuator_rate_mm_per_s=20,
        ),
    )
    calls, records = 0, []

    def history(plan, weights):
        nonlocal calls
        if calls >= 4:
            raise ValueError("Original two-family/two-variant control boundary changed")
        readonly = weights.view()
        readonly.flags.writeable = False
        result = BoundaryHistory(plan, readonly)
        if calls >= 2:
            (result,), record = CACHE["cache_history_prefixes"](
                [result], capacity_bytes=capacity // 2, max_lags=32
            )
            records.append(record)
        calls += 1
        return result

    try:
        inputs = MIXED["CachedInputs"](kernel, kernel, environment)
        original = MIXED["cached_experiment"](inputs)
        experiment = FunctionType(
            original.__code__,
            {**original.__globals__, "BoundaryHistory": history},
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        experiment.__kwdefaults__ = original.__kwdefaults__
        with threadpool_limits(limits=1, user_api="blas"):
            result = experiment(output, frozen, environment, begun)
        if (
            calls != 4
            or len(records) != 2
            or any(r["cached_lags"] != 32 for r in records)
            or sum(r["cached_bytes"] for r in records) > capacity
        ):
            raise ValueError("Complete pair cache or explicit memory accounting changed")
        if (
            any(fingerprint(p) != digest for p, digest in frozen.items())
            or environment_record(case) != environment
        ):
            raise ValueError("Original/cached inputs, numerical source or runtime changed")
        with (
            np.load(output / "original-arrays.npz") as first,
            np.load(output / "mixed-arrays.npz") as second,
        ):
            equality = {key: bool(np.array_equal(first[key], second[key])) for key in first.files}
            max_abs = {key: float(np.max(abs(first[key] - second[key]))) for key in first.files}
        result.update(
            status="passed_complete_coarse_prefix_cache_sliding_and_initialized_ad",
            cache_records=records,
            same_saved_kernel_for_both_variants=True,
            array_bitwise_equality=equality,
            maximum_absolute_array_differences=max_abs,
            policy_sources_and_saved_kernel_unchanged=True,
            total_seconds_including_controls=perf_counter() - begun,
            full_original_goal_achieved=False,
            scope="Complete 128-step coarse accelerated sliding; same saved fixed-bulk kernel, selected healed initialization and recorded loading derivatives. No full event, material/geometry/stopping-time AD, isolated runtime or FEM advantage claim.",
        )
        write_json(output / "summary.json", result)
    except BaseException as error:
        write_json(output / "failure.json", dict(error=repr(error), seconds=perf_counter() - begun))
        raise


if __name__ == "__main__":
    main()

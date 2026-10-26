"""Compare equivalent complete-history bar gradients and compiler memory plans."""

import argparse
import json
import runpy
import subprocess
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from finiteblock_bem.io import environment_record, fingerprint, write_json

ROOT = Path(__file__).resolve().parents[2]
BAR_PATH = Path(__file__).with_name("radiation_bar_experiment.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline-revision", default="b7f007b")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-temporary-mib", type=float, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.rounds < 1
        or not np.isfinite(args.max_temporary_mib)
        or args.max_temporary_mib <= 0
    ):
        raise ValueError("Require a new output directory, positive rounds and finite memory budget")
    summary_path = args.run / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (
        summary["source_sha256"] != fingerprint(BAR_PATH)
        or not summary["fine_history_ad"]["passed"]
    ):
        raise ValueError("Require a source-matching full fine-history parameter check")
    frozen = {str(p.resolve()): fingerprint(p) for p in [BAR_PATH, summary_path, Path(__file__)]}
    options = dict(
        micro_dt=summary["micro_dt_s"],
        duration=summary["duration_s"],
        round_trip=2 * summary["one_way_transit_s"],
        radiation=summary["radiation_mpa_s_per_mm"],
        pressure=summary["pressure_mpa"],
        peak=summary["peak_applied_traction_mpa"],
        rise=summary["loading_rise_s"],
    )
    theta = jnp.array(summary["parameters"])
    args.output.mkdir(parents=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.baseline_revision}^{{commit}}"],
        cwd=ROOT,
        text=True,
    ).strip()
    baseline_path = args.output / "baseline_bar.py"
    baseline_path.write_bytes(
        subprocess.check_output(
            ["git", "show", f"{revision}:bem/tools/radiation_bar_experiment.py"], cwd=ROOT
        )
    )
    baseline, current = runpy.run_path(str(baseline_path)), runpy.run_path(str(BAR_PATH))
    old_loss, _ = baseline["fine_delay_loss"](**options)
    loss, raw = current["fine_delay_loss"](**options)

    def reference(parameters):
        return raw(parameters)[-1, 2]

    functions = {
        "array_adjoint": jax.value_and_grad(old_loss),
        "reverse_ring_adjoint": jax.value_and_grad(loss),
        "automatic_reverse_scan": jax.value_and_grad(reference),
        "forward_parameter_jvp": jax.jacfwd(reference),
    }
    compiled, records = {}, {}
    for name, function in functions.items():
        start = perf_counter()
        program = jax.jit(function).lower(theta).compile()
        compilation = perf_counter() - start
        plan = program.memory_analysis()
        memory = (
            None
            if plan is None
            else {
                key: getattr(plan, key)
                for key in [
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                    "alias_size_in_bytes",
                ]
            }
        )
        record = dict(compilation_seconds=compilation, compiler_memory_plan=memory, warm_seconds=[])
        if memory is None or memory["temp_size_in_bytes"] > args.max_temporary_mib * 1024**2:
            record["status"] = "compiled_not_executed_without_sufficient_temporary_budget"
        else:
            start = perf_counter()
            value = program(theta)
            jax.block_until_ready(value)
            record["warmup_seconds"] = perf_counter() - start
            gradient = value if name == "forward_parameter_jvp" else value[1]
            record["gradient"] = np.asarray(gradient).tolist()
            if not np.allclose(
                gradient, summary["fine_history_ad"]["forward"], rtol=1e-5, atol=1e-12
            ):
                raise ValueError(f"Gradient guard failed for {name}")
            record["status"] = "executed_gradient_guard_passed"
            compiled[name] = program
        records[name] = record
        write_json(args.output / "progress.json", records)
    order = list(compiled)
    schedule = (order + order[::-1]) * args.rounds
    for name in schedule:
        start = perf_counter()
        value = compiled[name](theta)
        jax.block_until_ready(value)
        records[name]["warm_seconds"].append(perf_counter() - start)
    for record in records.values():
        if record["warm_seconds"]:
            record["median_warm_seconds"] = float(np.median(record["warm_seconds"]))
    if any(fingerprint(p) != digest for p, digest in frozen.items()):
        raise ValueError("Benchmark source or the checked trajectory changed")
    result = dict(
        records=records,
        schedule=schedule,
        baseline_revision=revision,
        baseline_source_sha256=fingerprint(baseline_path),
        frozen_input_sha256=frozen,
        environment=environment_record(),
        scope="One process, all variants compiled and warmed before mirrored timing order; compiler array plans are not process RSS. This bar-only AD comparison is not an event-16 or same-accuracy 2D FEM benchmark.",
        same_accuracy_fem_advantage_verified=False,
    )
    write_json(args.output / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "environment"}, indent=2))


if __name__ == "__main__":
    main()

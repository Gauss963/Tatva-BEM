"""Same alternating original raw/Schur gates, including cold JIT and all tables."""

import argparse
import json
import resource
import runpy
import sys
from pathlib import Path
from time import perf_counter

TOOLS = Path(__file__).resolve().parent


def main():
    begun = perf_counter()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    jit = runpy.run_path(str(TOOLS / "jax_ray_galerkin.py"))
    benchmark = runpy.run_path(str(TOOLS / "benchmark_ray_bessel.py"))
    imports_seconds = perf_counter() - begun
    try:
        benchmark["main"](
            assembler=jit["JaxRayGalerkin"],
            extra_sources=(Path(__file__), TOOLS / "jax_ray_galerkin.py"),
            backend_name="jax_x64_ray_batch_original_equations",
        )
    finally:
        if args.output.is_dir():
            record = dict(
                import_and_jax_startup_seconds=imports_seconds,
                wrapper_seconds_including_imports=perf_counter() - begun,
                compiled_signatures=jit["COMPILED_SIGNATURES"],
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                compilation_and_synchronization_in_assembly_times=True,
                full_event_or_fem_speed_claim=False,
            )
            with (args.output / "jax-execution.json").open("x") as stream:
                json.dump(record, stream, indent=2, allow_nan=False)
                stream.write("\n")


if __name__ == "__main__":
    main()

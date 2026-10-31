"""Use the same alternating raw/Schur control for the paired-ray backend.

The shared benchmark retains its legacy timing key ``ray``; the backend
identity and all extra numerical sources are recorded explicitly. Costs
include table construction on every assembly, rather than a warmed table.
"""

import runpy
from pathlib import Path

TOOLS = Path(__file__).resolve().parent


def main():
    paired = runpy.run_path(str(TOOLS / "paired_ray_galerkin.py"))
    benchmark = runpy.run_path(str(TOOLS / "benchmark_ray_bessel.py"))
    benchmark["main"](
        assembler=paired["PairedRayGalerkin"],
        extra_sources=(Path(__file__), TOOLS / "paired_ray_galerkin.py"),
        backend_name="paired_ray_bessel_and_factored_contractions",
    )


if __name__ == "__main__":
    main()

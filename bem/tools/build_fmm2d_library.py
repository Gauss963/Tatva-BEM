"""Build a pinned optional FMM2D library locally and check its Cauchy ABI.

No system installation or Python extension build is performed. The generated
source, compiler log and binary stay inside a new explicitly selected folder.
"""

import argparse
import io
import json
import os
import runpy
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.io import fingerprint, write_json

COMMIT = "550dae5b77b1e006c8ffae37fc832f8c2b536871"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cores", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.cores < 1:
        raise ValueError("Choose a new output directory and a positive core count")
    fc, cc, make = (shutil.which(name) for name in ["gfortran", "cc", "make"])
    if not all([fc, cc, make]):
        raise ValueError("gfortran, a C compiler and make are required")
    started = perf_counter()
    args.output.mkdir(parents=True)
    tool = Path(__file__)
    cauchy_path = tool.with_name("cauchy_reuse_experiment.py")
    frozen = {str(p.resolve()): fingerprint(p) for p in [tool, cauchy_path]}
    record = dict(
        source_commit=COMMIT,
        source_repository="https://github.com/flatironinstitute/fmm2d",
        cores_requested=args.cores,
        hard_cpu_affinity_enforced=False,
        frozen_inputs_sha256=frozen,
    )
    try:
        url = f"https://codeload.github.com/flatironinstitute/fmm2d/tar.gz/{COMMIT}"
        with urllib.request.urlopen(url, timeout=60) as response:
            archive = response.read()
        archive_path = args.output / "source.tar.gz"
        archive_path.write_bytes(archive)
        record["source_archive_sha256"] = fingerprint(archive_path)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(args.output, filter="data")
        source = args.output / f"fmm2d-{COMMIT}"
        if not (source / "src/laplace/cfmm2dwrap_vec.f").is_file():
            raise ValueError("Pinned archive does not contain the expected Fortran interface")
        command = [
            make,
            "lib",
            f"-j{args.cores}",
            "FAST_KER=OFF",
            "OMP=ON",
            f"FC={fc}",
            f"CC={cc}",
            "FFLAGS=-fPIC -O3 -std=legacy -w -fallow-argument-mismatch -fopenmp",
            "CFLAGS=-fPIC -O3 -std=c99",
            "DYLIBS=-lm -lgomp",
        ]
        record.update(
            build_command=command,
            fortran_compiler_version=subprocess.check_output([fc, "--version"], text=True),
            c_compiler_version=subprocess.check_output([cc, "--version"], text=True),
            integer_interface="32-bit default INTEGER; no integer-promotion flags",
        )
        build_started = perf_counter()
        with (args.output / "build.log").open("x") as log:
            subprocess.run(command, cwd=source, check=True, stdout=log, stderr=subprocess.STDOUT)
        record["build_seconds"] = perf_counter() - build_started
        library = source / "lib/libfmm2d.so"
        record.update(library=str(library.resolve()), library_sha256=fingerprint(library))
        os.environ["OMP_NUM_THREADS"] = str(args.cores)
        local = runpy.run_path(str(cauchy_path))
        backend = local["fortran_fmm2d_backend"](library)
        circle = local["CircleCauchy"](512, 0.97)
        half = np.stack(
            (
                1 / (2 - circle.sources[:257]),
                1 / (3 + circle.sources[:257]),
                np.exp(0.2 * circle.sources[:257]),
            ),
            axis=1,
        )
        targets = np.array([0.0, 0.7 + 0.2j, 0.8 - 0.1j, 0.97 + 1e-10j])
        with threadpool_limits(limits=1, user_api="blas"):
            expected = circle.evaluate(half, targets, derivative=True)
            actual = circle.evaluate(half, targets, derivative=True, backend=backend)
        differences = []
        for candidate, reference in zip(actual, expected, strict=True):
            differences.append(float(np.max(abs(candidate - reference))))
            np.testing.assert_allclose(candidate, reference, rtol=1e-11, atol=1e-12)
        if any(fingerprint(path) != digest for path, digest in frozen.items()):
            raise ValueError("Library builder or Cauchy wrapper changed during build")
        if fingerprint(library) != record["library_sha256"]:
            raise ValueError("Built library changed during its ABI check")
        record.update(
            status="completed_library_and_small_cauchy_check",
            value_max_absolute_difference=differences[0],
            derivative_max_absolute_difference=differences[1],
            total_seconds=perf_counter() - started,
            full_original_event_reproduced=False,
            scope="Pinned external library build and small dense/FMM Cauchy value/derivative check; no full BEM speed or accuracy claim",
        )
    except Exception as error:
        record.update(
            status="failed_library_build_or_check",
            error=str(error),
            total_seconds=perf_counter() - started,
        )
        write_json(args.output / "summary.json", record)
        raise
    write_json(args.output / "summary.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

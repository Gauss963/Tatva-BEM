"""Controlled cold transfer benchmark: reference pairs vs prepared batches."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import bdf2_symbol
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("FDBEM_CORES", "1")))
    parser.add_argument("--graded", action="store_true")
    args = parser.parse_args()
    options = (
        dict(
            max_panel_length=40,
            contact_panel_length=20,
            contact_end_length=2,
            corner_panel_length=2,
        )
        if args.graded
        else dict(max_panel_length=80)
    )
    model = load_nodal_model(
        ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml", **options
    )
    frequencies = (
        bdf2_symbol(
            1e-10 ** (1 / 16384) * np.exp(-1j * np.array([0, 0.05, 0.2, 0.5, 1, 2, 2.8, np.pi]))
        )
        / 8e-6
    )
    records, answers = [], {}
    # Exchange execution order. Include geometry preparation in each cold call.
    with threadpool_limits(limits=1, user_api="blas"):
        for repeat, order in enumerate([[False, True], [True, False]]):
            for prepared in order:
                start = perf_counter()
                transfer = NodalInterfaceTransfer(model, prepared=prepared)
                preparation = perf_counter() - start
                with ThreadPoolExecutor(args.workers) as pool:
                    result = np.stack(list(pool.map(transfer.evaluate, frequencies)))
                seconds = perf_counter() - start
                answers[prepared] = result
                record = dict(
                    repeat=repeat,
                    prepared=prepared,
                    geometry_seconds=preparation,
                    total_seconds=seconds,
                )
                records.append(record)
                print(json.dumps(record), flush=True)
    difference = answers[True] - answers[False]
    record = dict(
        records=records,
        frequencies_real=frequencies.real.tolist(),
        frequencies_imag=frequencies.imag.tolist(),
        relative_frobenius_difference=float(
            np.linalg.norm(difference) / np.linalg.norm(answers[False])
        ),
        max_absolute_difference=float(np.max(abs(difference))),
        workers=args.workers,
        quadrature=12,
        mesh_options=options,
        panels=[model.reference.moving.count, model.reference.stationary.count],
        environment=environment_record(),
        scope="Identical transfer matrices; not an equal-accuracy FEM speed claim",
    )
    output = (
        ROOT / "outputs" / ("galerkin-benchmark-graded" if args.graded else "galerkin-benchmark")
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", record)


if __name__ == "__main__":
    main()

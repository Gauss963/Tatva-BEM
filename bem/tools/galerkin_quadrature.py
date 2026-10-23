"""Compare Galerkin integration orders on the actual event-16 CQ contour."""

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
    parser.add_argument(
        "--fine", action="store_true", help="Contact 5 / outer 20 mm; orders 6/8/12"
    )
    args = parser.parse_args()
    model = load_nodal_model(
        ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml",
        max_panel_length=20 if args.fine else 40,
        contact_panel_length=5 if args.fine else 20,
        contact_end_length=2,
        corner_panel_length=2,
    )
    angles = np.array([0, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, np.pi])
    frequencies = bdf2_symbol(1e-10 ** (1 / 16384) * np.exp(-1j * angles)) / 8e-6
    answers, records = {}, []
    orders = [6, 8, 12] if args.fine else [8, 12, 16]
    with threadpool_limits(limits=1, user_api="blas"):
        for order in orders:
            start = perf_counter()
            transfer = NodalInterfaceTransfer(model, quadrature=order)
            with ThreadPoolExecutor(int(os.environ.get("FDBEM_CORES", "1"))) as pool:
                answers[order] = np.stack(list(pool.map(transfer.evaluate, frequencies)))
            del transfer
            record = dict(order=order, seconds=perf_counter() - start)
            records.append(record)
            print(json.dumps(record), flush=True)
    weights = model.constraints.contact_weights
    scale = 1 / np.r_[weights, weights, model.constraints.platen_width]
    errors = {}
    reference_order = orders[-1]
    for order in orders[:-1]:
        errors[str(order)] = [
            float(np.linalg.norm(scale[:, None] * (a - b)) / np.linalg.norm(scale[:, None] * b))
            for a, b in zip(answers[order], answers[reference_order], strict=True)
        ]
    output = ROOT / "outputs" / ("galerkin-quadrature-fine" if args.fine else "galerkin-quadrature")
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "summary.json",
        dict(
            records=records,
            angles=angles.tolist(),
            **{f"errors_relative_to_{reference_order}": errors},
            scope="Sampled traction-scaled matrix differences; not a global quadrature error bound",
            environment=environment_record(),
        ),
    )
    print(json.dumps(errors), flush=True)


if __name__ == "__main__":
    main()

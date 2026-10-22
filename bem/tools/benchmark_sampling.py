"""Controlled cold, uncached dense/adaptive frequency assembly on the same grid."""

import argparse
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from finiteblock_bem.case import load_case
from finiteblock_bem.condensation import InterfaceTransfer, plan_interface
from finiteblock_bem.cq import convolution_weights
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.sampling import SamplingOptions, parsimonious_weights

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reverse", action="store_true", help="Reverse the method order")
    args = parser.parse_args()
    case_path = ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    case = load_case(case_path, max_panel_length=40)
    dt, steps, workers, quadrature = 8e-6, 2048, 4, 12
    plan = plan_interface(
        case, dt, steps, quadrature=quadrature, workers=workers, max_memory_mb=1024
    )[0][0]
    wc = case.moving.lengths[case.moving_contact]
    records, arrays = [], {}
    methods = ["adaptive", "dense"] if args.reverse else ["dense", "adaptive"]
    for method in methods:
        wall, cpu = perf_counter(), process_time()
        transfer = InterfaceTransfer(case, quadrature=quadrature)
        if method == "dense":
            weights = convolution_weights(transfer.evaluate, plan, workers=workers)
            metadata = {"transfer_evaluations": plan.fft_size // 2 + 1}
        else:
            weights, metadata = parsimonious_weights(
                transfer.evaluate,
                plan,
                options=SamplingOptions(),
                workers=workers,
                row_scale=1 / np.r_[wc, wc, transfer.constraints.platen_width],
            )
        records.append(
            {
                "method": method,
                "wall_s": perf_counter() - wall,
                "cpu_s": process_time() - cpu,
                "sampling": metadata,
            }
        )
        arrays[method] = weights
        print(json.dumps(records[-1]), flush=True)
    result = {
        "scope": "Same grid, material, workers and BLAS limits; kernel generation including geometry, no cache. No other simulation was running.",
        "dt_s": dt,
        "steps": steps,
        "quadrature": quadrature,
        "panels": [case.moving.count, case.stationary.count],
        "workers": workers,
        "results": records,
        "weight_relative_frobenius": float(
            np.linalg.norm(arrays["adaptive"] - arrays["dense"]) / np.linalg.norm(arrays["dense"])
        ),
        "weight_max_absolute": float(np.max(np.abs(arrays["adaptive"] - arrays["dense"]))),
        "environment": environment_record(case_path),
    }
    output = ROOT / (
        "outputs/sampling-benchmark-reverse" if args.reverse else "outputs/sampling-benchmark"
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", result)


if __name__ == "__main__":
    main()

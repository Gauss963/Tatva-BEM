"""Compare Galerkin integration orders on the actual event-16 CQ contour."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import bdf2_symbol, plan_cq
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model
from finiteblock_bem.nodal_dynamics import plan_nodal_history

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fine", action="store_true", help="Contact 5 / outer 20 mm; orders 6/8/12"
    )
    parser.add_argument(
        "--run", type=Path, help="Restore the exact mesh/time contour of a saved run"
    )
    parser.add_argument("--orders", type=int, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dt", type=float, help="Audit a proposed timestep on the saved mesh")
    parser.add_argument("--duration", type=float, help="Full contour horizon for --dt")
    args = parser.parse_args()
    if args.run is not None:
        original = json.loads((args.run / "run_manifest.json").read_text())["arguments"]
        model = load_nodal_model(
            original["case"],
            max_panel_length=original["h"],
            contact_panel_length=original["contact_h"],
            contact_end_length=original["contact_end_h"],
            corner_panel_length=original["corner_h"],
            restart_knot_window=original.get("restart_knot_window"),
            restart_knot_subdivisions=original.get("restart_knot_subdivisions", 1),
        )
        dt = original["dt"] if args.dt is None else args.dt
        original_steps = original["steps"] or round(original["duration"] / original["dt"])
        duration = original_steps * original["dt"] if args.duration is None else args.duration
        steps = round(duration / dt)
        plan = plan_cq(2 * len(model.moving_contact) + 1, dt, steps, max_memory_mb=65536)
        radius = plan.radius
    else:
        model = load_nodal_model(
            ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml",
            max_panel_length=20 if args.fine else 40,
            contact_panel_length=5 if args.fine else 20,
            contact_end_length=2,
            corner_panel_length=2,
        )
        if args.dt is not None or args.duration is not None:
            raise ValueError("Timestep/horizon overrides require --run for an exact source mesh")
        radius, dt = 1e-10 ** (1 / 16384), 8e-6
    angles = np.array([0, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, np.pi])
    frequencies = bdf2_symbol(radius * np.exp(-1j * angles)) / dt
    answers, records = {}, []
    orders = args.orders or ([6, 8, 12] if args.fine else [8, 12, 16])
    if orders != sorted(set(orders)) or any(order < 4 or order % 2 for order in orders):
        raise ValueError("Orders must be strictly increasing even integers >=4")
    with threadpool_limits(limits=1, user_api="blas"):
        for order in orders:
            plan_nodal_history(
                model,
                dt,
                1,
                workers=int(os.environ.get("FDBEM_CORES", "1")),
                quadrature=order,
                max_memory_mb=8192,
            )
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
    output = args.output or ROOT / "outputs" / (
        "galerkin-quadrature-fine" if args.fine else "galerkin-quadrature"
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "summary.json",
        dict(
            records=records,
            angles=angles.tolist(),
            actual_s_real=frequencies.real.tolist(),
            actual_s_imag=frequencies.imag.tolist(),
            dt_s=dt,
            source_run=None if args.run is None else str(args.run.resolve()),
            **{f"errors_relative_to_{reference_order}": errors},
            scope="Sampled traction-scaled matrix differences; not a global quadrature error bound",
            environment=environment_record(),
        ),
    )
    print(json.dumps(errors), flush=True)


if __name__ == "__main__":
    main()

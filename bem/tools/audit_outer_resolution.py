"""Compare outer-boundary meshes at the same retained contact coordinates.

Uses sampled points on the requested CQ contour and fixed linear contact
probes. This is a spatial-resolution diagnostic, not a global error bound.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import bdf2_symbol, plan_cq
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model
from finiteblock_bem.nodal_dynamics import plan_nodal_history

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--outer-sizes", nargs="+", type=float, default=[20, 10, 5])
    parser.add_argument("--dt", type=float, default=4e-6)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/outer-resolution")
    args = parser.parse_args()
    original = json.loads((args.run / "run_manifest.json").read_text())["arguments"]
    args.output.mkdir(parents=True, exist_ok=True)
    sizes = sorted(set(args.outer_sizes), reverse=True)
    target_frequencies = np.array([1000, 5000, 10000, 20000, 40000])
    matrices, responses, records = {}, {}, []
    reference_y = reference_weights = None
    workers = int(os.environ.get("FDBEM_CORES", "1"))
    for length in sizes:
        model = load_nodal_model(
            original["case"],
            max_panel_length=length,
            contact_panel_length=original["contact_h"],
            contact_end_length=original["contact_end_h"],
            corner_panel_length=original["corner_h"],
            restart_knot_window=original.get("restart_knot_window"),
            restart_knot_subdivisions=original.get("restart_knot_subdivisions", 1),
        )
        y = model.reference.moving.starts[model.moving_contact, 1]
        weights = model.constraints.contact_weights
        if records:
            np.testing.assert_array_equal(y, reference_y)
            np.testing.assert_array_equal(weights, reference_weights)
        else:
            reference_y, reference_weights = y.copy(), weights.copy()
        n = len(y)
        plan = plan_cq(2 * n + 1, args.dt, round(0.045 / args.dt), max_memory_mb=65536)
        _, peak = plan_nodal_history(
            model,
            args.dt,
            1,
            workers=workers,
            quadrature=original["quadrature"],
            max_memory_mb=8192,
        )
        indices = np.rint(target_frequencies * plan.fft_size * args.dt).astype(int)
        s = bdf2_symbol(plan.radius * np.exp(-2j * np.pi * indices / plan.fft_size)) / args.dt
        start = perf_counter()
        transfer = NodalInterfaceTransfer(model, quadrature=original["quadrature"])
        with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(workers) as pool:
            matrices[length] = np.stack(list(pool.map(transfer.evaluate, s)))
        del transfer
        forcing = np.zeros((2 * n + 1, 5))
        forcing[-1, 0] = model.constraints.platen_width
        forcing[:n, 1] = weights
        for column, mode in enumerate([0, 8, 24], start=2):
            forcing[n : 2 * n, column] = weights * (
                np.ones(n) if mode == 0 else np.sin(mode * np.pi * y / y[-1])
            )
        tangent = np.diag(
            np.r_[
                weights * model.reference.contact.normal_penalty,
                weights * model.reference.contact.tangential_penalty,
                model.reference.spring_stiffness * model.constraints.platen_width,
            ]
        )
        with threadpool_limits(limits=1, user_api="blas"):
            responses[length] = np.stack(
                [solve(z + tangent, forcing, check_finite=False) for z in matrices[length]]
            )
        np.savez(
            args.output / f"outer{length:g}.npz",
            s=s,
            impedance=matrices[length],
            closed_stick_response=responses[length],
        )
        record = dict(
            outer_h_mm=length,
            panels=[model.reference.moving.count, model.reference.stationary.count],
            seconds=perf_counter() - start,
            workspace_estimate_mib=peak / 1024**2,
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    row_scale = 1 / np.r_[reference_weights, reference_weights, model.constraints.platen_width]
    response_scale = np.sqrt(1 / row_scale)
    errors = {}
    finest = sizes[-1]
    for length in sizes[:-1]:
        impedance = matrices[length] - matrices[finest]
        response = responses[length] - responses[finest]
        errors[str(length)] = dict(
            relative_impedance_frobenius=(
                np.linalg.norm(impedance * row_scale[None, :, None], axis=(1, 2))
                / np.linalg.norm(matrices[finest] * row_scale[None, :, None], axis=(1, 2))
            ).tolist(),
            per_probe_relative_line_l2=(
                np.linalg.norm(response * response_scale[None, :, None], axis=1)
                / np.linalg.norm(responses[finest] * response_scale[None, :, None], axis=1)
            ).tolist(),
        )
    write_json(
        args.output / "summary.json",
        dict(
            records=records,
            reference_outer_h_mm=finest,
            errors=errors,
            target_frequencies_hz=target_frequencies.tolist(),
            actual_s_real=s.real.tolist(),
            actual_s_imag=s.imag.tolist(),
            probe_order=[
                "platen",
                "uniform_normal",
                "uniform_shear",
                "shear_mode_8",
                "shear_mode_24",
            ],
            scope="Sampled CQ-contour comparison against finest outer mesh; fixed linear closed/sticking contact; not a global error bound",
            environment=environment_record(),
        ),
    )
    print(json.dumps(errors), flush=True)


if __name__ == "__main__":
    main()

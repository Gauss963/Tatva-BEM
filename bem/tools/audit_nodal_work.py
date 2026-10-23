"""Independent offline FFT reconstruction of retained-coordinate bulk work.

The result concerns perturbations about the fixed static lift. It excludes
the initial prestress, external normal-load work and nonlinear contact energy;
it is not a total mechanical energy conservation proof.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.nodal import load_nodal_model
from finiteblock_bem.nodal_dynamics import build_nodal_history
from finiteblock_bem.sampling import SamplingOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    folder = args.run.resolve()
    summary = json.loads((folder / "summary.json").read_text())
    manifest = json.loads((folder / "run_manifest.json").read_text())["arguments"]
    if not summary.get("boundary_formulation", "").startswith("P1/P0"):
        raise ValueError("This audit requires the nodal Galerkin force backend")
    if summary["status"] != "completed_requested_horizon" or summary.get(
        "source_changed_during_run"
    ):
        raise ValueError("A complete forward run from frozen numerical sources is required")
    environment = environment_record(Path(manifest["case"]))
    for field in ["source_sha256", "case_sha256", "restart_data_sha256"]:
        if environment[field] != summary["environment"][field]:
            raise ValueError(f"{field} differs; rerun forward before auditing")
    model = load_nodal_model(
        manifest["case"],
        max_panel_length=manifest["h"],
        contact_panel_length=manifest["contact_h"],
        contact_end_length=manifest["contact_end_h"],
        corner_panel_length=manifest["corner_h"],
        restart_knot_window=manifest.get("restart_knot_window"),
        restart_knot_subdivisions=manifest.get("restart_knot_subdivisions", 1),
    )
    history = build_nodal_history(
        model,
        summary["dt_s"],
        summary["completed_steps"],
        workers=manifest["workers"],
        quadrature=manifest["quadrature"],
        sampling=(
            SamplingOptions(relative_tolerance=manifest["sampling_rtol"])
            if manifest["sampling"] == "adaptive"
            else None
        ),
        cache_directory=manifest["cache"],
        max_memory_mb=manifest["max_memory_mb"],
    )
    with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
        time = data["time"]
        q = np.column_stack(
            (data["contact_jumps"] - data["contact_jumps"][0], data["platen"] - data["platen"][0])
        )
    # One zero-padded FFT, independent of OnlineConvolution's dyadic algorithm.
    # Bound the temporary matrix spectrum to four output rows.
    length = next_fast_len(2 * len(q) - 1)
    spectrum = rfft(q, n=length, axis=0)
    force = np.empty_like(q)
    for first in range(0, q.shape[1], 4):
        weight = rfft(history.weights[:, first : first + 4], n=length, axis=0)
        force[:, first : first + 4] = irfft(
            np.einsum("fij,fj->fi", weight, spectrum), n=length, axis=0
        )[: len(q)]
    direct_errors = []
    for index in sorted({0, 1, 2, len(q) // 2, len(q) - 1}):
        if index >= len(q):
            continue
        exact = np.einsum("kij,kj->i", history.weights[: index + 1], q[index::-1])
        direct_errors.append(float(np.max(abs(force[index] - exact))))
        np.testing.assert_allclose(force[index], exact, rtol=1e-8, atol=1e-7)
    dt = history.plan.dt
    velocity = np.zeros_like(q)
    velocity[1] = (1.5 * q[1] - 2 * q[0]) / dt
    velocity[2:] = (1.5 * q[2:] - 2 * q[1:-1] + 0.5 * q[:-2]) / dt
    work = np.cumsum(np.sum(force * velocity, axis=1)) * dt
    if not np.all(np.isfinite(work)):
        raise FloatingPointError("Nonfinite bulk perturbation work")
    record = {
        "scope": "Bulk perturbation work about fixed lift; not total nonlinear energy",
        "method": "Independent offline zero-padded FFT and BDF2 velocity",
        "minimum_cumulative_work_mpa_mm2": float(work.min()),
        "final_cumulative_work_mpa_mm2": float(work[-1]),
        "completed_steps": len(q) - 1,
        "max_force_difference_from_direct_sum": max(direct_errors),
        "environment": environment,
    }
    np.savez(folder / "bulk_work.npz", time=time, cumulative_work=work)
    write_json(folder / "bulk_work.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

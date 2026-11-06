"""Read-only disk check for full startup actions plus a short Radau correction."""

import argparse
import json
import math
import runpy
import shutil
from dataclasses import asdict
from pathlib import Path

from finiteblock_bem.cq import CQPlan, plan_cq

TOOLS = Path(__file__).resolve().parent
RECIPROCAL = runpy.run_path(str(TOOLS / "reciprocal_radau_storage.py"))
GIB = 1024**3


def grid_steps(time, dt):
    if not math.isfinite(time) or time <= 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Require positive finite times and time step")
    steps = round(time / dt)
    if steps < 1 or not math.isclose(time, steps * dt, rel_tol=1e-12, abs_tol=0):
        raise ValueError("The requested time must lie on the declared time grid")
    return steps


def required_data(full, correction, block):
    if (full.dofs, full.dt) != (correction.dofs, correction.dt):
        raise ValueError("Full startup and correction must use the same trace and time step")
    d = full.dofs
    return {
        "reciprocal_raw_branches": 2 * RECIPROCAL["reciprocal_size"](correction, block),
        "physical_correction_weights": 2 * correction.weight_bytes,
        "physical_correction_wrap": d**2 * 8,
        "full_startup_actions": (full.fft_size // 2 + 1) * 2 * d * 2 * 16,
        "full_startup_histories": (full.steps + 1) * 2 * d * 2 * 8,
    }


def inspect_volumes(spectra, time, data, spectra_reserve, time_reserve):
    volumes = {}
    by_volume = [
        (spectra, data["reciprocal_raw_branches"], spectra_reserve),
        (time, sum(data.values()) - data["reciprocal_raw_branches"], time_reserve),
    ]
    for folder, payload, reserve in by_volume:
        folder = folder.resolve()
        if folder.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
            raise ValueError("Choose independent BEM directories outside protected Tatva")
        if not folder.is_dir() or not math.isfinite(reserve) or reserve < 0:
            raise ValueError("Require existing volume directories and finite nonnegative reserves")
        dev = folder.stat().st_dev
        entry = volumes.setdefault(
            dev,
            dict(
                paths=[],
                free_bytes=shutil.disk_usage(folder).free,
                payload_bytes=0,
                reserve_bytes=0,
            ),
        )
        entry["paths"].append(str(folder))
        entry["payload_bytes"] += payload
        entry["reserve_bytes"] += math.ceil(reserve * GIB)
    for entry in volumes.values():
        required = entry["payload_bytes"] + entry["reserve_bytes"]
        entry["margin_bytes"] = entry["free_bytes"] - required
    return list(volumes.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "kernel", type=Path, help="Completed original kernel supplying the trace size"
    )
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--earliest-correction", type=float, required=True)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--spectra-volume", type=Path, required=True)
    parser.add_argument("--time-volume", type=Path, required=True)
    parser.add_argument("--spectra-reserve-gib", type=float, required=True)
    parser.add_argument("--time-reserve-gib", type=float, required=True)
    args = parser.parse_args()
    source = json.loads((args.kernel / "summary.json").read_text())
    if source.get("status") != "completed_compact_radau_resolvent_kernel_only" or (
        source.get("time_method") != "radau2"
    ):
        raise ValueError("Require a completed original Radau kernel")
    parent = CQPlan(**source["plan"])
    steps = grid_steps(args.duration, args.dt)
    first = grid_steps(args.earliest_correction, args.dt)
    if first >= steps or args.duration > source["full_steps"] * parent.dt + 1e-12:
        raise ValueError("Require a positive correction horizon within the source duration")
    full = plan_cq(parent.dofs, args.dt, steps, max_memory_mb=1e9, disk_spectrum=True)
    correction = plan_cq(parent.dofs, args.dt, steps - first, max_memory_mb=1e9, disk_spectrum=True)
    data = required_data(full, correction, args.block)
    volumes = inspect_volumes(
        args.spectra_volume, args.time_volume, data, args.spectra_reserve_gib, args.time_reserve_gib
    )
    passes = all(v["margin_bytes"] >= 0 for v in volumes)
    print(
        json.dumps(
            dict(
                status="passed_disk_payload_preflight"
                if passes
                else "insufficient_disk_payload_space",
                source_kernel=str(args.kernel.resolve()),
                source_mesh=source["source_mesh_arguments"],
                startup_plan=asdict(full),
                correction_plan=asdict(correction),
                reciprocal_block=args.block,
                first_allowed_correction_step=first,
                first_allowed_correction_requires_run_time_check=True,
                payload_data_bytes=data,
                total_payload_data_bytes=sum(data.values()),
                volumes=volumes,
                scope="Disk payload plus selected reserves. No allocation, sampling or simulation.",
            ),
            indent=2,
        )
    )
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())

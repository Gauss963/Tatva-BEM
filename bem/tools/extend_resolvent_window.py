"""Retain more CQ lags from the exact same completed frequency circle.

Refined contact quadrature may create correction sources during the sticking
prefix. A late-start 20 ms correction window is then insufficient for a 45 ms
event. This tool never changes dt, radius, FFT size or frequency matrices.
It writes a new complete lag file and verifies the old prefix independently.
"""

import argparse
import gc
import json
import resource
import runpy
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.fft import set_workers

from finiteblock_bem.cq import CQPlan
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
READ = runpy.run_path(str(TOOLS / "replay_resolvent.py"))
NATIVE = runpy.run_path(str(TOOLS / "native_spectrum.py"))
CONVERT = runpy.run_path(str(TOOLS / "prepare_native_resolvent.py"))


def extended_plan(plan, steps, *, fft_rows=4):
    if (
        not isinstance(steps, int)
        or isinstance(steps, bool)
        or steps <= plan.steps
        or 2 * (steps + 1) > plan.fft_size
        or fft_rows < 1
    ):
        raise ValueError("Require a longer window fitting the unchanged zero-padded FFT circle")
    rows = min(fft_rows, plan.dofs)
    estimate = (
        4 * plan.fft_size * rows * plan.dofs * 8
        + (steps + 1) * rows * plan.dofs * 8
        + 64 * plan.dofs**2 * 16
    )
    return replace(
        plan,
        steps=steps,
        weight_bytes=(steps + 1) * plan.dofs**2 * 8,
        estimated_peak_bytes=estimate,
    )


def extend_weights(spectrum, plan, steps, path, line, *, fft_rows=4, max_workspace_mb=3072):
    full = extended_plan(plan, steps, fft_rows=fft_rows)
    line = np.asarray(line, dtype=float)
    if line.shape != (plan.dofs,) or not np.all(np.isfinite(line)) or np.any(line <= 0):
        raise ValueError("Require positive finite work-coordinate line weights")
    value = NATIVE["invert_spectrum_to_file"](
        spectrum, full, path, fft_rows=fft_rows, max_workspace_mb=max_workspace_mb
    )
    value._mmap.close()
    value = CONVERT["physical_weights_file_inplace"](path, line)
    return full, value


def compare_prefix(old, new, *, block_lags=64):
    if (
        old.ndim != 3
        or new.shape[1:] != old.shape[1:]
        or len(new) <= len(old)
        or old.dtype != np.float64
        or new.dtype != np.float64
    ):
        raise ValueError("Require matching physical float64 matrix prefixes and a longer target")
    maximum, equal = 0.0, True
    for first in range(0, len(old), block_lags):
        last = min(first + block_lags, len(old))
        a, b = (
            np.asarray(old[first:last]),
            np.asarray(new[first:last]),
        )
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("Nonfinite prefix weights")
        equal = equal and np.array_equal(a, b)
        maximum = max(maximum, float(np.max(abs(a - b))))
    return dict(compared_lags=len(old), bitwise_equal=equal, maximum_absolute_difference=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fft-rows", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-workspace-mb", type=float, default=3072)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a new output outside the protected Tatva tree")
    if args.workers < 1 or not np.isfinite(args.max_workspace_mb) or args.max_workspace_mb <= 0:
        raise ValueError("Require positive workers and finite workspace")
    begun = perf_counter()
    metadata_path = args.kernel / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["status"] != "completed_resolvent_kernel_only":
        raise ValueError("Require a completed original elastic-resolvent kernel")
    storage, options = metadata["spectrum_storage"], metadata["source_mesh_arguments"]
    if (
        storage.get("coordinates") != "diag(sqrt(w)) * L * diag(sqrt(w))"
        or not storage["all_frequency_matrices_written"]
    ):
        raise ValueError("Require a complete work-coordinate resolvent spectrum")
    plan = CQPlan(**metadata["plan"])
    full = extended_plan(plan, metadata["full_steps"], fft_rows=args.fft_rows)
    if full.estimated_peak_bytes > args.max_workspace_mb * 1024**2:
        raise ValueError("Complete-lag IFFT exceeds the explicit array workspace")
    environment = environment_record(options["case"])
    for key in ("source_sha256", "case_sha256", "restart_data_sha256"):
        if environment[key] != metadata["environment"][key]:
            raise ValueError("Prepared operator package or physical input changed")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    wc = model.constraints.contact_weights
    line = np.r_[wc, wc, model.constraints.platen_width]
    if len(line) != plan.dofs or metadata["full_steps"] * plan.dt != options["duration"]:
        raise ValueError("Retained geometry or full horizon differs from the source")
    frozen_paths = [
        Path(__file__),
        metadata_path,
        TOOLS / "native_spectrum.py",
        TOOLS / "prepare_native_resolvent.py",
        TOOLS / "bounded_weight_conversion.py",
        TOOLS / "replay_resolvent.py",
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in frozen_paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    required = (
        full.weight_bytes
        + (args.kernel / "elastic.npz").stat().st_size
        + 2 * plan.dofs**2 * 8
        + 65536
    )
    free = shutil.disk_usage(output.parent).free
    if free < required:
        raise ValueError("Insufficient disk for every lag and copied startup artifacts")
    preflight = dict(
        status="preflight_only_not_a_complete_kernel",
        source_kernel=str(args.kernel.resolve()),
        unchanged_frequency_circle=dict(dt=plan.dt, radius=plan.radius, fft_size=plan.fft_size),
        old_correction_steps=plan.steps,
        full_correction_steps=full.steps,
        required_disk_bytes=required,
        available_disk_bytes=free,
        estimated_ifft_array_bytes=full.estimated_peak_bytes,
        max_workspace_mb=args.max_workspace_mb,
        scope="Array estimate, not an OS RSS bound; no new frequency evaluations or physical trajectory",
        frozen_input_sha256=frozen,
        source_spectrum_sha256=storage["sha256"],
    )
    output.mkdir()
    write_json(output / "preflight.json", preflight)
    if args.preflight_only:
        print(output / "preflight.json", flush=True)
        return
    source_path = Path(storage["path"])
    source_stat = source_path.stat()
    for path, expected in [
        (source_path, storage["sha256"]),
        (args.kernel / "weights.npy", metadata["weights_sha256"]),
        (args.kernel / "elastic.npz", metadata["elastic_sha256"]),
    ]:
        if READ["digest_file"](path) != expected:
            raise ValueError(f"Source data digest differs: {path.name}")
    hashing_seconds = perf_counter() - begun
    old = np.load(args.kernel / "weights.npy", mmap_mode="r", allow_pickle=False)
    spectrum = READ["SavedDiskSpectrum"](source_path, plan)
    started = perf_counter()
    with set_workers(args.workers):
        full, weights = extend_weights(
            spectrum,
            plan,
            full.steps,
            output / "weights.npy",
            line,
            fft_rows=args.fft_rows,
            max_workspace_mb=args.max_workspace_mb,
        )
    inversion_seconds = perf_counter() - started
    prefix = compare_prefix(old, weights)
    if not prefix["bitwise_equal"]:
        raise ValueError(f"Extended weights changed the old physical prefix: {prefix}")
    law = model.reference.contact
    cdiag = np.r_[wc * law.normal_penalty, wc * law.tangential_penalty, 0]
    impedance = np.load(args.kernel / "initial_impedance.npy", allow_pickle=False)
    matrix = impedance + np.diag(cdiag)
    matrix[-1, -1] += model.reference.spring_stiffness * model.constraints.platen_width
    composition = float(np.linalg.norm(weights[0] @ matrix - np.eye(plan.dofs), ord=np.inf))
    if composition >= 1e-6:
        raise ValueError("Extended lag-zero inverse composition failed")
    old._mmap.close()
    weights._mmap.close()
    del old, weights, spectrum
    gc.collect()
    for name in ("elastic.npz", "initial_impedance.npy"):
        shutil.copyfile(args.kernel / name, output / name)
    new_digest = READ["digest_file"](output / "weights.npy")
    if (source_path.stat().st_size, source_path.stat().st_mtime_ns) != (
        source_stat.st_size,
        source_stat.st_mtime_ns,
    ):
        raise ValueError("Read-only source spectrum changed during extension")
    if any(fingerprint(path) != value for path, value in frozen.items()):
        raise ValueError("Frozen code or source metadata changed")
    extension_seconds = perf_counter() - begun
    result = dict(metadata)
    result.update(
        plan=asdict(full),
        correction_steps=full.steps,
        weights_sha256=new_digest,
        total_seconds=metadata["total_seconds"] + extension_seconds,
        correction_window_extended_without_new_frequency_evaluations=True,
        source_kernel=str(args.kernel.resolve()),
        source_kernel_summary_sha256=fingerprint(metadata_path),
        old_physical_prefix_comparison=prefix,
        initial_composition_inf_error=composition,
        extension_timing_seconds=dict(
            initial_hashing_and_preflight=hashing_seconds,
            ifft_and_physical_conversion=inversion_seconds,
            total_including_checks=extension_seconds,
        ),
        recorded_upstream_preparation_seconds=metadata["total_seconds"],
        timing_scope="Source cost plus this extension are both recorded; source execution was not an isolated fair speed benchmark",
        peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        frozen_input_sha256=frozen,
        environment=environment,
        scope="Same full frequency circle with every lag through the original horizon; nonlinear refined-contact trajectory still requires validation",
    )
    write_json(output / "summary.json", result)
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

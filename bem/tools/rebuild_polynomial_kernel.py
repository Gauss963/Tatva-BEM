"""Rebuild a prepared L kernel/startup from a grid-verified compact spectrum."""

import argparse
import json
import runpy
import shutil
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.cq import CQPlan, invert_spectrum
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectrum", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-memory-mb", type=float, default=4096)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new empty output directory")
    started = perf_counter()
    script_sha = fingerprint(Path(__file__))
    reader_path = Path(__file__).with_name("polynomial_spectrum.py")
    reader_sha = fingerprint(reader_path)
    namespace = runpy.run_path(str(reader_path))
    spectrum = namespace["PolynomialSpectrum"](args.spectrum)
    provenance = spectrum.metadata["provenance"]
    source = Path(provenance["source_kernel"])
    if fingerprint(source / "summary.json") != provenance["source_summary_sha256"]:
        raise ValueError("Source kernel metadata differs")
    metadata = json.loads((source / "summary.json").read_text())
    plan = CQPlan(**metadata["plan"])
    if (
        metadata["status"] != "completed_resolvent_kernel_only"
        or metadata["spectrum_storage"] != provenance["source_spectrum"]
    ):
        raise ValueError("A matching completed kernel is required")
    if metadata["spectrum_storage"]["coordinates"] != "diag(sqrt(w)) * L * diag(sqrt(w))":
        raise ValueError("Expected work-coordinate resolvent spectrum")
    if spectrum.shape != (plan.fft_size // 2 + 1, plan.dofs, plan.dofs):
        raise ValueError("Compressed frequency grid differs from the original plan")
    estimate = (
        plan.weight_bytes
        + 4 * plan.fft_size * min(4, plan.dofs) * plan.dofs * 8
        + 12 * (metadata["full_steps"] + 1) * plan.dofs * 8
    )
    if estimate > args.max_memory_mb * 1024**2:
        raise ValueError("Rebuild arrays exceed the requested memory budget")
    for name, key in [("weights.npy", "weights_sha256"), ("elastic.npz", "elastic_sha256")]:
        if fingerprint(source / name) != metadata[key]:
            raise ValueError(f"Changed source data: {name}")
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    for name in ["source_sha256", "case_sha256", "restart_data_sha256"]:
        if environment[name] != metadata["environment"][name]:
            raise ValueError(f"Source differs in {name}")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    weights = invert_spectrum(spectrum, plan)
    line_weights = np.r_[
        model.constraints.contact_weights,
        model.constraints.contact_weights,
        model.constraints.platen_width,
    ]
    scale = np.sqrt(line_weights[:, None] * line_weights[None, :])
    for first in range(0, len(weights), 64):
        weights[first : first + 64] /= scale
    original = np.load(source / "weights.npy", mmap_mode="r", allow_pickle=False)
    weight_difference = max(
        float(np.max(abs(weights[k : k + 64] - original[k : k + 64])))
        for k in range(0, len(weights), 64)
    )
    del original
    helper_path = Path(__file__).with_name("replay_resolvent.py")
    helper_sha = fingerprint(helper_path)
    helper = runpy.run_path(str(helper_path))
    with np.load(source / "elastic.npz", allow_pickle=False) as data:
        time, ramp, old_elastic = data["time"], data["uninterrupted_actuator"], data["elastic"]
    rate = model.reference.actuator_rate
    if options.get("diagnostic_actuator_increment") is not None:
        rate = options["diagnostic_actuator_increment"] / time[-1]
    spring = model.reference.spring_stiffness * model.constraints.platen_width
    column = spectrum[:, :, -1] / np.sqrt(line_weights * line_weights[-1])
    elastic = helper["causal_elastic_baseline"](
        column, replace(plan, steps=metadata["full_steps"]), spring * rate * time
    )
    cdiag = np.r_[
        model.constraints.contact_weights * model.reference.contact.normal_penalty,
        model.constraints.contact_weights * model.reference.contact.tangential_penalty,
        0,
    ]
    tangent = np.diag(cdiag)
    tangent[-1, -1] += spring
    impedance = np.load(source / "initial_impedance.npy", allow_pickle=False)
    composition = float(
        np.linalg.norm(weights[0] @ (impedance + tangent) - np.eye(plan.dofs), ord=np.inf)
    )
    if composition > 1e-6:
        raise ValueError("Rebuilt instantaneous inverse fails composition")
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "weights.npy", weights, allow_pickle=False)
    np.savez(args.output / "elastic.npz", time=time, elastic=elastic, uninterrupted_actuator=ramp)
    shutil.copy2(source / "initial_impedance.npy", args.output / "initial_impedance.npy")
    if (
        script_sha != fingerprint(Path(__file__))
        or reader_sha != fingerprint(reader_path)
        or helper_sha != fingerprint(helper_path)
    ):
        raise ValueError("Rebuild implementation changed during execution")
    result = {
        key: metadata[key]
        for key in [
            "source_run",
            "source_mesh_arguments",
            "dt_s",
            "full_steps",
            "correction_steps",
            "plan",
            "history_dofs",
            "environment",
            "helper_math_sha256",
        ]
    }
    measurement = args.spectrum / "process_measurement.json"
    compression_seconds = (
        json.loads(measurement.read_text())["pause_seconds"] if measurement.is_file() else None
    )
    elapsed = perf_counter() - started
    result.update(
        status="completed_resolvent_kernel_only",
        spectrum_storage={
            **spectrum.record(),
            "coordinates": metadata["spectrum_storage"]["coordinates"],
        },
        initial_composition_inf_error=composition,
        maximum_time_weight_difference=weight_difference,
        maximum_elastic_startup_difference=float(np.max(abs(elastic - old_elastic))),
        weights_sha256=fingerprint(args.output / "weights.npy"),
        elastic_sha256=fingerprint(args.output / "elastic.npz"),
        prototype_sha256=script_sha,
        helper_sha256=helper_sha,
        frequency_sampling=dict(
            method="reconstruction_from_completed_grid",
            source_sampling=metadata["frequency_sampling"],
        ),
        total_seconds=None
        if compression_seconds is None
        else metadata["total_seconds"] + compression_seconds + elapsed,
        timing_components=dict(
            source_build_seconds=metadata["total_seconds"],
            compression_process_seconds=compression_seconds,
            rebuild_seconds=elapsed,
        ),
        timing_scope="Dependency-inclusive observed times, source build may overlap diagnostics; not a controlled cold benchmark",
        scope="Reconstructed complete elastic startup and correction weights from compressed full contour; no source BEM solve or causal history term removed",
    )
    write_json(args.output / "summary.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["environment", "source_mesh_arguments"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

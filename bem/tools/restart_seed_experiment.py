"""Capture the original float64 restart and reuse fixed-geometry bulk kernels.

Only seed values may change. The archived knot positions, actuator, healed
memory, geometry, material, contact law and prepared time kernel stay fixed.
This is an explicit input-fidelity experiment, not a relaxed cache check.
"""

import argparse
import hashlib
import json
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np

from finiteblock_bem.friction import ContactState
from finiteblock_bem.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_restart_seed(model, path):
    """Use a finite float64 seed on exactly the same original interpolation knots."""
    path = Path(path).resolve()
    checksum = digest(path)
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != {"y", "plastic_slip", "cumulative_slip", "actuator"}:
            raise ValueError(
                "Restart seed must contain y, plastic_slip, cumulative_slip and actuator"
            )
        data = {key: source[key] for key in source.files}
    if any(a.dtype != np.float64 or not np.all(np.isfinite(a)) for a in data.values()):
        raise ValueError("All restart seed arrays must be finite float64")
    y, z, c, actuator = (data[k] for k in ["y", "plastic_slip", "cumulative_slip", "actuator"])
    stock = Path(model.reference.payload["source"]["restart_interface"])
    if not stock.is_absolute():
        stock = model.case_path.parent / stock
    default_y = np.genfromtxt(stock, delimiter=",", names=True)["y"]
    if y.ndim != 1 or not np.array_equal(y, default_y) or np.any(np.diff(y) <= 0):
        raise ValueError("Restart knots must be exactly the archived knots; geometry cannot change")
    if z.shape != y.shape or c.shape != y.shape or np.any(c != 0):
        raise ValueError(
            "A matching plastic-slip vector and zero healed cumulative slip are required"
        )
    if actuator.shape != () or actuator.item() != model.reference.restart_actuator:
        raise ValueError("Restart actuator must match the prepared loading exactly")
    nodes = model.reference.moving.starts[model.moving_contact, 1]
    if nodes[0] != y[0] or nodes[-1] != y[-1]:
        raise ValueError("Restart interpolation and contact interval differ")
    updated = replace(model, restart=ContactState(np.interp(nodes, y, z), np.interp(nodes, y, c)))
    provenance_path = path.with_suffix(".json")
    provenance = None
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("seed_file_sha256") != checksum:
            raise ValueError("Restart provenance identifies different seed arrays")
    if digest(path) != checksum:
        raise ValueError("Restart seed changed while reading")
    return updated, dict(
        path=str(path),
        sha256=checksum,
        source_provenance=provenance,
        provenance_path=None if provenance is None else str(provenance_path),
        provenance_sha256=None if provenance is None else digest(provenance_path),
        maximum_plastic_change_from_packaged_mm=float(
            np.max(abs(updated.restart.plastic_slip - model.restart.plastic_slip))
        ),
        unchanged_knots_geometry_and_bulk_kernel=True,
        scope="Explicit float64 values on the original seed knots; static lift is recomputed, complete time kernel is unchanged",
    )


def export_original_restart(rerun, output):
    """Copy only the verified event-15 seed; never copy or modify its full HDF5."""
    import h5py

    rerun, output = Path(rerun).resolve(), Path(output).resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists():
        raise ValueError("Choose a new seed output folder inside this checkout's bem/outputs")
    summary_path, config_path = rerun / "summary.json", rerun / "resolved_run_config.json"
    summary, config = [json.loads(p.read_text()) for p in [summary_path, config_path]]
    if not (
        summary["status"] == "completed"
        and summary["full_original_horizon"]
        and summary["source_and_protected_input_unchanged"]
        and summary.get("computational_dtype") == config["dtype"] == "float64"
        and config["friction_law"] == "slip-weakening"
        and config["reset_slip_weakening_at_shear_start"]
        and config["healing_fraction"] == 1
        and not summary["restart_projected_to_coarse_mesh"]
    ):
        raise ValueError("A verified original-precision, fully healed FEM rerun is required")
    helper = runpy.run_path(str(Path(__file__).with_name("run_fem_reference.py")))
    restart = summary["actual_restart"]
    if helper["restart_seed_record"](restart["path"]) != restart:
        raise ValueError("Original restart differs from the seed used by the verified FEM run")
    source_path, completed_path = Path(restart["path"]), rerun / "simulation.h5"
    frozen = {
        str(p): digest(p)
        for p in [
            summary_path,
            config_path,
            Path(__file__),
            Path(__file__).with_name("run_fem_reference.py"),
        ]
    }
    before = [p.stat() for p in [source_path, completed_path]]
    with h5py.File(source_path, "r") as source, h5py.File(completed_path, "r") as target:
        group = source["final_state"]
        z = np.asarray(group["plastic_slip"])
        actuator = np.asarray(group.attrs["applied_shear_displacement"])
        y = np.asarray(source["interface/contact_line_y"][:], dtype=np.float64)
        if z.dtype != np.float64 or actuator.dtype != np.float64 or z.shape != y.shape:
            raise ValueError("Original HDF5 must contain an aligned float64 restart")
        order = np.argsort(y)
        y, z = y[order], z[order]
        selected = (
            target["interface_high_rate"]
            if "interface_high_rate" in target
            else target["interface"]
        )
        np.testing.assert_array_equal(y, selected["contact_line_y"][:])
        np.testing.assert_array_equal(
            z.astype(np.float32), selected["plastic_slip"][0].astype(np.float32)
        )
    after = [p.stat() for p in [source_path, completed_path]]
    if any(
        (a.st_size, a.st_mtime_ns) != (b.st_size, b.st_mtime_ns)
        for a, b in zip(before, after, strict=True)
    ) or any(digest(p) != sha for p, sha in frozen.items()):
        raise ValueError("Restart capture source changed during read-only export")
    output.mkdir()
    path = output / "restart_seed.npz"
    np.savez(path, y=y, plastic_slip=z, cumulative_slip=np.zeros_like(z), actuator=actuator)
    record = dict(
        status="captured_original_float64_restart",
        seed_file_sha256=digest(path),
        original_restart=restart,
        source_hdf5_size_and_mtime_unchanged=True,
        completed_fem=str(rerun),
        initial_saved_plastic_equal_at_storage_precision=True,
        cumulative_slip_reset_to_zero=True,
        frozen_input_sha256=frozen,
        scope="Original float64 event-15 plastic slip and actuator; healed memory reset as specified by event-16. Not a bulk checkpoint or evidence of BEM equivalence.",
    )
    write_json(path.with_suffix(".json"), record)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rerun", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_original_restart(args.rerun, args.output), flush=True)


if __name__ == "__main__":
    main()

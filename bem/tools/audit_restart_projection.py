"""Quantify the initial P1 seed field lost between retained bulk nodes.

Original float64 values and their interpolated coarse P1 field coincide at
retained nodes but need not coincide between them. The trial-shear difference
reported here holds the jump fixed; it does not predict the recomputed static
equilibrium or a full nonlinear event.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model


def seed_field_difference(source_y, source_z, retained_y, tangential_penalty):
    y, z, coarse = (np.asarray(v, dtype=float) for v in (source_y, source_z, retained_y))
    if (
        y.ndim != 1
        or coarse.ndim != 1
        or len(y) < 2
        or len(coarse) < 2
        or z.shape != y.shape
        or not all(np.all(np.isfinite(v)) for v in (y, z, coarse))
        or np.any(np.diff(y) <= 0)
        or np.any(np.diff(coarse) <= 0)
        or not np.array_equal(y[[0, -1]], coarse[[0, -1]])
        or not np.isfinite(tangential_penalty)
        or tangential_penalty <= 0
    ):
        raise ValueError("Require finite increasing P1 grids on the same interval and positive kt")
    knots = np.union1d(y, coarse)
    original = np.interp(knots, y, z)
    retained = np.interp(coarse, y, z)
    interpolated = np.interp(knots, coarse, retained)
    difference = original - interpolated
    a, b = difference[:-1], difference[1:]
    square_integral = np.dot(np.diff(knots), (a * a + a * b + b * b) / 3)
    largest = np.argmax(abs(difference))
    return dict(
        original_seed_nodes=len(y),
        bulk_contact_nodes=len(coarse),
        maximum_seed_difference_mm=float(abs(difference[largest])),
        p1_l2_seed_difference_mm_sqrt_mm=float(np.sqrt(square_integral)),
        largest_error_y_mm=float(knots[largest]),
        maximum_frozen_jump_trial_shear_difference_mpa=float(
            tangential_penalty * abs(difference[largest])
        ),
        scope="Exact P1 difference on the knot union; trial shear holds jumps fixed. No new static solve or full-event result.",
    ), dict(y=knots, original_plastic_slip=original, coarse_p1_plastic_slip=interpolated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--restart-seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a new projection audit output outside protected Tatva")
    metadata_path = args.kernel / "summary.json"
    frozen = {
        str(p.resolve()): fingerprint(p) for p in (Path(__file__), metadata_path, args.restart_seed)
    }
    metadata = json.loads(metadata_path.read_text())
    o = metadata["source_mesh_arguments"]
    environment = environment_record(o["case"])
    keys = ("source_sha256", "case_sha256", "restart_data_sha256")
    if any(environment[key] != metadata["environment"][key] for key in keys):
        raise ValueError("Prepared retained geometry or original physical inputs changed")
    model = load_nodal_model(
        o["case"],
        max_panel_length=o["h"],
        contact_panel_length=o["contact_h"],
        contact_end_length=o["contact_end_h"],
        corner_panel_length=o["corner_h"],
        restart_knot_window=o.get("restart_knot_window"),
        restart_knot_subdivisions=o.get("restart_knot_subdivisions", 1),
    )
    with np.load(args.restart_seed, allow_pickle=False) as data:
        if any(data[key].dtype != np.float64 for key in ("y", "plastic_slip")):
            raise ValueError("Original P1 projection audit requires float64 source arrays")
        result, fields = seed_field_difference(
            data["y"],
            data["plastic_slip"],
            model.reference.moving.starts[model.moving_contact, 1],
            model.reference.contact.tangential_penalty,
        )
    final_environment = environment_record(o["case"])
    if any(fingerprint(path) != value for path, value in frozen.items()) or any(
        final_environment[key] != environment[key] for key in keys
    ):
        raise ValueError("Projection audit inputs changed")
    output.mkdir(parents=True)
    np.savez(output / "seed_fields.npz", **fields)
    write_json(
        output / "summary.json",
        dict(
            status="completed_p1_projection_audit",
            frozen_input_sha256=frozen,
            environment=environment,
            original_tatva_readonly=True,
            **result,
        ),
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

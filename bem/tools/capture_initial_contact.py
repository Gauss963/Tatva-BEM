"""Capture saved FEM static frames with explicit storage-rounding bounds.

This reads completed local reference artifacts. Casting the saved fields to
float64 does not recover a true float64 static solution. Optional BEM initial
rows are read without allocating their complete transient fields.
"""

import argparse
import hashlib
import json
import runpy
import zipfile
from pathlib import Path

import numpy as np

from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent


def first_npz_row(path, key):
    """Read one C-order float64 matrix row through the standard NPY header."""
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }
        if version not in readers:
            raise ValueError("Unsupported NPY header version")
        shape, fortran, dtype = readers[version](stream)
        if fortran or len(shape) != 2 or min(shape) < 1 or dtype != np.float64:
            raise ValueError("Require a nonempty C-order float64 trajectory matrix")
        count = shape[1] * 8
        if archive.getinfo(key + ".npy").file_size != stream.tell() + shape[0] * count:
            raise ValueError("NPZ member size does not match its trajectory header")
        data = stream.read(count)
        if len(data) != count:
            raise ValueError("Incomplete initial trajectory row")
        answer = np.frombuffer(data, dtype=dtype).copy()
        if not np.all(np.isfinite(answer)):
            raise ValueError("Nonfinite initial trajectory field")
        return answer


def rounding_ulp(values):
    """Conservative full-ulp bound, including powers of two and signed zero."""
    values = np.asarray(values)
    if values.dtype.kind != "f" or not np.all(np.isfinite(values)):
        raise ValueError("Storage bounds require finite floating-point arrays")
    ulp = np.spacing(abs(values)).astype(float)
    if not np.all(np.isfinite(ulp)):
        raise ValueError("No finite storage-rounding interval")
    return ulp


def read_static_frame(path, kn, kt, *, expected_step=None):
    import h5py

    path = Path(path)
    if not all(np.isfinite(v) and v > 0 for v in [kn, kt]):
        raise ValueError("Require positive finite physical penalties")
    before = path.stat()
    with h5py.File(path, "r") as stream:
        normal = np.flatnonzero(stream["phase_id"][:] == 1)
        if not len(normal):
            raise ValueError("No saved normal-phase frame")
        frame = int(normal[-1])
        step = int(stream["step_id"][frame])
        if expected_step is not None and step != expected_step:
            raise ValueError("Saved frame is not the recorded normal-phase endpoint")
        group, high = stream["interface"], stream["interface_high_rate"]
        matches = np.flatnonzero((high["phase_id"][:] == 1) & (high["step_id"][:] == step))
        if len(matches) != 1:
            raise ValueError("No unique high-rate frame at the same normal step")
        index = int(matches[0])
        y = np.asarray(group["contact_line_y"][:], float)
        if y.ndim != 1 or len(y) < 2 or not np.all(np.isfinite(y)) or np.any(np.diff(y) <= 0):
            raise ValueError("Require ordered finite contact coordinates")
        np.testing.assert_array_equal(y, high["contact_line_y"][:])
        master, slave = (group[k + "_nodes"][:] for k in ["master", "slave"])
        if any(a.shape != y.shape or a.dtype.kind not in "iu" for a in [master, slave]):
            raise ValueError("Contact-node indices do not match the stored field")
        pairs = [("moving", master), ("stationary", slave)]
        displacement, coordinates = [], []
        for body, ids in pairs:
            coords = stream[body + "/coords"]
            if np.any(ids < 0) or np.any(ids >= len(coords)):
                raise ValueError("Contact-node index outside the body")
            coordinates.append(coords[:][ids])
            displacement.append(stream[body + "/displacement"][frame][ids])
        um, us = displacement
        cm, cs = coordinates
        if any(a.shape != (len(y), 2) for a in [um, us, cm, cs]):
            raise ValueError("Require aligned two-dimensional body fields")
        plastic = group["plastic_slip"][frame]
        np.testing.assert_array_equal(plastic, high["plastic_slip"][index])
        mu, strength, cumulative = (
            high[k][index] for k in ["friction_coefficient", "friction_strength", "cumulative_slip"]
        )
        if any(a.shape != y.shape for a in [plastic, mu, strength, cumulative]):
            raise ValueError("Initial state fields do not match contact nodes")
        ulps = [rounding_ulp(a) for a in [um, us, cm, cs, plastic, mu, strength, cumulative]]
        uerr, serr, cmerr, cserr, zerr, muerr, strengtherr, _ = ulps
        if np.any(mu.astype(float) - muerr <= 0) or np.any(strength < 0):
            raise ValueError(
                "Pressure reconstruction requires positive mu and nonnegative strength"
            )
        un = um[:, 0].astype(float) - us[:, 0].astype(float)
        un -= cs[:, 0].astype(float) - cm[:, 0].astype(float)
        ut = um[:, 1].astype(float) - us[:, 1].astype(float)
        pressure = strength.astype(float) / mu.astype(float)
        shear = kt * (ut - plastic.astype(float))
        pressure_from_u = kn * np.maximum(un, 0)
        ratio_bound = (strengtherr + abs(pressure) * muerr) / (mu.astype(float) - muerr)
        ratio_bound += 4 * np.finfo(float).eps * abs(pressure)
        u_bound = kn * (uerr[:, 0] + serr[:, 0] + cmerr[:, 0] + cserr[:, 0])
        shear_bound = kt * (uerr[:, 1] + serr[:, 1] + zerr)
        if np.any(abs(pressure_from_u - pressure) > u_bound + ratio_bound):
            raise ValueError("Displacement and strength/mu pressures exceed storage bounds")
        names = [
            v.decode() if isinstance(v, bytes) else str(v) for v in stream["history_columns"][:]
        ]
        row = stream["history"][frame]
        platen = row[names.index("loading_face_displacement")]
        actuator = row[names.index("applied_shear_displacement")]
        rounding_ulp(np.array([platen, actuator]))
        arrays = dict(
            y=y,
            pressure=pressure,
            shear=shear,
            normal_jump=un,
            tangential_jump=ut,
            plastic_slip=plastic.astype(float),
            cumulative_slip=cumulative.astype(float),
            pressure_from_displacement=pressure_from_u,
            pressure_storage_bound=ratio_bound,
            displacement_pressure_storage_bound=u_bound,
            shear_storage_bound=shear_bound,
            platen=np.asarray(platen, dtype=float),
            actuator=np.asarray(actuator, dtype=float),
        )
        provenance = dict(
            source=str(path.resolve()),
            source_bytes=before.st_size,
            source_mtime_ns=before.st_mtime_ns,
            frame=frame,
            step_id=step,
            high_rate_frame=index,
            pressure_strength_vs_displacement_maximum_difference_mpa=float(
                np.max(abs(pressure - pressure_from_u))
            ),
            maximum_pressure_storage_bound_mpa=float(np.max(ratio_bound)),
            maximum_displacement_pressure_storage_bound_mpa=float(np.max(u_bound)),
            maximum_shear_storage_bound_mpa=float(np.max(shear_bound)),
            healed_memory_maximum_mm=float(np.max(cumulative)),
            stored_dtypes={
                k: str(v.dtype)
                for k, v in zip(
                    [
                        "moving_u",
                        "stationary_u",
                        "moving_coordinates",
                        "stationary_coordinates",
                        "plastic_slip",
                        "friction_coefficient",
                        "friction_strength",
                        "cumulative_slip",
                    ],
                    [um, us, cm, cs, plastic, mu, strength, cumulative],
                    strict=True,
                )
            },
        )
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Read-only FEM file changed during capture")
    return arrays, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fem_runs", type=Path, nargs="+")
    parser.add_argument("--bem-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a fresh output outside protected Tatva")
    paths = [
        Path(__file__),
        TOOLS / "compare_fem_mesh.py",
        TOOLS / "compare_refined_fem_history.py",
    ]
    for folder in args.fem_runs:
        paths.extend(
            folder / name
            for name in ["summary.json", "resolved_run_config.json", "reference_manifest.json"]
        )
    paths.extend(folder / "summary.json" for folder in args.bem_runs)
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    mesh = runpy.run_path(str(TOOLS / "compare_fem_mesh.py"))
    validate = runpy.run_path(str(TOOLS / "compare_refined_fem_history.py"))[
        "validate_physical_parameters"
    ]
    output.mkdir(parents=True)
    records = []
    for i, folder in enumerate(args.fem_runs):
        summary = json.loads((folder / "summary.json").read_text())
        config = json.loads((folder / "resolved_run_config.json").read_text())
        if not (
            summary["status"] == "completed"
            and summary["full_original_horizon"]
            and summary["source_and_protected_input_unchanged"]
            and summary["computational_dtype"] == config["dtype"] == "float64"
            and config["normal_phase_mode"] == "static"
            and config["healing_fraction"] == 1
            and config["reset_slip_weakening_at_shear_start"]
        ):
            raise ValueError("Require a completed float64 FEM event with healed static preload")
        if fingerprint(folder / "reference_manifest.json") != summary["reference_manifest_sha256"]:
            raise ValueError("Completed FEM reference manifest changed")
        arrays, provenance = read_static_frame(
            folder / "simulation.h5",
            config["normal_penalty"],
            config["tangential_penalty"],
            expected_step=summary["summary"]["pressure_steps"],
        )
        target = output / f"fem-{i}-{folder.name}.npz"
        np.savez(target, **arrays)
        comparisons = []
        for primal in args.bem_runs:
            bem = json.loads((primal / "summary.json").read_text())
            if (
                bem["status"] != "completed_requested_horizon"
                or bem["static_initialization"] != "fem-active-set"
                or bem["case"]["source"]["revision"] != summary["source_revision"]
            ):
                raise ValueError("Require a complete BEM run with the Tatva healed initialization")
            validate(bem, summary, config)
            with np.load(primal / "trajectory.npz", allow_pickle=False) as data:
                by = data["contact_y"].copy()
            fields = {k: first_npz_row(primal / "trajectory.npz", k) for k in ["pressure", "shear"]}
            comparisons.append(
                dict(
                    run=str(primal.resolve()),
                    **{
                        k: mesh["error_report"](arrays["y"], arrays[k][None], by, value[None])
                        for k, value in fields.items()
                    },
                    initial_row_sha256={
                        k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in fields.items()
                    },
                )
            )
        records.append(
            dict(
                **provenance,
                snapshot=str(target),
                snapshot_sha256=fingerprint(target),
                old_bem_initial_comparisons=comparisons,
            )
        )
    if any(fingerprint(p) != v for p, v in frozen.items()):
        raise ValueError("Capture source or metadata changed")
    write_json(
        output / "summary.json",
        dict(
            status="captured_saved_static_frames_with_storage_bounds",
            records=records,
            frozen_input_sha256=frozen,
            original_tatva_readonly=True,
            same_accuracy_fem_advantage_verified=False,
            scope="Saved FEM static frames cast to float64; explicit full-ulp storage bounds, not true float64 static solutions or dynamic convergence",
        ),
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

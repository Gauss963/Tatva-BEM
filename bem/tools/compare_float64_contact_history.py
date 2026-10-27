"""Compare true float64 endpoint pressure and signed shear over a complete event.

The original FEM integration is unchanged; the explicit output adapter observes
its current displacement and committed plastic offset. Friction strength is
compared separately from signed shear. Integer phase steps set the shear clock.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
COMPARE = runpy.run_path(str(TOOLS / "compare_refined_fem_history.py"))
DIGEST = runpy.run_path(str(TOOLS / "replay_resolvent.py"))["digest_file"]
FIELDS = (
    "pressure",
    "elastic_shear",
    "plastic_slip",
    "cumulative_slip",
    "friction_coefficient",
    "friction_strength",
)


def observation_grid(stream, summary):
    """Use only actual shear observations, after the source's normal handoff."""
    group, high = stream["contact_observations"], stream["interface_high_rate"]
    count = int(stream.attrs["saved_interface_frames"])
    if (
        group.attrs["schema_version"] != 1
        or int(group.attrs["written_frames"]) != count
        or group["phase_id"].shape != (count,)
        or group["step_id"].dtype.kind not in "iu"
        or group["absolute_step"].dtype.kind not in "iu"
    ):
        raise ValueError("Require complete integer-clock contact observations")
    phases, steps, absolute = (group[k][:] for k in ["phase_id", "step_id", "absolute_step"])
    if not np.array_equal(phases, high["phase_id"][:]) or not np.array_equal(
        steps, high["step_id"][:]
    ):
        raise ValueError("Observer clock differs from the original high-rate writer")
    dt = summary["summary"]["dt"]
    normal_steps = summary["summary"]["pressure_steps"]
    if (
        not np.isfinite(dt)
        or dt <= 0
        or np.any(~np.isin(phases, [0, 1, 2]))
        or np.any(np.diff(phases) < 0)
        or np.any(steps < 0)
        or np.any((phases == 0) & ((np.arange(count) != 0) | (steps != 0)))
    ):
        raise ValueError("Invalid actual FEM step clock")
    expected = steps + np.where(phases == 2, normal_steps, 0)
    if (
        not np.array_equal(absolute, expected)
        or group["time_s"].dtype != np.float64
        or not np.array_equal(group["time_s"][:], expected.astype(float) * dt)
    ):
        raise ValueError("Observation times disagree with integer source steps")
    indices = np.flatnonzero(phases == 2)
    if (
        len(indices) < 2
        or np.any(np.diff(steps[indices]) <= 0)
        or steps[indices][0] <= 0
        or steps[indices][-1] != summary["summary"]["shear_steps"]
    ):
        raise ValueError("Require ordered shear observations through the final source step")
    y = group["y"][:]
    if (
        group["y"].dtype != np.float64
        or y.ndim != 1
        or len(y) < 2
        or not np.all(np.isfinite(y))
        or np.any(np.diff(y) <= 0)
    ):
        raise ValueError("Require the true float64 ordered contact coordinate grid")
    for key in FIELDS:
        if group[key].dtype != np.float64 or group[key].shape != (count, len(y)):
            raise ValueError("Require true float64 observations on the complete interface")
    # Do not relabel the last normal row as shear t=0: the original source
    # subsequently resets weakening memory at the normal-to-shear handoff.
    return indices, steps[indices].astype(float) * dt, y


def validate_case(bem, fem, config):
    COMPARE["validate_physical_parameters"](bem, fem, config)
    legacy, geometry = fem["summary"]["legacy"], bem["case"]["geometry"]
    material = bem["case"]["material"]
    expected_material = dict(
        young=material["young"], poisson=material["poisson"], rho=material["density"]
    )
    for body in ["moving", "stationary"]:
        actual = legacy["materials"][body + "-block"]
        if dict(young=actual["E"], poisson=actual["nu"], rho=actual["rho"]) != expected_material:
            raise ValueError("Finite-body material parameters differ")
        shape = legacy["geometry"][body]
        if (
            shape["origin"] != geometry[body + "_origin"]
            or shape["dimensions"] != geometry[body + "_size"]
        ):
            raise ValueError("Finite-body dimensions or positions differ")
    for bk, fk in [
        ("chamfer_length", "moving_leading_chamfer_along_fault"),
        ("chamfer_depth", "moving_leading_chamfer_perpendicular"),
    ]:
        if geometry[bk] != fem["summary"][fk]:
            raise ValueError("Contact endpoint geometry differs")
    if bem["case"]["loading"]["restart_actuator"] != fem["actual_restart"]["actuator_mm"]:
        raise ValueError("Restart actuator differs")


def bem_field(data, name):
    if name == "friction_strength":
        return data["friction_coefficient"] * data["pressure"]
    return data["shear" if name == "elastic_shear" else name]


def compare_fields(stream, fem, data, bem, *, row_chunk=128):
    indices, ft, fy = observation_grid(stream, fem)
    bt, by = data["time"], data["contact_y"]
    expected = np.arange(bem["completed_steps"] + 1) * bem["dt_s"]
    if bt.dtype != np.float64 or not np.array_equal(bt, expected):
        raise ValueError("BEM trajectory does not match its recorded integer step clock")
    errors = {}
    for key in FIELDS:
        actual = bem_field(data, key)
        if actual.dtype != np.float64:
            raise ValueError("BEM contact fields must retain true float64 storage")
        errors[key] = COMPARE["compare_saved_p1_fields"](
            (ft, fy, stream["contact_observations"][key][indices]),
            (bt, by, actual),
            row_chunk=row_chunk,
        )
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bem_run", type=Path)
    parser.add_argument("fem_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-chunk", type=int, default=128)
    args = parser.parse_args()
    begun = perf_counter()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a fresh comparison outside protected Tatva")
    paths = [
        Path(__file__),
        *(
            TOOLS / name
            for name in [
                "compare_refined_fem_history.py",
                "compare_fem_mesh.py",
                "fem_reference_io.py",
                "replay_resolvent.py",
                "fem_contact_observations.py",
            ]
        ),
        args.bem_run / "summary.json",
        args.fem_run / "summary.json",
        args.fem_run / "resolved_run_config.json",
        args.fem_run / "reference_manifest.json",
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    bem, fem, config, manifest = (
        json.loads(p.read_text())
        for p in [
            args.bem_run / "summary.json",
            args.fem_run / "summary.json",
            args.fem_run / "resolved_run_config.json",
            args.fem_run / "reference_manifest.json",
        ]
    )
    observer = fem.get("contact_observations")
    if (
        bem["status"] != "completed_requested_horizon"
        or bem["duration_s"] != 0.045
        or not bem["all_history_retained"]
        or fem["status"] != "completed"
        or not fem["full_original_horizon"]
        or fem["computational_dtype"] != "float64"
        or config["dtype"] != "float64"
        or not fem["source_and_protected_input_unchanged"]
        or bem["case"]["source"]["revision"] != fem["source_revision"]
        or observer is None
        or observer != manifest.get("contact_observations")
        or observer["sha256"] != fingerprint(TOOLS / "fem_contact_observations.py")
        or fingerprint(args.fem_run / "reference_manifest.json") != fem["reference_manifest_sha256"]
    ):
        raise ValueError("Require complete source-frozen events with verified float64 observations")
    validate_case(bem, fem, config)
    trajectory, hdf = args.bem_run / "trajectory.npz", args.fem_run / "simulation.h5"
    hashes = {str(p.resolve()): DIGEST(p) for p in [trajectory, hdf]}
    with h5py.File(hdf, "r") as stream, np.load(trajectory, allow_pickle=False) as data:
        if abs(data["time"][-1] - 0.045) > 1e-12:
            raise ValueError("BEM trajectory does not cover the original horizon")
        errors = compare_fields(stream, fem, data, bem, row_chunk=args.row_chunk)
    if any(DIGEST(p) != value for p, value in hashes.items()) or any(
        fingerprint(p) != value for p, value in frozen.items()
    ):
        raise ValueError("Comparison sources or datasets changed")
    output.mkdir(parents=True)
    write_json(
        output / "summary.json",
        dict(
            status="completed_float64_contact_space_time_comparison",
            bem_run=str(args.bem_run.resolve()),
            fem_run=str(args.fem_run.resolve()),
            frozen_input_sha256=frozen,
            dataset_sha256=hashes,
            field_errors=errors,
            clock="Shear phase integer step times actual dt; no fitted shift or float32 timestamp subtraction",
            omitted_initial_interval="Before the first actual shear observation; pre-handoff normal state is not relabelled",
            shear_scope="Signed endpoint kt*(jump - committed plastic offset), zero when open; BEM uses last constitutive microstep",
            friction_strength_scope="Separate positive bound mu*pressure, never substituted for signed shear",
            integration_scope="Exact norms of all saved P1 interpolants on their whole shared interface and time interval",
            stability_scope="Saved output does not bound unsaved oscillations or prove nonlinear stability",
            comparison_seconds_including_hashes=perf_counter() - begun,
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
        ),
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

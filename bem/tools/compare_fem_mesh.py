"""Compare a completed FEM spatial control at the original event-16 time step.

The h2 baseline must already have passed the archived-field verification.
Only mesh size and the independently checked P1 restart projection may differ.
Stored histories use identical step IDs, with no temporal alignment by fitting.
Spatial and space-time errors integrate the saved P1 interpolants exactly;
they do not certify unsaved oscillations or continuous-solution convergence.
"""

import argparse
import json
import runpy
from pathlib import Path

import h5py
import numpy as np

from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
FIELDS = ("plastic_slip", "cumulative_slip", "friction_coefficient", "friction_strength")
READER = runpy.run_path(str(TOOLS / "fem_reference_io.py"))
ALIGN = runpy.run_path(str(TOOLS / "compare_fem_reference.py"))["aligned_shear_rows"]
SEED = runpy.run_path(str(TOOLS / "run_fem_reference.py"))


def interpolate_nodes(nodes, values, knots):
    """Interpolate all saved rows on common knots without extrapolation."""
    nodes, values, knots = (np.asarray(v, dtype=float) for v in (nodes, values, knots))
    if (
        nodes.ndim != 1
        or knots.ndim != 1
        or len(nodes) < 2
        or len(knots) < 2
        or values.ndim != 2
        or values.shape[1] != len(nodes)
        or not all(np.all(np.isfinite(v)) for v in (nodes, values, knots))
        or np.any(np.diff(nodes) <= 0)
        or np.any(np.diff(knots) <= 0)
        or knots[0] < nodes[0]
        or knots[-1] > nodes[-1]
    ):
        raise ValueError("Require finite P1 rows on ordered nodes without extrapolation")
    left = np.clip(np.searchsorted(nodes, knots, side="right") - 1, 0, len(nodes) - 2)
    fraction = (knots - nodes[left]) / (nodes[left + 1] - nodes[left])
    return values[:, left] * (1 - fraction) + values[:, left + 1] * fraction


def p1_inner(a, b, widths):
    """Exact spatial inner products, one for each leading row."""
    return np.sum(
        widths
        * (
            2 * a[..., :-1] * b[..., :-1]
            + a[..., :-1] * b[..., 1:]
            + a[..., 1:] * b[..., :-1]
            + 2 * a[..., 1:] * b[..., 1:]
        )
        / 6,
        axis=-1,
    )


def error_report(reference_y, reference, actual_y, actual, *, time=None):
    """P1-space / P1-time L2 differences with explicit zero-reference behavior."""
    if not np.array_equal(np.asarray(reference_y)[[0, -1]], np.asarray(actual_y)[[0, -1]]):
        raise ValueError("Both P1 fields must cover exactly the same contact interval")
    knots = np.union1d(reference_y, actual_y)
    a = interpolate_nodes(reference_y, reference, knots)
    b = interpolate_nodes(actual_y, actual, knots)
    if a.shape != b.shape or len(a) < 1:
        raise ValueError("P1 fields must have matching nonempty saved rows")
    widths, delta = np.diff(knots), b - a
    target2, error2 = p1_inner(a, a, widths), p1_inner(delta, delta, widths)

    def ratio(numerator, denominator):
        return None if denominator == 0 else float(np.sqrt(numerator / denominator))

    result = dict(
        final_relative_p1_l2=ratio(error2[-1], target2[-1]),
        final_absolute_p1_l2=float(np.sqrt(error2[-1])),
        final_maximum_absolute_difference=float(np.max(abs(delta[-1]))),
        final_reference_maximum=float(np.max(a[-1])),
        final_actual_maximum=float(np.max(b[-1])),
        final_reference_p1_mean=float(
            np.sum(widths * (a[-1, :-1] + a[-1, 1:]) / 2) / np.sum(widths)
        ),
        final_actual_p1_mean=float(np.sum(widths * (b[-1, :-1] + b[-1, 1:]) / 2) / np.sum(widths)),
        maximum_saved_absolute_difference=float(np.max(abs(delta))),
    )
    if time is not None:
        time = np.asarray(time, dtype=float)
        if (
            time.shape != (len(a),)
            or len(time) < 2
            or not np.all(np.isfinite(time))
            or np.any(np.diff(time) <= 0)
        ):
            raise ValueError("Space-time errors require strictly increasing aligned sample times")

        def space_time(norm2, values):
            cross = p1_inner(values[:-1], values[1:], widths)
            return np.sum(np.diff(time) / 3 * (norm2[:-1] + cross + norm2[1:]))

        result.update(
            relative_space_time_l2=ratio(space_time(error2, delta), space_time(target2, a)),
            absolute_space_time_l2=float(np.sqrt(space_time(error2, delta))),
            time_interpretation="P1 interpolants of saved rows; unsaved steps are not bounded",
        )
    return result


def validate_mesh_pair(baseline, refined):
    """Reject a nominal mesh control that changes physics, dt, source or precision."""
    for record in (baseline, refined):
        summary, config, manifest = (record[k] for k in ("summary", "config", "manifest"))
        if not (
            summary.get("status") == "completed"
            and summary.get("full_original_horizon")
            and summary.get("source_and_protected_input_unchanged")
            and summary.get("computational_dtype") == config.get("dtype") == "float64"
            and summary.get("jax_enable_x64")
            and summary.get("source_revision") == manifest["source"]["revision"]
            and summary.get("source_archive_sha256") == manifest["source"]["archive_sha256"]
            and summary["summary"]["dt"] == config["time_step_override"]
            and summary["summary"]["mesh_size"] == config["mesh_size"]
            and summary["summary"]["pressure_time"] == config["normal_phase_time"] == 0.002
            and summary["summary"]["shear_time"] == config["shear_phase_time"] == 0.045
        ):
            raise ValueError(
                "Require completed original-horizon float64 runs with verified sources"
            )
    a, b = baseline["config"], refined["config"]
    if not (a["mesh_size"] == 2 and 0 < b["mesh_size"] < a["mesh_size"]):
        raise ValueError("Require the original h2 baseline and a strictly finer spatial control")
    excluded = {"mesh_size", "restart_from_file"}
    if {k: v for k, v in a.items() if k not in excluded} != {
        k: v for k, v in b.items() if k not in excluded
    }:
        raise ValueError("Only mesh size and the verified restart path may differ")
    ma, mb = baseline["manifest"], refined["manifest"]
    if not (
        ma["source"] == mb["source"]
        and ma["protected_input_sha256"] == mb["protected_input_sha256"]
        and ma["source_restart"] == mb["source_restart"]
        and ma["restart_projection"] is None
        and mb["restart_projection"] is not None
        and mb["restart_projection"]["source_plastic_dtype"] == "float64"
        and mb["restart_projection"]["source_restart"] == ma["source_restart"]
        and ma["actual_restart"] == ma["source_restart"]
    ):
        raise ValueError(
            "Spatial controls must use the same archived input and float64 source seed"
        )


def read_final_fields(path, config):
    """Recover float64 endpoint fields directly from final_state, not saved float32 rows."""
    with h5py.File(path, "r") as stream:
        moving, stationary = (
            np.asarray(stream[f"{body}/coords"][:], dtype=float)
            for body in ("moving", "stationary")
        )
        interface, final = stream["interface"], stream["final_state"]
        master, slave = (
            np.asarray(interface[f"{kind}_nodes"][:], dtype=int) for kind in ("master", "slave")
        )
        y = np.asarray(interface["contact_line_y"][:], dtype=float)
        if any(final[key].dtype != np.float64 for key in ("u", "plastic_slip", "cumulative_slip")):
            raise ValueError("Endpoint control requires true float64 final_state arrays")
        u, z, c = (final[key][:] for key in ("u", "plastic_slip", "cumulative_slip"))
        if (
            u.shape != (2 * (len(moving) + len(stationary)),)
            or z.shape != y.shape
            or c.shape != y.shape
            or master.shape != y.shape
            or slave.shape != y.shape
            or np.any(master < 0)
            or np.any(master >= len(moving))
            or np.any(slave < 0)
            or np.any(slave >= len(stationary))
            or not np.array_equal(moving[master, 1], y)
            or not np.array_equal(stationary[slave, 1], y)
        ):
            raise ValueError("Final-state node pairs and body displacement layout do not align")
        um, us = u[: 2 * len(moving)].reshape(-1, 2), u[2 * len(moving) :].reshape(-1, 2)
        gap = stationary[slave, 0] - moving[master, 0]
        pressure = config["normal_penalty"] * np.maximum(um[master, 0] - us[slave, 0] - gap, 0)
        trial = config["tangential_penalty"] * (um[master, 1] - us[slave, 1] - z)
        fields = dict(
            plastic_slip=z, cumulative_slip=c, pressure=pressure, elastic_trial_shear=trial
        )
        if not all(np.all(np.isfinite(value)) for value in fields.values()):
            raise ValueError("Nonfinite final-state fields")
        return y, fields


def first_crossing(time, values, threshold):
    indices = np.flatnonzero(np.asarray(values) > threshold)
    return float(time[indices[0]]) if len(indices) else None


def load_record(path):
    return dict(
        path=path.resolve(),
        **{
            key: json.loads((path / name).read_text())
            for key, name in (
                ("summary", "summary.json"),
                ("config", "resolved_run_config.json"),
                ("manifest", "reference_manifest.json"),
            )
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("refined", type=Path)
    parser.add_argument("--field-verification", type=Path, required=True)
    parser.add_argument("--bem-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "outputs"):
        raise ValueError("Choose a new comparison folder in this checkout's bem/outputs")
    baseline, refined = (load_record(path) for path in (args.baseline, args.refined))
    validate_mesh_pair(baseline, refined)
    frozen_paths = [
        Path(__file__),
        TOOLS / "fem_reference_io.py",
        TOOLS / "compare_fem_reference.py",
        TOOLS / "run_fem_reference.py",
        args.field_verification,
    ]
    for record in (baseline, refined):
        frozen_paths.extend(
            record["path"] / name
            for name in ("summary.json", "resolved_run_config.json", "reference_manifest.json")
        )
    for run in args.bem_runs:
        frozen_paths.extend([run / "summary.json", run / "trajectory.npz"])
    frozen = {str(p.resolve()): fingerprint(p) for p in frozen_paths}
    verification = json.loads(args.field_verification.read_text())
    if not (
        verification["original_fem_reference_verified"]
        and Path(verification["rerun"]["path"]).resolve() == baseline["path"] / "simulation.h5"
        and verification["frozen_input_sha256"][str(baseline["path"] / "summary.json")]
        == frozen[str(baseline["path"] / "summary.json")]
    ):
        raise ValueError("The h2 baseline must have passed the archived-field verification")
    traces, final_fields, seed_checks, files, restart_records = [], [], [], [], []
    for record in (baseline, refined):
        for path, digest in record["manifest"]["protected_input_sha256"].items():
            if fingerprint(path) != digest:
                raise ValueError("Archived case input changed after the simulation")
            frozen[str(Path(path).resolve())] = digest
        if (
            fingerprint(record["path"] / "reference_manifest.json")
            != record["summary"]["reference_manifest_sha256"]
        ):
            raise ValueError("FEM manifest changed after the simulation")
        for key in ("source_restart", "actual_restart"):
            saved = record["manifest"][key]
            if SEED["restart_seed_record"](saved["path"]) != saved:
                raise ValueError("FEM source or projected restart changed after the simulation")
            restart_records.append(saved)
        path = record["path"] / "simulation.h5"
        before = path.stat()
        trace = READER["read_fem_trace"](path, fields=FIELDS)
        if record is baseline and any(
            trace["provenance"][key] != verification["rerun"][key]
            for key in ("bytes", "mtime_ns", "saved_frames")
        ):
            raise ValueError("The verified h2 trace changed")
        if record is refined:
            source = record["manifest"]["source_restart"]
            expected, actuator, projection = SEED["project_restart_seed"](
                source["path"], trace["y"], source
            )
            with h5py.File(record["config"]["restart_from_file"], "r") as stream:
                group = stream["final_state"]
                if (
                    group["plastic_slip"].dtype != np.float64
                    or not np.array_equal(group["plastic_slip"][:], expected)
                    or group.attrs["applied_shear_displacement"] != actuator
                    or projection != record["manifest"]["restart_projection"]
                ):
                    raise ValueError("Refined restart is not the original float64 P1 projection")
        else:
            with h5py.File(record["config"]["restart_from_file"], "r") as stream:
                expected = stream["final_state/plastic_slip"][:]
        # Report initialization changes. A correct input projection does not
        # imply the subsequent static normal solver preserves every seed value.
        indices = [0, np.flatnonzero(trace["phase_id"] == 2)[0]]
        seed_checks.append(
            dict(
                input_projection_verified=True,
                first_frame_minus_input_maximum_mm=float(
                    np.max(abs(trace["interface_fields"]["plastic_slip"][indices[0]] - expected))
                ),
                first_shear_minus_input_maximum_mm=float(
                    np.max(abs(trace["interface_fields"]["plastic_slip"][indices[1]] - expected))
                ),
                caveat="Saved initial fields are float32; static initialization may also change plastic slip",
            )
        )
        traces.append(trace)
        final_fields.append(read_final_fields(path, record["config"]))
        files.append((path, before))
    rows = ALIGN(*traces)
    time = traces[0]["history"]["time"][rows[0]]
    if not np.array_equal(time, traces[1]["history"]["time"][rows[1]]):
        raise ValueError("Same step IDs must also have exactly matching saved times")
    history = {}
    if set(traces[0]["history"]) != set(traces[1]["history"]):
        raise ValueError("FEM history columns differ")
    for key in traces[0]["history"]:
        a, b = (t["history"][key][r] for t, r in zip(traces, rows, strict=True))
        # A spatially constant function reduces the P1 norm to the time norm.
        history[key] = error_report(
            [0, 1],
            np.repeat(a[:, None], 2, axis=1),
            [0, 1],
            np.repeat(b[:, None], 2, axis=1),
            time=time,
        )
    fields = {}
    for key in (*FIELDS, "pressure_reconstructed"):
        values = []
        for trace, indices in zip(traces, rows, strict=True):
            data = trace["interface_fields"]
            if key == "pressure_reconstructed":
                mu = data["friction_coefficient"][indices]
                if np.any(mu <= 0):
                    raise ValueError("Reconstructing pressure requires strictly positive stored mu")
                values.append(data["friction_strength"][indices] / mu)
            else:
                values.append(data[key][indices])
        fields[key] = error_report(traces[0]["y"], values[0], traces[1]["y"], values[1], time=time)
    endpoint = {
        key: error_report(
            final_fields[0][0],
            final_fields[0][1][key][None],
            final_fields[1][0],
            final_fields[1][1][key][None],
        )
        for key in final_fields[0][1]
    }
    bem_records = []
    for run in args.bem_runs:
        summary = json.loads((run / "summary.json").read_text())
        if summary["status"] != "completed_requested_horizon":
            raise ValueError("BEM endpoint comparisons require completed requested horizons")
        with np.load(run / "trajectory.npz", allow_pickle=False) as data:
            if abs(float(data["time"][-1]) - 0.045) > 1e-10:
                raise ValueError("BEM endpoint must cover the same nominal 45 ms shear phase")
            matches = []
            for y, values in final_fields:
                matches.append(
                    {
                        key: error_report(y, values[key][None], data["contact_y"], data[bkey][-1:])
                        for key, bkey in (
                            ("plastic_slip", "plastic_slip"),
                            ("cumulative_slip", "cumulative_slip"),
                            ("pressure", "pressure"),
                            ("elastic_trial_shear", "shear"),
                        )
                    }
                )
        bem_records.append(
            dict(
                run=str(run.resolve()),
                versus_h2=matches[0],
                versus_refined=matches[1],
                schedule=summary.get("schedule"),
                dt_s=summary["dt_s"],
            )
        )
    for path, before in files:
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("FEM trajectory changed during the comparison")
    if any(fingerprint(path) != value for path, value in frozen.items()):
        raise ValueError("Comparison code or input artifacts changed")
    if any(SEED["restart_seed_record"](saved["path"]) != saved for saved in restart_records):
        raise ValueError("FEM source or projected restart changed during the comparison")
    events = []
    for record, trace, indices in zip((baseline, refined), traces, rows, strict=True):
        t = trace["history"]["time"][indices] - record["config"]["normal_phase_time"]
        events.append(
            dict(
                mesh_mm=record["config"]["mesh_size"],
                first_saved_slip_above_1e_7_mm_at_s=first_crossing(
                    t, trace["history"]["max_slip"][indices], 1e-7
                ),
                first_saved_actuator_hold_s=first_crossing(
                    t, trace["history"]["shear_loading_stopped"][indices], 0.5
                ),
                actual_final_shear_time_s=float(t[-1]),
                timing_seconds=record["summary"]["timing_seconds"],
            )
        )
    result = dict(
        status="completed_same_dt_spatial_comparison",
        fem_mesh_convergence_verified=False,
        same_accuracy_fem_advantage_verified=False,
        compared_shear_rows=len(time),
        exact_same_steps_and_saved_times=True,
        actual_dt_s=baseline["summary"]["summary"]["dt"],
        baseline=traces[0]["provenance"],
        refined=traces[1]["provenance"],
        initialization=seed_checks,
        events=events,
        scalar_history=history,
        stored_interface_fields=fields,
        true_float64_final_state=endpoint,
        bem_endpoints=bem_records,
        frozen_input_sha256=frozen,
        scope="Two FEM meshes at fixed original dt; no Richardson order, continuum error bound or speed advantage follows from two grids",
        pressure_scope="Saved pressure is strength/mu from float32 values; final pressure is kn*positive(normal jump minus geometric gap) from float64 u",
        shear_scope="Endpoint elastic_trial_shear is kt*(tangential jump-z), not an independently re-evaluated return map",
        timing_scope="Raw run times retained; known scheduling pauses and overlap controls must be assessed separately",
    )
    args.output.mkdir(parents=True)
    write_json(args.output / "summary.json", result)
    print(args.output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

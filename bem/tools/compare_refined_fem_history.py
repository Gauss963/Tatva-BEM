"""Compare saved FEM/BEM fields on the union of physical space/time knots.

No fitted time shift or extrapolation. The integrals are exact for the saved
tensor-product P1 interpolants; unsaved oscillations remain unbounded.
FEM tractions are not inferred from friction strength: only pressure is
reconstructed from its recorded positive friction coefficient.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
MESH = runpy.run_path(str(TOOLS / "compare_fem_mesh.py"))
READ = runpy.run_path(str(TOOLS / "fem_reference_io.py"))
FIELDS = ("plastic_slip", "cumulative_slip", "friction_coefficient", "friction_strength")


def validate_field(time, y, values):
    time, y, values = (np.asarray(v, dtype=float) for v in (time, y, values))
    if (
        time.ndim != 1
        or y.ndim != 1
        or len(time) < 2
        or len(y) < 2
        or values.shape != (len(time), len(y))
        or not all(np.all(np.isfinite(v)) for v in (time, y, values))
        or np.any(np.diff(time) <= 0)
        or np.any(np.diff(y) <= 0)
    ):
        raise ValueError("Require finite P1 fields on strictly increasing space/time grids")
    return time, y, values


def sample_rows(time, y, values, requested_time, knots):
    left = np.clip(np.searchsorted(time, requested_time, side="right") - 1, 0, len(time) - 2)
    fraction = (requested_time - time[left]) / (time[left + 1] - time[left])
    rows = values[left] * (1 - fraction[:, None]) + values[left + 1] * fraction[:, None]
    return MESH["interpolate_nodes"](y, rows, knots)


def compare_saved_p1_fields(reference, actual, *, row_chunk=128):
    a, b = (validate_field(*field) for field in (reference, actual))
    if not isinstance(row_chunk, int) or row_chunk < 1:
        raise ValueError("Require positive comparison row chunks")
    if not np.array_equal(a[1][[0, -1]], b[1][[0, -1]]):
        raise ValueError("Fields must cover the same complete contact interval")
    start, stop = max(a[0][0], b[0][0]), min(a[0][-1], b[0][-1])
    if stop <= start:
        raise ValueError("Saved fields have no time interval in common")
    union = np.union1d(a[0], b[0])
    time = np.union1d([start, stop], union[(union > start) & (union < stop)])
    y = np.union1d(a[1], b[1])
    widths, inner = np.diff(y), MESH["p1_inner"]
    error_integral, reference_integral, maximum = 0.0, 0.0, 0.0
    last_error, last_reference = 0.0, 0.0
    for first in range(0, len(time) - 1, row_chunk):
        tt = time[first : first + row_chunk + 1]
        aa, bb = (sample_rows(*field, tt, y) for field in (a, b))
        delta = bb - aa
        maximum = max(maximum, float(np.max(abs(delta))))
        en, rn = inner(delta, delta, widths), inner(aa, aa, widths)
        error_integral += float(
            np.sum(np.diff(tt) / 3 * (en[:-1] + inner(delta[:-1], delta[1:], widths) + en[1:]))
        )
        reference_integral += float(
            np.sum(np.diff(tt) / 3 * (rn[:-1] + inner(aa[:-1], aa[1:], widths) + rn[1:]))
        )
        last_error, last_reference = float(en[-1]), float(rn[-1])
    return dict(
        relative_space_time_p1_l2=None
        if reference_integral == 0
        else float(np.sqrt(error_integral / reference_integral)),
        absolute_space_time_p1_l2=float(np.sqrt(error_integral)),
        reference_squared_space_time_integral=reference_integral,
        error_squared_space_time_integral=error_integral,
        maximum_saved_interpolant_absolute_difference=maximum,
        final_shared_time_relative_p1_l2=None
        if last_reference == 0
        else float(np.sqrt(last_error / last_reference)),
        common_time_interval_s=[float(start), float(stop)],
        reference_saved_interval_s=[float(a[0][0]), float(a[0][-1])],
        actual_saved_interval_s=[float(b[0][0]), float(b[0][-1])],
        temporal_union_knots=len(time),
        spatial_union_knots=len(y),
        all_saved_knots_in_overlap_retained=True,
        fitted_time_shift_s=0.0,
        extrapolation_used=False,
        scope="Exact norms of saved P1 interpolants on the common interval; no unsaved-step error bound",
    )


def fem_step_clock(trace, summary, config):
    indices = np.flatnonzero(trace["phase_id"] == 2)
    steps = trace["step_id"][indices]
    dt = summary["summary"]["dt"]
    if len(steps) < 2 or np.any(np.diff(steps) <= 0) or dt != config["time_step_override"]:
        raise ValueError("Require ordered shear step IDs and the recorded actual FEM dt")
    absolute = summary["summary"]["pressure_steps"] * dt + steps.astype(float) * dt
    # The file stores timestamps as float32. Reconstruct from integer steps
    # only after verifying that this exact clock rounds to every saved value.
    np.testing.assert_array_equal(
        absolute.astype(np.float32).astype(float), trace["history"]["time"][indices]
    )
    return indices, absolute - config["normal_phase_time"]


def validate_physical_parameters(bem, fem, config):
    resolved = fem["summary"]
    expected = dict(
        normal_penalty=resolved["normal_penalty"],
        tangential_penalty=resolved["tangential_penalty"],
        static_friction=resolved["legacy"]["friction"]["mu_s"],
        kinetic_friction=resolved["mu_k"],
        characteristic_slip=resolved["critical_slip"],
    )
    if bem["case"]["contact"] != expected:
        raise ValueError("Resolved contact physical parameters differ")
    if (
        config["friction_law"] != "slip-weakening"
        or config["mu_s_start_fraction"] != 1
        or config["mu_s_end_fraction"] != 1
        or any(
            config[key] != 0
            for key in [
                "pw_length",
                "loading_edge_nucleation_length",
                "leading_edge_creep_length",
                "leading_edge_guard_length",
                "leading_edge_tangential_taper_length",
            ]
        )
    ):
        raise ValueError("Require the same uniform slip-weakening contact law")
    for bk, fk in [
        ("normal_pressure", "normal_stress"),
        ("spring_stiffness", "shear_loading_stiffness"),
        ("actuator_rate", "shear_post_ramp_rate"),
    ]:
        if bem["case"]["loading"][bk] != resolved[fk]:
            raise ValueError("Physical loading parameters differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bem_run", type=Path)
    parser.add_argument("fem_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-chunk", type=int, default=128)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a fresh comparison outside protected Tatva")
    begun = perf_counter()
    paths = [
        Path(__file__),
        TOOLS / "compare_fem_mesh.py",
        TOOLS / "fem_reference_io.py",
        TOOLS / "replay_resolvent.py",
        args.bem_run / "summary.json",
        args.bem_run / "history.csv",
        args.fem_run / "summary.json",
        args.fem_run / "resolved_run_config.json",
        args.fem_run / "reference_manifest.json",
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    bem, fem, config = (
        json.loads(p.read_text())
        for p in [
            args.bem_run / "summary.json",
            args.fem_run / "summary.json",
            args.fem_run / "resolved_run_config.json",
        ]
    )
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
    ):
        raise ValueError("Require complete source-frozen BEM and float64 FEM events")
    validate_physical_parameters(bem, fem, config)
    manifest = args.fem_run / "reference_manifest.json"
    if fingerprint(manifest) != fem["reference_manifest_sha256"]:
        raise ValueError("FEM reference manifest differs from the completed run")
    trajectory, fem_file = args.bem_run / "trajectory.npz", args.fem_run / "simulation.h5"
    digest = runpy.run_path(str(TOOLS / "replay_resolvent.py"))["digest_file"]
    trajectory_hash = digest(trajectory)
    before = fem_file.stat()
    trace = READ["read_fem_trace"](fem_file, fields=FIELDS)
    indices, ft = fem_step_clock(trace, fem, config)
    fields = {key: value[indices] for key, value in trace["interface_fields"].items()}
    if np.any(fields["friction_coefficient"] <= 0):
        raise ValueError("Pressure reconstruction needs strictly positive saved mu")
    field_errors, scalar_errors = {}, {}
    with np.load(trajectory, allow_pickle=False) as data:
        bt, by = data["time"], data["contact_y"]
        np.testing.assert_array_equal(bt, np.arange(bem["completed_steps"] + 1) * bem["dt_s"])
        if bt[0] != 0 or abs(bt[-1] - 0.045) > 1e-12:
            raise ValueError("BEM trajectory does not cover the complete recorded event")
        for key in [*FIELDS, "pressure_reconstructed"]:
            if key == "friction_strength":
                actual = data["friction_coefficient"] * data["pressure"]
            elif key == "pressure_reconstructed":
                actual = data["pressure"]
            else:
                actual = data[key]
            reference = (
                fields["friction_strength"] / fields["friction_coefficient"]
                if key == "pressure_reconstructed"
                else fields[key]
            )
            field_errors[key] = compare_saved_p1_fields(
                (ft, trace["y"], reference), (bt, by, actual), row_chunk=args.row_chunk
            )
            del actual, reference
    history = np.genfromtxt(args.bem_run / "history.csv", delimiter=",", names=True)
    for bk, fk in [
        ("max_cumulative_slip_mm", "max_slip"),
        ("mean_contact_pressure_mpa", "avg_sigma_n"),
        ("mean_contact_shear_mpa", "avg_tau"),
        ("platen_mm", "loading_face_displacement"),
        ("actuator_mm", "applied_shear_displacement"),
        ("applied_shear_mpa", "applied_shear"),
    ]:
        scalar_errors[bk] = compare_saved_p1_fields(
            (ft, [0, 1], np.repeat(trace["history"][fk][indices, None], 2, axis=1)),
            (history["time_s"], [0, 1], np.repeat(history[bk][:, None], 2, axis=1)),
            row_chunk=args.row_chunk,
        )
    after = fem_file.stat()
    if (
        (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
        or digest(trajectory) != trajectory_hash
        or any(fingerprint(p) != value for p, value in frozen.items())
    ):
        raise ValueError("Comparison sources or data changed")
    output.mkdir(parents=True)
    write_json(
        output / "summary.json",
        dict(
            status="completed_saved_space_time_comparison",
            bem_run=str(args.bem_run.resolve()),
            fem_run=str(args.fem_run.resolve()),
            frozen_input_sha256=frozen,
            bem_trajectory_sha256=trajectory_hash,
            fem_provenance=trace["provenance"],
            clock="Recorded dt and integer phase steps, verified against every stored float32 timestamp",
            field_errors=field_errors,
            scalar_errors=scalar_errors,
            comparison_seconds_including_hashes=perf_counter() - begun,
            pressure_scope="Saved strength/mu; endpoint storage errors require separate float64 comparison. Shear is not reconstructed from strength",
            endpoint_scope="Shared-time saved P1 interpolants; separate true float64 final_state comparison is required",
            stability_scope="Saved fields alone do not bound unsaved oscillations or prove nonlinear stability",
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
        ),
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

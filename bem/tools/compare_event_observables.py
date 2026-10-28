"""Compare complete event-16 arrival brackets, propagation and contact gauges.

Default input requires true float64 FEM endpoint observations. The explicit
legacy mode reports float32 rounding uncertainty and cannot certify the new
float64 target. It never infers endpoint pressure/shear from friction strength.
"""

import argparse
import csv
import json
import runpy
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
OBS = runpy.run_path(str(TOOLS / "event_observables.py"))
CONTACT = runpy.run_path(str(TOOLS / "compare_float64_contact_history.py"))
DIGEST = CONTACT["DIGEST"]
DC = 0.0125959
THRESHOLDS = {"micro": 1e-7, "Dc10": DC / 10, "Dc": DC}
GAUGE_FRACTIONS = np.array([0.1, 0.5, 0.9])
SEED = TOOLS.parent / "src/finiteblock_bem/resources/data/ev16_restart_float64.npz"
SEED_RECORD = SEED.with_suffix(".json")


def validate_event_policy(bem, config, manifest, seed_record):
    """Do not accept matching curves obtained from a different seed/hold policy."""
    expected = dict(
        reset_slip_weakening_at_shear_start=True,
        stop_shear_loading_on_rupture=True,
        shear_loading_stop_slip=None,
        shear_loading_stop_velocity=None,
        shear_loading_stop_min_y=0.0,
        shear_loading_stop_max_y=480.0,
        shear_loading_stop_coverage_fraction=1.0,
    )
    if (
        any(key not in config or config[key] != value for key, value in expected.items())
        or bem["case"]["loading"]["stop_coverage"] != 1.0
    ):
        raise ValueError("Require the original healed-memory, whole-interface coverage-stop policy")
    seed = bem.get("restart_seed_override") or {}
    if (
        seed.get("sha256") != seed_record["seed_file_sha256"]
        or manifest["source_restart"]["seed_sha256"]
        != seed_record["source_plastic_and_actuator_sha256"]
        or bem["case"]["source"]["revision"] != seed_record["source_revision"]
    ):
        raise ValueError("Require the same original float64 event-15 plastic-slip/actuator seed")


def validate_metadata(bem, fem, config, manifest, fem_run, *, legacy):
    if (
        bem.get("status") != "completed_requested_horizon"
        or bem.get("duration_s") != 0.045
        or not bem.get("all_history_retained")
        or bem.get("source_changed_during_run", False)
        or fem.get("status") != "completed"
        or not fem.get("full_original_horizon")
        or fem.get("computational_dtype") != config.get("dtype")
        or config.get("dtype") != "float64"
        or not fem.get("source_and_protected_input_unchanged")
        or bem["case"]["source"]["revision"] != fem.get("source_revision")
        or fem["source_revision"] != manifest["source"]["revision"]
        or fem["source_archive_sha256"] != manifest["source"]["archive_sha256"]
        or fem["summary"]["dt"] != config["time_step_override"]
        or fem["summary"]["shear_time"] != 0.045
        or fingerprint(fem_run / "reference_manifest.json") != fem["reference_manifest_sha256"]
    ):
        raise ValueError("Require complete original-horizon source-frozen float64 computations")
    CONTACT["validate_case"](bem, fem, config)
    seed_record = json.loads(SEED_RECORD.read_text())
    if fingerprint(SEED) != seed_record["seed_file_sha256"]:
        raise ValueError("Packaged original float64 restart differs from its recorded values")
    validate_event_policy(bem, config, manifest, seed_record)
    if bem["case"]["contact"]["characteristic_slip"] != DC:
        raise ValueError("Arrival thresholds require the original event-16 Dc")
    if not legacy:
        observer = fem.get("contact_observations")
        if (
            observer is None
            or observer != manifest.get("contact_observations")
            or observer["sha256"] != fingerprint(TOOLS / "fem_contact_observations.py")
        ):
            raise ValueError("Require verified float64 observer; legacy input needs explicit mode")


def read_fem_event(stream, fem, *, legacy):
    high = stream["interface_high_rate"]
    count = int(stream.attrs["saved_interface_frames"])
    if count != len(high["history"]) or count < 2:
        raise ValueError("Require a complete high-rate FEM history")
    if legacy:
        if "contact_observations" in stream:
            raise ValueError("Use the default mode when true endpoint observations are available")
        phases, steps = high["phase_id"][:], high["step_id"][:]
        if (
            phases.shape != (count,)
            or steps.shape != (count,)
            or steps.dtype.kind not in "iu"
            or np.any(np.diff(phases) < 0)
            or np.any(~np.isin(phases, [0, 1, 2]))
            or np.any(steps < 0)
        ):
            raise ValueError("Require the actual integer FEM phase clock")
        indices = np.flatnonzero(phases == 2)
        if (
            len(indices) < 2
            or steps[indices][0] <= 0
            or np.any(np.diff(steps[indices]) <= 0)
            or steps[indices][-1] != fem["summary"]["shear_steps"]
        ):
            raise ValueError("Require all actual shear observations through the final step")
        time = steps[indices].astype(float) * fem["summary"]["dt"]
        y = high["contact_line_y"][:].astype(float)
        cumulative = high["cumulative_slip"][indices]
        if cumulative.dtype != np.float32:
            raise ValueError("Legacy mode specifically describes float32 stored cumulative slip")
        fields = {}
    else:
        indices, time, y = CONTACT["observation_grid"](stream, fem)
        group = stream["contact_observations"]
        cumulative = group["cumulative_slip"][indices]
        fields = {k: group[k][indices] for k in ["pressure", "elastic_shear"]}
    names = [v.decode() if isinstance(v, bytes) else str(v) for v in high["history_columns"][:]]
    history = high["history"][:]
    if (
        history.shape != (count, len(names))
        or len(set(names)) != len(names)
        or not np.all(np.isfinite(history))
    ):
        raise ValueError("Invalid complete FEM scalar history")
    # Validate the original storage clock before using step*dt; the rounded
    # normal duration is never subtracted to define shear t=0.
    steps = high["step_id"][indices].astype(float)
    absolute = (steps + fem["summary"]["pressure_steps"]) * fem["summary"]["dt"]
    expected = absolute.astype(high["history"].dtype)
    if not np.array_equal(expected, history[indices, names.index("time")]):
        raise ValueError("Original saved timestamps disagree with the integer FEM step clock")
    stopped = history[indices, names.index("shear_loading_stopped")]
    if np.any(~np.isin(stopped, [0, 1])) or np.any(np.diff(stopped) < 0):
        raise ValueError("Require the actual monotone categorical FEM loading-stop flag")
    stop = OBS["crossing_brackets"](time, stopped[:, None], 0.5)[0]
    return dict(time=time, y=y, cumulative=cumulative, fields=fields, stop=stop)


def read_bem_event(data, bem):
    time, y = data["time"], data["contact_y"]
    if (
        time.dtype != np.float64
        or y.dtype != np.float64
        or not np.array_equal(time, np.arange(bem["completed_steps"] + 1) * bem["dt_s"])
        or abs(time[-1] - 0.045) > 1e-12
        or any(data[k].dtype != np.float64 for k in ["cumulative_slip", "pressure", "shear"])
    ):
        raise ValueError("Require true float64 BEM fields on the complete integer step clock")
    loading = CONTACT["validate_bem_loading"](data, bem)
    hold = bem["actuator_hold_time_s"]
    stop = dict(
        state="unreached" if hold is None else "bounded",
        lower_s=float(time[-1]) if hold is None else hold,
        upper_s=hold,
    )
    return dict(
        time=time,
        y=y,
        cumulative=data["cumulative_slip"],
        fields={"pressure": data["pressure"], "elastic_shear": data["shear"]},
        stop=stop,
        loading_check=loading,
    )


def sampled_passage(event, threshold, knots=None):
    values = event["cumulative"]
    lower, upper = OBS["storage_bounds"](values)
    if knots is not None:
        values, lower, upper = (
            OBS["sample_columns"](event["y"], v, knots) for v in (values, lower, upper)
        )
    return OBS["crossing_brackets"](event["time"], values, threshold, bounds=(lower, upper))


def duration_gate(reference, actual, tolerance):
    if reference["duration_interval_s"] is None or actual["duration_interval_s"] is None:
        return dict(status="inconclusive", reason="Saved arrivals do not resolve both directions")
    if reference["direction"] != actual["direction"]:
        return dict(status="fail", reason="Resolved gauge arrival directions differ")
    r, a = (
        dict(zip(("lower_s", "upper_s"), record["duration_interval_s"], strict=True))
        for record in (reference, actual)
    )
    return OBS["relative_interval_gate"](r, a, tolerance)


def compare_events(reference, actual, *, time_tolerance=0.02, travel_tolerance=0.05):
    if not np.array_equal(reference["y"][[0, -1]], actual["y"][[0, -1]]):
        raise ValueError("Require exactly the same full physical contact interval")
    knots = np.union1d(reference["y"], actual["y"])
    gauges = knots[0] + GAUGE_FRACTIONS * (knots[-1] - knots[0])
    thresholds, tables = {}, {}
    for name, threshold in THRESHOLDS.items():
        native_r, native_a = (sampled_passage(e, threshold) for e in (reference, actual))
        first_r, first_a = (
            OBS["first_event"](e["y"], p)
            for e, p in zip((reference, actual), (native_r, native_a), strict=True)
        )
        common_r, common_a = (sampled_passage(e, threshold, knots) for e in (reference, actual))
        curve = [
            OBS["relative_interval_gate"](r, a, time_tolerance)
            for r, a in zip(common_r, common_a, strict=True)
        ]
        gauges_r, gauges_a = (sampled_passage(e, threshold, gauges) for e in (reference, actual))
        travel = []
        for left, right in [(0, 1), (1, 2), (0, 2)]:
            r, a = (
                OBS["gauge_travel"](gauges[left], gauges[right], p[left], p[right])
                for p in (gauges_r, gauges_a)
            )
            travel.append(
                dict(
                    reference=r, actual=a, duration_comparison=duration_gate(r, a, travel_tolerance)
                )
            )
        thresholds[name] = dict(
            threshold_mm=threshold,
            native_first_event=dict(
                reference=first_r,
                actual=first_a,
                comparison=OBS["relative_interval_gate"](first_r, first_a, time_tolerance),
                possible_location_difference=OBS["location_difference"](first_r, first_a),
            ),
            common_physical_knots=dict(
                count=len(knots),
                arrival_gate_counts={
                    s: sum(v["status"] == s for v in curve)
                    for s in ["pass", "fail", "inconclusive"]
                },
                maximum_proven_absolute_time_difference_s=max(
                    (
                        v["minimum_absolute_difference_s"]
                        for v in curve
                        if v["minimum_absolute_difference_s"] is not None
                    ),
                    default=None,
                ),
                maximum_possible_absolute_time_difference_s=max(
                    (
                        v["maximum_absolute_difference_s"]
                        for v in curve
                        if v["maximum_absolute_difference_s"] is not None
                    ),
                    default=None,
                ),
                unbounded_comparisons=sum(
                    v["maximum_absolute_difference_s"] is None for v in curve
                ),
            ),
            fixed_gauge_arrivals=[
                dict(
                    y_mm=float(y),
                    reference=r,
                    actual=a,
                    comparison=OBS["relative_interval_gate"](r, a, time_tolerance),
                )
                for y, r, a in zip(gauges, gauges_r, gauges_a, strict=True)
            ],
            gauge_pair_propagation=travel,
        )
        tables[name] = (knots, common_r, common_a, curve)
    waveforms = {}
    for key in reference["fields"]:
        r, a = (
            OBS["sample_columns"](e["y"], e["fields"][key], gauges) for e in (reference, actual)
        )
        waveforms[key] = [
            dict(y_mm=float(y), **OBS["waveform_error"](reference["time"], rr, actual["time"], aa))
            for y, rr, aa in zip(gauges, r.T, a.T, strict=True)
        ]
    return dict(
        thresholds=thresholds,
        loading_stop=dict(
            reference=reference["stop"],
            actual=actual["stop"],
            comparison=OBS["relative_interval_gate"](
                reference["stop"], actual["stop"], time_tolerance
            ),
        ),
        contact_gauge_waveforms=waveforms,
        gauge_positions_mm=gauges.tolist(),
        time_relative_tolerance=time_tolerance,
        propagation_duration_relative_tolerance=travel_tolerance,
    ), tables


def write_tables(output, tables):
    for name, (knots, reference, actual, comparison) in tables.items():
        with (output / f"{name}_arrival_brackets.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "y_mm",
                    "fem_state",
                    "fem_lower_s",
                    "fem_upper_s",
                    "bem_state",
                    "bem_lower_s",
                    "bem_upper_s",
                    "gate",
                    "minimum_absolute_difference_s",
                    "maximum_absolute_difference_s",
                ]
            )
            for y, r, a, c in zip(knots, reference, actual, comparison, strict=True):
                writer.writerow(
                    [
                        y,
                        r["state"],
                        r["lower_s"],
                        r["upper_s"],
                        a["state"],
                        a["lower_s"],
                        a["upper_s"],
                        c["status"],
                        c["minimum_absolute_difference_s"],
                        c["maximum_absolute_difference_s"],
                    ]
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bem_run", type=Path)
    parser.add_argument("fem_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-diagnostic", action="store_true")
    parser.add_argument("--time-rtol", type=float, default=0.02)
    parser.add_argument("--travel-rtol", type=float, default=0.05)
    args = parser.parse_args()
    begun = perf_counter()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a fresh output outside protected Tatva")
    paths = [
        Path(__file__),
        SEED,
        SEED_RECORD,
        *(
            TOOLS / name
            for name in [
                "event_observables.py",
                "compare_float64_contact_history.py",
                "compare_refined_fem_history.py",
                "compare_fem_mesh.py",
                "fem_reference_io.py",
                "compare_fem_reference.py",
                "run_fem_reference.py",
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
    bem, fem, config, manifest = (json.loads(p.read_text()) for p in paths[-4:])
    validate_metadata(bem, fem, config, manifest, args.fem_run, legacy=args.legacy_diagnostic)
    trajectory, hdf = args.bem_run / "trajectory.npz", args.fem_run / "simulation.h5"
    hashes = {str(p.resolve()): DIGEST(p) for p in [trajectory, hdf]}
    with h5py.File(hdf, "r") as stream, np.load(trajectory, allow_pickle=False) as data:
        reference = read_fem_event(stream, fem, legacy=args.legacy_diagnostic)
        actual = read_bem_event(data, bem)
        metrics, tables = compare_events(
            reference, actual, time_tolerance=args.time_rtol, travel_tolerance=args.travel_rtol
        )
    if any(DIGEST(p) != v for p, v in hashes.items()) or any(
        fingerprint(p) != v for p, v in frozen.items()
    ):
        raise ValueError("Comparison inputs changed during evaluation")
    output.mkdir(parents=True)
    write_tables(output, tables)
    write_json(
        output / "summary.json",
        dict(
            status="completed_legacy_event_diagnostic"
            if args.legacy_diagnostic
            else "completed_float64_event_observables",
            bem_run=str(args.bem_run.resolve()),
            fem_run=str(args.fem_run.resolve()),
            frozen_input_sha256=frozen,
            dataset_sha256=hashes,
            metrics=metrics,
            bem_loading_check=actual["loading_check"],
            fem_cumulative_storage_dtype=str(reference["cumulative"].dtype),
            fem_maximum_saved_shear_interval_s=float(np.max(np.diff(reference["time"]))),
            legacy_float32_coordinate_uncertainty_bounded=False if args.legacy_diagnostic else None,
            eligible_for_current_float64_observation_comparison=not args.legacy_diagnostic,
            scope="Sampling and f32 rounding brackets for monotone cumulative slip; no PDE/spatial/integration/unsaved-state or formal floating-point error bound",
            first_event_scope="Native nodes separately from P1 common physical knots; all compatible earliest native locations retained",
            propagation_scope="Fixed 10/50/90-percent gauges; average first-passage speed may span multiple fronts, not instantaneous local rupture velocity",
            waveform_scope="True endpoint contact pressure and signed shear only; does not reconstruct complete interior waves",
            stop_scope="FEM categorical flag bracket; BEM verified first accepted coverage endpoint; no fitted time shift",
            omitted_initial_interval="No pre-handoff normal observation is relabelled as shear t=0; crossings before first shear sample remain censored",
            comparison_seconds_including_hashes=perf_counter() - begun,
            same_accuracy_fem_advantage_verified=False,
        ),
    )
    print(
        json.dumps(
            dict(status="completed", output=str(output), loading_stop=metrics["loading_stop"])
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a reproducible comparison from local BEM and archived FEM artifacts."""

import argparse
import json
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from finiteblock_bem.comparison import relative_l2
from finiteblock_bem.io import write_json

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "validation/static_bem2d/reference"


def read_history(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def first_crossing(time, values, threshold):
    found = np.flatnonzero(values > threshold)
    return float(time[found[0]]) if len(found) else None


def p1_profile_error(y, values, reference_y, reference):
    """Exact L2 norm of two piecewise-linear profiles on their common knots."""
    knots = np.union1d(y, reference_y)
    if not np.allclose([y[0], y[-1]], [reference_y[0], reference_y[-1]], atol=1e-8):
        raise ValueError("P1 profiles must cover the same physical interval")
    actual, target = np.interp(knots, y, values), np.interp(knots, reference_y, reference)

    def square_integral(v):
        return np.sum(np.diff(knots) * (v[:-1] ** 2 + v[:-1] * v[1:] + v[1:] ** 2) / 3)

    return float(np.sqrt(square_integral(actual - target) / square_integral(target)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "docs")
    parser.add_argument(
        "--coarse-fem",
        "--fem-rerun",
        dest="coarse_fem",
        type=Path,
        help="Optional local archived FEM rerun directory, at any mesh size",
    )
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    figures = args.report_dir / "figures"
    figures.mkdir(exist_ok=True)
    fem = read_history(REFERENCE / "fem_history.csv")
    keep = fem["time"] >= 0.002
    fem = fem[keep]
    fem_time = fem["time"] - 0.002
    fem_hold = first_crossing(fem_time, fem["shear_loading_stopped"], 0.5)
    fem_metrics = {
        "first_stored_slip_over_1e-7_mm_s": first_crossing(fem_time, fem["max_slip"], 1e-7),
        "actuator_hold_s": fem_hold,
        "final_max_cumulative_slip_mm": float(fem["max_slip"][-1]),
        "final_mean_pressure_mpa": float(fem["avg_sigma_n"][-1]),
        "final_mean_shear_mpa": float(fem["avg_tau"][-1]),
        "final_platen_mm": float(fem["loading_face_displacement"][-1]),
        "final_applied_shear_mpa": float(fem["applied_shear"][-1]),
        "original_recorded_wall_seconds": 6969.130398500711,
        "original_recorded_shear_steady_seconds": 6630.270908067934,
        "original_core_count": "not recorded; these are not controlled 4-core timings",
    }
    mappings = [
        ("max_cumulative_slip_mm", "max_slip", "Maximum cumulative slip [mm]"),
        ("mean_contact_shear_mpa", "avg_tau", "Mean contact shear [MPa]"),
        ("mean_contact_pressure_mpa", "avg_sigma_n", "Mean contact pressure [MPa]"),
        ("platen_mm", "loading_face_displacement", "Platen displacement [mm]"),
        ("applied_shear_mpa", "applied_shear", "Applied spring shear [MPa]"),
        ("actuator_mm", "applied_shear_displacement", "Actuator displacement [mm]"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True, constrained_layout=True)
    for ax, (_, fem_key, label) in zip(axes.flat, mappings, strict=True):
        ax.plot(fem_time * 1e3, fem[fem_key], color="black", lw=1.4, label="FEM h=2 mm (archive)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    coarse_record = None
    if args.coarse_fem is not None:
        coarse_record = json.loads((args.coarse_fem / "summary.json").read_text())
        reader = runpy.run_path(str(Path(__file__).with_name("fem_reference_io.py")))
        trace = reader["read_fem_trace"](args.coarse_fem / "simulation.h5")
        coarse = trace["history"]
        coarse_record["comparison_trace"] = trace["provenance"]
        normal_time = coarse_record["summary"]["pressure_time"]
        # A rounded normal-end timestamp can exceed pressure_time. Phase IDs
        # exclude its pre-reset cumulative slip from shear nucleation metrics.
        keep = trace["phase_id"] == 2
        tt = coarse["time"][keep] - normal_time
        for ax, (_, fem_key, _) in zip(axes.flat, mappings, strict=True):
            ax.plot(
                tt * 1e3,
                coarse[fem_key][keep],
                color="tab:green",
                ls="--",
                lw=1,
                label=f"FEM h={coarse_record['summary']['mesh_size']:g} mm (local)",
            )
        coarse_record["first_stored_slip_over_1e-7_mm_s"] = first_crossing(
            tt, coarse["max_slip"][keep], 1e-7
        )
    records = []
    profile_figure, profile_axes = None, None
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        label = (
            f"BEM {summary['panels'][0]}+{summary['panels'][1]} panels, "
            f"dt={summary['dt_s'] * 1e6:g} us"
        )
        if summary.get("schedule") == "recorded_fem_actuator_diagnostic":
            label += ", FEM actuator replay"
        history = read_history(run / "history.csv")
        with np.load(run / "trajectory.npz", allow_pickle=False) as trajectory:
            strength = trajectory["friction_coefficient"] * trajectory["pressure"]
            max_strength_ratio = float(
                np.max(abs(trajectory["shear"]) / np.maximum(strength, 1e-20))
            )
            weights = (
                trajectory["contact_weights"]
                if "contact_weights" in trajectory
                else np.linalg.norm(
                    trajectory["moving_ends"] - trajectory["moving_starts"], axis=1
                )[trajectory["moving_contact"]]
            )
            friction_work = float(
                np.sum(
                    trajectory["shear"][1:] * np.diff(trajectory["plastic_slip"], axis=0) * weights
                )
            )
            state_checks = {
                "all_saved_fields_finite": all(
                    np.all(np.isfinite(trajectory[key])) for key in trajectory.files
                ),
                "minimum_contact_pressure_mpa": float(np.min(trajectory["pressure"])),
                "minimum_cumulative_increment_mm": float(
                    np.min(np.diff(trajectory["cumulative_slip"], axis=0))
                ),
            }
            profile_errors = None
            if (
                summary.get("boundary_formulation", "").startswith("P1/P0")
                and abs(trajectory["time"][-1] - 0.045) < 1e-8
            ):
                reference = read_history(REFERENCE / "frame_09039.csv")
                fields = ["cumulative_slip", "pressure", "shear"]
                reference_fields = ["cumulative_slip", "p", "tau"]
                profile_errors = {
                    key: p1_profile_error(
                        trajectory["contact_y"],
                        trajectory[key][-1],
                        reference["y"],
                        reference[other],
                    )
                    for key, other in zip(fields, reference_fields, strict=True)
                }
                if profile_figure is None:
                    profile_figure, profile_axes = plt.subplots(
                        1, 3, figsize=(12, 4), layout="constrained"
                    )
                    for ax, other in zip(profile_axes, reference_fields, strict=True):
                        ax.plot(reference["y"], reference[other], "k-", label="FEM h=2")
                for ax, key in zip(profile_axes, fields, strict=True):
                    ax.plot(trajectory["contact_y"], trajectory[key][-1], label=label)
                    ax.set(
                        xlabel="Interface y [mm]",
                        ylabel={
                            "cumulative_slip": "Cumulative slip [mm]",
                            "pressure": "Pressure [MPa]",
                            "shear": "Shear [MPa]",
                        }[key],
                    )
                    ax.grid(alpha=0.2)
                    ax.legend(fontsize=7)
        time = history["time_s"]
        errors = {}
        for ax, (bem_key, fem_key, _) in zip(axes.flat, mappings, strict=True):
            ax.plot(time * 1e3, history[bem_key], lw=1.1, label=label)
            reference = np.interp(time, fem_time, fem[fem_key])
            weights = np.ones(len(time))
            weights[[0, -1]] = 0.5
            errors[bem_key] = relative_l2(history[bem_key], reference, weights)
        endpoint = {
            bem_key: {
                "bem": float(history[bem_key][-1]),
                "fem_at_same_time": float(np.interp(time[-1], fem_time, fem[fem_key])),
                "difference": float(
                    history[bem_key][-1] - np.interp(time[-1], fem_time, fem[fem_key])
                ),
            }
            for bem_key, fem_key, _ in mappings
        }
        records.append(
            {
                "name": run.name,
                "status": summary["status"],
                "duration_s": float(time[-1]),
                "panels": summary["panels"],
                "dt_s": summary["dt_s"],
                "first_stored_slip_over_1e-7_mm_s": first_crossing(
                    time, history["max_cumulative_slip_mm"], 1e-7
                ),
                "actuator_hold_s": summary["actuator_hold_time_s"],
                "endpoint": endpoint,
                "history_relative_l2": errors,
                "timing_seconds": summary["timing_seconds"],
                "max_residual_mpa": summary["max_boundary_contact_residual_mpa"],
                "max_strength_ratio": max_strength_ratio,
                "friction_work_mpa_mm2": friction_work,
                "final_p1_profile_relative_l2": profile_errors,
                "environment": summary["environment"],
                "source_changed_during_run": summary.get("source_changed_during_run"),
                "state_checks": state_checks,
            }
        )
    for ax in axes[-1]:
        ax.set_xlabel("Time from shear-phase start [ms]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Event 16: full-horizon BEM vs archived FEM\n"
        "Different spatial/temporal discretization and static restart projection"
    )
    fig.savefig(figures / "event16_comparison.png", dpi=170)
    plt.close(fig)
    if profile_figure is not None:
        profile_figure.savefig(figures / "event16_profiles.png", dpi=160)
        plt.close(profile_figure)
    record = {
        "fem": fem_metrics,
        "bem": records,
        "comparison_scope": "45 ms physical loading; BEM resolution and initial field differ from FEM",
        "full_original_event_reproduced": False,
    }
    if coarse_record is not None:
        record["coarse_fem_full_run"] = coarse_record
    cold = ROOT / "outputs/cold-kernel-timings.json"
    if cold.exists():
        record["cold_kernel_assembly"] = json.loads(cold.read_text())
    for name, key in [
        ("static-refinement", "static_refinement"),
        ("ad-benchmark", "autodiff"),
        ("fem-4c-timing", "fem_4core_short_timing"),
    ]:
        path = ROOT / "outputs" / name / "summary.json"
        if path.exists():
            item = json.loads(path.read_text())
            if name == "fem-4c-timing":
                item = {k: v for k, v in item.items() if k != "summary"} | {
                    "steady_chunk_timings": item["summary"]["steady_chunk_timings"],
                    "dt_s": item["summary"]["dt"],
                    "shear_steps": item["summary"]["shear_steps"],
                }
            record[key] = item
    if "static_refinement" in record:
        refinements = record["static_refinement"]["refinements"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        fem_initial = read_history(REFERENCE / "frame_00039.csv")
        axes[0].plot(fem_initial["y"], fem_initial["p"], "k-", label="FEM h=2 nodes")
        axes[1].plot(fem_initial["y"], fem_initial["tau"], "k-", label="FEM h=2 nodes")
        for h in [40, 10, 2]:
            data = read_history(ROOT / f"outputs/static-refinement/h{h}.csv")
            axes[0].plot(data["y_mm"], data["bem_pressure_mpa"], label=f"BEM h={h}")
            axes[1].plot(data["y_mm"], data["bem_shear_mpa"], label=f"BEM h={h}")
        axes[0].set_ylabel("Initial pressure [MPa]")
        axes[1].set_ylabel("Initial shear [MPa]")
        for ax in axes[:2]:
            ax.set_xlabel("Interface y [mm]")
            ax.grid(alpha=0.25)
        for key, label in [("pressure_relative_l2", "Pressure"), ("shear_relative_l2", "Shear")]:
            axes[2].loglog(
                [r["h_mm"] for r in refinements], [r[key] for r in refinements], "o-", label=label
            )
        axes[2].set_xlabel("BEM maximum panel length [mm]")
        axes[2].set_ylabel("Relative L2 vs projected FEM h=2")
        axes[2].grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        axes[2].legend(fontsize=8)
        fig.savefig(figures / "static_refinement.png", dpi=170)
        plt.close(fig)
    write_json(args.report_dir / "event16_results.json", record)
    print(
        json.dumps(
            {
                "fem": fem_metrics,
                "bem": [{k: v for k, v in r.items() if k != "environment"} for r in records],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Locate saved contact opening, signed-traction changes and spring separation.

Uses the unchanged verified float64 field comparison and its actual clocks.
Spatial measures are exact for the saved P1 interpolants; they do not bound
events between saved frames or establish nonlinear stability.
"""

import argparse
import json
import runpy
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
PLOT = runpy.run_path(str(TOOLS / "plot_float64_contact_fields.py"))
COMPARE = PLOT["COMPARE"]


def negative_length_fraction(y, values, *, include_zero=False):
    """Exact negative length of a spatial P1 function, without a threshold."""
    h = np.diff(y)
    a, b = values[:, :-1], values[:, 1:]
    lengths = np.zeros_like(a)
    lengths[(a < 0) & (b < 0)] = 1
    first, second = (a < 0) & (b >= 0), (a >= 0) & (b < 0)
    np.divide(-a, b - a, out=lengths, where=first)
    np.divide(-b, a - b, out=lengths, where=second)
    if include_zero:
        lengths[(a == 0) & (b == 0)] = 1
    return np.sum(lengths * h, axis=1) / np.sum(h)


def absolute_p1_mean(y, values):
    """Integrate |P1| exactly, splitting every sign-changing interval."""
    a, b = values[:, :-1], values[:, 1:]
    integral = (abs(a) + abs(b)) / 2
    crossing = ((a < 0) & (b > 0)) | ((a > 0) & (b < 0))
    np.divide(a * a + b * b, 2 * (abs(a) + abs(b)), out=integral, where=crossing)
    return np.sum(integral * np.diff(y), axis=1) / (y[-1] - y[0])


def measures(y, fields, initial_plastic, initial_cumulative):
    """Keep signs and plastic variation distinct from net displacement."""
    normal, shear, plastic, cumulative, strength = fields
    mean = PLOT["p1_mean"]
    net = absolute_p1_mean(y, plastic - initial_plastic)
    variation = mean(y, cumulative - initial_cumulative)
    return dict(
        minimum_normal_jump_mm=np.min(normal, axis=1),
        open_p1_length_fraction=negative_length_fraction(y, normal, include_zero=True),
        negative_shear_p1_length_fraction=negative_length_fraction(y, shear),
        mean_signed_shear_mpa=mean(y, shear),
        mean_absolute_shear_mpa=absolute_p1_mean(y, shear),
        mean_strength_mpa=mean(y, strength),
        mean_cumulative_increment_mm=variation,
        mean_absolute_net_plastic_increment_mm=net,
        mean_variation_excess_mm=variation - net,
    )


def bem_arrays(data):
    """Read the writer's flat [normal nodes, tangent nodes] jump columns."""
    time, y, jumps = data["time"], data["contact_y"], data["contact_jumps"]
    if jumps.shape != (len(time), 2 * len(y)):
        raise ValueError("Saved BEM jumps must contain flat normal/tangent node blocks")
    fields = [
        jumps[:, : len(y)],
        data["shear"],
        data["plastic_slip"],
        data["cumulative_slip"],
        data["friction_coefficient"] * data["pressure"],
    ]
    gap = data["actuator"] - data["platen"]
    return time, y, fields, gap, np.zeros_like(gap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose fresh diagnostic output outside protected Tatva")
    evidence_path = args.comparison.resolve() / "summary.json"
    evidence = json.loads(evidence_path.read_text())
    if (
        evidence["status"] != "completed_float64_contact_space_time_comparison"
        or not evidence["bem_loading_check"]["no_fitted_time_shift"]
    ):
        raise ValueError("Require the complete unshifted float64 field comparison")
    frozen = {
        **evidence["frozen_input_sha256"],
        str(evidence_path): fingerprint(evidence_path),
        str(Path(__file__).resolve()): fingerprint(__file__),
        str(TOOLS / "plot_float64_contact_fields.py"): fingerprint(
            TOOLS / "plot_float64_contact_fields.py"
        ),
    }

    def verify():
        if any(fingerprint(p) != h for p, h in frozen.items()) or any(
            COMPARE["DIGEST"](p) != h for p, h in evidence["dataset_sha256"].items()
        ):
            raise ValueError("Verified comparison, source or datasets changed")

    verify()
    target = evidence["field_errors"]["pressure"]["common_time_interval_s"][-1]
    runs = {name: Path(evidence[key]) for name, key in [("bem", "bem_run"), ("fem", "fem_run")]}
    summaries = {name: json.loads((p / "summary.json").read_text()) for name, p in runs.items()}
    traces, records, grids = {}, {}, {}
    for label in ["bem", "fem"]:
        if label == "bem":
            with np.load(runs[label] / "trajectory.npz", allow_pickle=False) as data:
                time, y, fields, gap, gap_bound = bem_arrays(data)
        else:
            with h5py.File(runs[label] / "simulation.h5", "r") as stream:
                indices, time, y = COMPARE["observation_grid"](stream, summaries[label])
                group = stream["contact_observations"]
                fields = [
                    group[k][indices]
                    for k in [
                        "normal_jump",
                        "elastic_shear",
                        "plastic_slip",
                        "cumulative_slip",
                        "friction_strength",
                    ]
                ]
                scalar = stream["interface_high_rate/history"]
                names = [
                    x.decode() if isinstance(x, bytes) else str(x)
                    for x in stream["interface_high_rate/history_columns"][:]
                ]
                actuator = scalar[indices, names.index("applied_shear_displacement")].astype(float)
                platen = scalar[indices, names.index("loading_face_displacement")].astype(float)
                gap = actuator - platen
                gap_bound = 2 * np.finfo(scalar.dtype).eps * (abs(actuator) + abs(platen))
        if (
            not all(a.dtype == np.float64 and a.shape == (len(time), len(y)) for a in fields)
            or not all(np.all(np.isfinite(a)) for a in [*fields, gap, gap_bound])
            or np.any(np.diff(time) <= 0)
            or np.any(np.diff(y) <= 0)
        ):
            raise ValueError("Require finite aligned actual clocks and true float64 contact fields")
        initial = fields[2][0].copy(), fields[3][0].copy()
        values = measures(y, fields, *initial)
        shared = measures(
            y, [PLOT["shared_row"](time, a, target)[None, :] for a in fields], *initial
        )
        opened = np.flatnonzero(values["open_p1_length_fraction"] > 0)
        spring_open = np.flatnonzero(gap < -gap_bound)
        records[label] = dict(
            saved_samples=len(time),
            first_native_time_s=float(time[0]),
            last_native_time_s=float(time[-1]),
            first_saved_contact_opening_time_s=None if not len(opened) else float(time[opened[0]]),
            maximum_saved_open_p1_length_fraction=float(np.max(values["open_p1_length_fraction"])),
            minimum_saved_normal_jump_mm=float(np.min(values["minimum_normal_jump_mm"])),
            first_saved_definite_spring_opening_time_s=None
            if not len(spring_open)
            else float(time[spring_open[0]]),
            minimum_saved_spring_gap_mm=float(np.min(gap)),
            maximum_spring_scalar_storage_roundoff_bound_mm=float(np.max(gap_bound)),
            shared_final_time_s=float(target),
            shared_final_measures={k: float(v[0]) for k, v in shared.items()},
            maximum_saved_mean_variation_excess_mm=float(
                np.max(values["mean_variation_excess_mm"])
            ),
        )
        traces[label] = dict(
            time=time, spring_gap_mm=gap, spring_gap_storage_bound_mm=gap_bound, **values
        )
        grids[label] = y
    output.mkdir(parents=True)
    np.savez(
        output / "contact_modes.npz",
        **{f"{label}_{key}": value for label, row in traces.items() for key, value in row.items()},
        **{f"{label}_y": y for label, y in grids.items()},
    )
    colors = {"bem": "#0072B2", "fem": "#D55E00"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for label, row in traces.items():
        time, color = row["time"] * 1000, colors[label]
        axes[0, 0].plot(time, row["open_p1_length_fraction"], color=color, label=label.upper())
        axes[0, 1].plot(time, row["spring_gap_mm"], color=color)
        axes[1, 0].plot(
            time,
            abs(row["mean_signed_shear_mpa"]),
            color=color,
            label=label.upper() + " |mean shear|",
        )
        axes[1, 0].plot(
            time,
            row["mean_absolute_shear_mpa"],
            color=color,
            ls="--",
            label=label.upper() + " mean |shear|",
        )
        axes[1, 1].plot(time, row["mean_variation_excess_mm"], color=color)
    axes[0, 0].set(title="Open contact length / total length", ylabel="Fraction")
    axes[0, 1].set(title="Unilateral spring gap", ylabel="mm")
    axes[0, 1].axhline(0, color="0.4", lw=0.7)
    axes[1, 0].set(title="Signed shear cancellation", ylabel="MPa")
    axes[1, 1].set(
        title="Plastic variation minus absolute net increment", ylabel="Spatial P1 mean [mm]"
    )
    for axis in axes.flat:
        axis.set(xlabel="Actual shear-phase time [ms]", xlim=(0, target * 1000))
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    axes[1, 0].legend(fontsize=8)
    figure.suptitle("Saved contact and spring states; no fitted shift or smoothing")
    figure.text(
        0.5,
        -0.02,
        "Spatial measures refer to saved P1 fields. Unsaved opening or reversals are not bounded; stability remains unverified.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(output / "contact_modes.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    verify()
    write_json(
        output / "summary.json",
        dict(
            status="completed_saved_contact_and_spring_mode_diagnostic",
            records=records,
            frozen_inputs_sha256=frozen,
            dataset_sha256=evidence["dataset_sha256"],
            all_contact_endpoints_retained=True,
            no_fitted_time_shift=True,
            no_temporal_smoothing=True,
            scope="Opening and sign measures of saved spatial P1 interpolants, plastic variation relative to each first native sample. Negative shear does not by itself identify slip direction. FEM spring uses original stored scalar precision with an explicit storage bound. No bound on unsaved events, time-step convergence, causal mechanism or nonlinear stability.",
            full_original_event_reproduced=False,
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
        ),
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

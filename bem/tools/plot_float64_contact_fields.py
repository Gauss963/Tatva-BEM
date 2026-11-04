"""Plot the verified, unshifted full-event BEM/FEM contact comparison.

All interface endpoints are retained. Final profiles use the same shared
time as the P1 field comparison. Spatial means are exact P1 integrals;
the plotted time traces cannot bound oscillations between saved frames.
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
COMPARE = runpy.run_path(str(TOOLS / "compare_float64_contact_history.py"))


def shared_row(time, values, target):
    if not time[0] <= target <= time[-1]:
        raise ValueError("Final profile must lie in the actual shared time interval")
    right = min(max(int(np.searchsorted(time, target, side="right")), 1), len(time) - 1)
    left = right - 1
    fraction = (target - time[left]) / (time[right] - time[left])
    return (1 - fraction) * values[left] + fraction * values[right]


def p1_mean(y, values):
    widths = np.diff(y)
    return np.sum(0.5 * (values[:, :-1] + values[:, 1:]) * widths, axis=1) / (y[-1] - y[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose fresh figures outside protected Tatva")
    evidence_path = args.comparison / "summary.json"
    evidence = json.loads(evidence_path.read_text())
    if (
        evidence["status"] != "completed_float64_contact_space_time_comparison"
        or set(evidence["field_errors"]) != set(COMPARE["FIELDS"])
        or not evidence["bem_loading_check"]["no_fitted_time_shift"]
    ):
        raise ValueError("Require the complete verified float64 field comparison")
    frozen = {
        **evidence["frozen_input_sha256"],
        str(evidence_path.resolve()): fingerprint(evidence_path),
        str(Path(__file__).resolve()): fingerprint(__file__),
    }

    def verify():
        if any(fingerprint(path) != digest for path, digest in frozen.items()):
            raise ValueError("Verified comparison or plot source changed")
        if any(
            COMPARE["DIGEST"](Path(path)) != digest
            for path, digest in evidence["dataset_sha256"].items()
        ):
            raise ValueError("Verified contact datasets changed")

    verify()
    bem_run, fem_run = (Path(evidence[k]) for k in ["bem_run", "fem_run"])
    bem, fem = (json.loads((p / "summary.json").read_text()) for p in [bem_run, fem_run])
    counts = "/".join(map(str, bem.get("segment_substeps", [])))
    discretization = f"BEM: {bem['dt_s'] * 1e6:g} us"
    if counts:
        discretization += f", {counts} substeps"
    discretization += f"; FEM: {fem['summary']['dt'] * 1e9:.3g} ns, original float64 state"
    target = evidence["field_errors"]["pressure"]["common_time_interval_s"][-1]
    profiles, means, peaks = {"BEM": [], "FEM": []}, {"BEM": [], "FEM": []}, {}
    with (
        h5py.File(fem_run / "simulation.h5", "r") as stream,
        np.load(bem_run / "trajectory.npz", allow_pickle=False) as data,
    ):
        indices, ft, fy = COMPARE["observation_grid"](stream, fem)
        bt, by = data["time"], data["contact_y"]
        for field in COMPARE["FIELDS"]:
            bvalues = COMPARE["bem_field"](data, field)
            fvalues = stream["contact_observations"][field][indices]
            for label, time, y, values in [("BEM", bt, by, bvalues), ("FEM", ft, fy, fvalues)]:
                profiles[label].append(shared_row(time, values, target))
                means[label].append(p1_mean(y, values))
                if field == "cumulative_slip":
                    peaks[label] = np.max(values, axis=1)
    profiles = {key: np.asarray(value) for key, value in profiles.items()}
    means = {key: np.asarray(value) for key, value in means.items()}
    if not all(
        np.all(np.isfinite(v)) for v in [*profiles.values(), *means.values(), *peaks.values()]
    ):
        raise ValueError("Require finite complete saved contact fields")
    output.mkdir(parents=True)
    np.savez(
        output / "plot_data.npz",
        fields=np.asarray(COMPARE["FIELDS"]),
        shared_final_time=target,
        bem_time=bt,
        fem_time=ft,
        bem_y=by,
        fem_y=fy,
        bem_final=profiles["BEM"],
        fem_final=profiles["FEM"],
        bem_mean=means["BEM"],
        fem_mean=means["FEM"],
        bem_peak_cumulative=peaks["BEM"],
        fem_peak_cumulative=peaks["FEM"],
    )
    colors = {"BEM": "#0072B2", "FEM": "#D55E00"}
    titles = [
        "Pressure",
        "Signed elastic shear",
        "Plastic offset",
        "Cumulative slip",
        "Friction coefficient",
        "Friction strength",
    ]
    units = ["MPa", "MPa", "mm", "mm", "1", "MPa"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.5), layout="constrained")
    for index, (field, axis) in enumerate(zip(COMPARE["FIELDS"], axes.flat, strict=True)):
        for label, y in [("BEM", by), ("FEM", fy)]:
            axis.plot(y, profiles[label][index], color=colors[label], lw=1.4, label=label)
        error = evidence["field_errors"][field]["final_shared_time_relative_p1_l2"] * 100
        axis.set(
            title=f"{titles[index]} | final P1 L2 difference {error:.2f}%",
            xlabel="Contact coordinate [mm]",
            ylabel=units[index],
            xlim=(by[0], by[-1]),
        )
        if field == "cumulative_slip":
            if min(np.min(profiles[label][index]) for label in profiles) > 0:
                axis.set_yscale("log")
                axis.set_ylabel("mm (log scale)")
            else:
                axis.set_yscale("symlog", linthresh=0.001)
                axis.set_ylim(bottom=0)
                axis.set_ylabel("mm (linear below 0.001; log above)")
        axis.grid(alpha=0.2)
    axes.flat[0].legend()
    figure.suptitle(
        f"Complete event: final contact profiles at shared t = {target * 1000:.3f} ms\n{discretization}"
    )
    figure.text(
        0.5,
        -0.02,
        "All contact endpoints retained. Original 5% field gate is not passed; physical convergence remains unverified.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(output / "final_contact_profiles.png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for axis, index in zip(axes.flat[:3], [0, 1, 3], strict=True):
        for label, time in [("BEM", bt), ("FEM", ft)]:
            axis.plot(time * 1000, means[label][index], color=colors[label], lw=1.2, label=label)
        axis.set(title="Spatial P1 mean: " + titles[index].lower(), ylabel=units[index])
    for label, time in [("BEM", bt), ("FEM", ft)]:
        axes.flat[3].plot(time * 1000, peaks[label], color=colors[label], lw=1.2, label=label)
    axes.flat[3].set(
        title="Maximum saved cumulative slip",
        ylabel="mm (linear below 0.001; log above)",
        yscale="symlog",
    )
    axes.flat[3].set_yscale("symlog", linthresh=0.001)
    axes.flat[3].set_ylim(bottom=0)
    for axis in axes.flat:
        axis.set(xlabel="Shear-phase time [ms]", xlim=(0, 45))
        axis.grid(alpha=0.2)
    axes.flat[0].legend()
    figure.suptitle(
        "Complete saved contact histories; original clocks and loading stop, no fitted shift"
    )
    figure.text(
        0.5,
        -0.02,
        "P1 spatial means use the whole interface. Saved traces do not bound unsaved oscillations or prove nonlinear stability.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(output / "contact_time_traces.png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    verify()
    write_json(
        output / "figure_manifest.json",
        dict(
            status="completed_verified_float64_contact_figures",
            comparison_sha256=fingerprint(evidence_path),
            source_sha256=frozen,
            dataset_sha256=evidence["dataset_sha256"],
            shared_final_time_s=target,
            all_contact_endpoints_retained=True,
            no_time_shift=True,
            no_temporal_smoothing=True,
            full_original_event_reproduced=False,
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
        ),
    )
    print(output / "figure_manifest.json", flush=True)


if __name__ == "__main__":
    main()

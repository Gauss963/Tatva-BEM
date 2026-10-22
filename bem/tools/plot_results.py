"""Export scientific verification plots from local, saved solver artifacts."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plot_bar(folder, summary):
    values = np.genfromtxt(folder / "wave_reflection.csv", delimiter=",", names=True)
    t = values["time"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    fig.suptitle(
        "Finite elastic bar: wave arrival and fixed-end reflection\n"
        "2D elastodynamic BEM / BDF2 convolution quadrature versus analytic delayed waves"
    )
    for ax, key, title in [
        (axes[0, 0], "loading", "Loading face"),
        (axes[0, 1], "probe", "Interior-x boundary probe"),
    ]:
        ax.plot(t, values[f"{key}_exact"], "k-", label="Analytic", lw=1.5)
        ax.plot(t, values[f"{key}_displacement"], "--", label="CQ-BEM", lw=1.3)
        ax.set(xlabel="Time (L/cp)", ylabel="Displacement", title=title)
        ax.legend()
    axes[1, 0].plot(
        t, values["loading_displacement"] - values["loading_exact"], label="Loading face"
    )
    axes[1, 0].plot(t, values["probe_displacement"] - values["probe_exact"], label="Probe")
    axes[1, 0].set(
        xlabel="Time (L/cp)",
        ylabel="Displacement error",
        title="Error including the reflected wave",
    )
    axes[1, 0].legend()
    axes[1, 1].plot(t, np.gradient(values["loading_displacement"], t), label="CQ-BEM velocity")
    axes[1, 1].plot(t, np.gradient(values["loading_exact"], t), "k--", label="Analytic velocity")
    axes[1, 1].axvline(2, color="gray", ls=":", label="First return: 2L/cp")
    axes[1, 1].set(
        xlabel="Time (L/cp)",
        ylabel="Loading-face velocity",
        title="Velocity reversal after clamp reflection",
    )
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.savefig(folder / "verification.png", dpi=180)
    plt.close(fig)
    return [
        f"Loading-face relative L2 error: {100 * summary['loading_relative_l2']:.3f}%.",
        f"Probe relative L2 error: {100 * summary['probe_relative_l2']:.3f}%.",
        f"Pre-arrival numerical signal / probe peak: {100 * summary['precursor_relative_max']:.5f}%.",
    ]


def plot_blocks(folder, summary):
    with np.load(folder / "trajectory.npz") as data:
        t = data["time"] * 1e3
        y = data["contact_y"]
        p = data["pressure"]
        tau = data["shear"]
        slip = data["cumulative_slip"]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        name = (
            "Original-rate event-16 transient"
            if summary["schedule"].startswith("event16")
            else "Cold-start accelerated-loading diagnostic"
        )
        fig.suptitle(
            f"{name}\nFully elastodynamic finite blocks; spatial/temporal convergence remains unverified"
        )
        for ax, field, title, units in [
            (axes[0, 0], p, "Contact pressure", "MPa"),
            (axes[0, 1], tau, "Resisting shear traction", "MPa"),
            (axes[0, 2], slip, "Cumulative plastic slip", "mm"),
        ]:
            image = ax.pcolormesh(t, y, field.T, shading="nearest", cmap="viridis")
            fig.colorbar(image, ax=ax, label=units)
            ax.set(
                xlabel="Time since initialization (ms)", ylabel="Along-fault y (mm)", title=title
            )
        axes[1, 0].plot(t, data["actuator"], label="Actuator")
        axes[1, 0].plot(t, data["platen"], label="Rigid platen")
        axes[1, 0].set(xlabel="Time (ms)", ylabel="Displacement (mm)", title="Spring loading")
        axes[1, 0].legend()
        history = np.genfromtxt(folder / "history.csv", delimiter=",", names=True)
        axes[1, 1].plot(t, history["mean_contact_pressure_mpa"], label="Mean pressure")
        axes[1, 1].plot(t, history["mean_contact_shear_mpa"], label="Mean shear")
        axes[1, 1].set(xlabel="Time (ms)", ylabel="Traction (MPa)", title="Interface history")
        axes[1, 1].legend()
        axes[1, 2].plot(y, p[-1], label="Pressure")
        axes[1, 2].plot(y, tau[-1], label="Shear")
        axes[1, 2].set(
            xlabel="Along-fault y (mm)", ylabel="Traction (MPa)", title="Last valid contact profile"
        )
        axes[1, 2].legend()
        for ax in axes[1]:
            ax.grid(alpha=0.2)
        fig.savefig(folder / "verification.png", dpi=180)
        plt.close(fig)
    return [
        f"Completed time: {summary['duration_s'] * 1000:.4f} ms.",
        f"Completed steps: {summary['completed_steps']}.",
        f"Maximum cumulative slip: {summary['max_cumulative_slip_mm']:.6g} mm.",
        f"Maximum applied-strength ratio: {summary['max_contact_strength_ratio']:.6g}.",
        f"Maximum residual: {summary['max_boundary_contact_residual_mpa']:.3e} MPa.",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="bar-reference")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name):
        raise ValueError("Invalid output name")
    folder = ROOT / "outputs" / args.name
    summary = json.loads((folder / "summary.json").read_text())
    if summary["status"] == "completed_analytic_wave_benchmark":
        facts = plot_bar(folder, summary)
    elif summary["status"] in {"completed_requested_horizon", "nonlinear_step_failed"}:
        facts = plot_blocks(folder, summary)
    else:
        raise ValueError("The selected run contains no valid trajectory to plot")
    report = ["# Verification result", "", f"Status: **{summary['status']}**.", ""]
    report += ["- " + fact for fact in facts]
    report += [
        "",
        "**The complete original FEM event has not been reproduced.**",
        "A successful short diagnostic is not evidence of spatial or temporal convergence of event 16.",
        "See summary.json for formulation, complete-history flag, loading schedule, initial-state mapping, and code fingerprints.",
        "",
    ]
    (folder / "REPORT.md").write_text("\n".join(report))
    print(folder / "verification.png")


if __name__ == "__main__":
    main()

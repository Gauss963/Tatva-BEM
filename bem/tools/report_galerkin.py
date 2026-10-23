"""Curate small Galerkin validation records and scientific plots from local runs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from finiteblock_bem.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = {
        "static_audit": "galerkin-audit/summary.json",
        "corner_refinement": "galerkin-corner-refinement/summary.json",
        "wave": "galerkin-wave/summary.json",
        "assembly_benchmark": "galerkin-benchmark-graded/summary.json",
        "quadrature": "galerkin-quadrature/summary.json",
        "short_sliding": "galerkin-sliding-32/summary.json",
        "short_adjoint": "galerkin-sliding-32/adjoint.json",
        "fem_static_refinement": "fem-static-refinement.json",
        "read_only_audit": "research-readonly-audit.json",
        "event16_galerkin_q8": "galerkin-corners-dt8-q8/summary.json",
    }
    results = {
        "date": "2026-10-23",
        "scope": "Galerkin bulk/short dynamics and ongoing event validation; no equal-accuracy FEM speed claim",
        "full_original_event_reproduced": False,
    }
    for key, path in files.items():
        target = ROOT / "outputs" / path
        if target.exists():
            results[key] = json.loads(target.read_text())
    write_json(ROOT / "docs/galerkin_results.json", results)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for h, dt in [(0.2, 0.04), (0.1, 0.02)]:
        with np.load(ROOT / f"outputs/galerkin-wave/h{h}-dt{dt}.npz") as data:
            axes[0, 0].plot(data["time"], data["loading"], label=f"h={h}, dt={dt}")
            axes[0, 1].plot(data["time"], data["probe"], label=f"h={h}, dt={dt}")
            if h == 0.1:
                axes[0, 0].plot(data["time"], data["loading_exact"], "k--", lw=1, label="analytic")
                axes[0, 1].plot(data["time"], data["probe_exact"], "k--", lw=1, label="analytic")
    axes[0, 0].set_title("Finite bar: loaded face")
    axes[0, 1].set_title("Finite bar: same probe x=0.5")
    for ax in axes[0]:
        ax.set_xlabel("Time [L/cp]")
        ax.set_ylabel("Displacement")
    seed = np.genfromtxt(
        ROOT / "src/finiteblock_bem/resources/data/ev16_restart_interface.csv",
        delimiter=",",
        names=True,
    )
    for ax in axes[1]:
        ax.plot(seed["y"], seed["p"], "k-", label="FEM h=2")
        for folder, label in [("fem-static-h1", "FEM h=1"), ("fem-static-h05", "FEM h=0.5")]:
            data = np.genfromtxt(
                ROOT / "outputs" / folder / "interface.csv", delimiter=",", names=True
            )
            ax.plot(data["y_mm"], data["p_mpa"], "--", label=label)
        for size in [2, 1, 0.5]:
            data = np.genfromtxt(
                ROOT / f"outputs/galerkin-corner-refinement/corner{size}-contact5.csv",
                delimiter=",",
                names=True,
            )
            ax.plot(data["y_mm"], data["p_mpa"], label=f"SG end={size}, contact=5")
        ax.set_xlabel("Interface y [mm]")
        ax.set_ylabel("Initial pressure [MPa]")
    axes[1, 0].set(xlim=(-0.3, 12), ylim=(0, 65), title="Static lower endpoint: mesh dependence")
    axes[1, 1].set(xlim=(470, 480.3), ylim=(0, 700), title="Static upper endpoint: mesh dependence")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    figures = ROOT / "docs/figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "galerkin_validation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

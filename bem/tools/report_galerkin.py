"""Curate small Galerkin validation records and scientific plots from local runs."""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from finiteblock_bem.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def plot_static_onset(figures):
    studies = []
    source = ROOT / "outputs/fem-static-onset"
    record = json.loads((source / "summary.json").read_text())
    for row in record["records"]:
        studies.append((f"FEM h={row['h_mm']}", source / f"h{row['h_mm']}.csv"))
    studies.extend(
        [
            (
                "SG end0.5/contact5",
                ROOT / "outputs/galerkin-corner-refinement/corner0.5-contact5.csv",
            ),
            (
                "SG + local 0.5mm patch",
                ROOT
                / "outputs/galerkin-corner-refinement-aligned8-sub4-end0.5/corner0.5-contact5.csv",
            ),
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for index, (name, path) in enumerate(studies):
        data = np.genfromtxt(path, delimiter=",", names=True)
        finite = np.isfinite(data["elastic_yield_time_s"])
        axes[0].plot(
            data["y_mm"][finite],
            1e3 * data["elastic_yield_time_s"][finite],
            ".-",
            lw=0.8,
            ms=3,
            label=name,
        )
        k = int(np.argmin(data["elastic_yield_time_s"]))
        t = 1e3 * data["elastic_yield_time_s"][k]
        axes[1].plot(t, index, "o")
        axes[1].annotate(
            f"y={data['y_mm'][k]:.3g} mm",
            (t, index),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set(
        xlim=(0, 30),
        ylim=(15, 80),
        xlabel="Interface y [mm]",
        ylabel="Static yield prediction [ms]",
    )
    axes[0].legend(fontsize=7)
    axes[1].set(
        xlim=(20, 32),
        ylim=(-0.5, len(studies) - 0.4),
        yticks=range(len(studies)),
        yticklabels=[s[0] for s in studies],
        xlabel="First static yield prediction [ms]",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Frozen-plastic static study; not dynamic rupture onset")
    fig.savefig(figures / "galerkin_static_onset.png", dpi=160)
    plt.close(fig)


def main():
    files = {
        "static_audit": "galerkin-audit/summary.json",
        "corner_refinement": "galerkin-corner-refinement/summary.json",
        "wave": "galerkin-wave/summary.json",
        "assembly_benchmark": "galerkin-benchmark-graded/summary.json",
        "quadrature": "galerkin-quadrature/summary.json",
        "short_sliding": "galerkin-sliding-32/summary.json",
        "short_adjoint": "galerkin-sliding-32/adjoint.json",
        "short_bulk_work": "galerkin-sliding-32/bulk_work.json",
        "fem_static_refinement": "fem-static-refinement.json",
        "fem_static_onset": "fem-static-onset/summary.json",
        "fem_packaged_input_replay": "fem-static-onset-packaged-replay/summary.json",
        "density_refinement": "galerkin-density/summary.json",
        "bdf2_domain": "bdf2-domain/summary.json",
        "read_only_audit": "research-readonly-audit.json",
        "event16_cold_run": "galerkin-corners-dt8-q8/summary.json",
        "event16_galerkin_q8": "galerkin-corners-dt8-q8-final/summary.json",
        "event16_bulk_work": "galerkin-corners-dt8-q8-final/bulk_work.json",
        "event16_adjoint": "galerkin-corners-dt8-q8-final/adjoint.json",
        "event16_comparison": "galerkin-comparison-final/event16_results.json",
        "quadrature_fine": "galerkin-quadrature-fine/summary.json",
        "seed_aligned": "galerkin-corner-refinement-aligned8/summary.json",
        "seed_patch": "galerkin-corner-refinement-aligned8-sub4-end0.5/summary.json",
        "patch_preflight": "galerkin-patch-preflight.json",
        "fine_run_configuration": "galerkin-patch5-dt8-q6/run_manifest.json",
        "fine_run_result_if_completed": "galerkin-patch5-dt8-q6/summary.json",
        "fine_event_comparison": "galerkin-space-comparison/event16_results.json",
        "fine_bulk_work": "galerkin-patch5-dt8-q6/bulk_work.json",
        "fine_adjoint": "galerkin-patch5-dt8-q6/adjoint.json",
        "fine_streamed_replay": "galerkin-patch5-dt8-q6-streamed/summary.json",
        "fine_streamed_adjoint": "galerkin-patch5-dt8-q6-streamed/adjoint.json",
        "history_streaming_equivalence": "history-streaming-fine-equivalence.json",
        "history_storage_coarse": "history-storage-dynamic-final/summary.json",
        "history_storage_fine": "history-storage-fine-production/summary.json",
        "cache_migration_fine": "history-migration-fine.json",
        "bessel_prototype": "bessel-experiment.json",
        "front_comparison": "front-comparison/summary.json",
        "front_table_replay": "front-comparison-replay/summary.json",
    }
    results = {
        "date": "2026-10-23",
        "scope": "Galerkin bulk/short dynamics and ongoing event validation; no equal-accuracy FEM speed claim",
        "full_original_event_reproduced": False,
    }
    for key, path in files.items():
        target = ROOT / "outputs" / path
        if target.exists():
            item = json.loads(target.read_text())
            if key in ["event16_comparison", "fine_event_comparison"]:
                item = {
                    k: item[k]
                    for k in ["fem", "bem", "comparison_scope", "full_original_event_reproduced"]
                }
            results[key] = item
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
    plot_static_onset(figures)
    for name, target in [
        ("event16_comparison.png", "galerkin_event16.png"),
        ("event16_profiles.png", "galerkin_profiles.png"),
    ]:
        source = ROOT / "outputs/galerkin-comparison-final/figures" / name
        if source.exists():
            shutil.copyfile(source, figures / target)
    source = ROOT / "outputs/front-comparison/fronts.png"
    if source.exists():
        shutil.copyfile(source, figures / "galerkin_fronts.png")
    for name, target in [
        ("event16_comparison.png", "galerkin_space_event16.png"),
        ("event16_profiles.png", "galerkin_space_profiles.png"),
    ]:
        source = ROOT / "outputs/galerkin-space-comparison/figures" / name
        if source.exists():
            shutil.copyfile(source, figures / target)


if __name__ == "__main__":
    main()

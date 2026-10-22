"""Publish small measured artifacts for the literature-driven development step."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from finiteblock_bem.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def main():
    passivity = read("outputs/passivity-audit/summary.json")
    quadrature = read("outputs/passivity-audit/quadrature.json")
    benchmark = read("outputs/sampling-benchmark/summary.json")
    comparison = read("outputs/parsimonious-corners-dt8-final/sampling_comparison.json")
    cold = read("outputs/parsimonious-corners-dt8/summary.json")
    data = {
        "date": "2026-10-23",
        "goal_achieved": False,
        "reason": "FEM rupture mismatch and sampled bulk passivity failures remain unresolved",
        "passivity": passivity,
        "quadrature_and_full_boundary_work_check": quadrature,
        "controlled_kernel_benchmark": benchmark,
        "controlled_kernel_benchmark_reverse_order": read(
            "outputs/sampling-benchmark-reverse/summary.json"
        ),
        "full_horizon_sampling_comparison": comparison,
        "full_horizon_cold_sampling": cold,
        "cold_timing_scope": "The full-horizon exploratory build overlapped brief serial passivity audits; use the separate controlled benchmark for speed comparisons.",
        "cache_rekey_after_resource_accounting_fix": read("outputs/sampling-cache-rekey.json"),
        "next_refinement_preflight_only": read(
            "outputs/refinement-resource-probe-updated/summary.json"
        ),
    }
    adjoint = ROOT / "outputs/parsimonious-corners-dt8-final/adjoint.json"
    if adjoint.exists():
        data["conditional_adjoint"] = json.loads(adjoint.read_text())
        reference = read("outputs/interface-corners-dt8-final/adjoint.json")
        data["adjoint_absolute_difference_from_dense"] = np.abs(
            np.array(data["conditional_adjoint"]["gradient"]) - reference["gradient"]
        ).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for mesh in dict.fromkeys(r["mesh"] for r in passivity["records"]):
        selected = [
            r
            for r in passivity["records"]
            if r["mesh"] == mesh and r["sigma_per_s"] == 200 and r["frequency_hz"] > 0
        ]
        x = np.array([r["frequency_hz"] for r in selected])
        axes[0].semilogx(x, [r["min_over_spectral_radius"] for r in selected], "o-", label=mesh)
        axes[1].loglog(x, [r["reciprocity_relative_frobenius"] for r in selected], "o-")
    axes[0].axhline(0, color="k", lw=0.7)
    axes[0].set(
        ylabel="min eig(H) / spectral radius(H)", title="Negative values fail sampled passivity"
    )
    axes[1].set(ylabel="Relative reciprocity defect", title="Work-conjugate velocity impedance")
    for ax in axes:
        ax.set_xlabel("Frequency [Hz]")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Linear bulk only: Re(s)=200 /s; contact stability is not certified")
    fig.savefig(ROOT / "docs/figures/passivity_audit.png", dpi=160)
    plt.close(fig)
    write_json(ROOT / "docs/literature_results.json", data)
    print(ROOT / "docs/literature_results.json")


if __name__ == "__main__":
    main()

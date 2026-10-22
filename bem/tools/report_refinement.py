"""Publish small reproducible spatial-refinement and conditional-adjoint artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from finiteblock_bem.comparison import relative_l2
from finiteblock_bem.io import environment_record, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    fem = np.genfromtxt(
        ROOT.parent / "validation/static_bem2d/reference/fem_history.csv", delimiter=",", names=True
    )
    fem = fem[fem["time"] >= 0.002]
    tf = fem["time"] - 0.002
    mappings = [
        ("max_cumulative_slip_mm", "max_slip", "Maximum cumulative slip [mm]"),
        ("mean_contact_shear_mpa", "avg_tau", "Mean contact shear [MPa]"),
        ("mean_contact_pressure_mpa", "avg_sigma_n", "Mean contact pressure [MPa]"),
        ("platen_mm", "loading_face_displacement", "Platen displacement [mm]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, layout="constrained")
    for ax, (_, key, label) in zip(axes.flat, mappings, strict=True):
        ax.plot(tf * 1e3, fem[key], "k-", lw=1.2, label="FEM 2 mm (archive)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    records = []
    current = environment_record()
    for run in args.runs:
        s = json.loads((run / "summary.json").read_text())
        controls = json.loads((run / "run_manifest.json").read_text())["arguments"]
        if s["source_changed_during_run"]:
            raise ValueError(f"Frozen-source rerun required: {run}")
        for name, sha in s["environment"]["source_sha256"].items():
            if current["source_sha256"].get(name) != sha:
                raise ValueError(f"{run}: numerical source {name} no longer matches")
        h = np.genfromtxt(run / "history.csv", delimiter=",", names=True)
        with np.load(run / "trajectory.npz", allow_pickle=False) as a:
            nc = len(a["contact_y"])
            nm, ns = len(a["moving_starts"]), len(a["stationary_starts"])
            first = np.flatnonzero(np.max(a["cumulative_slip"], axis=1) > 1e-7)
            location = (
                None
                if not len(first)
                else float(a["contact_y"][np.argmax(a["cumulative_slip"][first[0]])])
            )
        label = (
            f"Corners 2 mm / contact <=20 mm ({nm}+{ns} panels)"
            if controls.get("corner_h")
            else f"Uniform <=20 mm ({nm}+{ns} panels)"
        )
        errors = {}
        for ax, (key, fkey, _) in zip(axes.flat, mappings, strict=True):
            ax.plot(h["time_s"] * 1e3, h[key], lw=1, label=label)
            weights = np.ones(len(h))
            weights[[0, -1]] = 0.5
            errors[key] = relative_l2(h[key], np.interp(h["time_s"], tf, fem[fkey]), weights)
        size = 2 * nc + 1
        record = {
            "name": run.name,
            "controls": controls,
            "summary": s,
            "first_slip_time_s": None if not len(first) else float(h["time_s"][first[0]]),
            "first_slip_y_mm": location,
            "history_relative_l2": errors,
            "retained_weight_bytes": (s["requested_steps"] + 1) * size**2 * 8,
            "equivalent_full_weight_bytes": (s["requested_steps"] + 1) * 4 * (nm**2 + ns**2) * 8,
        }
        if (run / "adjoint.json").exists():
            record["adjoint"] = json.loads((run / "adjoint.json").read_text())
            with np.load(run / "adjoint.npz", allow_pickle=False) as a:
                sensitivity = a["actuator_gradient"]
            fig2, ax2 = plt.subplots(figsize=(10, 3.5), layout="constrained")
            ax2.plot(h["time_s"] * 1e3, sensitivity, lw=0.8)
            ax2.set(
                xlabel="Shear time [ms]",
                ylabel="d(final mean c) / d(D_n) [mm/mm]",
                title="Conditional discrete adjoint: fixed bulk, lift and recorded loading",
            )
            ax2.grid(alpha=0.2)
            fig2.savefig(ROOT / "docs/figures/actuator_sensitivity.png", dpi=160)
            plt.close(fig2)
            for filename, key in [
                ("adjoint-mus-eps1e-6.json", "mu_s_larger_step_check"),
                ("adjoint-mus-eps1e-7.json", "mu_s_smaller_step_check"),
            ]:
                if (run / filename).exists():
                    record[key] = json.loads((run / filename).read_text())[
                        "finite_difference_checks"
                    ]
        records.append(record)
    for ax in axes[-1]:
        ax.set_xlabel("Shear time [ms]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Spatial refinement with exact interface condensation\n"
        "Same 45 ms loading policy; resolution convergence is not established"
    )
    fig.savefig(ROOT / "docs/figures/event16_refinement.png", dpi=160)
    plt.close(fig)
    output = {"date": "2026-10-23", "runs": records, "full_original_event_reproduced": False}
    for folder, key in [
        ("elastic-onset", "elastic_onset"),
        ("condensation-ad", "short_ad_comparison"),
    ]:
        path = ROOT / "outputs" / folder / "summary.json"
        if path.exists():
            output[key] = json.loads(path.read_text())
    cold = ROOT / "outputs/interface-corners-dt8/summary.json"
    if cold.exists():
        s = json.loads(cold.read_text())
        output["corner_cold_kernel_measurement"] = {
            "seconds": s["timing_seconds"]["kernel_build_or_cache_load"],
            "kernel_workers": s["kernel_workers"],
            "environment": s["environment"],
            "note": "Development kernel build; subsequent forward rerun used the same cached transfer and corrected nonlinear globalization.",
        }
    cold = ROOT / "outputs/interface-h20-cold-measurement.json"
    if cold.exists():
        s = json.loads(cold.read_text())
        output["uniform_cold_kernel_measurement"] = {
            "seconds": s["timing_seconds"]["kernel_build_or_cache_load"],
            "kernel_workers": s["kernel_workers"],
            "environment": s["environment"],
            "note": "Earlier cold build of the identical linear transfer; final forward rerun used its cached weights with fixed lift traction.",
        }
    rekey = ROOT / "outputs/cache-rekey-audit.json"
    if rekey.exists():
        output["local_cache_rekey"] = {
            "reason": "Cache key now fingerprints the frequency-transfer class separately from the contact equilibrium class. Transfer source and all bulk dependencies were checked byte-identical before reusing weights.",
            "records": json.loads(rekey.read_text()),
        }
    write_json(ROOT / "docs/refinement_results.json", output)
    print(ROOT / "docs/refinement_results.json")


if __name__ == "__main__":
    main()

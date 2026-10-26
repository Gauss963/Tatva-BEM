"""Plot contact-grid and seed-field controls with distinct physical labels."""

import argparse
import json
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--fem-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose new comparison figures outside protected Tatva")
    records = [json.loads((p / "summary.json").read_text()) for p in args.runs]
    first = records[0]
    for r in records:
        if r["status"] != "completed_requested_horizon" or r["duration_s"] != 0.045:
            raise ValueError("Require completed full refined-contact events")
        for key in ("case", "dt_s", "bulk_contact_nodes", "panels", "constitutive_substeps"):
            if r[key] != first[key]:
                raise ValueError(f"Contact-only controls differ in {key}")
        if r["restart_seed_override"]["sha256"] != first["restart_seed_override"]["sha256"]:
            raise ValueError("Contact-only controls use different source seed values")
    paths = [Path(__file__), TOOLS / "fem_reference_io.py", TOOLS / "compare_fem_mesh.py"]
    paths.extend(p / "summary.json" for p in [*args.runs, *args.fem_runs])
    paths.extend(p / "resolved_run_config.json" for p in args.fem_runs)
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    data_paths = [p / "trajectory.npz" for p in args.runs] + [
        p / "simulation.h5" for p in args.fem_runs
    ]
    stats = {str(p.resolve()): (p.stat().st_size, p.stat().st_mtime_ns) for p in data_paths}
    mapping = [
        ("max_cumulative_slip_mm", "max_slip", "Maximum cumulative slip [mm]"),
        ("mean_contact_pressure_mpa", "avg_sigma_n", "Mean pressure [MPa]"),
        ("mean_contact_shear_mpa", "avg_tau", "Mean shear [MPa]"),
        ("platen_mm", "loading_face_displacement", "Platen displacement [mm]"),
        ("actuator_mm", "applied_shear_displacement", "Actuator displacement [mm]"),
        ("applied_shear_mpa", "applied_shear", "Applied spring shear [MPa]"),
    ]
    profiles = [
        ("cumulative_slip", "Cumulative slip [mm]"),
        ("pressure", "Pressure [MPa]"),
        ("shear", "Shear [MPa]"),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True, layout="constrained")
    endpoint, profile_axes = plt.subplots(1, 3, figsize=(14, 6), layout="constrained")
    reader = runpy.run_path(str(TOOLS / "fem_reference_io.py"))
    final_reader = runpy.run_path(str(TOOLS / "compare_fem_mesh.py"))
    for index, folder in enumerate(args.fem_runs):
        record = json.loads((folder / "summary.json").read_text())
        if record["status"] != "completed" or not record["full_original_horizon"]:
            raise ValueError("Require completed original-horizon FEM references")
        trace = reader["read_fem_trace"](folder / "simulation.h5")
        selected = trace["phase_id"] == 2
        time = trace["history"]["time"][selected] - record["summary"]["pressure_time"]
        label = f"FEM h={record['summary']['mesh_size']:g} mm"
        color = "black" if index == 0 else "0.4"
        style = "-" if index == 0 else "-."
        for ax, (_, key, _) in zip(axes.flat, mapping, strict=True):
            ax.plot(
                time * 1e3, trace["history"][key][selected], style, color=color, lw=1.5, label=label
            )
        config = json.loads((folder / "resolved_run_config.json").read_text())
        y, fields = final_reader["read_final_fields"](folder / "simulation.h5", config)
        for ax, key in zip(
            profile_axes, ("cumulative_slip", "pressure", "elastic_trial_shear"), strict=True
        ):
            ax.plot(y, fields[key], style, color=color, lw=1.5, label=label)
    subdivisions = sorted({r["subdivisions"] for r in records})
    palette = {count: plt.get_cmap("tab10")(index) for index, count in enumerate(subdivisions)}
    labels = []
    for folder, record in zip(args.runs, records, strict=True):
        mode = record["seed_interpolation"]
        label = f"BEM subdivisions={record['subdivisions']}, {record['contact_nodes']} states, {mode} seed"
        labels.append(dict(run=str(folder.resolve()), label=label))
        style = "--" if mode == "original-p1" else "-"
        color = palette[record["subdivisions"]]
        history = np.genfromtxt(folder / "history.csv", delimiter=",", names=True)
        if not np.all(np.isfinite(history["time_s"])) or abs(history["time_s"][-1] - 0.045) > 1e-12:
            raise ValueError("Invalid complete BEM history time grid")
        for ax, (key, _, _) in zip(axes.flat, mapping, strict=True):
            ax.plot(history["time_s"] * 1e3, history[key], style, color=color, lw=1.1, label=label)
        with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
            for ax, (key, _) in zip(profile_axes, profiles, strict=True):
                field = data[key]
                value = field[-1].copy()
                del field
                ax.plot(data["contact_y"], value, style, color=color, lw=1.1, label=label)
    for ax, (_, _, title) in zip(axes.flat, mapping, strict=True):
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Time from shear start [ms]")
    for ax, (_, title) in zip(profile_axes, profiles, strict=True):
        ax.set_xlabel("Contact coordinate y [mm]")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
    for fig, first_ax in ((figure, axes[0, 0]), (endpoint, profile_axes[0])):
        fig.legend(*first_ax.get_legend_handles_labels(), loc="outside lower center", fontsize=8)
    figure.suptitle("Fixed bulk/time grid: contact integration and original seed-field controls")
    endpoint.suptitle("Nominal 45 ms endpoint; FEM uses true float64 final_state")
    output.mkdir(parents=True)
    figure.savefig(output / "contact_refinement_histories.png", dpi=170)
    endpoint.savefig(output / "contact_refinement_endpoints.png", dpi=170)
    plt.close(figure)
    plt.close(endpoint)
    if any(fingerprint(p) != value for p, value in frozen.items()) or any(
        (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != value for p, value in stats.items()
    ):
        raise ValueError("Figure source or input data changed")
    write_json(
        output / "figure_manifest.json",
        dict(
            status="completed_contact_refinement_figures",
            frozen_input_sha256=frozen,
            source_size_and_mtime_ns=stats,
            bem_curves=labels,
            initial_seed_field_modes_shown_separately=True,
            scope="Saved histories and endpoints; physical convergence and timing require the numerical comparison reports",
        ),
    )
    print(output / "figure_manifest.json", flush=True)


if __name__ == "__main__":
    main()

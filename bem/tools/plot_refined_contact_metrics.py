"""Plot measured full-event endpoint metrics from the published evidence JSON."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from finiteblock_bem.io import fingerprint, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a new figure directory outside protected Tatva")
    before = fingerprint(args.evidence)
    data = json.loads(args.evidence.read_text())
    reference = data["early_endpoint_comparison"]["data"]["runs"][0]["fields"]["cumulative_slip"]
    rows = data["endpoint_metrics_versus_fem_h1"]
    subdivisions = np.array([r["subdivisions"] for r in rows])
    fields = [("cumulative_slip", "Cumulative slip"), ("pressure", "Pressure"), ("shear", "Shear")]
    values = np.array([[r[key + "_relative_p1_l2"] * 100 for r in rows] for key, _ in fields])
    slip = np.array([r["max_slip_mm"] for r in rows])
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(slip)):
        raise ValueError("Plot requires finite measured endpoint metrics")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    x = np.arange(len(rows))
    for column, (_, label) in enumerate(fields):
        axes[0].plot(x, values[column], marker="o", label=label)
    axes[0].set_ylabel("Endpoint P1 L2 difference from FEM h1 [%]")
    axes[0].set_title("Stress profiles remain different")
    axes[0].legend()
    axes[1].plot(x, slip, marker="o", label="BEM maximum slip")
    axes[1].axhline(
        reference["fem_h1"]["final_reference_maximum"], color="0.35", ls="--", label="FEM h1"
    )
    axes[1].axhline(
        reference["fem_h2"]["final_reference_maximum"], color="black", ls=":", label="FEM h2"
    )
    axes[1].set_ylabel("Maximum cumulative slip [mm]")
    axes[1].set_title("Small change in slip is not full convergence")
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(
            x,
            [
                f"{n}\n"
                + (
                    "preserved offset"
                    if r["static_initialization"] == "preserve-offset"
                    else "healed Coulomb"
                )
                for n, r in zip(subdivisions, rows, strict=True)
            ],
        )
        ax.set_xlabel("Contact subdivisions / static initialization")
        ax.grid(alpha=0.25)
    figure.suptitle("Complete 45 ms event: fixed bulk and 2 us time grid; no excluded endpoints")
    output.mkdir(parents=True)
    figure.savefig(output / "contact_integration_metrics.png", dpi=170)
    plt.close(figure)
    if fingerprint(args.evidence) != before:
        raise ValueError("Endpoint evidence changed during plotting")
    write_json(
        output / "figure_manifest.json",
        dict(
            status="completed_measured_endpoint_metric_figure",
            evidence_path=str(args.evidence.resolve()),
            evidence_sha256=before,
            figure_source_sha256=fingerprint(__file__),
            rows=rows,
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
        ),
    )
    print(output / "contact_integration_metrics.png", flush=True)


if __name__ == "__main__":
    main()

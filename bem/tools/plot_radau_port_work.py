"""Plot signed incremental work and its separately visible residual bound."""

import argparse
import json
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
DIGEST = runpy.run_path(str(TOOLS / "replay_resolvent.py"))["digest_file"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Fresh PNG path")
    args = parser.parse_args()
    output = args.output.resolve()
    targets = [output, output.with_suffix(".svg"), output.with_suffix(".json")]
    if (
        output.suffix.lower() != ".png"
        or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve())
        or any(p.exists() for p in targets)
    ):
        raise ValueError("Choose a fresh PNG output outside protected Tatva")
    summary_path, history_path = args.evidence / "summary.json", args.evidence / "work_history.npz"
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "completed_incremental_radau_bulk_port_work_diagnostic":
        raise ValueError("Require completed signed-work evidence")
    if DIGEST(history_path) != summary["history_sha256"]:
        raise ValueError("Saved work history changed")
    with np.load(history_path, allow_pickle=False) as data:
        time = data["endpoint_time"] * 1e6
        work = data["cumulative_work"]
        weighted = data["radius_weighted_cumulative_work"]
        bound = data["cumulative_residual_error_bound"]
    if not all(np.all(np.isfinite(v)) and v.shape == time.shape for v in [work, weighted, bound]):
        raise ValueError("Incomplete or nonfinite work history")
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.3), sharex=True, constrained_layout=True)
    title = "Incremental RK bulk-port work"
    if not summary["complete_prepared_horizon_checked"]:
        title += " — short diagnostic"
    fig.suptitle(title, fontsize=15)
    axes[0].plot(time, work, color="#174b73", label="Signed stage-weighted work", linewidth=2)
    axes[0].fill_between(time, work - bound, work + bound, color="#174b73", alpha=0.2)
    axes[0].plot(
        time, weighted, color="#a46a25", linestyle="--", label="CQ radius weighting (diagnostic)"
    )
    axes[0].set_ylabel(r"Work ($\mathrm{MPa}\,\mathrm{mm}^2$)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[0].text(
        0.98,
        0.04,
        "Residual uncertainty shown separately below",
        transform=axes[0].transAxes,
        ha="right",
        fontsize=9,
        color="#52606d",
    )
    axes[1].plot(time, bound, color="#6d435a", linewidth=2)
    axes[1].set_ylabel(r"Cumulative error bound ($\mathrm{MPa}\,\mathrm{mm}^2$)")
    axes[1].set_xlabel("Shear time (µs)")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].text(
        0.98,
        0.05,
        "From the independently audited force residual",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=9,
        color="#52606d",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(time[0], time[-1])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(targets[0], dpi=180)
    fig.savefig(targets[1])
    plt.close(fig)
    write_json(
        targets[2],
        dict(
            source_summary=str(summary_path.resolve()),
            source_summary_sha256=fingerprint(summary_path),
            work_history_sha256=summary["history_sha256"],
            plotting_source_sha256=fingerprint(Path(__file__)),
            output_files={p.name: DIGEST(p) for p in targets[:2]},
            scope="Incremental boundary work with a force-residual bound; no interior energy or FEM stability/speed certificate",
        ),
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()

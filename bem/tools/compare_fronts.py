"""Compare first passage of cumulative slip at common physical interface nodes.

FEM HDF5 is opened read-only. A small arrival table can instead be supplied
for offline replay. Interpolated arrivals are estimates inside the recorded
crossing brackets, not a claim of sub-frame temporal resolution.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from finiteblock_bem.io import write_json

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "validation/static_bem2d/reference"
DC = 0.0125959
THRESHOLDS = {"micro": 1e-7, "Dc10": 0.1 * DC, "Dc": DC}


def arrivals(time, cumulative, threshold):
    """Return first saved, interpolated, and left-bracket crossing times."""
    time = np.asarray(time, dtype=float)
    cumulative = np.asarray(cumulative, dtype=float)
    if np.any(np.diff(time) <= 0) or cumulative.shape[0] != len(time):
        raise ValueError("Strictly increasing times must match the saved cumulative field")
    reached = cumulative > threshold
    first = np.argmax(reached, axis=0)
    left = np.maximum(first - 1, 0)
    columns = np.arange(cumulative.shape[1])
    saved, lower = time[first].copy(), time[left].copy()
    denominator = cumulative[first, columns] - cumulative[left, columns]
    fraction = np.divide(
        threshold - cumulative[left, columns],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    interpolated = lower + np.clip(fraction, 0, 1) * (saved - lower)
    for value in [saved, lower, interpolated]:
        value[~reached.any(axis=0)] = np.nan
    return dict(saved=saved, interpolated=interpolated, lower=lower)


def write_table(path, y, records):
    columns = {"y_mm": y}
    for name, values in records.items():
        columns.update({f"{name}_{key}_s": value for key, value in values.items()})
    np.savetxt(
        path,
        np.column_stack(list(columns.values())),
        delimiter=",",
        header=",".join(columns),
        comments="",
        fmt="%.17g",
    )


def read_fem(table):
    if table is not None:
        data = np.genfromtxt(table, delimiter=",", names=True)
        metadata = json.loads(table.with_suffix(".json").read_text())
        if metadata["thresholds_mm"] != THRESHOLDS:
            raise ValueError("Reference arrival thresholds differ")
        return (
            data["y_mm"],
            {
                name: {key: data[f"{name}_{key}_s"] for key in ["saved", "interpolated", "lower"]}
                for name in THRESHOLDS
            },
            metadata,
        )
    import h5py

    provenance = json.loads((REFERENCE / "provenance.json").read_text())
    path = Path(provenance["hdf5"]["path"])
    before = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        provenance["hdf5"]["bytes"],
        provenance["hdf5"]["mtime_ns"],
    ):
        raise ValueError("FEM source differs from the recorded reference")
    with h5py.File(path, "r") as data:
        columns = [v.decode() if isinstance(v, bytes) else v for v in data["history_columns"][:]]
        shear = data["phase_id"][:] == 2
        time = data["history"][:, columns.index("time")][shear] - 0.002
        cumulative = data["interface/cumulative_slip"][:][shear]
        y = data["interface/contact_line_y"][:]
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Reference file changed during the read")
    record = {name: arrivals(time, cumulative, value) for name, value in THRESHOLDS.items()}
    metadata = dict(
        thresholds_mm=THRESHOLDS,
        source_hdf5=provenance["hdf5"],
        source_hdf5_unchanged=True,
        time_origin_s=0.002,
        frame_spacing_max_s=float(np.max(np.diff(time))),
        scope="First cumulative-slip passage at saved FEM interface nodes",
    )
    return y, record, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--reference-table", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/front-comparison")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    y, reference, metadata = read_fem(args.reference_table)
    write_table(args.output / "fem_arrivals.csv", y, reference)
    write_json(args.output / "fem_arrivals.json", metadata)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, (name, threshold) in zip(axes, THRESHOLDS.items(), strict=True):
        ax.plot(y, reference[name]["interpolated"] * 1000, "k-", label="FEM h=2 mm")
        ax.set(
            xlabel="Interface y [mm]", ylabel="First passage [ms]", title=f"c > {threshold:g} mm"
        )
        ax.grid(alpha=0.2)
    results = []
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        if summary["case"]["contact"]["characteristic_slip"] != DC:
            raise ValueError("This reference uses the original event-16 characteristic slip")
        with np.load(run / "trajectory.npz", allow_pickle=False) as data:
            contact_y = data["contact_y"]
            if not np.allclose(contact_y[[0, -1]], y[[0, -1]], atol=1e-8):
                raise ValueError("Reference and BEM must cover the same interval")
            right = np.clip(np.searchsorted(contact_y, y, side="right"), 1, len(contact_y) - 1)
            left = right - 1
            alpha = (y - contact_y[left]) / (contact_y[right] - contact_y[left])
            cumulative = (1 - alpha) * data["cumulative_slip"][:, left]
            cumulative += alpha * data["cumulative_slip"][:, right]
            records = {
                name: arrivals(data["time"], cumulative, threshold)
                for name, threshold in THRESHOLDS.items()
            }
        write_table(args.output / f"{run.name}.csv", y, records)
        metrics = {}
        label = f"BEM {sum(summary['panels'])} panels, dt={summary['dt_s'] * 1e6:g} us"
        for ax, name in zip(axes, THRESHOLDS, strict=True):
            target, actual = reference[name]["interpolated"], records[name]["interpolated"]
            keep = np.isfinite(target) & np.isfinite(actual)
            difference = 1000 * (actual[keep] - target[keep])
            metrics[name] = dict(
                common_reached_nodes=int(np.count_nonzero(keep)),
                fem_unreached_nodes=int(np.count_nonzero(~np.isfinite(target))),
                bem_unreached_nodes=int(np.count_nonzero(~np.isfinite(actual))),
                max_abs_difference_ms=float(np.max(abs(difference))) if len(difference) else None,
                rms_difference_ms=float(np.sqrt(np.mean(difference**2)))
                if len(difference)
                else None,
                bem_first_passage_ms=(
                    float(np.nanmin(actual) * 1000) if np.isfinite(actual).any() else None
                ),
                fem_first_passage_ms=(
                    float(np.nanmin(target) * 1000) if np.isfinite(target).any() else None
                ),
            )
            ax.plot(y, actual * 1000, label=label)
        results.append(dict(name=run.name, metrics=metrics))
    for ax in axes:
        ax.legend(fontsize=7)
    figure.suptitle("Cumulative-slip first passage; spatial P1 and within-frame time interpolation")
    figure.savefig(args.output / "fronts.png", dpi=170)
    plt.close(figure)
    record = dict(
        reference=metadata,
        results=results,
        scope="Saved-frame crossing estimates; not instantaneous front speed or sub-step accuracy",
    )
    write_json(args.output / "summary.json", record)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

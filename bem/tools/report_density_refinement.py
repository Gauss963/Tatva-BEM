"""Compare fixed-trace density controls with each completed float64 FEM mesh.

All profile norms integrate the two P1 functions on their common knots. End
regions remain in the full norm; separate contributions locate the error rather
than defining a more favorable acceptance interval. This is a static diagnostic.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from finiteblock_bem.io import fingerprint, write_json


def validate_profile(y, values):
    y, values = np.asarray(y), np.asarray(values)
    if (
        y.ndim != 1
        or len(y) < 2
        or values.shape != y.shape
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(values))
        or np.any(np.diff(y) <= 0)
    ):
        raise ValueError("Profiles require ordered finite coordinates and finite values")
    return y, values


def profile_integrals(y, values, reference_y, reference, *, lower, upper):
    """Exact squared L2 error and target norm of two physical P1 profiles."""
    y, values = validate_profile(y, values)
    reference_y, reference = validate_profile(reference_y, reference)
    if not np.isfinite(lower + upper) or not max(y[0], reference_y[0]) <= lower < upper <= min(
        y[-1], reference_y[-1]
    ):
        raise ValueError("The integration interval must be covered by both profiles")
    knots = np.unique(
        np.r_[
            lower,
            upper,
            y[(y > lower) & (y < upper)],
            reference_y[(reference_y > lower) & (reference_y < upper)],
        ]
    )
    actual = np.interp(knots, y, values)
    target = np.interp(knots, reference_y, reference)

    def square(v):
        return float(np.sum(np.diff(knots) * (v[:-1] ** 2 + v[:-1] * v[1:] + v[1:] ** 2) / 3))

    return square(actual - target), square(target)


def compare_profiles(actual, reference, end_window):
    y, yr = actual["y_mm"], reference["y_mm"]
    if not np.array_equal(y[[0, -1]], yr[[0, -1]]):
        raise ValueError("Static profiles must cover the same complete contact interval")
    lower, upper = float(y[0]), float(y[-1])
    if not np.isfinite(end_window) or not 0 < 2 * end_window < upper - lower:
        raise ValueError("End window must leave a nonempty contact interior")
    intervals = [
        (lower, lower + end_window),
        (lower + end_window, upper - end_window),
        (upper - end_window, upper),
    ]
    result = {}
    for field in ["p_mpa", "tau_mpa"]:
        parts = []
        for a, b in intervals:
            error, norm = profile_integrals(
                y, actual[field], yr, reference[field], lower=a, upper=b
            )
            parts.append(dict(interval_mm=[a, b], squared_error=error, squared_reference_norm=norm))
        error = sum(p["squared_error"] for p in parts)
        norm = sum(p["squared_reference_norm"] for p in parts)
        result[field] = dict(
            relative_l2=None if norm == 0 else float(np.sqrt(error / norm)),
            squared_error=error,
            squared_reference_norm=norm,
            regions=[
                dict(p, fraction_of_squared_error=p["squared_error"] / error if error else 0.0)
                for p in parts
            ],
        )
    return result


def main():
    # Plotting is optional; importing numerical diagnostics must not require it.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--fem-static", type=Path, required=True)
    parser.add_argument("--end-window", type=float, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new static comparison output directory")
    study_path, fem_path = args.study / "summary.json", args.fem_static / "summary.json"
    study, fem = [json.loads(p.read_text()) for p in [study_path, fem_path]]
    if fem.get("model_dtype") != "float64" or fem.get("historical_control"):
        raise ValueError("Require the authored float64 FEM static control")
    if len(fem["records"]) != len(fem["effective_run_configs"]):
        raise ValueError("The FEM static study is incomplete")
    seed_hashes = [
        v
        for k, v in fem["protected_input_sha256"].items()
        if Path(k).name == "ev16_restart_interface.csv"
    ]
    if seed_hashes != [study["environment"]["restart_data_sha256"]]:
        raise ValueError("Static controls use different restart seed data")
    inputs = [Path(__file__), study_path, fem_path]
    profiles, comparisons = [], []
    fem_profiles = []
    for row in fem["records"]:
        if row["model_dtype"] != "float64" or not row["prediction_valid"]:
            raise ValueError("Invalid FEM static control")
        path = args.fem_static / f"h{row['h_mm']}.csv"
        inputs.append(path)
        data = np.genfromtxt(path, delimiter=",", names=True)
        fem_profiles.append((row, data))
        profiles.append((f"FEM h={row['h_mm']:g} mm", data, dict(ls="--", lw=1.1)))
    for row in study["records"]:
        if not row["prediction_valid"]:
            raise ValueError("Invalid BEM static control")
        path = args.study / f"factor{row['density_factor']}.csv"
        inputs.append(path)
        data = np.genfromtxt(path, delimiter=",", names=True)
        profiles.append((f"BEM density x{row['density_factor']}", data, dict(lw=1.1)))
        comparisons.append(
            dict(
                density_factor=row["density_factor"],
                density_panels=row["density_panels"],
                predicted_y_mm=row["predicted_first_y_mm"],
                predicted_shear_time_s=row["predicted_shear_time_s"],
                construction_and_static_seconds=row["construction_and_static_seconds"],
                fem_meshes=[
                    dict(h_mm=fr["h_mm"], profiles=compare_profiles(data, fd, args.end_window))
                    for fr, fd in fem_profiles
                ],
            )
        )
    frozen = {str(p.resolve()): fingerprint(p) for p in inputs}
    args.output.mkdir(parents=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    y = profiles[0][1]["y_mm"]
    limits = [(y[0], y[-1]), (y[0], y[0] + args.end_window), (y[-1] - args.end_window, y[-1])]
    for row, field in enumerate(["p_mpa", "tau_mpa"]):
        for col, interval in enumerate(limits):
            ax = axes[row, col]
            for label, data, style in profiles:
                ax.plot(data["y_mm"], data[field], label=label, **style)
            ax.set(
                xlim=interval,
                xlabel="Contact y [mm]",
                ylabel="Pressure [MPa]" if row == 0 else "Shear [MPa]",
            )
            if col:
                visible = np.concatenate(
                    [
                        data[field][(data["y_mm"] >= interval[0]) & (data["y_mm"] <= interval[1])]
                        for _, data, _ in profiles
                    ]
                )
                padding = max(float(np.ptp(visible)) * 0.08, 0.5)
                ax.set_ylim(float(visible.min()) - padding, float(visible.max()) + padding)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Fixed P1 trace / refined P0 density: static profiles, not dynamic rupture")
    fig.savefig(args.output / "density_static_profiles.png", dpi=170)
    plt.close(fig)
    if any(fingerprint(p) != sha for p, sha in frozen.items()):
        raise ValueError("Static comparison input changed")
    write_json(
        args.output / "summary.json",
        dict(
            records=comparisons,
            retained_history_dofs=study["retained_history_dofs"],
            fem_static_predictions=fem["records"],
            frozen_input_sha256=frozen,
            end_window_mm=args.end_window,
            norm="Exact physical P1 L2 on union of knots; full contact includes both ends",
            full_original_event_reproduced=False,
            scope="Static fixed-plastic comparison. FEM h2 is the event reference; finer FEM meshes are distinct controls, not certified converged solutions.",
        ),
    )


if __name__ == "__main__":
    main()

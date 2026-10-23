"""Audit how saved CQ steps resolve the original slip-weakening transition.

The work comparison integrates mu(c) exactly while holding pressure at that
step's endpoint. It diagnoses constitutive time resolution; it is not a bound
on the coupled trajectory error or a total energy balance.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from finiteblock_bem.io import fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]


def audit(run, output):
    summary = json.loads((run / "summary.json").read_text())
    law = summary["case"]["contact"]
    dc = law["characteristic_slip"]
    mus, muk = law["static_friction"], law["kinetic_friction"]
    with np.load(run / "trajectory.npz", allow_pickle=False) as data:
        c, p, weights = data["cumulative_slip"], data["pressure"], data["contact_weights"]
        before, after = c[:-1], c[1:]
        increment = after - before
        if np.min(increment) < -1e-12:
            raise ValueError("Cumulative slip must be nondecreasing")
        np.testing.assert_allclose(
            increment, abs(np.diff(data["plastic_slip"], axis=0)), rtol=1e-10, atol=1e-12
        )
        expected_mu = mus - (mus - muk) * np.minimum(before / dc, 1)
        is_previous = bool(
            np.allclose(data["friction_coefficient"][1:], expected_mu, rtol=1e-12, atol=1e-12)
        )
        low, high = np.minimum(before, dc), np.minimum(after, dc)
        integrated_mu = muk * increment + (mus - muk) * (high - low - (high**2 - low**2) / (2 * dc))
        applied_mu = data["friction_coefficient"][1:] * increment
        actual_work = np.sum(data["shear"][1:] * np.diff(data["plastic_slip"], axis=0) * weights)
        np.testing.assert_allclose(actual_work, np.sum(applied_mu * p[1:] * weights), rtol=1e-10)
        gap = (applied_mu - integrated_mu) * p[1:] * weights
        if is_previous and np.min(gap) < -1e-9:
            raise AssertionError(
                "A decreasing previous-state coefficient cannot under-integrate mu"
            )
        skipped = np.any((before < 0.1 * dc) & (after > 0.9 * dc), axis=0)
        progress = np.max((high - low) / dc, axis=0)
        order = np.argsort(progress)
        fraction = np.cumsum(weights[order])
        fraction /= fraction[-1]
        line_quantiles = progress[order][np.searchsorted(fraction, [0.5, 0.9, 1])]
        local_ratio = np.max(increment / dc, axis=0)
        local_gap = np.sum(gap, axis=0)
        initial_scale = 0.5 * (mus - muk) * dc * float(np.sum(p[0] * weights))
        record = dict(
            name=run.name,
            dt_s=summary["dt_s"],
            pressure_mean_initial_mpa=float(np.average(p[0], weights=weights)),
            uses_previous_cumulative_strength=is_previous,
            maximum_increment_over_Dc=float(np.max(local_ratio)),
            maximum_increment_while_weakening_over_Dc=float(
                np.max(np.where(before < dc, increment / dc, 0))
            ),
            maximum_weakening_progress_in_one_step=float(np.max(progress)),
            per_node_maximum_progress_quantiles=dict(
                zip(
                    ["50_percent", "90_percent", "100_percent"],
                    np.quantile(progress, [0.5, 0.9, 1]).tolist(),
                    strict=True,
                )
            ),
            lumped_line_weighted_maximum_progress_quantiles=dict(
                zip(
                    ["50_percent", "90_percent", "100_percent"],
                    line_quantiles.tolist(),
                    strict=True,
                )
            ),
            nodes_skipping_10_to_90_percent=int(np.count_nonzero(skipped)),
            total_nodes=len(weights),
            line_fraction_skipping_10_to_90_percent=float(
                np.sum(weights[skipped]) / np.sum(weights)
            ),
            skipping_positions_mm=data["contact_y"][skipped].tolist(),
            applied_friction_work_mpa_mm2=float(actual_work),
            fixed_pressure_integrated_strength_work_mpa_mm2=float(
                np.sum(integrated_mu * p[1:] * weights)
            ),
            fixed_pressure_work_gap_mpa_mm2=float(np.sum(gap)),
            gap_fraction_of_applied_work=(
                float(np.sum(gap) / actual_work) if actual_work else None
            ),
            initial_pressure_weakening_excess_scale_mpa_mm2=initial_scale,
            gap_over_initial_weakening_excess_scale=(
                float(np.sum(gap) / initial_scale) if initial_scale else None
            ),
            trajectory_sha256=fingerprint(run / "trajectory.npz"),
        )
        np.savetxt(
            output / f"{run.name}.csv",
            np.column_stack((data["contact_y"], local_ratio, progress, local_gap)),
            delimiter=",",
            header="y_mm,max_increment_over_Dc,max_transition_progress_in_one_step,fixed_pressure_work_gap_mpa_mm2",
            comments="",
        )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/weakening-resolution")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(
        results=[audit(run, args.output) for run in args.runs],
        scope="Exact mu(c) integral with endpoint pressure held fixed; not total energy or coupled error bound",
        reference_integral="mu_k*(c_new-c_old) + (mu_s-mu_k)*[b-a-(b*b-a*a)/(2*Dc)], a=min(c_old,Dc), b=min(c_new,Dc)",
    )
    write_json(args.output / "summary.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

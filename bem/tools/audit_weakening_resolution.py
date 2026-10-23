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


def updated_local_condition(pressure, weakening, sliding, kt, mus, muk, dc):
    """Diagnostic for the OPTIONAL end-state return, not the current solver.

    The local denominator is kt-p*(mus-muk)/Dc. A nonpositive value on a
    weakening/sliding branch fails the current updated path's sufficient
    monotonicity check. It does not itself prove multiple roots at that q,
    nor instability of the original previous-state scheme.
    """
    pressure, weakening, sliding = (
        np.asarray(pressure, dtype=float),
        np.asarray(weakening, dtype=bool),
        np.asarray(sliding, dtype=bool),
    )
    kt, mus, muk, dc = (np.asarray(value, dtype=float) for value in [kt, mus, muk, dc])
    if pressure.shape != weakening.shape or pressure.shape != sliding.shape:
        raise ValueError("Pressure and branch masks must have matching shapes")
    slope = (np.asarray(mus) - muk) / dc
    eta = pressure * slope / kt
    denominator = kt - pressure * slope
    selected = weakening & sliding
    bad = selected & (denominator <= 0)
    critical = np.divide(kt, slope, out=np.full_like(slope, np.inf, dtype=float), where=slope > 0)
    return (
        dict(
            maximum_eta_while_weakening=float(np.max(eta[weakening]))
            if np.any(weakening)
            else None,
            maximum_eta_while_weakening_and_sliding=float(np.max(eta[selected]))
            if np.any(selected)
            else None,
            minimum_denominator_while_weakening_and_sliding=float(np.min(denominator[selected]))
            if np.any(selected)
            else None,
            nonpositive_denominator_weakening_sliding_entries=int(np.count_nonzero(bad)),
            nonpositive_denominator_nodes=int(np.count_nonzero(np.any(bad, axis=0))),
            critical_pressure_min_mpa=float(np.min(critical))
            if np.all(np.isfinite(critical))
            else None,
            interpretation="Sufficient monotonicity check used by the optional updated-state path; failure does not itself prove multiple roots at the accepted q or instability of the previous-state method",
        ),
        bad,
        eta,
    )


def updated_endpoint_candidates(pressure, trial, old_c, kt, mus, muk, dc):
    """Count admissible stick/soft/kinetic endpoint candidates, without selecting one."""
    mu = mus - (mus - muk) * np.minimum(old_c / dc, 1)
    slope = np.where(old_c < dc, (mus - muk) / dc, 0)
    denominator = kt - pressure * slope
    excess = trial - pressure * mu
    soft_increment = np.divide(
        excess, denominator, out=np.zeros_like(excess), where=denominator != 0
    )
    soft = (old_c < dc) & (soft_increment > 0) & (old_c + soft_increment < dc)
    kinetic_increment = (trial - pressure * muk) / kt
    kinetic = (kinetic_increment > 0) & (old_c + kinetic_increment >= dc)
    count = (trial <= pressure * mu).astype(int) + soft + kinetic
    count = np.where(pressure > 0, count, 1)
    flat = (pressure > 0) & (old_c < dc) & (denominator == 0) & (excess == 0)
    return count, flat


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
        if "contact_jumps" in data:
            trial = law["tangential_penalty"] * (
                data["contact_jumps"][1:, len(weights) :] - data["plastic_slip"][:-1]
            )
            sliding = (p[1:] > 0) & (abs(trial) > expected_mu * p[1:])
            branch_scope = "Accepted jumps and previous state, exact sliding predicate"
        else:
            sliding = abs(np.diff(data["plastic_slip"], axis=0)) > 0
            branch_scope = "Saved plastic increment proxy; exact jumps unavailable"
        local_condition, bad, eta = updated_local_condition(
            p[1:],
            before < dc,
            sliding,
            law["tangential_penalty"],
            mus,
            muk,
            dc,
        )
        local_condition["branch_activity_scope"] = branch_scope
        if "contact_jumps" in data:
            counts, flat = updated_endpoint_candidates(
                p[1:], abs(trial), before, law["tangential_penalty"], mus, muk, dc
            )
            local_condition["accepted_q_with_multiple_updated_endpoint_candidates"] = int(
                np.count_nonzero((counts > 1) | flat)
            )
            local_condition["accepted_q_with_no_updated_endpoint_candidate"] = int(
                np.count_nonzero(counts == 0)
            )
            local_condition["accepted_q_with_flat_updated_endpoint_continuum"] = int(
                np.count_nonzero(flat)
            )
        locations = np.argwhere(bad)
        local_condition["first_entries"] = [
            dict(
                time_s=float(data["time"][i + 1]),
                y_mm=float(data["contact_y"][j]),
                pressure_mpa=float(p[i + 1, j]),
                eta=float(eta[i, j]),
                prior_c_over_Dc=float(before[i, j] / dc),
                new_c_over_Dc=float(after[i, j] / dc),
            )
            for i, j in locations[:32]
        ]
        local_condition["entry_list_truncated"] = len(locations) > 32
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
            updated_local_map_condition=local_condition,
        )
        np.savetxt(
            output / f"{run.name}.csv",
            np.column_stack(
                (
                    data["contact_y"],
                    local_ratio,
                    progress,
                    local_gap,
                    np.max(np.where(before < dc, eta, 0), axis=0),
                )
            ),
            delimiter=",",
            header="y_mm,max_increment_over_Dc,max_transition_progress_in_one_step,fixed_pressure_work_gap_mpa_mm2,max_eta_while_weakening",
            comments="",
        )
    return record


def audit_saved_fem_condition(output):
    """Read small interface datasets; never open the reference for writing."""
    import h5py

    provenance = json.loads(
        (ROOT / "src/finiteblock_bem/resources/data/source_provenance.json").read_text()
    )
    source = provenance["hdf5"]
    path = Path(source["path"])
    before = path.stat()
    if (before.st_size, before.st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
        raise ValueError("FEM archive differs from the recorded source")
    with h5py.File(path, "r") as data:
        if data.attrs["friction_law"] != "slip-weakening":
            raise ValueError("Strength/mu pressure reconstruction requires slip weakening")
        phase = data["phase_id"][:] == 2
        mu_dataset = data["interface/friction_coefficient"]
        mu = mu_dataset[:][phase].astype(float)
        strength = data["interface/friction_strength"][:][phase].astype(float)
        if np.any(mu <= 0):
            raise ValueError("Positive saved friction coefficient required")
        pressure = strength / mu
        kt = data["interface/tangential_penalty_profile"][:]
        dc = data["interface/critical_slip_profile"][:]
        mus, muk = [
            data["interface/" + name][:] for name in ["mu_static_profile", "mu_kinetic_profile"]
        ]
        # Apply the SAME storage quantization to mu_k. A float32 0.6 is
        # slightly greater than float64 0.6 and is still fully weakened.
        kinetic_saved = muk.astype(mu_dataset.dtype).astype(float)
        weakening = mu > kinetic_saved
        plastic = data["interface/plastic_slip"][:][phase].astype(float)
        sliding_proxy = np.zeros_like(weakening)
        sliding_proxy[1:] = abs(np.diff(plastic, axis=0)) > 0
        result, bad, eta = updated_local_condition(
            pressure, weakening, sliding_proxy, kt, mus, muk, dc
        )
        result["maximum_saved_pressure_mpa"] = float(pressure.max())
        result["saved_frames"] = len(pressure)
        result["pressure_reconstruction"] = (
            "saved friction_strength / applied friction_coefficient; source uses strength=mu*normal_traction for this law"
        )
        result["branch_activity_scope"] = (
            "Saved applied mu identifies weakening; saved plastic increments are only a sliding proxy, not every FEM microstep"
        )
        result["scope"] = (
            "Saved frames only, including float32 storage error; maxima between frames unknown; not a global or microstep stability proof"
        )
        result["source_hdf5"] = source
        y = data["interface/contact_line_y"][:]
        np.savetxt(
            output / "fem_softening_condition.csv",
            np.column_stack(
                (
                    y,
                    np.max(eta, axis=0),
                    np.max(pressure, axis=0),
                    np.any(bad, axis=0),
                    np.max(np.where(weakening, eta, 0), axis=0),
                )
            ),
            delimiter=",",
            comments="",
            header="y_mm,max_saved_eta,max_saved_pressure_mpa,any_nonpositive_weakening_sliding_denominator,max_eta_while_weakening",
        )
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("FEM source changed during the read")
    result["source_hdf5_unchanged"] = True
    return result


def plot_conditions(runs, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for run in runs:
        data = np.genfromtxt(output / f"{run.name}.csv", delimiter=",", names=True)
        summary = json.loads((run / "summary.json").read_text())
        label = f"BEM {summary['contact_nodes']} nodes, {summary['dt_s'] * 1e6:g} us"
        for ax in axes:
            ax.plot(data["y_mm"], data["max_eta_while_weakening"], label=label)
    fem = output / "fem_softening_condition.csv"
    if fem.is_file():
        data = np.genfromtxt(fem, delimiter=",", names=True)
        for ax in axes:
            ax.plot(
                data["y_mm"], data["max_eta_while_weakening"], "k--", label="FEM h=2, saved frames"
            )
    for ax in axes:
        ax.axhline(1, color="tab:red", ls=":", label="updated local denominator = 0")
        ax.set(
            xlabel="Interface y [mm]", ylabel=r"max $\eta=p(\mu_s-\mu_k)/(k_t D_c)$ while weakening"
        )
        ax.grid(alpha=0.25)
    axes[1].set_xlim(460, 480)
    axes[1].set_title("Upper contact endpoint")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Optional updated-state local condition at recorded states\nNot a global stability test of the previous-state solver"
    )
    fig.savefig(output / "softening_condition.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/weakening-resolution")
    parser.add_argument(
        "--fem-local-condition",
        action="store_true",
        help="Read saved FEM interface strength/mu to audit the updated-map condition",
    )
    parser.add_argument("--plot-softening-condition", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(
        results=[audit(run, args.output) for run in args.runs],
        scope="Exact mu(c) integral with endpoint pressure held fixed; not total energy or coupled error bound",
        reference_integral="mu_k*(c_new-c_old) + (mu_s-mu_k)*[b-a-(b*b-a*a)/(2*Dc)], a=min(c_old,Dc), b=min(c_new,Dc)",
    )
    if args.fem_local_condition:
        record["saved_fem_updated_local_condition"] = audit_saved_fem_condition(args.output)
    if args.plot_softening_condition:
        plot_conditions(args.runs, args.output)
    write_json(args.output / "summary.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

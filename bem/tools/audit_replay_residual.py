"""Check a replay in the ORIGINAL complete CQ perturbation equations.

The FFT below acts on the original Z weights, independently of the replay's
resolvent history and virtual forcing. Initial static residual is reported
separately. This does not certify a total energy balance or FEM equivalence.
"""

import argparse
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from finiteblock_bem.cq import CQPlan, plan_cq
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model
from finiteblock_bem.nodal_dynamics import nodal_cache_key
from finiteblock_bem.sampling import SamplingOptions

ROOT = Path(__file__).resolve().parents[1]


def original_bulk_force(weights, q):
    """Zero-padded linear convolution; no damped circular forcing shortcut."""
    length = next_fast_len(2 * len(q) - 1, real=True)
    inputs = rfft(q, n=length, axis=0)
    force = np.zeros_like(q)
    for row in range(0, q.shape[1], 4):
        kernel = rfft(weights[:, row : row + 4], n=length, axis=0)
        force[:, row : row + 4] = irfft(
            np.einsum("fij,fj->fi", kernel, inputs, optimize=False), n=length, axis=0
        )[: len(q)]
    return force


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--max-memory-mb", type=float, default=12288)
    args = parser.parse_args()
    source = json.loads((args.original / "summary.json").read_text())
    replay = json.loads((args.replay / "summary.json").read_text())
    options = json.loads((args.original / "run_manifest.json").read_text())["arguments"]
    environment = environment_record(options["case"])
    for name in ["source_sha256", "case_sha256", "restart_data_sha256"]:
        if (
            source["environment"][name] != environment[name]
            or replay["environment"][name] != environment[name]
        ):
            raise ValueError(f"Input differs in {name}")
    if source["source_changed_during_run"] or source["status"] != "completed_requested_horizon":
        raise ValueError("A complete source-frozen forward trajectory is required")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    if "cq_plan" in source["spectrum_storage"]:
        plan = CQPlan(**source["spectrum_storage"]["cq_plan"])
    else:
        plan = plan_cq(
            2 * len(model.moving_contact) + 1,
            source["dt_s"],
            source["completed_steps"],
            max_memory_mb=65536,
        )
    sampling = (
        SamplingOptions(relative_tolerance=options["sampling_rtol"])
        if options["sampling"] == "adaptive"
        else None
    )
    cache = Path(options["cache"])
    if not cache.is_absolute():
        cache = ROOT.parent / cache
    cache /= f"galerkin-{nodal_cache_key(model, plan, options['quadrature'], sampling)}.npy"
    if not cache.is_file() or not cache.with_suffix(".json").is_file():
        raise ValueError("Matching original Z weights are required")
    length = next_fast_len(2 * (plan.steps + 1) - 1, real=True)
    estimate = (
        plan.weight_bytes
        + 3 * length * 4 * plan.dofs * 16
        + 20 * (plan.steps + 1) * plan.dofs * 8
        + 512 * 1024**2
    )
    if estimate > args.max_memory_mb * 1024**2:
        raise ValueError(f"Residual audit needs about {estimate / 1024**2:.1f} MiB")
    fields = [
        "time",
        "pressure",
        "shear",
        "plastic_slip",
        "cumulative_slip",
        "contact_jumps",
        "platen",
        "actuator",
    ]
    with np.load(args.original / "trajectory.npz", allow_pickle=False) as data:
        original = SimpleNamespace(**{key: data[key] for key in fields})
    with np.load(args.replay / "trajectory.npz", allow_pickle=False) as data:
        actual = SimpleNamespace(**{key: data[key] for key in fields})
    np.testing.assert_array_equal(actual.time, original.time)
    q = np.c_[actual.contact_jumps - original.contact_jumps[0], actual.platen - original.platen[0]]
    weights = np.load(cache, mmap_mode="r", allow_pickle=False)
    force = original_bulk_force(weights, q)
    errors = []
    for index in sorted({0, 1, len(q) // 2, len(q) - 1}):
        direct = np.einsum("kij,kj->i", weights[: index + 1], q[index::-1], optimize=False)
        errors.append(float(np.max(abs(direct - force[index]))))
        np.testing.assert_allclose(direct, force[index], rtol=1e-9, atol=1e-7)
    del weights
    n = actual.pressure.shape[1]
    wc = model.constraints.contact_weights
    width = model.constraints.platen_width
    force[:, :n] += (actual.pressure - original.pressure[0]) * wc
    force[:, n : 2 * n] += (actual.shear - original.shear[0]) * wc
    force[:, -1] += (
        model.reference.spring_stiffness
        * width
        * (q[:, -1] - actual.actuator + original.actuator[0])
    )
    residual = abs(force) / np.r_[wc, wc, width]
    history = np.genfromtxt(args.original / "history.csv", delimiter=",", names=True)
    initial = float(history["residual_mpa"][0])
    branch_function = runpy.run_path(str(Path(__file__).with_name("run_adjoint.py")))[
        "exact_contact_branches"
    ]
    original_branches = branch_function(model.reference, original)
    actual_branches = branch_function(model.reference, actual)
    branch_changes = {
        name: int(np.count_nonzero(value != original_branches[name]))
        for name, value in actual_branches.items()
    }
    maximum = float(np.max(residual))
    record = dict(
        maximum_original_perturbation_residual_mpa=maximum,
        original_static_residual_mpa=initial,
        residual_plus_static_bound_mpa=maximum + initial,
        original_balance_tolerance_mpa=1e-7,
        passes_original_balance_tolerance=bool(maximum + initial < 1e-7),
        maximum_force_fft_direct_difference=max(errors),
        exact_branch_changes=branch_changes,
        original_trajectory_sha256=fingerprint(args.original / "trajectory.npz"),
        replay_trajectory_sha256=fingerprint(args.replay / "trajectory.npz"),
        environment=environment,
        validator_sha256=fingerprint(Path(__file__)),
        scope="Independent original full-history CQ perturbation balance, relative to the accepted initial force balance; not energy or FEM equivalence",
    )
    np.savez(
        args.replay / "original_residual.npz",
        time=actual.time,
        max_residual_mpa=np.max(residual, axis=1),
    )
    write_json(args.replay / "original_residual.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

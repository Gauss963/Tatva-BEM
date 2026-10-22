"""Differentiate a saved interface run with fixed recorded loading and lift."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from finiteblock_bem.adjoint import trajectory_adjoint
from finiteblock_bem.case import load_case
from finiteblock_bem.condensation import build_condensed_history
from finiteblock_bem.dynamics import LoadingSchedule, initialize_restart, simulate
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.sampling import SamplingOptions

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--finite-difference", action="store_true")
    parser.add_argument(
        "--fd-scale", type=float, default=1.0, help="Scale all central-difference steps"
    )
    parser.add_argument(
        "--fd-parameters", nargs="+", choices=["mu_s", "mu_k", "Dc"], default=["mu_s", "mu_k", "Dc"]
    )
    args = parser.parse_args()
    folder = args.run.resolve()
    summary = json.loads((folder / "summary.json").read_text())
    manifest = json.loads((folder / "run_manifest.json").read_text())["arguments"]
    case_path = args.case or Path(manifest["case"])
    if not case_path.is_file():
        case_path = ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    if fingerprint(case_path) != summary["environment"]["case_sha256"]:
        raise ValueError("Provide the original case with --case")
    current = environment_record(case_path)
    for name, sha in current["source_sha256"].items():
        if sha != summary["environment"]["source_sha256"].get(name):
            raise ValueError(f"Numerical code {name} differs from this run; rerun forward first")
    if current["restart_data_sha256"] != summary["environment"]["restart_data_sha256"]:
        raise ValueError("Restart snapshot differs from this run; rerun forward first")
    if summary.get("source_changed_during_run"):
        raise ValueError("Rerun forward from a frozen source before validating gradients")
    case = load_case(
        case_path,
        max_panel_length=manifest.get("h"),
        contact_panel_length=manifest.get("contact_h"),
        contact_end_length=manifest.get("contact_end_h"),
        corner_panel_length=manifest.get("corner_h"),
    )
    quadrature = manifest["quadrature"]
    lift = initialize_restart(case, quadrature=quadrature)
    with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
        if "contact_jumps" not in data:
            raise ValueError("Adjoint requires --formulation interface output")
        saved = {
            key: data[key]
            for key in (
                "time",
                "contact_jumps",
                "platen",
                "plastic_slip",
                "cumulative_slip",
                "actuator",
            )
        }
        base_pressure = data["pressure"].copy()
        for prefix, mesh in [("moving", case.moving), ("stationary", case.stationary)]:
            for field in ("starts", "ends"):
                np.testing.assert_allclose(
                    data[f"{prefix}_{field}"], getattr(mesh, field), rtol=0, atol=1e-12
                )
        np.testing.assert_array_equal(data["moving_contact"], case.moving_contact)
        np.testing.assert_array_equal(data["stationary_contact"], case.stationary_contact)
    table = np.genfromtxt(folder / "history.csv", delimiter=",", names=True)
    result = SimpleNamespace(**saved, diagnostics=summary, residual_mpa=table["residual_mpa"])
    kwargs = dict(
        quadrature=quadrature,
        workers=manifest["workers"],
        max_memory_mb=manifest["max_memory_mb"],
        cache_directory=manifest["cache"],
        sampling=(
            SamplingOptions(
                relative_tolerance=manifest["sampling_rtol"], degree=manifest["sampling_degree"]
            )
            if manifest.get("sampling") == "adaptive"
            else None
        ),
    )
    history = build_condensed_history(case, summary["dt_s"], summary["completed_steps"], **kwargs)
    answer = trajectory_adjoint(case, lift, history, result)
    np.savez(
        folder / "adjoint.npz",
        **{key: value for key, value in answer.items() if isinstance(value, np.ndarray)},
    )
    record = {key: value for key, value in answer.items() if not isinstance(value, np.ndarray)}
    record["gradient"] = answer["gradient"].tolist()
    if args.finite_difference:
        if not np.isfinite(args.fd_scale) or args.fd_scale <= 0:
            raise ValueError("Positive finite --fd-scale required")
        schedule = LoadingSchedule(
            lambda _: case.normal_pressure,
            lambda t: float(np.interp(t, result.time, result.actuator)),
            "fixed_recorded_loading",
        )
        checks = []
        for index, field, epsilon in [
            (2, "static_friction", 1e-6),
            (3, "kinetic_friction", 1e-6),
            (4, "characteristic_slip", 1e-7),
        ]:
            if answer["parameter_order"][index] not in args.fd_parameters:
                continue
            epsilon *= args.fd_scale
            losses = []
            branch_changes = []
            for sign in [-1, 1]:
                varied = replace(
                    case,
                    contact=replace(
                        case.contact, **{field: getattr(case.contact, field) + sign * epsilon}
                    ),
                )
                trial = simulate(
                    varied,
                    summary["dt_s"],
                    summary["completed_steps"],
                    schedule=schedule,
                    lift=lift,
                    formulation="interface",
                    backend="jax",
                    **kwargs,
                )
                if trial.diagnostics["status"] != "completed_requested_horizon":
                    raise RuntimeError(
                        "Perturbed forward failed; difference is not a gradient check"
                    )
                losses.append(
                    float(
                        np.average(
                            trial.cumulative_slip[-1],
                            weights=case.moving.lengths[case.moving_contact],
                        )
                    )
                )
                base_slide = np.abs(np.diff(result.plastic_slip, axis=0)) > 1e-10
                trial_slide = np.abs(np.diff(trial.plastic_slip, axis=0)) > 1e-10
                branch_changes.append(
                    {
                        "sliding_entries": int(np.count_nonzero(base_slide != trial_slide)),
                        "opening_entries": int(
                            np.count_nonzero((base_pressure > 1e-10) != (trial.pressure > 1e-10))
                        ),
                    }
                )
            finite = (losses[1] - losses[0]) / (2 * epsilon)
            checks.append(
                {
                    "parameter": answer["parameter_order"][index],
                    "epsilon": epsilon,
                    "adjoint": float(answer["gradient"][index]),
                    "central_difference": finite,
                    "relative_error": abs(finite - answer["gradient"][index])
                    / max(abs(finite), 1e-12),
                    "branch_changes_minus_plus": branch_changes,
                }
            )
            print(json.dumps(checks[-1]), flush=True)
        record["finite_difference_checks"] = checks
    record["environment"] = environment_record(case_path)
    write_json(folder / "adjoint.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

"""Fresh original-algebra audit at EVERY frequency of a guarded small kernel.

Optionally form the independent dense causal stage weights and compare a
complete source trajectory with prepared observations. This is not a full
fine-event audit: explicit frequency and array budgets reject oversized work.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.fft import irfft
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import CQPlan
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
METHOD = runpy.run_path(str(TOOLS / "interior_raw_sampling.py"))
REFERENCE = runpy.run_path(str(TOOLS / "reconstruct_interior_reference.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))
DENSE = runpy.run_path(str(TOOLS / "interior_radau.py"))
DIGEST = REFERENCE["DIGEST"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kernel", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-run", type=Path)
    p.add_argument("--observations", type=Path)
    p.add_argument("--max-frequency-evaluations", type=int, default=2048)
    p.add_argument("--max-workspace-mb", type=float, default=512)
    p.add_argument("--rtol", type=float, default=1e-7)
    p.add_argument("--atol", type=float, default=1e-12)
    args = p.parse_args()
    begun = perf_counter()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose fresh audit output outside protected Tatva")
    metadata = json.loads((args.kernel / "summary.json").read_text())
    plan = CQPlan(**metadata["plan"])
    shape = metadata["family_spectra"][0]["shape"]
    half, m, n = shape
    if (
        metadata["status"] != "completed_empirical_raw_interior_radau_spectra"
        or 2 * half > args.max_frequency_evaluations
        or any(not np.isfinite(v) or v <= 0 for v in [args.rtol, args.atol, args.max_workspace_mb])
        or (args.observations is not None and args.source_run is None)
    ):
        raise ValueError(
            "Invalid interior audit configuration or direct evaluation budget exceeded"
        )
    peak = (
        0
        if args.source_run is None
        else (half * 16 + 2 * plan.fft_size * 8 + (plan.steps + 1) * 8) * 4 * m * n
    )
    if peak > args.max_workspace_mb * 1024**2:
        raise ValueError("Dense independent audit exceeds array budget")
    for p, v in metadata["frozen_inputs_sha256"].items():
        if DIGEST(p) != v:
            raise ValueError("Prepared source changed")
    environment = environment_record(metadata["source_mesh_arguments"]["case"])
    if any(
        environment[k] != metadata["environment"][k]
        for k in ["source_sha256", "case_sha256", "restart_data_sha256", "numpy", "scipy"]
    ):
        raise ValueError("Prepared environment changed")
    protected = [args.kernel / "summary.json", Path(__file__)]
    if args.source_run is not None:
        source = json.loads((args.source_run / "summary.json").read_text())
        source_kernel_path = Path(source["prepared_kernel"]) / "summary.json"
        source_kernel = json.loads(source_kernel_path.read_text())
        mesh_keys = [
            "case",
            "h",
            "contact_h",
            "contact_end_h",
            "corner_h",
            "quadrature",
            "restart_knot_window",
            "restart_knot_subdivisions",
        ]
        if (
            source_kernel["plan"] != metadata["plan"]
            or source["completed_steps"] != plan.steps
            or any(
                source_kernel["source_mesh_arguments"].get(k)
                != metadata["source_mesh_arguments"].get(k)
                for k in mesh_keys
            )
            or any(
                source_kernel["environment"][k] != metadata["environment"][k]
                for k in ["source_sha256", "case_sha256", "restart_data_sha256"]
            )
        ):
            raise ValueError("Independent trajectory must cover the exact complete model and plan")
        protected.extend(
            [
                args.source_run / "summary.json",
                args.source_run / "stages.npz",
                source_kernel_path,
            ]
        )
    if args.observations is not None:
        observed = json.loads((args.observations / "summary.json").read_text())
        if (
            Path(observed["source_run"]).resolve() != args.source_run.resolve()
            or Path(observed["prepared_kernel"]).resolve() != args.kernel.resolve()
        ):
            raise ValueError("Observations have a different source trajectory or kernel")
        protected.extend(
            [
                args.observations / "summary.json",
                args.observations / "observations.npz",
            ]
        )
    maps = []
    try:
        for group in ["raw_branch_spectra", "family_spectra"]:
            for r in metadata[group]:
                if DIGEST(r["path"]) != r["sha256"]:
                    raise ValueError("Prepared spectrum bytes changed")
                maps.append(np.load(r["path"], mmap_mode="r", allow_pickle=False))
                protected.append(Path(r["path"]))
        before = {str(p.resolve()): DIGEST(p) for p in protected}
        raw, compact = maps[:2], maps[2:]
        options = metadata["source_mesh_arguments"]
        model = load_nodal_model(
            options["case"],
            max_panel_length=options["h"],
            contact_panel_length=options["contact_h"],
            contact_end_length=options["contact_end_h"],
            corner_panel_length=options["corner_h"],
            restart_knot_window=options.get("restart_knot_window"),
            restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
        )
        points = [metadata["probe_plan"][k] for k in ["moving", "stationary"]]
        direct = METHOD["FIELD"]["InteriorRetainedTransfer"](
            model,
            points,
            quadrature=options["quadrature"],
            probe_quadrature=metadata["probe_quadrature"],
        )
        tolerances = METHOD["RAW"]["RawSamplingOptions"](
            output_rtol=args.rtol, output_atol=args.atol
        )
        spectrum = np.empty((half, 2 * m, 2 * n), complex) if args.source_run is not None else None
        maximum = 0.0
        last = 0.0
        with threadpool_limits(limits=1, user_api="blas"):
            for k in range(half):
                z = plan.radius * np.exp(-2j * np.pi * k / plan.fft_size)
                modes = RADAU["lower_half_radau_modes"](z, 1.0)
                fresh = [REFERENCE["pack"](direct.evaluate(s / plan.dt), m // 8, n) for s in modes]
                for i, value in enumerate(fresh):
                    error = METHOD["maximum_row_error"]([value], [raw[i][k]], tolerances)
                    maximum = max(maximum, error)
                a, b = modes
                families = [
                    (b * fresh[0] - a * fresh[1]) / (b - a),
                    (fresh[1] - fresh[0]) / (b - a),
                ]
                for i, value in enumerate(families):
                    if k in [0, half - 1]:
                        value = value.real
                    error = METHOD["maximum_row_error"]([value], [compact[i][k]], tolerances)
                    maximum = max(maximum, error)
                if spectrum is not None:
                    spectrum[k] = np.kron(np.eye(2), families[0]) + np.kron(
                        DENSE["RK"]["radau_symbol"](z), families[1]
                    )
                if not np.isfinite(maximum) or maximum > 1:
                    raise ValueError(
                        f"Independent original map audit failed at {k}: normalized error {maximum}"
                    )
                if perf_counter() - last > 30:
                    print(
                        json.dumps(dict(checked_frequency=k, maximum_normalized_error=maximum)),
                        flush=True,
                    )
                    last = perf_counter()
        result = dict(
            status="passed_fresh_original_algebra_at_every_interior_frequency",
            plan=metadata["plan"],
            physical_frequency_evaluations=2 * half,
            maximum_normalized_global_and_row_error=maximum,
            relative_tolerance=args.rtol,
            absolute_tolerance=args.atol,
            full_frequency_grid_checked=True,
            original_free_body_algebra_reference=True,
            array_estimate_bytes=peak,
        )
        if args.source_run is not None:
            with np.load(args.source_run / "stages.npz", allow_pickle=False) as saved:
                q = saved["displacement"]
            if q.dtype != np.float64 or q.shape != (plan.steps + 1, 2, n) or np.any(q[0] != 0):
                raise ValueError("Invalid original perturbation trajectory")
            weights = irfft(spectrum, n=plan.fft_size, axis=0)[: plan.steps + 1]
            weights *= plan.radius ** (-np.arange(plan.steps + 1))[:, None, None]
            values = DENSE["apply_weights"](weights, q.reshape(plan.steps + 1, -1)).reshape(
                plan.steps + 1, 2, m
            )
            result["dense_causal_reference_source_steps"] = plan.steps
            if args.observations:
                with np.load(args.observations / "observations.npz", allow_pickle=False) as saved:
                    actual = saved["perturbation"]
                error = float(np.linalg.norm(actual - values) / max(np.linalg.norm(values), 1e-30))
                if error > 1e-7:
                    raise ValueError(
                        "Complete causal observation history differs from independent original map"
                    )
                result["complete_observation_relative_l2"] = error
        if any(DIGEST(p) != v for p, v in before.items()):
            raise ValueError("Audit input changed during independent reconstruction")
        if any(DIGEST(p) != v for p, v in metadata["frozen_inputs_sha256"].items()):
            raise ValueError("Prepared numerical source changed during audit")
        result.update(
            source_sha256=before,
            inputs_verified_before_and_after=True,
            seconds_including_hashes=perf_counter() - begun,
            complete_fine_event_or_fem_verified=False,
            initialized_observation_gradient_verified=False,
            same_accuracy_fem_advantage_verified=False,
        )
        write_json(output, result)
        print(output, flush=True)
    finally:
        for a in maps:
            a._mmap.close()


if __name__ == "__main__":
    main()

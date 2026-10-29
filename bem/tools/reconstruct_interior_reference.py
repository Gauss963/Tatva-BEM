"""Reconstruct physical interior probes from a complete compact Radau run.

Uses the bounded history backend and a guarded DIRECT frequency sampler.
The latter is a small correctness reference, not the efficient full-event
sampler. No fitted phase, nodal-force density substitution or history cutoff.
"""

import argparse
import json
import runpy
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.cq import CQPlan
from finiteblock_bem.cq_storage import DiskSpectrum
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
FIELD = runpy.run_path(str(TOOLS / "interior_fields.py"))
HISTORY = runpy.run_path(str(TOOLS / "rectangular_cq_history.py"))
DENSE = runpy.run_path(str(TOOLS / "interior_radau.py"))
RAW = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))
DIGEST = HISTORY["DIGEST"]


class RectangularSpectrum(DiskSpectrum):
    """Use the original complete-matrix bounded I/O for a rectangular shape."""

    def __init__(self, path, shape):
        self.path, self.shape, self.dtype = Path(path), tuple(shape), np.dtype("complex128")
        self.written = np.zeros(shape[0], dtype=bool)
        self.pending_writes = 0
        with self.path.open("xb") as stream:
            np.lib.format.write_array_header_2_0(
                stream, dict(descr=self.dtype.str, fortran_order=False, shape=self.shape)
            )
            self.offset = stream.tell()
            stream.truncate(self.offset + int(np.prod(shape)) * 16)


def pack(response, points, inputs):
    return np.concatenate(
        [
            response[k].reshape(points, n, inputs)
            for k, n in [("displacement", 2), ("velocity", 2), ("stress", 4)]
        ],
        axis=1,
    ).reshape(8 * points, inputs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--probe-plan", type=Path, default=TOOLS.parent / "examples/event16_interior_probes.json"
    )
    p.add_argument("--output-rows", type=int, default=1)
    p.add_argument("--max-workspace-mb", type=float, default=2048)
    p.add_argument("--max-direct-frequency-evaluations", type=int, default=256)
    p.add_argument("--dense-check", action="store_true")
    p.add_argument("--check-ad", action="store_true")
    args = p.parse_args()
    begun = perf_counter()
    output = args.output.resolve()
    if args.output_rows < 1 or not np.isfinite(args.max_workspace_mb) or args.max_workspace_mb <= 0:
        raise ValueError("Require positive output rows and finite workspace budget")
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh reconstruction output outside protected Tatva")
    source = args.run.resolve()
    summary = json.loads((source / "summary.json").read_text())
    if summary["status"] != "completed_requested_horizon":
        raise ValueError("Require a complete source trajectory")
    if summary.get("initial_lift_record") is None:
        raise ValueError("Require the source run's saved complete original static lift")
    kernel = Path(summary["prepared_kernel"])
    if DIGEST(kernel / "summary.json") != summary["kernel_summary_sha256"]:
        raise ValueError("Source kernel receipt changed")
    metadata = json.loads((kernel / "summary.json").read_text())
    plan = CQPlan(**metadata["plan"])
    if summary["completed_steps"] != plan.steps:
        raise ValueError("Require the complete kernel horizon, without a shortened prefix")
    options = metadata["source_mesh_arguments"]
    environment = environment_record(options["case"])
    for key in ["source_sha256", "case_sha256", "restart_data_sha256", "numpy", "scipy"]:
        if (
            environment[key] != metadata["environment"][key]
            or environment[key] != summary["environment"][key]
        ):
            raise ValueError("Numerical source/runtime changed")
    count = plan.fft_size // 2 + 1
    evaluations = 2 * count * (2 if args.dense_check else 1)
    if (
        args.max_direct_frequency_evaluations < 1
        or evaluations > args.max_direct_frequency_evaluations
    ):
        raise ValueError(
            "Direct interior sampling exceeds its evaluation budget; full events need an efficient sampler"
        )
    probes = json.loads(args.probe_plan.read_text())
    points = [np.asarray(probes[k], dtype=float) for k in ["moving", "stationary"]]
    total = sum(len(a) for a in points)
    channels = 8 * total
    preflight = HISTORY["resources"](plan, plan.dofs, channels, args.output_rows)
    if preflight["array_peak_bytes"] > args.max_workspace_mb * 1024**2:
        raise ValueError("Interior reconstruction exceeds its array workspace budget")
    needed = (
        2 * count * channels * plan.dofs * 16 + 8 * (plan.steps + 1) * channels * 8 + 64 * 1024**2
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if needed > shutil.disk_usage(output.parent).free:
        raise ValueError("Insufficient disk for complete rectangular spectra")
    paths = [
        source / "summary.json",
        source / "stages.npz",
        Path(summary["initial_lift_record"]["path"]),
        kernel / "summary.json",
        args.probe_plan.resolve(),
        Path(__file__),
        TOOLS / "interior_fields.py",
        TOOLS / "rectangular_cq_history.py",
        TOOLS / "interior_radau.py",
        TOOLS / "radau_raw_sampling.py",
    ]
    frozen = {str(a.resolve()): DIGEST(a) for a in paths}
    lift_path = Path(summary["initial_lift_record"]["path"])
    if DIGEST(lift_path) != summary["initial_lift_record"]["sha256"]:
        raise ValueError("Original static lift changed")
    with np.load(source / "stages.npz", allow_pickle=False) as saved:
        q, stage_time, initial_residual = (
            saved["displacement"],
            saved["stage_time"],
            saved["initial_residual_force"],
        )
    expected_time = np.zeros((plan.steps + 1, 2))
    expected_time[1:] = (np.arange(plan.steps)[:, None] + DENSE["RK"]["C"]) * plan.dt
    if (
        q.shape != (plan.steps + 1, 2, plan.dofs)
        or q.dtype != np.float64
        or not np.all(np.isfinite(q))
        or np.any(q[0] != 0)
        or initial_residual.shape != (plan.dofs,)
        or initial_residual.dtype != np.float64
        or not np.all(np.isfinite(initial_residual))
        or not np.array_equal(expected_time, stage_time)
    ):
        raise ValueError("Invalid original perturbation stages or physical stage clocks")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    transfer = FIELD["InteriorRetainedTransfer"](model, points, quadrature=options["quadrature"])

    def evaluate(s):
        return pack(transfer.evaluate(s), total, plan.dofs)

    output.mkdir()
    shape = (count, channels, plan.dofs)
    disks = [RectangularSpectrum(output / f"family{i}.npy", shape) for i in range(2)]
    for k in range(count):
        z = plan.radius * np.exp(-2j * np.pi * k / plan.fft_size)
        modes = RAW["lower_half_radau_modes"](z, 1.0)
        first, second = [evaluate(s / plan.dt) for s in modes]
        a, b = modes
        families = [(b * first - a * second) / (b - a), (second - first) / (b - a)]
        for i, values in enumerate(families):
            if k in [0, count - 1]:
                if np.linalg.norm(values.imag) > 1e-10 * max(np.linalg.norm(values.real), 1):
                    raise ValueError("Interior compact family violates endpoint symmetry")
                values = values.real.astype(np.complex128)
            disks[i][k] = values
    receipts = [d.record() for d in disks]
    fixed = HISTORY["FixedRectangularCQ"](
        receipts, plan, output_rows=args.output_rows, max_workspace_mb=args.max_workspace_mb
    )
    checks = {}
    try:
        perturbation = fixed.apply(q)
        if args.dense_check:
            weights, dense_meta = DENSE["rectangular_weights"](
                evaluate,
                plan.dt,
                plan.steps,
                plan.dofs,
                channels,
                max_memory_mb=args.max_workspace_mb,
            )
            expected = DENSE["apply_weights"](weights, q.reshape(plan.steps + 1, -1)).reshape(
                perturbation.shape
            )
            difference = np.linalg.norm(perturbation - expected) / max(
                np.linalg.norm(expected), 1e-30
            )
            if not np.isfinite(difference) or difference > 1e-7:
                raise ValueError("Original dense stage-map CQ differs")
            checks["dense_relative_l2"] = float(difference)
            checks["dense_cq"] = dense_meta
        rng = np.random.default_rng(563)
        cotangent = rng.normal(size=perturbation.shape)
        direction = rng.normal(size=q.shape)
        direction[0] = 0
        reverse = fixed.apply(cotangent, transpose=True)
        tangent = fixed.apply(direction)
        left, right = np.vdot(reverse, direction), np.vdot(cotangent, tangent)
        relative = float(abs(left - right) / max(abs(left), abs(right), 1e-30))
        if relative > 1e-9:
            raise ValueError("Complete interior transpose dot product failed")
        checks["transpose_directional_relative_error"] = relative
        if args.check_ad:
            import jax
            import jax.numpy as jnp

            with jax.enable_x64(True):
                observed = fixed.as_jax()
                derivative = jax.jit(
                    jax.grad(lambda x: jnp.vdot(observed(x), jnp.asarray(cotangent)))
                )(jnp.asarray(q))
                relative = float(
                    np.linalg.norm(np.asarray(derivative) - reverse)
                    / max(np.linalg.norm(reverse), 1e-30)
                )
                if relative > 1e-9:
                    raise ValueError("JAX interior VJP differs from complete transpose")
                checks["jax_fixed_map_vjp_relative_error"] = relative
        fixed.verify()
    finally:
        fixed.close()
    with np.load(lift_path, allow_pickle=False) as saved:
        full = saved["full_displacement"]
    if full.dtype != np.float64 or full.shape != (
        sum(mesh.dofs for mesh in [model.reference.moving, model.reference.stationary]),
    ):
        raise ValueError("Require the original full float64 boundary static lift")
    static = []
    offset = 0
    for mesh, point in zip(
        [model.reference.moving, model.reference.stationary], points, strict=True
    ):
        field = FIELD["InteriorPotentials"](
            mesh, model.reference.material, point
        ).from_galerkin_trace(full[offset : offset + mesh.dofs], quadrature=options["quadrature"])
        static.append(
            np.concatenate(
                [field["displacement"], np.zeros_like(field["displacement"]), field["stress"]],
                axis=1,
            )
        )
        offset += mesh.dofs
    lift = np.vstack(static)
    values = perturbation.reshape(plan.steps + 1, 2, total, 8) + lift[None, None]
    np.savez(
        output / "observations.npz",
        stage_time=stage_time,
        points=np.vstack(points),
        body=np.repeat([0, 1], [len(a) for a in points]),
        value=values,
        static_lift=lift,
        perturbation=perturbation,
        source_initial_residual_force=initial_residual,
    )
    if any(DIGEST(p) != v for p, v in frozen.items()):
        raise ValueError("Reconstruction source changed")
    write_json(
        output / "summary.json",
        dict(
            status="completed_bounded_interior_reference_reconstruction",
            source_run=str(source),
            plan=metadata["plan"],
            probe_plan=probes,
            channels_per_point=["ux", "uy", "vx", "vy", "sxx", "syy", "sxy", "szz"],
            family_spectra=receipts,
            frozen_input_sha256=frozen,
            resources=preflight,
            checks=checks,
            direct_frequency_evaluations=evaluations,
            all_requested_lags_retained=True,
            finite_dft_wrap_retained=True,
            full_time_weight_files_stored=False,
            source_perturbation_and_initial_residual_unmodified=True,
            static_velocity_zero=True,
            fitted_time_shift_s=0.0,
            seconds_including_hashes=perf_counter() - begun,
            observations_sha256=DIGEST(output / "observations.npz"),
            efficient_full_event_sampler_implemented=False,
            full_original_event_reproduced=False,
            initialized_or_material_gradient_proven=False,
            same_accuracy_fem_advantage_verified=False,
        ),
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

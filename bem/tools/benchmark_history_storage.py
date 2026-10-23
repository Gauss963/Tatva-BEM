"""Validate production last-use FFT streaming against the cache-all reference.

Both retain the entire CQ history. The default streams output rows of a kernel
FFT that has no later use. Separate child
processes measure peak RSS; timings under other active jobs are exploratory.
"""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.cq import BoundaryHistory, OnlineConvolution, plan_cq
from finiteblock_bem.io import environment_record, write_json

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LegacyCacheAllConvolution(OnlineConvolution):
    """The original storage policy, with otherwise identical convolution code."""

    def __init__(self, history):
        super().__init__(history, fft_rows=None)


def verify_small_cases():
    records = []
    for steps in [1, 31, 64, 137, 257, 513]:
        for transpose in [False, True]:
            rng = np.random.default_rng(38)
            weights = rng.normal(size=(steps + 1, 7, 7))
            if transpose:
                weights = weights.transpose(0, 2, 1)
            history = BoundaryHistory(plan_cq(7, 0.01, steps), weights)
            inputs = rng.normal(size=(steps + 1, 7))
            for rows in [1, 4, 8]:
                online = OnlineConvolution(history, direct_block=2, fft_rows=rows)
                error = 0.0
                for n in range(steps + 1):
                    exact = history.past_traction(inputs, n)
                    actual = online.past(n)
                    error = max(error, float(np.max(abs(actual - exact))))
                    np.testing.assert_allclose(actual, exact, rtol=1e-11, atol=5e-13)
                    online.append(inputs[n])
                assert all(3 * level <= steps for level in online.transforms)
                records.append(
                    dict(steps=steps, transpose=transpose, fft_rows=rows, max_force_error=error)
                )
    return records


def load_supplied_history(args):
    with np.load(args.case_run / "trajectory.npz", allow_pickle=False) as saved:
        inputs = np.column_stack(
            (
                saved["contact_jumps"] - saved["contact_jumps"][0],
                saved["platen"] - saved["platen"][0],
            )
        )
        dt = float(saved["time"][1] - saved["time"][0])
    weights = np.load(args.weights, mmap_mode="r", allow_pickle=False)
    steps, dofs = len(inputs) - 1, inputs.shape[1]
    if weights.dtype != np.float64 or weights.shape != (steps + 1, dofs, dofs):
        raise ValueError("Supplied weights do not match the saved retained-coordinate trajectory")
    return BoundaryHistory(plan_cq(dofs, dt, steps, max_memory_mb=65536), weights), inputs


def worker(args):
    start = perf_counter()
    if args.case_run is not None:
        history, inputs = load_supplied_history(args)
    else:
        rng = np.random.default_rng(42)
        weights = rng.normal(size=(args.steps + 1, args.dofs, args.dofs))
        weights *= np.exp(-np.arange(args.steps + 1) / 400)[:, None, None] / np.sqrt(args.dofs)
        inputs = rng.normal(size=(args.steps + 1, args.dofs))
        history = BoundaryHistory(
            plan_cq(args.dofs, 0.01, args.steps, max_memory_mb=65536), weights
        )
    implementation = LegacyCacheAllConvolution if args.variant == "reference" else OnlineConvolution
    online = implementation(history)
    force = np.empty_like(inputs)
    for n in range(len(inputs)):
        force[n] = online.past(n)
        online.append(inputs[n])
    direct_error = 0.0
    for n in sorted({0, 1, 17, len(inputs) // 2, len(inputs) - 1}):
        if n >= len(inputs):
            continue
        direct = history.past_traction(inputs, n)
        direct_error = max(direct_error, float(np.max(abs(force[n] - direct))))
        np.testing.assert_allclose(force[n], direct, rtol=1e-9, atol=1e-7)
    record = dict(
        variant=args.variant,
        steps=history.plan.steps,
        dofs=history.plan.dofs,
        seconds=perf_counter() - start,
        max_direct_force_error=direct_error,
        cached_fft_bytes=sum(a.nbytes for a in online.transforms.values()),
        cached_levels=sorted(online.transforms),
        streamed_levels=getattr(online, "streamed_levels", []),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
    )
    np.savez(args.output / f"{args.variant}.npz", force=force)
    write_json(args.output / f"{args.variant}.json", record)
    print(json.dumps(record), flush=True)


def dynamic_worker(args):
    # Runtime substitutions are restricted to this validation child. The main
    # package files and the other running simulation remain unchanged.
    from finiteblock_bem import adjoint, nodal_dynamics
    from finiteblock_bem.nodal import (
        NodalInterfaceEquilibrium,
        initialize_nodal_restart,
        load_nodal_model,
    )

    history, _ = load_supplied_history(args)
    manifest = json.loads((args.case_run / "run_manifest.json").read_text())["arguments"]
    model = load_nodal_model(
        manifest["case"],
        max_panel_length=manifest["h"],
        contact_panel_length=manifest["contact_h"],
        contact_end_length=manifest["contact_end_h"],
        corner_panel_length=manifest["corner_h"],
        restart_knot_window=manifest.get("restart_knot_window"),
        restart_knot_subdivisions=manifest.get("restart_knot_subdivisions", 1),
    )
    start = perf_counter()
    lift = initialize_nodal_restart(model, quadrature=manifest["quadrature"])
    implementation = LegacyCacheAllConvolution if args.variant == "reference" else OnlineConvolution
    nodal_dynamics.OnlineConvolution = implementation
    adjoint.OnlineConvolution = implementation
    nodal_dynamics.build_nodal_history = lambda *a, **kw: history
    result = nodal_dynamics.simulate_nodal(
        model,
        history.plan.dt,
        history.plan.steps,
        lift=lift,
        quadrature=manifest["quadrature"],
        backend="jax",
        max_memory_mb=65536,
    )
    if result.diagnostics["status"] != "completed_requested_horizon":
        raise RuntimeError("Validation forward did not finish")
    equilibrium = NodalInterfaceEquilibrium(model, lift, history.instantaneous)
    answer = adjoint.trajectory_adjoint(
        model.contact_case(), lift, history, result, equilibrium=equilibrium
    )
    fields = {
        key: getattr(result, key)
        for key in [
            "pressure",
            "shear",
            "plastic_slip",
            "cumulative_slip",
            "platen",
            "contact_jumps",
        ]
    }
    fields.update(
        **{
            key: answer[key]
            for key in [
                "gradient",
                "actuator_gradient",
                "initial_plastic_slip_gradient",
                "initial_cumulative_slip_gradient",
            ]
        }
    )
    saved_differences = {}
    with np.load(args.case_run / "trajectory.npz", allow_pickle=False) as saved:
        for key in [
            "pressure",
            "shear",
            "plastic_slip",
            "cumulative_slip",
            "platen",
            "contact_jumps",
        ]:
            saved_differences[key] = float(np.max(abs(fields[key] - saved[key])))
            np.testing.assert_allclose(fields[key], saved[key], rtol=1e-9, atol=1e-10)
    np.savez(args.output / f"{args.variant}.npz", **fields)
    record = dict(
        variant=args.variant,
        steps=history.plan.steps,
        dofs=history.plan.dofs,
        seconds=perf_counter() - start,
        value_mm=answer["value_mm"],
        max_residual_mpa=float(np.max(result.residual_mpa)),
        saved_trajectory_differences=saved_differences,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        runtime_overrides="Selected online history implementation; explicitly supplied frozen CQ weights",
    )
    write_json(args.output / f"{args.variant}.json", record)
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["reference", "streamed"])
    parser.add_argument("--steps", type=int, default=4097)
    parser.add_argument("--dofs", type=int, default=64)
    parser.add_argument("--case-run", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/history-storage-experiment")
    args = parser.parse_args()
    if (args.case_run is None) != (args.weights is None) or (
        args.dynamic and args.case_run is None
    ):
        raise ValueError("Provide both --case-run and --weights for a saved model")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.variant is not None:
        (dynamic_worker if args.dynamic else worker)(args)
        return
    small = verify_small_cases()
    records = []
    for variant in ["reference", "streamed"]:
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--variant",
            variant,
            "--steps",
            str(args.steps),
            "--dofs",
            str(args.dofs),
            "--output",
            str(args.output),
        ]
        if args.case_run is not None:
            command.extend(["--case-run", str(args.case_run), "--weights", str(args.weights)])
        if args.dynamic:
            command.append("--dynamic")
        environment = os.environ.copy()
        for name in [
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ]:
            environment[name] = "1"
        subprocess.run(command, check=True, env=environment)
        records.append(json.loads((args.output / f"{variant}.json").read_text()))
    differences = {}
    with (
        np.load(args.output / "reference.npz") as reference,
        np.load(args.output / "streamed.npz") as streamed,
    ):
        for key in reference.files:
            differences[key] = float(np.max(abs(reference[key] - streamed[key])))
            np.testing.assert_allclose(
                streamed[key], reference[key], rtol=1e-7 if "gradient" in key else 1e-9, atol=1e-10
            )
    supplied_inputs = (
        {}
        if args.case_run is None
        else {
            str(path.resolve()): file_sha256(path)
            for path in [
                args.weights,
                args.weights.with_suffix(".json"),
                args.case_run / "trajectory.npz",
                args.case_run / "run_manifest.json",
            ]
        }
    )
    write_json(
        args.output / "summary.json",
        dict(
            records=records,
            small_checks=small,
            max_field_differences=differences,
            environment=environment_record(),
            supplied_input_sha256=supplied_inputs,
            prototype_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope="Exact production last-use row streaming versus original cache-all policy; timings may overlap other jobs",
        ),
    )


if __name__ == "__main__":
    main()

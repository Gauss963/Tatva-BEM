"""Frozen-plastic FEM onset study in our archived checkout, without dynamics.

Uses the unmodified source static solver twice per mesh. This is a static
prediction, not an event-16 dynamic onset. No external Tatva file is written.
Reads verified authored inputs by default. The old packaged float32 settings
remain available only through --historical-input, as an explicit control.
"""

import argparse
import hashlib
import json
import math
import os
import runpy
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
REVISION = "933ae1f3fe1c634391bdbd316e4efa5dbf0e6d5d"
REFERENCE_TOOL = runpy.run_path(str(Path(__file__).with_name("run_fem_reference.py")))


def static_inputs(*, historical=False, reference_input=None):
    """Distinguish the verified model input from the old statistics-derived control."""
    if historical:
        path = ROOT / "src/finiteblock_bem/resources/data/fem_static_onset_input.json"
        record = json.loads(path.read_text())
        if record["source_revision"] != REVISION:
            raise ValueError("Historical static input has a different source revision")
        case = record["case"]
        case = dict(
            **{k: v for k, v in case.items() if k not in {"geometry", "effective_friction"}},
            **case["geometry"],
        )
        paths, authority = [path], "Explicit historical statistics-derived control"
        solver = record["solver"]
    else:
        folder = (
            ROOT.parent / "validation/static_bem2d"
            if reference_input is None
            else Path(reference_input)
        )
        record = REFERENCE_TOOL["load_reference_input"](folder)
        case, solver = record["case"], record["solver"]
        paths = [folder / "input/case.toml", folder / "input/resolved.json"]
        authority = (
            "Authored/resolved solver inputs with verified TOML fingerprint and source revision"
        )
    return dict(
        case=case,
        solver=solver,
        authority=authority,
        historical_control=historical,
        input_sha256={str(p.resolve()): REFERENCE_TOOL["digest"](p) for p in paths},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", type=float, nargs="+", default=[2, 1, 0.5])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/fem-static-onset-authored-f64"
    )
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--reference-input", type=Path)
    parser.add_argument("--historical-input", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("FEM study output must remain inside this checkout's bem/outputs")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new empty static-study output directory")
    if args.cores < 1 or any(not math.isfinite(h) or h <= 0 for h in args.meshes):
        raise ValueError("Require positive mesh sizes and core count")
    if args.historical_input and args.reference_input is not None:
        raise ValueError("Historical control and authored-input directory are mutually exclusive")
    inputs = static_inputs(historical=args.historical_input, reference_input=args.reference_input)
    protected = {
        str(path.resolve()): REFERENCE_TOOL["digest"](path)
        for path in [Path(__file__), Path(__file__).with_name("run_fem_reference.py")]
    }
    protected.update(inputs["input_sha256"])
    for name in [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
    ]:
        os.environ[name] = str(args.cores)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={args.cores}"
    )
    os.environ["JAX_ENABLE_X64"] = "true" if inputs["solver"]["dtype"] == "float64" else "false"
    sys.dont_write_bytecode = True
    snapshot = ROOT / "outputs/fem-source" / REVISION
    archived_source = REFERENCE_TOOL["stage_snapshot"](ROOT.parent, snapshot)
    sys.path.insert(0, str(snapshot))
    from dataclasses import asdict, replace

    import numpy as np
    from tatva.pmma import static_normal
    from tatva.pmma.dynamics import RunConfig, build_case_model
    from tatva.pmma.model import (
        BlockSpec,
        FrictionReference,
        LoadingReference,
        Material,
        PMMAModelInput,
    )

    settings, legacy = inputs["solver"], inputs["case"]
    case = PMMAModelInput(
        BlockSpec(**legacy["moving"]),
        BlockSpec(**legacy["stationary"]),
        {k: Material(**v) for k, v in legacy["materials"].items()},
        FrictionReference(**legacy["friction"]),
        LoadingReference(**legacy["simulation"]),
    )
    seed_path = ROOT / "src/finiteblock_bem/resources/data/ev16_restart_interface.csv"
    protected[str(seed_path)] = REFERENCE_TOOL["digest"](seed_path)
    seed = np.genfromtxt(
        seed_path,
        delimiter=",",
        names=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    records = []
    effective_configs = []
    original = static_normal._block_stiffness
    for h in args.meshes:
        start = perf_counter()
        # This study never integrates in time. The archived h2 dynamic dt can
        # exceed the contact stability limit on a finer mesh; let the model
        # builder derive its own valid dt rather than reject a static solve.
        config = replace(
            RunConfig(**settings), mesh_size=h, restart_from_file=None, time_step_override=None
        )
        effective_configs.append(asdict(config))
        model = build_case_model(case, config)
        y = np.asarray(model["moving"].mesh.coords)[np.asarray(model["master_nodes"]), 1]
        z = np.interp(y, seed["y"], seed["plastic_slip"])
        matrices = {}

        def cached_stiffness(block, material):
            key = id(block)
            if key not in matrices:
                matrices[key] = original(block, material)
            return matrices[key]

        static_normal._block_stiffness = cached_stiffness
        actuator, delta = 10.806226665707777, 0.01
        try:
            base, other = [
                static_normal.solve_static_normal_phase(
                    model, config, actuator_displacement=d, plastic_slip=z
                )
                for d in [actuator, actuator + delta]
            ]
        finally:
            static_normal._block_stiffness = original
        for answer in [base, other]:
            if answer["sliding_nodes"] or answer["open_nodes"]:
                raise ValueError("Frozen-contact tangent is not valid for this static branch")
            np.testing.assert_array_equal(answer["plastic_slip"], z)
        p, tau = base["sigma_n"], base["tau"]
        dp, dtau = (other["sigma_n"] - p) / delta, (other["tau"] - tau) / delta
        mu = np.asarray(model["mu_s_profile"])
        onset = np.full(len(y), np.inf)
        for gap, slope in [(mu * p - tau, dtau - mu * dp), (mu * p + tau, -dtau - mu * dp)]:
            trial = np.full(len(y), np.inf)
            np.divide(gap, slope, out=trial, where=(slope > 0) & (gap >= 0))
            trial[(p + dp * trial <= 0) | (trial <= 0)] = np.inf
            onset = np.minimum(onset, trial)
        index = np.argmin(onset)
        opening = np.full(len(y), np.inf)
        np.divide(-p, dp, out=opening, where=dp < 0)
        record = dict(
            h_mm=h,
            model_dtype=config.dtype,
            moving_coordinates_dtype=str(model["moving"].mesh.coords.dtype),
            stiffness_matrix_dtypes=[str(matrix.dtype) for matrix in matrices.values()],
            predicted_shear_time_s=float(onset[index] / 20),
            predicted_y_mm=float(y[index]),
            prediction_valid=bool(np.isfinite(onset[index]) and opening.min() > onset[index]),
            lower_p_mpa=float(p[0]),
            upper_p_mpa=float(p[-1]),
            lower_tau_mpa=float(tau[0]),
            upper_tau_mpa=float(tau[-1]),
            platen_mm=base["loading_face_displacement"],
            static_residual=base["relative_residual"],
            seconds=perf_counter() - start,
        )
        records.append(record)
        np.savetxt(
            output / f"h{h}.csv",
            np.c_[y, p, tau, z, onset / 20],
            delimiter=",",
            comments="",
            header="y_mm,p_mpa,tau_mpa,plastic_slip_mm,elastic_yield_time_s",
        )
        print(json.dumps(record), flush=True)
    REFERENCE_TOOL["verify_snapshot"](snapshot, archived_source["files"])
    if any(REFERENCE_TOOL["digest"](path) != checksum for path, checksum in protected.items()):
        raise ValueError("Static-study source or protected input changed during execution")
    (output / "summary.json").write_text(
        json.dumps(
            dict(
                records=records,
                source_revision=REVISION,
                configuration_authority=inputs["authority"],
                historical_control=inputs["historical_control"],
                model_dtype=settings["dtype"],
                requested_cores=args.cores,
                hard_cpu_affinity=False,
                protected_input_sha256=protected,
                source_archive_sha256=archived_source["archive_sha256"],
                effective_run_configs=effective_configs,
                authored_time_step_override=settings.get("time_step_override"),
                time_step_policy="Automatic per-mesh model setup; no transient time integration performed",
                seed_scope="Fixed packaged float32-exported piecewise-linear plastic slip, shared with BEM; not the original float64 event-15 HDF5 restart",
                original_dynamic_event_reproduced=False,
                static_solver_sha256=hashlib.sha256(
                    Path(static_normal.__file__).read_bytes()
                ).hexdigest(),
                scope="Frozen-plastic static prediction, using fixed interpolated seed; not full dynamic rupture",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

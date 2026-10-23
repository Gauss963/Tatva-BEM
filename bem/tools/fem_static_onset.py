"""Frozen-plastic FEM onset study in our archived checkout, without dynamics.

Uses the unmodified source static solver twice per mesh. This is a static
prediction, not an event-16 dynamic onset. No external Tatva file is written.
Reads packaged model settings and the Git revision recorded by event 16.
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
REVISION = "933ae1f3fe1c634391bdbd316e4efa5dbf0e6d5d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", type=float, nargs="+", default=[2, 1, 0.5])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fem-static-onset")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("FEM study output must remain inside this checkout's bem/outputs")
    for name in [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
    ]:
        os.environ[name] = "4"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
    sys.dont_write_bytecode = True
    snapshot = ROOT / "outputs/fem-source" / REVISION
    if not snapshot.is_dir():
        archive = subprocess.run(
            ["git", "-C", str(ROOT.parent), "archive", REVISION, "tatva"],
            check=True,
            capture_output=True,
        ).stdout
        snapshot.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(snapshot, filter="data")
    sys.path.insert(0, str(snapshot))
    from dataclasses import replace

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

    input_path = ROOT / "src/finiteblock_bem/resources/data/fem_static_onset_input.json"
    model_input = json.loads(input_path.read_text())
    if model_input["source_revision"] != REVISION:
        raise ValueError("Packaged static settings and archived FEM source revision differ")
    settings, legacy = model_input["solver"], model_input["case"]
    case = PMMAModelInput(
        BlockSpec(**legacy["geometry"]["moving"]),
        BlockSpec(**legacy["geometry"]["stationary"]),
        {k: Material(**v) for k, v in legacy["materials"].items()},
        FrictionReference(**legacy["friction"]),
        LoadingReference(**legacy["simulation"]),
    )
    seed = np.genfromtxt(
        ROOT / "src/finiteblock_bem/resources/data/ev16_restart_interface.csv",
        delimiter=",",
        names=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    records = []
    original = static_normal._block_stiffness
    for h in args.meshes:
        start = perf_counter()
        config = replace(RunConfig(**settings), mesh_size=h, restart_from_file=None)
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
        base, other = [
            static_normal.solve_static_normal_phase(
                model, config, actuator_displacement=d, plastic_slip=z
            )
            for d in [actuator, actuator + delta]
        ]
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
        static_normal._block_stiffness = original
    (output / "summary.json").write_text(
        json.dumps(
            dict(
                records=records,
                source_revision=REVISION,
                input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
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

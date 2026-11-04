"""Check a candidate against the original JAX and independent NumPy stage law.

The original custom root may refine the supplied guess. This diagnostic
does not commit contact state or advance CQ history. Passing a local
equation is not a uniqueness, trajectory or nonlinear stability proof.
"""

import argparse
import json
import runpy
from pathlib import Path

import numpy as np

from finiteblock_bem.friction import ContactState, SlipWeakeningContact
from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
CAPTURE = runpy.run_path(str(TOOLS / "diagnose_radau_root.py"))


def numpy_stage_response(data, q, counts):
    """Replay both unchanged linear paths with the separate NumPy law."""
    n, d = len(data["plastic_slip"]), len(data["previous_displacement"])
    stages = np.asarray(q).reshape(2, d)
    previous = np.stack(
        (
            data["base_dn"] + data["previous_displacement"][:n],
            data["base_ds"] + data["previous_displacement"][n : 2 * n],
        ),
        axis=-1,
    )
    law = SlipWeakeningContact(*data["parameters"])
    state = ContactState(data["plastic_slip"], data["cumulative_slip"])
    saved, work = [], 0.0
    for stage, count in enumerate(counts):
        target = np.stack(
            (data["base_dn"] + stages[stage, :n], data["base_ds"] + stages[stage, n : 2 * n]),
            axis=-1,
        )
        for index in range(count):
            jump = previous + (index + 1) / count * (target - previous)
            response = law.evaluate(jump[:, 0], jump[:, 1], state)
            work += float(
                np.sum(
                    data["contact_weights"]
                    * response.shear
                    * (response.state.plastic_slip - state.plastic_slip)
                )
            )
            state = response.state
        saved.append(
            np.column_stack(
                (
                    response.pressure,
                    response.shear,
                    state.plastic_slip,
                    state.cumulative_slip,
                    response.friction_coefficient,
                )
            )
        )
        previous = target
    return np.stack(saved, axis=1), work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--key", default="random_hybrid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture, candidates, output = (
        p.resolve() for p in [args.capture, args.candidates, args.output]
    )
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh candidate audit outside protected Tatva")
    failure = json.loads(capture.with_name("failure.json").read_text())
    if CAPTURE["digest"](capture) != failure["failed_equilibrium_sha256"]:
        raise ValueError("Original captured equation changed")
    candidate_sha = CAPTURE["digest"](candidates)
    with np.load(capture, allow_pickle=False) as source:
        data = {k: source[k] for k in source.files if not k.startswith("accepted_")}
    with np.load(candidates, allow_pickle=False) as source:
        guess = source[args.key]
    if (
        guess.shape != data["forcing"].shape
        or guess.dtype != np.float64
        or not np.all(np.isfinite(guess))
    ):
        raise ValueError("Require an aligned finite float64 candidate")
    counts = tuple(failure["segment_substeps"])
    paths = [
        Path(__file__),
        TOOLS / "diagnose_radau_root.py",
        TOOLS / "identity_stage_tangent.py",
        TOOLS / "radau_contact_experiment.py",
        TOOLS / "substepped_contact_experiment.py",
        TOOLS.parent / "src/finiteblock_bem/friction.py",
    ]
    frozen = {str(p): fingerprint(p) for p in paths}
    system, _, state, _ = CAPTURE["captured_stage"](data, counts)
    if np.asarray(system.parameters).dtype != np.float64:
        raise ValueError("Original stage verification requires JAX x64")
    q, _ = system.implicit_displacement(
        system.parameters,
        state.plastic_slip,
        state.cumulative_slip,
        data["forcing"],
        guess,
        data["previous_displacement"],
    )
    q = np.asarray(q)
    jax_values = np.asarray(
        system.local_response(
            q,
            state.plastic_slip,
            state.cumulative_slip,
            system.parameters,
            data["previous_displacement"],
        )
    )
    jax_residual = np.asarray(
        system.residual(
            q,
            state.plastic_slip,
            state.cumulative_slip,
            system.parameters,
            data["forcing"],
            data["previous_displacement"],
        )
    )
    values, work = numpy_stage_response(data, q, counts)
    contact = np.concatenate(
        [
            np.r_[
                data["contact_weights"] * values[:, stage, 0],
                data["contact_weights"] * values[:, stage, 1],
                0,
            ]
            for stage in range(2)
        ]
    )
    numpy_residual = (data["matrix"] @ q + data["forcing"] + contact) / np.tile(
        data["reduced_weights"], 2
    )
    errors = [float(np.max(np.abs(v))) for v in [jax_residual, numpy_residual]]
    finite = all(
        np.all(np.isfinite(v)) for v in [q, jax_values, values, numpy_residual, jax_residual]
    )
    if not finite or max(errors) >= 1e-7:
        raise ValueError(f"Original local equation did not pass: {errors}")
    np.testing.assert_allclose(values, jax_values[:, :, :5], rtol=1e-11, atol=1e-12)
    if work < -1e-10:
        raise ValueError("Independent local friction work is negative")
    if (
        CAPTURE["digest"](capture) != failure["failed_equilibrium_sha256"]
        or CAPTURE["digest"](candidates) != candidate_sha
        or any(fingerprint(p) != h for p, h in frozen.items())
    ):
        raise ValueError("Frozen candidate or equation sources changed")
    output.mkdir(parents=True)
    np.savez(output / "verified_candidate_roots.npz", **{args.key: q})
    np.savez(output / "independent_state.npz", response=values, residual=numpy_residual)
    write_json(
        output / "summary.json",
        dict(
            status="passed_captured_original_stage_candidate",
            capture=str(capture),
            capture_sha256=failure["failed_equilibrium_sha256"],
            candidates=str(candidates),
            candidates_sha256=candidate_sha,
            key=args.key,
            segment_substeps=list(counts),
            failed_step=failure["failed_step"],
            original_jax_residual_mpa=errors[0],
            independent_numpy_residual_mpa=errors[1],
            maximum_absolute_local_response_difference=float(
                np.max(np.abs(values - jax_values[:, :, :5]))
            ),
            candidate_unchanged_by_original_root=bool(np.array_equal(q, guess)),
            minimum_pressure_mpa=float(values[:, :, 0].min()),
            maximum_cumulative_slip_mm=float(values[:, :, 3].max()),
            micro_friction_work_mpa_mm2=work,
            source_sha256=frozen,
            no_physical_state_committed=True,
            full_original_event_reproduced=False,
            same_accuracy_fem_advantage_verified=False,
        ),
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

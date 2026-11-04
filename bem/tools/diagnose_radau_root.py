"""Replay a captured two-stage equation with unchanged physical state/forcing.

Uses the existing diagnostic root strategies with a stage-shape adapter and
the validated optional identity tangent. This does not resume CQ history or
commit any candidate to a trajectory. Every trial vector is saved separately.
"""

import argparse
import hashlib
import json
import runpy
from pathlib import Path
from time import perf_counter
from types import FunctionType, SimpleNamespace

import jax.numpy as jnp
import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.friction import ContactState, SlipWeakeningContact
from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
POLICY = runpy.run_path(str(TOOLS / "identity_stage_tangent.py"))
PROBE = runpy.run_path(str(TOOLS / "diagnose_contact_root.py"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def captured_stage(data, counts):
    d, n = len(data["previous_displacement"]), len(data["plastic_slip"])
    if d != 2 * n + 1 or data["matrix"].shape != (2 * d, 2 * d):
        raise ValueError("Require the captured coupled normal/shear/platen stage equation")
    shapes = dict(
        forcing=(2 * d,),
        failed_displacement=(2 * d,),
        reduced_weights=(d,),
        base_dn=(n,),
        base_ds=(n,),
        cumulative_slip=(n,),
        contact_weights=(n,),
        parameters=(5,),
    )
    if any(data[key].shape != shape for key, shape in shapes.items()) or any(
        not np.all(np.isfinite(value)) for value in data.values()
    ):
        raise ValueError("Captured stage shapes or finite data differ")
    if any(np.any(data[key] <= 0) for key in ["contact_weights", "reduced_weights"]):
        raise ValueError("Captured work weights must be positive")
    reference = SimpleNamespace(
        base_dn=data["base_dn"],
        base_ds=data["base_ds"],
        base_platen=0.0,
        case=SimpleNamespace(contact=SlipWeakeningContact(*data["parameters"])),
        constraints=SimpleNamespace(
            size=d,
            normal_jump=np.pad(np.eye(n), ((0, 0), (0, n + 1))),
            tangential_jump=np.pad(np.eye(n), ((0, 0), (n, 1))),
            contact_weights=data["contact_weights"],
            reduced_weights=data["reduced_weights"],
        ),
    )
    system = POLICY["stage_equilibrium"](reference, data["matrix"], segment_substeps=counts)
    previous = jnp.asarray(data["previous_displacement"])

    def response(q, z, c, p, unused_previous):
        value = system.local_response(q, z, c, p, previous)
        return value[:, 1]

    # The old root harness uses previous_displacement as both a predictor and
    # a local-law argument. Its predictor has two stages, while this adapter
    # keeps the actual accepted single-stage previous displacement fixed.
    view = SimpleNamespace(
        parameters=system.parameters,
        base=system.base,
        residual=lambda q, z, c, p, f, unused: system.residual(q, z, c, p, f, previous),
        tangent=lambda q, z, c, p, unused: system.tangent(q, z, c, p, previous),
        local_response=response,
        previous_displacement=np.tile(data["previous_displacement"], 2),
        accepted_micro_work=[],
        accepted_integrated_strength_work=[],
    )
    state = ContactState(data["plastic_slip"].copy(), data["cumulative_slip"].copy())
    adapted = {**data, "previous_displacement": view.previous_displacement.copy()}
    return system, view, state, adapted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=["hybr", "lm", "trf", "forcing", "kinetic"], default=["lm"]
    )
    parser.add_argument("--max-nfev", type=int, default=200)
    args = parser.parse_args()
    capture, output = args.capture.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh owned root diagnostic directory")
    failure = json.loads(capture.with_name("failure.json").read_text())
    if digest(capture) != failure["failed_equilibrium_sha256"]:
        raise ValueError("Captured failed equation changed")
    counts = tuple(failure["segment_substeps"])
    with np.load(capture, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files if not key.startswith("accepted_")}
    sources = {
        str(p.resolve()): fingerprint(p)
        for p in [
            Path(__file__),
            TOOLS / "identity_stage_tangent.py",
            TOOLS / "diagnose_contact_root.py",
            TOOLS / "radau_contact_experiment.py",
        ]
    }
    output.mkdir(parents=True)
    write_json(
        output / "manifest.json",
        dict(
            capture=str(capture),
            capture_sha256=failure["failed_equilibrium_sha256"],
            source_sha256=sources,
            segment_substeps=counts,
            physical_state_not_committed=True,
        ),
    )
    begun, trials = perf_counter(), []

    def tracking(original, kind):
        def execute(*a, **k):
            result = original(*a, **k)
            name = f"trial-{len(trials):04d}-{kind}.npy"
            np.save(output / name, result.x, allow_pickle=False)
            trials.append(
                dict(
                    path=name,
                    method=k.get("method"),
                    nfev=result.nfev,
                    reported_success=bool(result.success),
                )
            )
            write_json(output / "trials.json", trials)
            return result

        return execute

    probe = PROBE["probe_roots"]
    probe = FunctionType(
        probe.__code__,
        {
            **probe.__globals__,
            "root": tracking(PROBE["root"], "root"),
            "least_squares": tracking(PROBE["least_squares"], "least-squares"),
        },
        probe.__name__,
        probe.__defaults__,
    )
    probe.__kwdefaults__ = PROBE["probe_roots"].__kwdefaults__
    try:
        with threadpool_limits(limits=1):
            system, view, state, adapted = captured_stage(data, counts)
            z, c = jnp.asarray(state.plastic_slip), jnp.asarray(state.cumulative_slip)
            q, previous = (
                jnp.asarray(data["failed_displacement"]),
                jnp.asarray(data["previous_displacement"]),
            )
            residual = float(
                jnp.max(abs(system.residual(q, z, c, system.parameters, data["forcing"], previous)))
            )
            np.testing.assert_allclose(residual, failure["residual_mpa"], rtol=1e-12, atol=1e-12)
            values = np.asarray(system.local_response(q, z, c, system.parameters, previous))
            write_json(
                output / "reproduced_failure.json",
                dict(
                    failed_step=failure["failed_step"],
                    stage_time_s=data["stage_time"].tolist(),
                    residual_mpa=residual,
                    previous_cumulative_min_mm=float(state.cumulative_slip.min()),
                    previous_cumulative_max_mm=float(state.cumulative_slip.max()),
                    previous_dc_coverage=float(
                        np.average(
                            state.cumulative_slip >= data["parameters"][-1],
                            weights=data["contact_weights"],
                        )
                    ),
                    trial_pressure_min_mpa=float(values[:, :, 0].min()),
                    trial_cumulative_max_mm=float(values[:, :, 3].max()),
                    trial_open_nodes_by_stage=np.sum(values[:, :, 0] <= 0, axis=0).tolist(),
                    tangent_policy=system.tangent_policy,
                ),
            )
            records, roots, distances = probe(
                view, state, adapted, methods=args.methods, max_nfev=args.max_nfev
            )
            if roots:
                np.savez(output / "candidate_roots.npz", **roots)
        if digest(capture) != failure["failed_equilibrium_sha256"] or any(
            fingerprint(p) != value for p, value in sources.items()
        ):
            raise ValueError("Captured equation or root diagnostic source changed")
        write_json(
            output / "summary.json",
            dict(
                status="found_root_of_captured_stage_equation" if roots else "no_root_recovered",
                records=records,
                accepted_root_distances_mm=distances,
                acceptance_tolerance_mpa=1e-7,
                segment_substeps=counts,
                seconds=perf_counter() - begun,
                original_state_and_forcing_unchanged=True,
                source_sha256=sources,
                full_original_event_reproduced=False,
                scope="Captured local stage equation only. Full accepted prefix retained in the original capture; no CQ restart, physical state commit, uniqueness or trajectory stability proof.",
            ),
        )
    except BaseException as error:
        write_json(output / "failure.json", dict(error=repr(error), seconds=perf_counter() - begun))
        raise


if __name__ == "__main__":
    main()

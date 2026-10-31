"""Record every fine FEM contact endpoint in verified dt/16 chunks.

The original step/scan is reused with additional post-state outputs. Every
native scalar row and all twelve final carry arrays must match the saved
original result bitwise before observations are committed. Path analysis
is deliberately separate, so a failed analysis cannot discard paid data.
"""

import argparse
import json
import runpy
import shutil
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from finiteblock_bem.io import environment_record, write_json

TOOLS = Path(__file__).resolve().parent
EXT = runpy.run_path(str(TOOLS / "extend_fem_timestep_window.py"))
WINDOW, CHUNKS, DIGEST = EXT["WINDOW"], EXT["CHUNKS"], EXT["DIGEST"]


def read_chunk(path):
    with np.load(path, allow_pickle=False) as a:
        arrays = {k: a[k] for k in a.files}
    meta = json.loads(str(arrays["metadata"]))
    if meta["payload_sha256"] != CHUNKS["payload_digest"](arrays) or meta["factor"] != 16:
        raise ValueError("Require an unchanged accepted dt/16 chunk payload")
    return arrays, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent", type=Path)
    p.add_argument("extension", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunks", type=int, nargs="+", default=[55, 70])
    p.add_argument("--pilot-fine-steps", type=int)
    p.add_argument("--max-output-mb", type=float, default=4096)
    args = p.parse_args()
    output, parent, extension = (v.resolve() for v in [args.output, args.parent, args.extension])
    if (
        output.exists()
        or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve())
        or output.is_relative_to(parent)
        or output.is_relative_to(extension)
        or args.chunks != sorted(set(args.chunks))
        or not args.chunks
        or min(args.chunks) < 1
        or (args.pilot_fine_steps is not None and not 2 <= args.pilot_fine_steps <= 1024)
        or not np.isfinite(args.max_output_mb)
        or args.max_output_mb <= 0
    ):
        raise ValueError("Require fresh owned output, ordered chunks and bounded native pilot")
    begun = perf_counter()
    compare = runpy.run_path(str(TOOLS / "compare_extended_fem_windows.py"))
    original, manifest, _ = compare["verified_extension"](extension)
    count = original["last_shear_step"] - original["first_shear_step"]
    record = next(r for r in original["records"] if r["factor"] == 16)
    if max(args.chunks) >= record["chunks"]:
        raise ValueError("Requested accepted-state chunk is outside the completed extension")
    paths = [extension / "factor16-chunks" / f"chunk-{i:05}.npz" for i in args.chunks]
    previous = [path.with_name(f"chunk-{i - 1:05}.npz") for path, i in zip(paths, args.chunks)]
    headers = [read_chunk(path)[1] for path in paths]
    rows = [
        (m["last"] - m["first"]) * 16 if args.pilot_fine_steps is None else args.pilot_fine_steps
        for m in headers
    ]
    if args.pilot_fine_steps is not None and len(paths) != 1:
        raise ValueError("Use one original chunk for an explicit bitwise observation pilot")
    with np.load(extension / "factor16.npz", allow_pickle=False) as a:
        y = a["contact_y"]
    logical = sum((n + 1) * len(y) * 8 * 8 + n * 17 * 8 for n in rows)
    if logical > args.max_output_mb * 1024**2:
        raise ValueError("Native observation arrays exceed the explicit output budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < logical + 1024**3:
        raise ValueError("Keep at least one GiB free beyond logical native observations")
    output.mkdir()
    frozen = dict(manifest["frozen_inputs_sha256"])
    for path in [*TOOLS.glob("*.py"), *paths, *previous, extension / "summary.json"]:
        frozen[str(path.resolve())] = DIGEST(path)
    observer = runpy.run_path(str(TOOLS / "fem_contact_path_observations.py"))
    environment = environment_record()
    write_json(
        output / "preflight.json",
        dict(
            chunks=args.chunks,
            pilot_fine_steps=args.pilot_fine_steps,
            logical_observation_bytes=logical,
            process_rss_hard_capped=False,
            frozen_inputs_sha256=frozen,
            environment=environment,
            full_event_convergence_or_bem_advantage=False,
        ),
    )
    try:
        files = manifest["frozen_inputs_sha256"]
        fem = next(Path(path).parent for path in files if path.endswith("reference_manifest.json"))
        checkpoint = next(Path(path) for path in files if path.endswith("checkpoint.npz"))
        context = EXT["setup_prefix"](
            WINDOW["main"],
            SimpleNamespace(
                fem=fem,
                checkpoint=checkpoint,
                output=output / "setup",
                original_steps=count,
                factors=[1, 2, 4],
            ),
        )
        if DIGEST(output / "setup/manifest.json") != manifest["context_sha256"]:
            raise ValueError("Fresh setup does not match the original complete FEM context")
        model, config = context["model"], context["config"]
        if (
            model["friction_law"] != "slip-weakening"
            or np.any(np.asarray(model["creep_weight_profile"]) != 0)
            or config.normal_relaxation_time is not None
            or config.quasistatic_damping_time is not None
        ):
            raise ValueError("Local path control requires the original undamped, noncreeping LSW")
        import jax
        import jax.numpy as jnp

        local = WINDOW["integrator_prefix"](
            context["dynamics"].run_simulation_dumped,
            lambda *a: {**model, "dt": record["dt_s"]},
        )(context["case"], config, output / "unused.h5")
        zero, prescribed = local["shear_phase_zero_dofs"], local["prescribed_dofs"]
        run = observer["observed_scan"](local["step"], model, zero, prescribed)
        observe = observer["contact_observer"](model)
        profiles = [
            model[k]
            for k in [
                "penalty_n",
                "penalty_t",
                "mu_s_profile",
                "mu_k_profile",
                "critical_slip_profile",
            ]
        ]
        if any(np.any(np.asarray(a) != np.asarray(a).flat[0]) for a in profiles):
            raise ValueError(
                "This path diagnostic requires five uniform original contact parameters"
            )
        parameters = np.array([np.asarray(a).flat[0] for a in profiles])
        results = []
        for path, prior, size in zip(paths, previous, rows, strict=True):
            expected, meta = read_chunk(path)
            initial, initial_meta = read_chunk(prior)
            if (
                initial_meta["last"] != meta["first"]
                or initial_meta["context_sha256"] != meta["context_sha256"]
                or meta["context_sha256"] != DIGEST(extension / "extension_manifest.json")
                or meta["previous_sha256"] != DIGEST(prior)
            ):
                raise ValueError("Accepted initial/final chunks do not share the original context")
            state = tuple(jnp.asarray(initial[f"carry_{i}"]) for i in range(12))
            loading = EXT["schedule"](model, meta["first"], meta["last"], 16)[:size]
            if len(loading) != size:
                raise ValueError("Native observation pilot exceeds the original complete chunk")
            write_json(
                output / "progress.json",
                dict(phase="native_replay", index=meta["index"], fine_steps=size),
            )
            started = perf_counter()
            final, (scalar, values) = run(state, jnp.asarray(loading))
            jax.block_until_ready(final)
            scalar, values = np.asarray(scalar), np.asarray(values)
            if args.pilot_fine_steps is None:
                wanted = tuple(expected[f"carry_{i}"] for i in range(12))
            else:
                wanted, original_scalar = local["advance_chunk"](
                    state, jnp.asarray(loading), zero, prescribed, "shear"
                )
                jax.block_until_ready(wanted)
                if not np.array_equal(original_scalar, scalar):
                    raise ValueError("Observed pilot changes native original scalar rows")
            carry_bits = [np.array_equal(a, b) for a, b in zip(final, wanted, strict=True)]
            scalar_bits = np.array_equal(scalar, expected["scalar"][:size])
            if not all(carry_bits) or not scalar_bits:
                raise ValueError("Added observations changed original numerical carry/scalar bits")
            fields = WINDOW["contact_fields"](final, model)
            endpoint_bits = (
                {k: np.array_equal(fields[k], expected[k]) for k in fields}
                if args.pilot_fine_steps is None
                else None
            )
            if endpoint_bits is not None and not all(endpoint_bits.values()):
                raise ValueError(
                    "Replayed endpoint differs from the original recorded contact fields"
                )
            endpoint_error = np.max(abs(values[-1] - np.stack(list(fields.values()))), axis=1)
            tolerance = np.array([1e-12, 1e-12, 1e-7, 1e-7, 1e-9, 1e-9, 1e-12, 1e-7])
            if not np.all(np.isfinite(values)) or np.any(endpoint_error > tolerance):
                raise ValueError(
                    "JAX post-state observer disagrees with the canonical endpoint output"
                )
            native = np.concatenate((np.asarray(observe(state))[None], values))
            time = meta["first"] * context["old_dt"] + np.arange(size + 1) * record["dt_s"]
            del values
            write_json(
                output / "progress.json", dict(phase="committing_observations", index=meta["index"])
            )
            target = output / f"native-chunk-{meta['index']:05}.npz"
            CHUNKS["atomic_npz"](
                target,
                dict(time_s=time, contact_y=y, fields=native, scalar=scalar, parameters=parameters),
            )
            results.append(
                dict(
                    index=meta["index"],
                    native_rows=size,
                    actual_shear_interval_s=[float(time[0]), float(time[-1])],
                    every_native_scalar_row_bitwise=scalar_bits,
                    all_final_carry_arrays_bitwise=carry_bits,
                    canonical_endpoint_fields_bitwise=endpoint_bits,
                    jax_observer_endpoint_maximum_absolute_errors=endpoint_error.tolist(),
                    projection_analysis_deferred=True,
                    output=str(target),
                    output_sha256=DIGEST(target),
                    seconds=perf_counter() - started,
                )
            )
            write_json(output / "completed_chunks.json", dict(results=results))
            del native, scalar, expected, initial, wanted, final
        if (
            any(DIGEST(path) != value for path, value in frozen.items())
            or environment_record() != environment
        ):
            raise ValueError(
                "Numerical source, runtime or original inputs changed during observation"
            )
        write_json(
            output / "summary.json",
            dict(
                status="passed_bitwise_native_contact_observation_pilot"
                if args.pilot_fine_steps is not None
                else "completed_bitwise_fem_native_contact_observations",
                results=results,
                frozen_inputs_sha256=frozen,
                environment=environment,
                every_observed_step_has_original_float64_scalars=True,
                all_original_contact_endpoints_retained=True,
                source_and_inputs_unchanged=True,
                full_event_convergence_or_bem_advantage_verified=False,
                original_tatva_readonly=True,
                seconds=perf_counter() - begun,
            ),
        )
    except BaseException as error:
        write_json(output / "failure.json", dict(error=repr(error), seconds=perf_counter() - begun))
        raise


if __name__ == "__main__":
    main()

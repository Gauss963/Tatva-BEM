"""Refine the original FEM time step over a checkpointed physical window.

This diagnostic shares the mesh, full accepted state and physical loading.
The original step/scan code is unchanged. It is a local time-refinement
control; it does not establish full-event or spatial convergence.
"""

import argparse
import ast
import inspect
import json
import runpy
import sys
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

TOOLS = Path(__file__).resolve().parent
CAPTURE = runpy.run_path(str(TOOLS / "run_checkpointed_fem_reference.py"))
DIGEST = CAPTURE["digest"]
OBSERVER = runpy.run_path(str(TOOLS / "fem_contact_observations.py"))
SCHEDULES = (
    "normal_schedule_shear",
    "shear_schedule",
    "normal_displacement_shear",
    "normal_velocity_shear",
    "shear_displacement_shear",
    "shear_velocity_shear",
    "normal_relaxation_shear",
)
FIELDS = OBSERVER["FIELDS"]


def integrator_prefix(function, builder):
    """Expose the original numerical closures, returning before output setup."""
    if DIGEST(function.__code__.co_filename) != OBSERVER["SOURCE_SHA256"]:
        raise ValueError("Require the pinned original FEM numerical source")
    tree = ast.parse(inspect.getsource(function))
    target = tree.body[0]
    indices = [
        i
        for i, n in enumerate(target.body)
        if isinstance(n, ast.FunctionDef) and n.name == "sample_stop_indices"
    ]
    if len(indices) != 1:
        raise ValueError("Original numerical/output boundary changed")
    numerical = target.body[: indices[0]]
    if not any(isinstance(n, ast.FunctionDef) and n.name == "advance_chunk" for n in numerical):
        raise ValueError("Original step/scan is missing")
    saved = ast.dump(ast.Module(body=numerical, type_ignores=[]), include_attributes=False)
    target.body = numerical + ast.parse("return locals()").body
    if (
        ast.dump(ast.Module(body=target.body[:-1], type_ignores=[]), include_attributes=False)
        != saved
    ):
        raise RuntimeError("Numerical prefix changed")
    namespace = {**function.__globals__, "build_case_model": builder}
    exec(
        compile(ast.fix_missing_locations(tree), function.__code__.co_filename + "|window", "exec"),
        namespace,
    )
    return namespace[function.__name__]


def rephase_half_velocity(carry, accepted_kick, old_dt, new_dt):
    """Preserve v(t)=v(t+h/2)-h*a(t)/2 using one actual original kick."""
    if not 0 < new_dt <= old_dt or not np.isfinite(old_dt):
        raise ValueError("Require finite positive refinement steps")
    if new_dt == old_dt:
        return carry
    velocity = carry[1] + (0.5 * (new_dt - old_dt)) * accepted_kick
    return (carry[0], velocity, *carry[2:])


def window_schedule(model, first, last, factor):
    """End-of-step loading; preserve all coincident original samples exactly."""
    if factor not in (1, 2, 4) or not 1 <= first < last:
        raise ValueError("Require a positive original-step window and factor 1/2/4")
    columns = []
    for name in SCHEDULES:
        values = np.asarray(model[name][first - 1 : last])
        if values.ndim != 1 or len(values) != last - first + 1 or not np.all(np.isfinite(values)):
            raise ValueError("Invalid complete source loading window")
        if factor == 1:
            columns.append(values[1:])
            continue
        # New endpoints divide each existing physical interval. Integer-aligned
        # old endpoints are assigned explicitly to preserve their bit patterns.
        left = np.repeat(values[:-1], factor)
        slope = np.repeat(np.diff(values), factor)
        fraction = np.tile(np.arange(1, factor + 1) / factor, last - first)
        refined = left + slope * fraction
        refined[factor - 1 :: factor] = values[1:]
        columns.append(refined)
    return np.column_stack(columns)


def contact_fields(carry, model):
    u, z, c, mu, strength = (np.asarray(carry[k]) for k in [0, 2, 3, 6, 8])
    master, slave = (np.asarray(model[k], dtype=int) for k in ["master_nodes", "slave_nodes"])
    offset = int(model["moving_offset"])
    dn = u[2 * master] - u[offset + 2 * slave] - np.asarray(model["interface_initial_gap"])
    ds = u[2 * master + 1] - u[offset + 2 * slave + 1]
    p = float(model["penalty_n"]) * np.maximum(dn, 0)
    tau = np.where(dn > 0, np.asarray(model["penalty_t"]) * (ds - z), 0)
    return dict(zip(FIELDS, [dn, ds, p, tau, z, c, mu, strength], strict=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fem", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--original-steps", type=int, default=26000)
    p.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4])
    args = p.parse_args()
    begun = perf_counter()
    output, fem, checkpoint = args.output.resolve(), args.fem.resolve(), args.checkpoint.resolve()
    if (
        output.exists()
        or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva"))
        or output.is_relative_to(fem)
        or output.is_relative_to(checkpoint.parent)
        or args.original_steps < 1
        or args.factors != [1, 2, 4]
    ):
        raise ValueError("Require fresh owned output and all three refinement factors")
    summary = json.loads((fem / "summary.json").read_text())
    manifest = json.loads((fem / "reference_manifest.json").read_text())
    if (
        summary["status"] != "completed"
        or not summary["full_original_horizon"]
        or summary["computational_dtype"] != "float64"
        or not summary["source_and_protected_input_unchanged"]
        or DIGEST(fem / "reference_manifest.json") != summary["reference_manifest_sha256"]
    ):
        raise ValueError("Require a complete unchanged float64 FEM reference")
    source = Path(manifest["source_snapshot_path"])
    frozen_paths = [
        Path(__file__),
        TOOLS / "run_checkpointed_fem_reference.py",
        TOOLS / "fem_contact_observations.py",
        TOOLS / "chamfer_mesh_refinement.py",
        TOOLS / "fem_static_onset.py",
        TOOLS / "run_fem_reference.py",
        fem / "summary.json",
        fem / "reference_manifest.json",
        fem / "resolved_run_config.json",
        checkpoint,
    ]
    for name, record in manifest["source"]["files"].items():
        file = source / name
        if DIGEST(file) != record["sha256"]:
            raise ValueError("Original archived FEM source changed")
        frozen_paths.append(file)
    if DIGEST(TOOLS / "chamfer_mesh_refinement.py") != manifest["mesh_adapter"]["sha256"]:
        raise ValueError("Reference mesh adapter changed")
    with np.load(checkpoint, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        initial = tuple(data[f"carry_{i}"].copy() for i in range(12))
    old_dt = metadata["dt"]
    if metadata["phase_id"] != 2 or old_dt != summary["summary"]["dt"] or initial[10]:
        raise ValueError("Require an original pre-hold shear checkpoint")
    config_values = json.loads((fem / "resolved_run_config.json").read_text())
    restart = Path(config_values["restart_from_file"])
    frozen_paths.append(restart)
    frozen = {str(a.resolve()): DIGEST(a) for a in frozen_paths}
    hdf = fem / "simulation.h5"
    hdf_hash = DIGEST(hdf)
    output.mkdir(parents=True)
    CAPTURE["write_json"](
        output / "manifest.json",
        dict(
            status="prepared_local_window_not_full_event",
            frozen_inputs_sha256=frozen,
            fem_hdf_sha256=hdf_hash,
            checkpoint_metadata=metadata,
            original_steps=args.original_steps,
            factors=args.factors,
            scope="Conditional on a shared original checkpoint; no spatial refinement or full-event time bound",
        ),
    )
    sys.path.insert(0, str(source))
    import jax
    import jax.numpy as jnp
    from tatva.pmma import dynamics
    from tatva.pmma.model import (
        BlockSpec,
        FrictionReference,
        LoadingReference,
        Material,
        PMMAModelInput,
    )

    jax.config.update("jax_enable_x64", True)
    legacy = summary["summary"]["legacy"]
    case = PMMAModelInput(
        BlockSpec(**legacy["geometry"]["moving"]),
        BlockSpec(**legacy["geometry"]["stationary"]),
        {k: Material(**v) for k, v in legacy["materials"].items()},
        FrictionReference(**legacy["friction"]),
        LoadingReference(**legacy["simulation"]),
    )
    config = dynamics.RunConfig(**config_values)
    builder = runpy.run_path(str(TOOLS / "chamfer_mesh_refinement.py"))[
        "make_chamfer_refined_builder"
    ](dynamics)
    base = integrator_prefix(dynamics.run_simulation_dumped, builder)(
        case, config, output / "unused.h5"
    )
    model = base["model"]
    if model["dt"] != old_dt or model["total_dofs"] != len(initial[0]):
        raise ValueError("Original checkpoint mesh/step differs")
    if config.friction_law != "slip-weakening" or config.normal_phase_mode != "static":
        raise ValueError("This local window is for the original undamped LSW shear")
    for key in ["normal_relaxation_time", "quasistatic_damping_time"]:
        if getattr(config, key) is not None:
            raise ValueError("Do not reinterpret a damped checkpoint")
    carry = tuple(jnp.asarray(a) for a in initial)
    start = metadata["step_id"]
    first_loading = window_schedule(model, start, start + 1, 1)
    common, _ = base["advance_chunk"](
        carry,
        jnp.asarray(first_loading),
        base["shear_phase_zero_dofs"],
        base["prescribed_dofs"],
        "shear",
    )
    jax.block_until_ready(common)
    kick = (common[1] - carry[1]) / old_dt
    first, last = start + 1, start + 1 + args.original_steps
    with h5py.File(hdf, "r") as h:
        g = h["contact_observations"]
        ids = np.flatnonzero(
            (g["phase_id"][:] == 2) & (g["step_id"][:] > first) & (g["step_id"][:] <= last)
        )
        saved_steps = g["step_id"][ids]
        expected = {k: g[k][ids] for k in FIELDS}
        y = g["y"][:]
        high = h["interface_high_rate/history"][ids]
    if len(ids) < 2:
        raise ValueError("Window must include at least two original saved observations")
    stops = sorted(set([*saved_steps.tolist(), last]))
    records = []
    for factor in args.factors:
        started = perf_counter()
        local = (
            base
            if factor == 1
            else integrator_prefix(
                dynamics.run_simulation_dumped, lambda *a: {**model, "dt": old_dt / factor}
            )(case, config, output / "unused.h5")
        )
        state = rephase_half_velocity(common, kick, old_dt, old_dt / factor)
        fields = {k: [] for k in FIELDS}
        rows, times, step_ids, all_rows = [], [], [], []
        previous = first
        for stop in stops:
            loading = window_schedule(model, previous, stop, factor)
            state, values = local["advance_chunk"](
                state,
                jnp.asarray(loading),
                local["shear_phase_zero_dofs"],
                local["prescribed_dofs"],
                "shear",
            )
            jax.block_until_ready(state)
            values = np.asarray(values)
            all_rows.append(values)
            if not np.all(np.isfinite(values)):
                raise ValueError("Nonfinite window trajectory")
            if stop in saved_steps:
                observed = contact_fields(state, model)
                for name in FIELDS:
                    fields[name].append(observed[name])
                rows.append(values[-1])
                times.append(stop * old_dt)
                step_ids.append(stop)
            previous = stop
            CAPTURE["write_json"](
                output / "progress.json",
                dict(
                    factor=factor,
                    original_step=stop,
                    requested_end_step=last,
                    elapsed_seconds=perf_counter() - begun,
                ),
            )
        fields = {k: np.asarray(v) for k, v in fields.items()}
        if not all(np.all(np.isfinite(a)) for a in [*fields.values(), *map(np.asarray, state)]):
            raise ValueError("Nonfinite accepted window state")
        identity = None
        if factor == 1:
            identity = {k: np.array_equal(fields[k], expected[k]) for k in FIELDS}
            # The original HDF5 clock is explicitly replaced by integer step*dt.
            legacy_rows = np.asarray(rows, dtype=np.float32)
            scalar_identity = np.array_equal(legacy_rows[:, 1:], high[:, 1:])
            CAPTURE["write_json"](
                output / "factor1-identity.json",
                dict(
                    contact_bitwise_checks=identity,
                    original_scalar_columns_1_to_16_bitwise=scalar_identity,
                    contact_maximum_absolute_errors={
                        k: float(np.max(abs(fields[k] - expected[k]))) for k in FIELDS
                    },
                ),
            )
        np.savez_compressed(
            output / f"factor{factor}.npz",
            time=np.asarray(times),
            contact_y=y,
            step_id=np.asarray(step_ids),
            scalar_history=np.asarray(rows),
            all_scalar_history=np.concatenate(all_rows),
            **fields,
            **{f"final_carry_{i}": np.asarray(a) for i, a in enumerate(state)},
        )
        if factor == 1 and (not all(identity.values()) or not scalar_identity):
            raise ValueError(
                "Original window failed bitwise contact/scalar identity; see retained audit/state"
            )
        records.append(
            dict(
                factor=factor,
                dt_s=old_dt / factor,
                seconds=perf_counter() - started,
                original_contact_bitwise_checks=identity,
                maximum_cumulative_slip_mm=float(np.max(np.asarray(state[3]))),
            )
        )
        CAPTURE["write_json"](output / "branches.json", dict(records=records))
    if any(DIGEST(a) != v for a, v in frozen.items()) or DIGEST(hdf) != hdf_hash:
        raise ValueError("FEM window inputs changed")
    CAPTURE["write_json"](
        output / "summary.json",
        dict(
            status="completed_local_fem_timestep_window",
            records=records,
            shared_original_step_after_checkpoint=1,
            first_shear_step=first,
            last_shear_step=last,
            physical_shear_interval_s=[first * old_dt, last * old_dt],
            observed_shared_samples=len(ids),
            all_source_inputs_unchanged=True,
            all_scalar_steps_within_window_retained=True,
            contact_observation_scope="Original shared saved endpoints; complete 12-array carry evolved at every substep",
            source_dt_before_window_unchanged=True,
            full_event_time_convergence_verified=False,
            rephasing="One actual source kick; preserve integer-time velocity, change future half-step clock only",
            loading="Piecewise-linear original physical endpoint schedule; coincident samples preserved bitwise",
            total_seconds=perf_counter() - begun,
        ),
    )


if __name__ == "__main__":
    main()

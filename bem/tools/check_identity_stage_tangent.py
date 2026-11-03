"""Verify optional contact assembly with saved kernels and actual fine inputs.

The coarse comparison retains complete initialized primal/JVP/VJP controls.
Optional fine probes use the original 531-DOF initialization and impedance;
their trial states are algebraic checks, not additional full trajectories.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter, process_time
from types import FunctionType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import BoundaryHistory
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
POLICY = runpy.run_path(str(TOOLS / "identity_stage_tangent.py"))
MIXED = runpy.run_path(str(TOOLS / "check_mixed_radau_nonlinear.py"))
CACHE = runpy.run_path(str(TOOLS / "history_prefix_cache.py"))


def bind(function, **globals_):
    result = FunctionType(
        function.__code__,
        {**function.__globals__, **globals_},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def fast_simulation(original, factory):
    return bind(
        original, CONTACT={**original.__globals__["CONTACT"], "RadauContactEquilibrium": factory}
    )


def fast_initialized_ad(module, factory):
    base = module["BASE"]
    context = base["context"]
    context = bind(
        context, CONTACT={**context.__globals__["CONTACT"], "RadauContactEquilibrium": factory}
    )
    initialized = bind(module["initialized_context"], BASE={**base, "context": context})
    return bind(module["initialized_gradient"], initialized_context=initialized)


def coarse_control(kernel, output, frozen, environment, begun):
    calls, policies = dict(simulate=0, ad=0, history=0), []

    def factory(*args, **kwargs):
        system = POLICY["stage_equilibrium"](*args, **kwargs)
        if not system.tangent_policy["applied"]:
            raise ValueError("Coarse control must exercise the identity specialization")
        policies.append(system.tangent_policy)
        return system

    def load(path, *args, **kwargs):
        module = runpy.run_path(path, *args, **kwargs)
        name = Path(path).name
        if name == "simulate_compact_radau.py":
            original = module["simulate_stages"]
            candidate = fast_simulation(original, factory)

            def simulate(*a, **k):
                index = calls["simulate"]
                calls["simulate"] += 1
                if index > 1:
                    raise ValueError("Original two-variant primal control boundary changed")
                return (original if index == 0 else candidate)(*a, **k)

            module["simulate_stages"] = simulate
        if name == "initialized_radau_adjoint.py":
            original, candidate = (
                module["initialized_gradient"],
                fast_initialized_ad(module, factory),
            )

            def ad(*a, **k):
                index = calls["ad"]
                calls["ad"] += 1
                if index > 3:
                    raise ValueError("Original two-variant derivative control boundary changed")
                return (original if index < 2 else candidate)(*a, **k)

            module["initialized_gradient"] = ad
        return module

    def history(plan, values):
        readonly = values.view()
        readonly.flags.writeable = False
        (cached,), _ = CACHE["cache_history_prefixes"](
            [BoundaryHistory(plan, readonly)], capacity_bytes=72 * 1024**2
        )
        calls["history"] += 1
        return cached

    experiment = MIXED["cached_experiment"](MIXED["CachedInputs"](kernel, kernel, environment))
    experiment = bind(experiment, runpy=SimpleNamespace(run_path=load), BoundaryHistory=history)
    result = experiment(output, frozen, environment, begun)
    if calls != dict(simulate=2, ad=4, history=4) or len(policies) != 3:
        raise ValueError("Complete original/candidate root and initialized AD paths were not used")
    with (
        np.load(output / "original-arrays.npz") as original,
        np.load(output / "mixed-arrays.npz") as candidate,
    ):
        result["array_value_equality"] = {
            k: bool(np.array_equal(original[k], candidate[k])) for k in original.files
        }
        result["maximum_absolute_array_difference"] = {
            k: float(np.max(abs(original[k] - candidate[k]))) for k in original.files
        }
    return {
        **result,
        "policies": policies,
        "same_prefix_cache_in_both_variants": True,
        "calls": calls,
    }


def fine_control(kernel, run, output):
    metadata = json.loads((kernel / "summary.json").read_text())
    args = metadata["source_mesh_arguments"]
    if metadata["plan"]["dofs"] != 531 or metadata["full_steps"] != 22500:
        raise ValueError("Fine probes require the original completed 531-DOF kernel")
    current = environment_record(args["case"])
    if any(
        current[key] != metadata["environment"][key]
        for key in ["source_sha256", "case_sha256", "restart_data_sha256", "numpy", "scipy"]
    ):
        raise ValueError("Fine probe physical inputs, package or runtime changed")
    path = kernel / "initial_impedance.npy"
    if fingerprint(path) != metadata["initial_impedance_sha256"]:
        raise ValueError("Original fine stage impedance changed")
    model = load_nodal_model(
        args["case"],
        max_panel_length=args["h"],
        contact_panel_length=args["contact_h"],
        contact_end_length=args["contact_end_h"],
        corner_panel_length=args["corner_h"],
        restart_knot_window=args.get("restart_knot_window"),
        restart_knot_subdivisions=args.get("restart_knot_subdivisions", 1),
    )
    refined = runpy.run_path(str(TOOLS / "refined_contact.py"))
    space = refined["ContactQuadrature"].build(
        model.reference.moving.starts[model.moving_contact, 1], 1
    )
    with np.load(run / "initial_lift.npz", allow_pickle=False) as saved:
        lift = SimpleNamespace(**{name: saved[name] for name in saved.files})
    reference = refined["dynamic_refined_reference"](model, space, lift, np.zeros((531, 531)))
    d, n = reference.constraints.size, len(reference.constraints.contact_weights)
    matrix = np.load(path, allow_pickle=False)
    matrix[[d - 1, 2 * d - 1], [d - 1, 2 * d - 1]] += reference.spring
    systems = {
        "original": POLICY["BASE"]["RadauContactEquilibrium"](
            reference, matrix, segment_substeps=(64, 128)
        ),
        "candidate": POLICY["stage_equilibrium"](reference, matrix, segment_substeps=(64, 128)),
    }
    if not systems["candidate"].tangent_policy["applied"]:
        raise ValueError("Fine probes did not activate the specialization")
    z, c = jnp.asarray(lift.plastic_slip), jnp.asarray(lift.cumulative_slip)
    previous, forcing = jnp.zeros(d), jnp.zeros(2 * d)
    parameters = systems["original"].parameters
    trials = dict(initial=np.zeros((2, d)), sliding=np.zeros((2, d)), opening=np.zeros((2, d)))
    trials["sliding"][:, n : 2 * n] = np.array([1e-4, 3e-4])[:, None]
    trials["opening"][0, :n] = -1.5 * reference.base_dn
    trials["opening"][1, n : 2 * n] = -2e-4
    directions = np.random.default_rng(904).normal(size=(2 * d, 3))

    @jax.jit
    def oracle(q, v):
        return jax.jvp(
            lambda value: systems["original"].residual(value, z, c, parameters, forcing, previous),
            (q,),
            (v,),
        )[1]

    result, warmup = {}, {}
    initial = jnp.asarray(trials["initial"].ravel())
    for name, system in systems.items():
        cpu, wall = process_time(), perf_counter()
        system.tangent(initial, z, c, parameters, previous).block_until_ready()
        warmup[name] = dict(cpu_seconds=process_time() - cpu, wall_seconds=perf_counter() - wall)
    for label, trial in trials.items():
        q, values, timings = jnp.asarray(trial.ravel()), {}, {name: [] for name in systems}
        for repeat in range(3):
            for name in list(systems) if repeat % 2 == 0 else list(systems)[::-1]:
                cpu = process_time()
                value = systems[name].tangent(q, z, c, parameters, previous)
                value.block_until_ready()
                timings[name].append(process_time() - cpu)
                values[name] = np.asarray(value)
        expected, actual = values["original"], values["candidate"]
        errors = {
            str(order): float(
                np.linalg.norm(actual - expected, ord=order) / np.linalg.norm(expected, ord=order)
            )
            for order in [None, np.inf]
        }
        jvps = {name: [] for name in systems}
        for direction in directions.T:
            exact = np.asarray(oracle(q, jnp.asarray(direction)))
            for name, value in values.items():
                jvps[name].append(
                    float(np.linalg.norm(value @ direction - exact) / np.linalg.norm(exact))
                )
        if (
            not all(np.all(np.isfinite(value)) for value in values.values())
            or max(errors.values()) > 1e-11
            or max(max(v) for v in jvps.values()) > 1e-11
        ):
            raise ValueError("Actual fine tangent or independent residual-JVP gate failed")
        np.savez(output / f"fine-{label}.npz", **values, trial=trial, directions=directions)
        result[label] = dict(
            matrix_relative_errors=errors, independent_jvp_relative_errors=jvps, cpu_seconds=timings
        )
    return dict(
        policy=systems["candidate"].tangent_policy,
        probes=result,
        warmup=warmup,
        no_frequency_or_static_reassembly=True,
        original_initial_lift_sha256=fingerprint(run / "initial_lift.npz"),
        physical_stage_impedance_sha256=metadata["initial_impedance_sha256"],
        scope="Actual fine initialization/matrix and 64/128 local microsteps; algebraic trial-state probes, not fine trajectories or a four-core isolated runtime comparison.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fine-kernel", type=Path)
    parser.add_argument("--fine-run", type=Path)
    args = parser.parse_args()
    kernel, output = args.kernel.resolve(), args.output.resolve()
    if (
        output.exists()
        or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve())
        or output.is_relative_to(kernel)
        or bool(args.fine_kernel) != bool(args.fine_run)
        or (
            args.fine_kernel is not None
            and any(output.is_relative_to(p.resolve()) for p in [args.fine_kernel, args.fine_run])
        )
    ):
        raise ValueError("Require fresh owned output and paired optional fine input paths")
    metadata = json.loads((kernel / "summary.json").read_text())
    if (
        metadata["full_steps"] != 128
        or metadata["plan"]["dofs"] > 64
        or metadata["plan"]["fft_size"] != 512
    ):
        raise ValueError("Use the saved complete coarse 128-step kernel")
    output.mkdir(parents=True)
    jax.config.update("jax_enable_x64", True)
    begun = perf_counter()
    case = MIXED["CONTROL"]["CASE"]
    paths = [
        *TOOLS.glob("*.py"),
        case,
        MIXED["CONTROL"]["SEED"],
        *[p for p in kernel.iterdir() if p.is_file()],
    ]
    if args.fine_kernel:
        paths.extend(
            [
                args.fine_kernel / "summary.json",
                args.fine_kernel / "initial_impedance.npy",
                args.fine_run / "initial_lift.npz",
            ]
        )
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    environment = environment_record(case)
    write_json(output / "manifest.json", dict(frozen_inputs_sha256=frozen, environment=environment))
    try:
        with threadpool_limits(limits=1, user_api="blas"):
            coarse = coarse_control(kernel, output, frozen, environment, begun)
            fine = (
                fine_control(args.fine_kernel, args.fine_run, output) if args.fine_kernel else None
            )
        if (
            any(fingerprint(p) != digest for p, digest in frozen.items())
            or environment_record(case) != environment
        ):
            raise ValueError("Original/candidate inputs, numerical source or runtime changed")
        write_json(
            output / "summary.json",
            dict(
                status="passed_identity_tangent_coarse_initialized_control",
                coarse=coarse,
                fine=fine,
                seconds=perf_counter() - begun,
                original_full_event_reproduced=False,
                same_accuracy_fem_advantage_verified=False,
            ),
        )
    except BaseException as error:
        write_json(output / "failure.json", dict(error=repr(error), seconds=perf_counter() - begun))
        raise


if __name__ == "__main__":
    main()

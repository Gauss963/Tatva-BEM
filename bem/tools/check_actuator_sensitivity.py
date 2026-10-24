"""Check selected actuator cotangents by independent forward-CQ JVPs.

Unlike the reverse adjoint or the resolvent inactive-prefix transpose, each
direction is a unit actuator pulse propagated forward through the original Z
history and local implicit tangents at the saved states. Loading event times,
bulk and static lift stay fixed. No finite parameter perturbations are used.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from finiteblock_bem.cq import BoundaryHistory, OnlineConvolution, plan_cq
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.jax_backend import JaxContactEquilibrium
from finiteblock_bem.nodal import (
    NodalInterfaceEquilibrium,
    initialize_nodal_restart,
    load_nodal_model,
)
from finiteblock_bem.nodal_dynamics import nodal_cache_key
from finiteblock_bem.sampling import SamplingOptions


def actuator_forward_gradient(
    system, history, result, reference, index, *, convolution_factory=None
):
    """One actuator input direction, with no reverse-time operations."""
    steps, d = len(result.time) - 1, history.plan.dofs
    if not 1 <= index <= steps:
        raise ValueError("Actuator index must lie after the fixed initial state")
    n = len(reference.constraints.contact_weights)
    q = np.c_[
        result.contact_jumps - np.r_[reference.base_dn, reference.base_ds],
        result.platen - reference.base_platen,
    ]
    wr = jnp.asarray(reference.constraints.reduced_weights)
    parameters = system.parameters

    @jax.jit
    def step(q, old_z, old_c, z_dot, c_dot, past_dot, pulse):
        old_force = jax.jvp(
            lambda z, c: system.residual(q, z, c, parameters, jnp.zeros(d)),
            (old_z, old_c),
            (z_dot, c_dot),
        )[1]
        rhs = -old_force - past_dot / wr
        rhs = rhs.at[-1].add(reference.spring * pulse / wr[-1])
        q_dot = jnp.linalg.solve(system.tangent(q, old_z, old_c, parameters), rhs)
        state_dot = jax.jvp(
            lambda q, z, c: system.local_response(q, z, c, parameters)[:, 2:4],
            (q, old_z, old_c),
            (q_dot, z_dot, c_dot),
        )[1]
        return q_dot, state_dot[:, 0], state_dot[:, 1]

    factory = OnlineConvolution if convolution_factory is None else convolution_factory
    online = factory(history)
    online.append(np.zeros(d))
    z_dot, c_dot = np.zeros(n), np.zeros(n)
    for k in range(1, steps + 1):
        q_dot, z_dot, c_dot = step(
            jnp.asarray(q[k]),
            jnp.asarray(result.plastic_slip[k - 1]),
            jnp.asarray(result.cumulative_slip[k - 1]),
            jnp.asarray(z_dot),
            jnp.asarray(c_dot),
            jnp.asarray(online.past(k)),
            jnp.asarray(float(k == index)),
        )
        q_dot, z_dot, c_dot = map(np.asarray, [q_dot, z_dot, c_dot])
        if not all(np.all(np.isfinite(value)) for value in [q_dot, z_dot, c_dot]):
            raise ValueError(f"Nonfinite actuator sensitivity at step {k}")
        online.append(q_dot)
    return float(np.average(c_dot, weights=reference.constraints.contact_weights))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fft-cache-mb", type=float, default=8)
    parser.add_argument("--max-memory-mb", type=float, default=4096)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new empty output directory")
    folder = args.run.resolve()
    summary = json.loads((folder / "summary.json").read_text())
    options = json.loads((folder / "run_manifest.json").read_text())["arguments"]
    environment = environment_record(options["case"])
    keys = ["source_sha256", "case_sha256", "restart_data_sha256"]
    if (
        summary["status"] != "completed_requested_horizon"
        or summary["source_changed_during_run"]
        or summary.get("history_representation") == "elastic_resolvent"
        or any(summary["environment"][key] != environment[key] for key in keys)
    ):
        raise ValueError("A completed source-frozen original Z run is required")
    if not np.isfinite(args.max_memory_mb) or args.max_memory_mb <= 0:
        raise ValueError("Finite positive array budget required")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    steps, d = summary["completed_steps"], summary["history_dofs"][0]
    if any(not 1 <= index <= steps for index in args.indices):
        raise ValueError("Actuator indices must lie in the saved simulated horizon")
    plan = plan_cq(d, summary["dt_s"], steps, max_memory_mb=1e9)
    sampling = (
        SamplingOptions(relative_tolerance=options["sampling_rtol"])
        if options["sampling"] == "adaptive"
        else None
    )
    cache = Path(options["cache"])
    if not cache.is_absolute():
        cache = Path(__file__).resolve().parents[2] / cache
    cache /= f"galerkin-{nodal_cache_key(model, plan, options['quadrature'], sampling)}.npy"
    if not cache.is_file() or not cache.with_suffix(".json").is_file():
        raise ValueError("Matching cached original Z coefficients are required; none are rebuilt")
    bounded_path = Path(__file__).with_name("bounded_history.py")
    bounded = runpy.run_path(str(bounded_path))
    estimate = bounded["estimate_online_arrays"](plan, steps, args.fft_cache_mb, fft_rows=4)
    estimate += 256 * 1024**2
    if estimate > args.max_memory_mb * 1024**2:
        raise ValueError(f"Forward sensitivity arrays need about {estimate / 1024**2:.1f} MiB")
    paths = [Path(__file__), bounded_path]
    paths += [
        folder / name
        for name in ["summary.json", "run_manifest.json", "trajectory.npz", "adjoint.npz"]
    ]
    frozen = {str(path): fingerprint(path) for path in paths}
    stat = cache.stat()
    weights = np.load(cache, mmap_mode="r", allow_pickle=False)
    if weights.shape != (steps + 1, d, d) or weights.dtype != np.float64:
        raise ValueError("Invalid original Z coefficient array")
    lift = initialize_nodal_restart(model, quadrature=options["quadrature"])
    reference = NodalInterfaceEquilibrium(model, lift, weights[0])
    system = JaxContactEquilibrium(reference)
    with np.load(folder / "trajectory.npz", allow_pickle=False) as saved:
        result = SimpleNamespace(
            **{
                name: saved[name]
                for name in ["time", "platen", "contact_jumps", "plastic_slip", "cumulative_slip"]
            }
        )
    with np.load(folder / "adjoint.npz", allow_pickle=False) as saved:
        target = saved["actuator_gradient"]
    answers = []
    for index in args.indices:
        started = perf_counter()
        actual = actuator_forward_gradient(
            system,
            BoundaryHistory(plan, weights),
            result,
            reference,
            index,
            convolution_factory=lambda h: bounded["BoundedOnlineConvolution"](
                h, cache_mb=args.fft_cache_mb
            ),
        )
        expected = float(target[index])
        record = dict(
            index=index,
            time_s=float(result.time[index]),
            forward_gradient=actual,
            original_reverse_gradient=expected,
            absolute_difference=abs(actual - expected),
            relative_difference=abs(actual - expected) / max(abs(expected), 1e-30),
            seconds_including_jit=perf_counter() - started,
            passes_original_comparison_tolerance=bool(
                np.isclose(actual, expected, rtol=1e-5, atol=1e-12)
            ),
        )
        answers.append(record)
        print(json.dumps(record), flush=True)
    after = cache.stat()
    current = environment_record(options["case"])
    if (
        any(current[key] != environment[key] for key in keys)
        or any(fingerprint(path) != value for path, value in frozen.items())
        or (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
    ):
        raise ValueError("Sensitivity inputs, implementation or original coefficients changed")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "summary.json",
        dict(
            status="completed_selected_actuator_forward_check",
            source_run=str(folder),
            records=answers,
            environment=environment,
            frozen_inputs_sha256=frozen,
            coefficient_path=str(cache),
            coefficient_file_size=stat.st_size,
            coefficient_mtime_ns=stat.st_mtime_ns,
            estimated_array_peak_mib=estimate / 1024**2,
            scope="Independent original-Z forward JVPs, fixed bulk/lift/recorded loading and selected contact branches",
            timing_scope="May overlap other jobs; not a controlled speed comparison",
        ),
    )


if __name__ == "__main__":
    main()

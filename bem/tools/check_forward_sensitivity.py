"""Independently check reverse-CQ gradients with full forward sensitivities.

JAX JVPs differentiate local equations at the recorded states; forward CQ
propagates five tangent histories. No finite parameter perturbations or Newton
iterate derivatives are used. Scope: fixed bulk, lift and recorded loading.
"""

import argparse
import hashlib
import json
import resource
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from finiteblock_bem.adjoint import trajectory_adjoint
from finiteblock_bem.cq import OnlineConvolution, plan_cq
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.jax_backend import JaxContactEquilibrium
from finiteblock_bem.nodal import (
    NodalInterfaceEquilibrium,
    initialize_nodal_restart,
    load_nodal_model,
)
from finiteblock_bem.nodal_dynamics import build_nodal_history, nodal_cache_key
from finiteblock_bem.sampling import SamplingOptions

ROOT = Path(__file__).resolve().parents[1]


def forward_gradient(system, history, result, reference, *, fft_rows=4):
    steps, dofs = history.plan.steps, history.plan.dofs
    n = len(reference.constraints.contact_weights)
    q = np.column_stack(
        (
            result.contact_jumps - np.r_[reference.base_dn, reference.base_ds],
            result.platen - reference.base_platen,
        )
    )
    parameters = system.parameters
    directions = jnp.eye(5)
    scale = jnp.array(reference.constraints.reduced_weights)

    @jax.jit
    def step(q, old_z, old_c, z_dot, c_dot, past_dot):
        def old_force(dz, dc, dp):
            return jax.jvp(
                lambda z, c, p: system.residual(q, z, c, p, jnp.zeros_like(q)),
                (old_z, old_c, parameters),
                (dz, dc, dp),
            )[1]

        forcing_dot = jax.vmap(old_force)(z_dot.T, c_dot.T, directions).T
        tangent = system.tangent(q, old_z, old_c, parameters)
        q_dot = jnp.linalg.solve(tangent, -forcing_dot - past_dot / scale[:, None])

        def update(dq, dz, dc, dp):
            return jax.jvp(
                lambda q, z, c, p: system.local_response(q, z, c, p)[:, 2:4],
                (q, old_z, old_c, parameters),
                (dq, dz, dc, dp),
            )[1]

        state_dot = jax.vmap(update)(q_dot.T, z_dot.T, c_dot.T, directions)
        return q_dot, state_dot[:, :, 0].T, state_dot[:, :, 1].T

    # The five independent source vectors reuse the SAME matrix-kernel FFTs.
    # Last-use row streaming bounds the large levels. All five streams still
    # share the transforms of levels that need reuse.
    transforms = {}
    online = [OnlineConvolution(history, fft_rows=fft_rows) for _ in range(5)]
    for stream in online:
        stream.transforms = transforms
        stream.append(np.zeros(dofs))
    z_dot = c_dot = np.zeros((n, 5))
    for index in range(1, steps + 1):
        past = np.column_stack([stream.past(index) for stream in online])
        q_dot, z_dot, c_dot = step(
            jnp.asarray(q[index]),
            jnp.asarray(result.plastic_slip[index - 1]),
            jnp.asarray(result.cumulative_slip[index - 1]),
            jnp.asarray(z_dot),
            jnp.asarray(c_dot),
            jnp.asarray(past),
        )
        q_dot, z_dot, c_dot = (np.asarray(value) for value in [q_dot, z_dot, c_dot])
        for parameter, stream in enumerate(online):
            stream.append(q_dot[:, parameter])
    return np.average(c_dot, weights=reference.constraints.contact_weights, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--max-memory-mb", type=float, default=8192)
    parser.add_argument("--cache-all-reference", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/forward-sensitivity-check")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    folder = args.run.resolve()
    summary = json.loads((folder / "summary.json").read_text())
    options = json.loads((folder / "run_manifest.json").read_text())["arguments"]
    environment = environment_record(options["case"])
    for name in ["source_sha256", "case_sha256", "restart_data_sha256"]:
        if environment[name] != summary["environment"][name]:
            raise ValueError(f"{name} differs; first replay the forward run with frozen sources")
    if summary["status"] != "completed_requested_horizon" or summary["source_changed_during_run"]:
        raise ValueError("A complete source-frozen trajectory is required")
    steps, dofs = summary["completed_steps"], summary["history_dofs"][0]
    if args.cache_all_reference:
        levels = int(steps).bit_length()
        largest = 1 << (levels - 1)
        cached = ((1 << levels) - 1 + levels) * dofs**2 * 16
        padding = 2 * largest * dofs**2 * 8 if 2 * largest > steps + 1 else 0
        kernel_peak = (steps + 1) * dofs**2 * 8 + cached + padding
    else:
        kernel_peak = plan_cq(
            dofs,
            summary["dt_s"],
            steps,
            max_memory_mb=65536,
            disk_spectrum=options.get("spectrum_directory") is not None,
        ).estimated_peak_bytes
    estimate = kernel_peak + 40 * (steps + 1) * dofs * 8 + 1024**3
    if estimate > args.max_memory_mb * 1024**2:
        raise ValueError(f"Shared-FFT forward validator needs about {estimate / 1024**2:.1f} MiB")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    sampling = (
        SamplingOptions(relative_tolerance=options["sampling_rtol"])
        if options["sampling"] == "adaptive"
        else None
    )
    plan = plan_cq(dofs, summary["dt_s"], steps, max_memory_mb=65536)
    cache = (
        Path(options["cache"])
        / f"galerkin-{nodal_cache_key(model, plan, options['quadrature'], sampling)}.npy"
    )
    if not cache.is_file() or not cache.with_suffix(".json").is_file():
        raise ValueError("Matching CQ cache required; this validator does not build kernels")
    history = build_nodal_history(
        model,
        plan.dt,
        steps,
        quadrature=options["quadrature"],
        sampling=sampling,
        cache_directory=options["cache"],
        max_memory_mb=65536,
        spectrum_directory=options.get("spectrum_directory"),
    )
    lift = initialize_nodal_restart(model, quadrature=options["quadrature"])
    reference = NodalInterfaceEquilibrium(model, lift, history.instantaneous)
    system = JaxContactEquilibrium(reference)
    with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
        saved = {
            key: data[key]
            for key in ["time", "contact_jumps", "platen", "plastic_slip", "cumulative_slip"]
        }
    table = np.genfromtxt(folder / "history.csv", delimiter=",", names=True)
    result = SimpleNamespace(**saved, diagnostics=summary, residual_mpa=table["residual_mpa"])
    started = perf_counter()
    forward = forward_gradient(
        system, history, result, reference, fft_rows=None if args.cache_all_reference else 4
    )
    forward_seconds = perf_counter() - started
    reverse = trajectory_adjoint(model.contact_case(), lift, history, result, equilibrium=reference)
    absolute = abs(forward - reverse["gradient"])
    record = dict(
        objective=reverse["objective"],
        parameter_order=reverse["parameter_order"],
        forward_gradient=forward.tolist(),
        reverse_gradient=reverse["gradient"].tolist(),
        absolute_difference=absolute.tolist(),
        relative_difference=(absolute / np.maximum(abs(reverse["gradient"]), 1e-15)).tolist(),
        forward_seconds_including_jit=forward_seconds,
        reverse_seconds_including_jit=reverse["seconds_including_jit"],
        estimated_memory_mib=estimate / 1024**2,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        source_run=str(folder),
        history_fft_rows=None if args.cache_all_reference else 4,
        environment=environment_record(options["case"]),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Fixed bulk/lift/recorded loading and active branches; full forward JVP/CQ vs reverse VJP/CQ; timings may overlap other jobs",
    )
    write_json(args.output / "summary.json", record)
    np.testing.assert_allclose(forward, reverse["gradient"], rtol=1e-7, atol=1e-12)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

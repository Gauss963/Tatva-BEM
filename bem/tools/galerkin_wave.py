"""Independent finite-bar reflection and boundary-work check for Galerkin CQ."""

import json
import os
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from finiteblock_bem.benchmarks import bar_displacement_exact, smooth_step
from finiteblock_bem.cq import BoundaryHistory, convolution_weights, plan_cq
from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.linear import solve_mixed
from finiteblock_bem.material import ElasticMaterial
from finiteblock_bem.mesh import polygon_boundary
from finiteblock_bem.nodal import edge_nodal_weights, edge_nodes

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = []
    output = ROOT / "outputs/galerkin-wave"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for h, dt in [(0.2, 0.04), (0.1, 0.02)]:
        start = perf_counter()
        material = ElasticMaterial(1, 0, 1)
        mesh = polygon_boundary(
            [[0, 0], [1, 0], [1, 0.5], [0, 0.5]], ["bottom", "fixed", "top", "loaded"], h
        )
        steps = round(3.6 / dt)
        plan = plan_cq(mesh.dofs, dt, steps, max_memory_mb=1024)
        prepared = PreparedGalerkinOperators(mesh, material, quadrature=12)
        weights = convolution_weights(
            lambda s: prepared.assemble(s).force_stiffness(),
            plan,
            workers=int(os.environ.get("FDBEM_CORES", "1")),
        )
        time = np.arange(steps + 1) * dt
        force = np.zeros((steps + 1, mesh.dofs))
        loading_weights = edge_nodal_weights(mesh, "loaded")
        force[:, ::2] = smooth_step(time, 0.3)[:, None] * loading_weights[None, :]
        prescribed = {
            int(d): np.zeros(steps + 1)
            for node in edge_nodes(mesh, "fixed")
            for d in [2 * node, 2 * node + 1]
        }
        result = solve_mixed(BoundaryHistory(plan, weights), prescribed, force)
        load = np.average(result.displacement[:, ::2], axis=1, weights=loading_weights)
        top = edge_nodes(mesh, "top")
        top = top[np.argsort(mesh.starts[top, 0])]
        probe_x = 0.5
        probe_u = np.array(
            [np.interp(probe_x, mesh.starts[top, 0], u[2 * top]) for u in result.displacement]
        )
        exact = bar_displacement_exact(time, 0.0)
        exact_probe = bar_displacement_exact(time, probe_x)
        # Force is conjugate to displacement. BDF2 velocity is consistent with
        # the same CQ symbol; cumulative bulk work is a diagnostic, not a full
        # nonlinear contact energy proof.
        u = result.displacement
        velocity = np.zeros_like(u)
        velocity[1] = (1.5 * u[1] - 2 * u[0]) / dt
        velocity[2:] = (1.5 * u[2:] - 2 * u[1:-1] + 0.5 * u[:-2]) / dt
        work = np.cumsum(np.sum(result.traction * velocity, axis=1)) * dt
        record = {
            "h": h,
            "dt": dt,
            "steps": steps,
            "panels": mesh.count,
            "loading_relative_l2": float(np.linalg.norm(load - exact) / np.linalg.norm(exact)),
            "probe_relative_l2": float(
                np.linalg.norm(probe_u - exact_probe) / np.linalg.norm(exact_probe)
            ),
            "probe_x": probe_x,
            "max_boundary_force_residual": result.max_boundary_residual,
            "min_cumulative_bulk_work": float(np.min(work)),
            "final_cumulative_bulk_work": float(work[-1]),
            "seconds": perf_counter() - start,
        }
        records.append(record)
        np.savez(
            output / f"h{h}-dt{dt}.npz",
            time=time,
            loading=load,
            loading_exact=exact,
            probe=probe_u,
            probe_exact=exact_probe,
            bulk_work=work,
            boundary_displacement=u,
        )
        axes[0].plot(time, load, label=f"Galerkin h={h}, dt={dt}")
        axes[1].plot(time, probe_u, label=f"Galerkin probe x={probe_x:g}")
        print(json.dumps(record), flush=True)
    axes[0].plot(time, exact, "k--", label="analytic loading face")
    axes[1].plot(time, exact_probe, "k--", label="analytic probe")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.set_ylabel("Displacement")
    axes[1].set_xlabel("Time")
    fig.suptitle("Independent finite-bar arrival and fixed-end reflection: Galerkin CQ")
    fig.savefig(output / "wave.png", dpi=160)
    plt.close(fig)
    write_json(
        output / "summary.json",
        {
            "records": records,
            "environment": environment_record(),
            "scope": "Linear analytic wave/reflection benchmark; no event-16 rupture claim",
        },
    )


if __name__ == "__main__":
    main()

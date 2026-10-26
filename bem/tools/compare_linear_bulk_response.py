"""Compare FEM/BEM closed-stick frequency responses before CQ time integration.

Both bodies retain their own spatial discretizations, supports, original
penalty contact, rigid platen and spring. FEM uses its original lumped mass.
The actuator perturbation is a linear tangent, not a nonlinear event replay.
"""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
STATIC = runpy.run_path(str(TOOLS / "fem_static_onset.py"))


def support_transform(total, fixed, face):
    """Eliminate fixed coordinates and tie every platen coordinate together."""
    fixed, face = (np.asarray(a, dtype=int) for a in (fixed, face))
    if len(face) == 0 or np.intersect1d(fixed, face).size:
        raise ValueError("A nonempty free rigid face is required")
    if np.any(np.r_[fixed, face] < 0) or np.any(np.r_[fixed, face] >= total):
        raise ValueError("Constraint coordinate out of range")
    free = np.setdiff1d(np.arange(total), np.r_[fixed, face])
    column = np.full(total, -1, dtype=int)
    column[free] = np.arange(len(free))
    column[face] = len(free)
    rows = np.flatnonzero(column >= 0)
    return sps.csr_matrix((np.ones(len(rows)), (rows, column[rows])), shape=(total, len(free) + 1))


def contact_jump(total, master, slave, offset):
    master, slave = (np.asarray(a, dtype=int) for a in (master, slave))
    if master.shape != slave.shape or master.ndim != 1 or len(master) == 0:
        raise ValueError("Contact node pairs must align")
    n = len(master)
    rows = np.tile(np.arange(2 * n), 2)
    cols = np.r_[2 * master, 2 * master + 1, offset + 2 * slave, offset + 2 * slave + 1]
    return sps.csr_matrix(
        (np.r_[np.ones(2 * n), -np.ones(2 * n)], (rows, cols)), shape=(2 * n, total)
    )


def profile_relative_error(y, actual, reference_y, reference):
    """Complex-amplitude P1 L2 error, integrated exactly on common knots."""
    y, actual, reference_y, reference = (
        np.asarray(value) for value in (y, actual, reference_y, reference)
    )
    for nodes, values in [(y, actual), (reference_y, reference)]:
        if (
            nodes.ndim != 1
            or len(nodes) < 2
            or values.shape != nodes.shape
            or not np.all(np.isfinite(nodes))
            or not np.all(np.isfinite(values))
            or np.any(np.diff(nodes) <= 0)
        ):
            raise ValueError("Require finite aligned responses on increasing interface nodes")
    if not np.allclose([y[0], y[-1]], [reference_y[0], reference_y[-1]], rtol=0, atol=1e-9):
        raise ValueError("Response profiles must cover the same interval")
    knots = np.union1d(y, reference_y)
    actual = np.interp(knots, y, actual)
    reference = np.interp(knots, reference_y, reference)

    def norm2(values):
        return np.sum(
            np.diff(knots)
            * (
                abs(values[:-1]) ** 2
                + np.real(values[:-1].conj() * values[1:])
                + abs(values[1:]) ** 2
            )
            / 3
        )

    return float(np.sqrt(norm2(actual - reference) / norm2(reference)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Prepared BEM kernel supplying its numerical mesh"
    )
    parser.add_argument("--fem-mesh", type=float, default=2)
    parser.add_argument("--frequencies", type=float, nargs="+", default=[0, 1000, 10000])
    parser.add_argument("--laplace-real", type=float, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new response-comparison directory")
    if (
        not np.isfinite(args.fem_mesh)
        or args.fem_mesh <= 0
        or not np.isfinite(args.laplace_real)
        or args.laplace_real <= 0
        or any(not np.isfinite(f) or f < 0 for f in args.frequencies)
    ):
        raise ValueError("Require positive mesh/real part and nonnegative frequencies")
    begun = perf_counter()
    inputs = STATIC["static_inputs"]()
    metadata_path = args.source / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    options = metadata["source_mesh_arguments"]
    frozen_paths = [
        metadata_path,
        Path(__file__),
        TOOLS / "fem_static_onset.py",
        TOOLS / "run_fem_reference.py",
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in frozen_paths}
    frozen.update(inputs["input_sha256"])
    environment = environment_record(options["case"])
    if any(
        environment[k] != metadata["environment"][k]
        for k in ["source_sha256", "case_sha256", "restart_data_sha256"]
    ):
        raise ValueError("BEM kernel package/case differs from the current source")
    if inputs["solver"]["dtype"] != "float64":
        raise ValueError("Require the original float64 FEM model")
    os.environ["JAX_ENABLE_X64"] = "true"
    sys.dont_write_bytecode = True
    snapshot = ROOT / "outputs/fem-source" / STATIC["REVISION"]
    archived = STATIC["REFERENCE_TOOL"]["stage_snapshot"](ROOT.parent, snapshot)
    sys.path.insert(0, str(snapshot))
    from dataclasses import replace

    from tatva.pmma import static_normal
    from tatva.pmma.dynamics import RunConfig, build_case_model
    from tatva.pmma.model import (
        BlockSpec,
        FrictionReference,
        LoadingReference,
        Material,
        PMMAModelInput,
    )

    legacy = inputs["case"]
    case = PMMAModelInput(
        BlockSpec(**legacy["moving"]),
        BlockSpec(**legacy["stationary"]),
        {k: Material(**v) for k, v in legacy["materials"].items()},
        FrictionReference(**legacy["friction"]),
        LoadingReference(**legacy["simulation"]),
    )
    config = replace(
        RunConfig(**inputs["solver"]),
        mesh_size=args.fem_mesh,
        restart_from_file=None,
        time_step_override=None,
    )
    model = build_case_model(case, config)
    if not model["shear_spring_rigid_face"] or model["shear_loading_mode"] != "spring-displacement":
        raise ValueError("Require the original rigid platen spring")
    moving, stationary = model["moving"], model["stationary"]
    offset = 2 * moving.n_nodes
    total = offset + 2 * stationary.n_nodes
    p = support_transform(total, model["fixed_dofs"], model["moving_shear_loading_dofs"])
    jump = contact_jump(total, model["master_nodes"], model["slave_nodes"], offset) @ p
    weights = np.asarray(model["interface_weights"], dtype=float)
    cdiag = np.r_[weights * float(model["penalty_n"]), weights * np.asarray(model["penalty_t"])]
    stiffness = sps.block_diag(
        [
            static_normal._block_stiffness(moving, model["moving_material"]),
            static_normal._block_stiffness(stationary, model["stationary_material"]),
        ],
        format="csr",
    )
    reduced = (p.T @ stiffness @ p + jump.T @ sps.diags(cdiag) @ jump).tocsc()
    spring = float(model["shear_loading_stiffness"]) * float(np.sum(model["force_shear_unit"]))
    reduced += sps.csc_matrix(([spring], ([p.shape[1] - 1], [p.shape[1] - 1])), shape=reduced.shape)
    mass = (p.T @ sps.diags(np.asarray(model["mass_flat"], dtype=float)) @ p).tocsc()
    rhs = np.zeros(p.shape[1])
    rhs[-1] = spring
    fy = np.asarray(moving.mesh.coords)[np.asarray(model["master_nodes"]), 1]
    bem = load_nodal_model(
        options["case"],
        **{
            out: options.get(key)
            for out, key in [
                ("max_panel_length", "h"),
                ("contact_panel_length", "contact_h"),
                ("contact_end_length", "contact_end_h"),
                ("corner_panel_length", "corner_h"),
                ("restart_knot_window", "restart_knot_window"),
                ("restart_knot_subdivisions", "restart_knot_subdivisions"),
            ]
        },
    )
    transfer = NodalInterfaceTransfer(bem, quadrature=options["quadrature"])
    by = bem.reference.moving.starts[bem.moving_contact, 1]
    n = len(by)
    law = bem.reference.contact
    bw = bem.constraints.contact_weights
    bspring = bem.reference.spring_stiffness * bem.constraints.platen_width
    np.testing.assert_allclose(bspring, spring, rtol=1e-13, atol=1e-10)
    bdiag = np.r_[bw * law.normal_penalty, bw * law.tangential_penalty, bspring]
    brhs = np.zeros(transfer.dofs)
    brhs[-1] = bspring
    args.output.mkdir(parents=True)
    records = []
    preparation = perf_counter() - begun
    for index, frequency in enumerate(args.frequencies):
        start = perf_counter()
        s = complex(args.laplace_real, 2 * np.pi * frequency)
        matrix = reduced + s * s * mass
        fu = spla.spsolve(matrix, rhs)
        fem_residual = float(np.linalg.norm(matrix @ fu - rhs) / np.linalg.norm(rhs))
        fj = np.asarray(jump @ fu)
        fp, ft = np.split(fj, 2)
        fp, ft = fp * float(model["penalty_n"]), ft * np.asarray(model["penalty_t"])
        z = transfer.evaluate(s)
        bmatrix = z + np.diag(bdiag)
        bu = np.linalg.solve(bmatrix, brhs)
        bem_residual = float(np.linalg.norm(bmatrix @ bu - brhs) / np.linalg.norm(brhs))
        if (
            not np.all(np.isfinite([fem_residual, bem_residual]))
            or max(fem_residual, bem_residual) > 1e-8
        ):
            raise ValueError("Linear response failed its independent equation residual")
        bp, bt = law.normal_penalty * bu[:n], law.tangential_penalty * bu[n : 2 * n]
        records.append(
            dict(
                frequency_hz=frequency,
                laplace_real_per_s=args.laplace_real,
                pressure_profile_relative_l2=profile_relative_error(by, bp, fy, fp),
                shear_profile_relative_l2=profile_relative_error(by, bt, fy, ft),
                platen_relative_error=float(abs(bu[-1] - fu[-1]) / abs(fu[-1])),
                fem_equation_relative_residual=fem_residual,
                bem_equation_relative_residual=bem_residual,
                seconds=perf_counter() - start,
            )
        )
        np.savez(
            args.output / f"response-{index}.npz",
            fem_y=fy,
            bem_y=by,
            fem_pressure=fp,
            fem_shear=ft,
            bem_pressure=bp,
            bem_shear=bt,
            fem_platen=fu[-1],
            bem_platen=bu[-1],
        )
        write_json(args.output / "progress.json", dict(records=records))
    if any(fingerprint(p) != digest for p, digest in frozen.items()):
        raise ValueError("Source code or authored input changed during response comparison")
    write_json(
        args.output / "summary.json",
        dict(
            status="completed_linear_stick_response_comparison",
            records=records,
            fem_mesh_mm=args.fem_mesh,
            fem_full_dofs=total,
            fem_reduced_dofs=p.shape[1],
            fem_contact_nodes=len(fy),
            bem_contact_nodes=n,
            fem_source=archived,
            frozen_input_sha256=frozen,
            environment=environment,
            cores_requested=int(os.environ.get("FDBEM_CORES", "1")),
            hard_cpu_affinity_enforced=False,
            timing_scope="Exploratory diagnostic timings; not a controlled forward-simulation or FEM speed benchmark",
            preparation_seconds=preparation,
            total_seconds_before_reporting=perf_counter() - begun,
            scope="Independent spatially discrete, continuous-time closed-stick actuator response; no CQ, slip weakening, branch/event sensitivity or FEM-equivalent nonlinear solution asserted",
            original_tatva_readonly=True,
            same_accuracy_fem_advantage_verified=False,
        ),
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

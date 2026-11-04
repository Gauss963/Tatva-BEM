"""Separate spatial bulk/contact errors from CQ damping and root selection.

Compare continuous-time actuator responses of closed-stick and closed,
positive fully weakened sliding tangents. The sliding tangent retains
normal-to-shear coupling and is nonsymmetric. It is a frozen control, not
the changing active set or a stability certificate for the nonlinear event.
"""

import argparse
import hashlib
import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
BULK = runpy.run_path(str(TOOLS / "compare_linear_bulk_response.py"))
STATIC = BULK["STATIC"]
RAW = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))


def correction_matrix(nodes, kn, kt, mu):
    """K_slide - K_stick in sqrt(line-weight) displacement coordinates."""
    correction = np.zeros((2 * nodes + 1, 2 * nodes + 1))
    index = np.arange(nodes)
    correction[nodes + index, index] = mu * kn
    correction[nodes + index, nodes + index] = -kt
    return correction


def resolvent_response(resolvent, correction, rhs):
    """Solve (L^-1 + correction)x=rhs without explicitly inverting L."""
    matrix = np.eye(len(rhs)) + resolvent @ correction
    load = resolvent @ rhs
    response = np.linalg.solve(matrix, load)
    residual = float(np.linalg.norm(matrix @ response - load) / np.linalg.norm(load))
    if not np.all(np.isfinite(response)) or not np.isfinite(residual) or residual > 1e-8:
        raise ValueError("Cached resolvent response failed its independent residual")
    return response, residual


def audit_local_tangents(law):
    """Check just the independent 2x2 maps with AD and the NumPy return law."""
    import jax
    import jax.numpy as jnp

    from finiteblock_bem.friction import ContactState
    from finiteblock_bem.jax_backend import contact_point

    kn, kt, mu = law.normal_penalty, law.tangential_penalty, law.kinetic_friction
    parameters = jnp.array([kn, kt, law.static_friction, mu, law.characteristic_slip])
    cumulative = 2 * law.characteristic_slip
    records = {}
    for kind, shear_jump, expected in [
        ("closed_stick", 0.0, [[kn, 0], [0, kt]]),
        ("positive_kinetic_sliding", 32 * mu / kt, [[kn, 0], [mu * kn, 0]]),
    ]:
        jump = jnp.array([16 / kn, shear_jump])
        actual = np.asarray(jax.jacfwd(contact_point)(jump, 0.0, cumulative, parameters))[:2]
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)
        response = law.evaluate(
            np.array([jump[0]]),
            np.array([jump[1]]),
            ContactState(np.zeros(1), np.array([cumulative])),
        )
        reference = np.array(
            [
                response.pressure[0],
                response.shear[0],
                response.state.plastic_slip[0],
                response.state.cumulative_slip[0],
                response.friction_coefficient[0],
            ]
        )
        value = np.asarray(contact_point(jump, 0.0, cumulative, parameters))
        np.testing.assert_allclose(value, reference, rtol=1e-13, atol=1e-12)
        records[kind] = dict(
            tangent=actual.tolist(),
            tangent_max_absolute_error=float(np.max(abs(actual - expected))),
            numpy_value_max_absolute_error=float(np.max(abs(value - reference))),
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode-indices", nargs="+", type=int, default=[0, 131, 1311])
    parser.add_argument("--branch", type=int, choices=[0, 1], default=0)
    parser.add_argument("--fresh-check-index", type=int, default=0)
    parser.add_argument("--fem-mesh", type=float, default=2)
    parser.add_argument("--fem-end-size", type=float, default=0.015625)
    parser.add_argument("--fem-grading-growth", type=float, default=1.2)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh response directory outside protected Tatva")
    if (
        len(set(args.mode_indices)) != len(args.mode_indices)
        or args.fresh_check_index not in args.mode_indices
        or not np.isfinite(args.fem_mesh)
        or not np.isfinite(args.fem_end_size)
        or not 0 < args.fem_end_size < args.fem_mesh
        or not np.isfinite(args.fem_grading_growth)
        or args.fem_grading_growth <= 1
    ):
        raise ValueError("Require distinct modes, an included fresh check and valid mesh grading")
    begun = perf_counter()
    metadata_path = args.kernel.resolve() / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    options, plan = metadata["source_mesh_arguments"], metadata["plan"]
    if (
        metadata["status"] != "completed_compact_radau_resolvent_kernel_only"
        or metadata["family_spectra_coordinates"] != "diag(sqrt(w)) * L * diag(sqrt(w))"
        or metadata["time_method"] != "radau2"
        or any(not 0 <= i <= plan["fft_size"] // 2 for i in args.mode_indices)
    ):
        raise ValueError("Require complete compact work-coordinate Radau spectra")
    environment = environment_record(options["case"])
    environment_keys = ["source_sha256", "case_sha256", "restart_data_sha256"]
    if any(environment[k] != metadata["environment"][k] for k in environment_keys):
        raise ValueError("Kernel package, case or restart source changed")
    inputs = STATIC["static_inputs"]()
    if inputs["solver"]["dtype"] != "float64":
        raise ValueError("Require authored float64 FEM input")
    paths = [
        Path(__file__),
        metadata_path,
        *[
            TOOLS / name
            for name in [
                "compare_linear_bulk_response.py",
                "fem_static_onset.py",
                "run_fem_reference.py",
                "chamfer_mesh_refinement.py",
                "radau_raw_sampling.py",
            ]
        ],
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    frozen.update(inputs["input_sha256"])
    storage = metadata["family_spectra"]
    if len(storage) != 2 or not all(x["all_frequency_matrices_written"] for x in storage):
        raise ValueError("Require both completely written source families")
    source_paths = [Path(x["path"]) for x in storage]
    source_stats = [(p.stat().st_size, p.stat().st_mtime_ns) for p in source_paths]
    spectra = [np.load(p, mmap_mode="r", allow_pickle=False) for p in source_paths]
    expected = (plan["fft_size"] // 2 + 1, plan["dofs"], plan["dofs"])
    if any(a.shape != expected or a.dtype != np.complex128 for a in spectra):
        raise ValueError("Source family dimensions/precision differ from their plan")
    output.mkdir(parents=True)
    write_json(
        output / "manifest.json",
        dict(
            frozen_inputs_sha256=frozen,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        ),
    )
    snapshot = ROOT / "outputs/fem-source" / STATIC["REVISION"]
    archived = STATIC["REFERENCE_TOOL"]["stage_snapshot"](ROOT.parent, snapshot)
    sys.path.insert(0, str(snapshot))
    import jax
    from tatva.pmma import dynamics, static_normal
    from tatva.pmma.dynamics import RunConfig
    from tatva.pmma.model import (
        BlockSpec,
        FrictionReference,
        LoadingReference,
        Material,
        PMMAModelInput,
    )

    jax.config.update("jax_enable_x64", True)
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
        fault_end_refinement_size=args.fem_end_size,
        fault_end_refinement_growth=args.fem_grading_growth,
    )
    builder = runpy.run_path(str(TOOLS / "chamfer_mesh_refinement.py"))[
        "make_chamfer_refined_builder"
    ](dynamics)
    model = builder(case, config)
    if not model["shear_spring_rigid_face"] or model["shear_loading_mode"] != "spring-displacement":
        raise ValueError("Require the authored rigid-platen spring")
    moving, stationary = model["moving"], model["stationary"]
    offset, total = 2 * moving.n_nodes, 2 * (moving.n_nodes + stationary.n_nodes)
    transform = BULK["support_transform"](
        total, model["fixed_dofs"], model["moving_shear_loading_dofs"]
    )
    jump = (
        BULK["contact_jump"](total, model["master_nodes"], model["slave_nodes"], offset) @ transform
    )
    fy = np.asarray(moving.mesh.coords)[np.asarray(model["master_nodes"]), 1]
    fw = np.asarray(model["interface_weights"], dtype=float)
    bulk = sparse.block_diag(
        [
            static_normal._block_stiffness(moving, model["moving_material"]),
            static_normal._block_stiffness(stationary, model["stationary_material"]),
        ],
        format="csr",
    )
    bulk = (transform.T @ bulk @ transform).tocsc()
    mass = (transform.T @ sparse.diags(np.asarray(model["mass_flat"])) @ transform).tocsc()
    spring = float(model["shear_loading_stiffness"]) * float(np.sum(model["force_shear_unit"]))
    bulk[-1, -1] += spring
    frhs = np.zeros(transform.shape[1])
    frhs[-1] = spring
    bem = load_nodal_model(
        options["case"],
        **{
            name: options.get(key)
            for name, key in [
                ("max_panel_length", "h"),
                ("contact_panel_length", "contact_h"),
                ("contact_end_length", "contact_end_h"),
                ("corner_panel_length", "corner_h"),
                ("restart_knot_window", "restart_knot_window"),
                ("restart_knot_subdivisions", "restart_knot_subdivisions"),
            ]
        },
    )
    by = bem.reference.moving.starts[bem.moving_contact, 1]
    n, nf = len(by), len(fy)
    law = bem.reference.contact
    kn, kt, mu = law.normal_penalty, law.tangential_penalty, law.kinetic_friction
    local_tangent_checks = audit_local_tangents(law)
    if plan["dofs"] != 2 * n + 1:
        raise ValueError("Cached coordinates must be normal jump, tangent jump and platen")
    np.testing.assert_allclose(model["penalty_n"], kn, rtol=1e-13, atol=0)
    np.testing.assert_allclose(np.asarray(model["penalty_t"]), kt, rtol=1e-13, atol=0)
    np.testing.assert_allclose(model["mu_k"], mu, rtol=1e-13, atol=0)
    line = np.r_[
        bem.constraints.contact_weights,
        bem.constraints.contact_weights,
        bem.constraints.platen_width,
    ]
    scale = np.sqrt(line)
    np.testing.assert_allclose(
        spring, bem.reference.spring_stiffness * line[-1], rtol=1e-13, atol=1e-10
    )
    brhs = np.zeros(plan["dofs"])
    brhs[-1] = spring
    work_rhs = brhs / scale
    slide_correction = correction_matrix(n, kn, kt, mu)
    contact_matrices = {
        "closed_stick": sparse.diags(np.r_[fw * kn, fw * kt]),
        "positive_kinetic_sliding": sparse.bmat(
            [
                [sparse.diags(fw * kn), None],
                [sparse.diags(fw * mu * kn), sparse.csr_matrix((nf, nf))],
            ],
            format="csr",
        ),
    }
    fem_stiffness = {
        key: (bulk + jump.T @ value @ jump).tocsc() for key, value in contact_matrices.items()
    }
    transfer, records, fresh_checks = None, [], []
    write_json(output / "progress.json", dict(phase="spatial_models_ready", fem_dofs=total))
    for index in args.mode_indices:
        zeta = plan["radius"] * np.exp(-2j * np.pi * index / plan["fft_size"])
        mode = RAW["lower_half_radau_modes"](zeta, 1.0)[args.branch]
        s = mode / plan["dt"]
        first, second = [np.array(a[index]) for a in spectra]
        resolvent = first + mode * second
        sample_sha = [
            hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in [first, second]
        ]
        fresh_z = None
        if index == args.fresh_check_index:
            transfer = NodalInterfaceTransfer(bem, quadrature=options["quadrature"])
            fresh_z = transfer.evaluate(s)
        for kind, correction in [
            ("closed_stick", np.zeros_like(slide_correction)),
            ("positive_kinetic_sliding", slide_correction),
        ]:
            start = perf_counter()
            matrix = fem_stiffness[kind] + s * s * mass
            fu = spsolve(matrix, frhs)
            fres = float(np.linalg.norm(matrix @ fu - frhs) / np.linalg.norm(frhs))
            bu_work, bres = resolvent_response(resolvent, correction, work_rhs)
            bu = bu_work / scale
            if not np.all(np.isfinite(fu)) or not np.isfinite(fres) or fres > 1e-8:
                raise ValueError("Original FEM spatial response failed its equation residual")
            if fresh_z is not None:
                ref = np.diag(np.r_[line[:n] * kn, line[n : 2 * n] * kt, spring])
                fresh_matrix = fresh_z + ref + scale[:, None] * correction * scale[None, :]
                fresh = np.linalg.solve(fresh_matrix, brhs)
                fresh_res = float(
                    np.linalg.norm(fresh_matrix @ fresh - brhs) / np.linalg.norm(brhs)
                )
                error = float(np.linalg.norm(bu - fresh) / np.linalg.norm(fresh))
                if not np.all(np.isfinite([fresh_res, error])) or fresh_res > 1e-8 or error > 1e-6:
                    raise ValueError(
                        "Cached coordinate/tangent response disagrees with fresh bulk assembly"
                    )
                fresh_checks.append(
                    dict(
                        index=index,
                        kind=kind,
                        relative_response_error=error,
                        fresh_relative_residual=fresh_res,
                    )
                )
            fj = np.asarray(jump @ fu)
            fp, ft = kn * fj[:nf], kt * fj[nf:]
            bp, bt = kn * bu[:n], kt * bu[n : 2 * n]
            if kind == "positive_kinetic_sliding":
                ft, bt = mu * fp, mu * bp
            record = dict(
                index=index,
                branch=args.branch,
                kind=kind,
                s_real_per_s=float(s.real),
                s_imag_per_s=float(s.imag),
                frequency_hz=float(s.imag / (2 * np.pi)),
                pressure_profile_relative_p1_l2=BULK["profile_relative_error"](by, bp, fy, fp),
                signed_shear_profile_relative_p1_l2=BULK["profile_relative_error"](by, bt, fy, ft),
                normal_jump_profile_relative_p1_l2=BULK["profile_relative_error"](
                    by, bu[:n], fy, fj[:nf]
                ),
                tangent_jump_profile_relative_p1_l2=BULK["profile_relative_error"](
                    by, bu[n : 2 * n], fy, fj[nf:]
                ),
                platen_relative_error=float(abs(bu[-1] - fu[-1]) / abs(fu[-1])),
                fem_equation_relative_residual=fres,
                bem_resolvent_relative_residual=bres,
                source_mode_bytes_sha256=sample_sha,
                seconds=perf_counter() - start,
            )
            records.append(record)
            np.savez(
                output / f"response-{index}-{kind}.npz",
                bem_y=by,
                fem_y=fy,
                bem_pressure=bp,
                fem_pressure=fp,
                bem_shear=bt,
                fem_shear=ft,
                bem_jump=bu[: 2 * n],
                fem_jump=fj,
                bem_platen=bu[-1],
                fem_platen=fu[-1],
            )
            write_json(
                output / "progress.json",
                dict(
                    phase="evaluating_spatial_responses", records=records, fresh_checks=fresh_checks
                ),
            )
    if any(fingerprint(p) != h for p, h in frozen.items()):
        raise ValueError("Authored inputs or response source changed")
    if source_stats != [(p.stat().st_size, p.stat().st_mtime_ns) for p in source_paths]:
        raise ValueError("Source spectrum changed while sampling modes")
    current = environment_record(options["case"])
    if any(current[k] != environment[k] for k in environment_keys):
        raise ValueError("Package or case changed during diagnostic")
    STATIC["REFERENCE_TOOL"]["verify_snapshot"](snapshot, archived["files"])
    write_json(
        output / "summary.json",
        dict(
            status="completed_frozen_stick_and_positive_sliding_response_comparison",
            records=records,
            fresh_checks=fresh_checks,
            local_contact_tangent_checks=local_tangent_checks,
            fem_full_dofs=total,
            fem_reduced_dofs=transform.shape[1],
            fem_contact_nodes=nf,
            bem_contact_nodes=n,
            fem_mesh_mm=args.fem_mesh,
            fem_end_size_mm=args.fem_end_size,
            frozen_inputs_sha256=frozen,
            environment=environment,
            source_expected_spectrum_records=storage,
            whole_spectrum_sha_not_recomputed_here=True,
            source_spectrum_size_and_mtime_unchanged=True,
            fresh_response_relative_tolerance=1e-6,
            equation_relative_tolerance=1e-8,
            all_contact_endpoints_retained=True,
            total_seconds_before_reporting=perf_counter() - begun,
            scope="Continuous-time spatial controls at saved actual Laplace modes, one independently reassembled mode; no CQ integration, event root selection, changing active set or nonlinear stability certificate. Whole-family SHA not recomputed in this diagnostic.",
            full_original_event_reproduced=False,
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
        ),
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

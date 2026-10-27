"""Numerically refine a pinned FEM static root using its original split forces.

The original solver chooses the healed Coulomb active set. Reuse its exact
elastic assembly, then correct its force defect without changing that set,
penalties, source files, model bindings or acceptance tolerance.
"""

import runpy
from pathlib import Path

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu

TOOLS = Path(__file__).resolve().parent


def correct_fixed_set(matrix, initial, residual, admissible, denominator, *, target=1e-9, steps=4):
    """Correct the measured split residual, preserving a prescribed contact set."""
    if not np.isfinite(denominator) or denominator <= 0 or not 0 < target < 1e-8 or steps < 0:
        raise ValueError("Require a positive load norm, stricter target and bounded steps")
    q = np.asarray(initial, dtype=float).copy()
    records, factor = [], None
    for iteration in range(steps + 1):
        force = np.asarray(residual(q))
        error = float(np.linalg.norm(force) / denominator)
        if not np.all(np.isfinite(q)) or not np.isfinite(error) or not admissible(q):
            raise ValueError("Static defect correction changed the original contact active set")
        records.append(dict(correction=iteration, original_relative_force_residual=error))
        if error < target:
            return q, records
        if iteration < steps:
            if factor is None:
                factor = splu(matrix.tocsc())
            q -= factor.solve(force)
    raise ValueError(f"Static defect correction did not reach its stricter target: {records}")


def solve_refined_static(module, model, config, *, actuator_displacement, plastic_slip):
    """Original static solve plus explicit fixed-active-set numerical correction."""
    if (
        config.dimension != 2
        or config.lock_shear_edge_during_normal
        or model["normal_loading_mode"] != "stress"
        or model["shear_loading_mode"] != "spring-displacement"
        or not model["shear_spring_rigid_face"]
    ):
        raise ValueError("Require the original 2D stress load and free rigid spring platen")
    adapter = runpy.run_path(str(TOOLS / "chamfer_mesh_refinement.py"))
    linear = runpy.run_path(str(TOOLS / "compare_linear_bulk_response.py"))
    original_solver, original_assembly = module.solve_static_normal_phase, module._block_stiffness
    assembled = {}

    def assembly(block, material):
        key = (id(block), id(material))
        if key not in assembled:
            assembled[key] = original_assembly(block, material)
        return assembled[key]

    # Only cache assembly while calling the exact original solver; do not
    # replace its constitutive/contact code or any source module binding.
    solver = adapter["rebind"](original_solver, {"_block_stiffness": assembly})
    answer = solver(
        model, config, actuator_displacement=actuator_displacement, plastic_slip=plastic_slip
    )
    moving, stationary = model["moving"], model["stationary"]
    offset, total = 2 * moving.n_nodes, 2 * (moving.n_nodes + stationary.n_nodes)
    k = sps.block_diag(
        [
            assembly(moving, model["moving_material"]),
            assembly(stationary, model["stationary_material"]),
        ],
        format="csr",
    )
    face = np.asarray(model["moving_shear_loading_dofs"], dtype=int)
    master, slave = (np.asarray(model[key], dtype=int) for key in ["master_nodes", "slave_nodes"])
    mx, my, sx, sy = 2 * master, 2 * master + 1, offset + 2 * slave, offset + 2 * slave + 1
    p = linear["support_transform"](total, model["fixed_dofs"], face)
    weights, gap = (
        np.asarray(model[key]) for key in ["interface_weights", "interface_initial_gap"]
    )
    kn, kt = float(model["penalty_n"]), np.asarray(model["penalty_t"])
    mu = np.asarray(model["mu_k_profile"]) + float(model.get("healing_fraction", 1)) * (
        np.asarray(model["mu_s_profile"]) - np.asarray(model["mu_k_profile"])
    )
    seed = np.asarray(plastic_slip)
    if seed.shape != master.shape or not np.all(np.isfinite(seed)):
        raise ValueError("Require the same finite original P1 restart offset")

    def contact_set(u):
        pressure = kn * (u[mx] - u[sx] - gap)
        trial = kt * (u[my] - u[sy] - seed)
        active = pressure > 0
        sliding = active & (abs(trial) > mu * pressure)
        return active, sliding, np.where(sliding, np.sign(trial), 0), pressure, trial

    active, sliding, direction, _, _ = contact_set(answer["u"])
    if sliding.sum() != answer["sliding_nodes"] or (~active).sum() != answer["open_nodes"]:
        raise ValueError("Reconstructed contact set differs from the original solver")
    normal = weights * kn * active
    tangential = weights * kt * (active & ~sliding)
    slide = weights * mu * kn * direction * (active & sliding)
    rows, cols, vals = [], [], []
    for aa, bb, vv in [
        (mx, mx, normal),
        (mx, sx, -normal),
        (sx, mx, -normal),
        (sx, sx, normal),
        (my, my, tangential),
        (my, sy, -tangential),
        (sy, my, -tangential),
        (sy, sy, tangential),
        (my, mx, slide),
        (my, sx, -slide),
        (sy, mx, -slide),
        (sy, sx, slide),
    ]:
        rows.append(aa)
        cols.append(bb)
        vals.append(vv)
    contact = sps.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(total, total)
    )
    spring = float(model["shear_loading_stiffness"]) * float(np.sum(model["force_shear_unit"]))
    force = np.asarray(model["force_normal"])
    rhs = force.copy()
    rhs[face] += spring * actuator_displacement / len(face)
    for index, value in [
        (mx, normal * gap),
        (sx, -normal * gap),
        (my, slide * gap),
        (sy, -slide * gap),
        (my, tangential * seed),
        (sy, -tangential * seed),
    ]:
        np.add.at(rhs, index, value)
    matrix = (p.T @ (k + contact) @ p).tocsc()
    matrix += sps.csc_matrix(([spring], ([p.shape[1] - 1], [p.shape[1] - 1])), shape=matrix.shape)
    columns = p.tocsc()
    # Read one representative coordinate per column. Averaging already tied
    # face values can change their last bit and corrupt this roundoff audit.
    q = np.asarray(answer["u"])[columns.indices[columns.indptr[:-1]]].copy()
    if not np.array_equal(p @ q, answer["u"]):
        raise ValueError("Original static displacement violates its support/platen constraints")
    denominator = max(np.linalg.norm(p.T @ force), 1e-30)

    def residual(q):
        u = p @ q
        r = k @ u + contact @ u - rhs
        r[face] += spring * u[face[0]] / len(face)
        return np.asarray(p.T @ r)

    def admissible(q):
        current = contact_set(p @ q)
        return all(np.array_equal(a, b) for a, b in zip(current[:3], (active, sliding, direction)))

    initial_error = float(np.linalg.norm(residual(q)) / denominator)
    if abs(initial_error - answer["relative_residual"]) > 1e-11:
        raise ValueError(
            f"Reconstructed split-force residual {initial_error:.17g} differs from "
            f"the original equation {answer['relative_residual']:.17g}"
        )
    q, records = correct_fixed_set(matrix, q, residual, admissible, denominator)
    u = p @ q
    _, _, _, pressure, trial = contact_set(u)
    pressure = np.where(active, pressure, 0)
    shear = np.where(sliding, direction * mu * pressure, trial) * active
    plastic = np.where(sliding, u[my] - u[sy] - shear / kt, seed)
    if (
        module.solve_static_normal_phase is not original_solver
        or module._block_stiffness is not original_assembly
    ):
        raise RuntimeError("Original FEM source bindings changed")
    return {
        **answer,
        "u": u,
        "sigma_n": pressure,
        "tau": shear,
        "plastic_slip": plastic,
        "cumulative_slip": abs(plastic),
        "tau_over_sigma": np.divide(
            abs(shear), pressure, out=np.zeros_like(pressure), where=pressure > 0
        ),
        "relative_residual": records[-1]["original_relative_force_residual"],
        "loading_face_displacement": float(np.mean(u[face])),
        "numerical_defect_correction": dict(
            records=records,
            original_relative_force_residual=answer["relative_residual"],
            maximum_displacement_correction_mm=float(np.max(abs(u - answer["u"]))),
            original_active_set_unchanged=True,
            original_source_bindings_unchanged=True,
            physical_parameters_or_tolerance_changed=False,
        ),
    }

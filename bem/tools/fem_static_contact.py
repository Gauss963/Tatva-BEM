"""Healed Coulomb preload with the same active-set policy as Tatva FEM.

The restart plastic slip is an input offset, not a constraint on sliding
points. Resolve opening/sliding at the healed strength, then retain the
resulting plastic offset and the prescribed healed weakening memory. No
dynamic slip-weakening increment is applied during this static solve.
"""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from finiteblock_bem.friction import ContactState


def solve_healed_static_contact(system, normal, actuator, state, *, max_iterations=100):
    """Return the active-set root, its updated offset and explicit diagnostics."""
    con, law = system.constraints, system.case.contact
    bn, bt = np.asarray(con.normal_jump), np.asarray(con.tangential_jump)
    wc = np.asarray(con.contact_weights)
    kn, kt = law.normal_penalty, law.tangential_penalty
    mu = law.coefficient(state.cumulative_slip)
    active = np.ones(len(wc), dtype=bool)
    sliding, direction = np.zeros(len(wc), dtype=bool), np.zeros(len(wc))
    external = con.reduce_force(system.external_traction(normal) - system.base_t)
    external[-1] += system.spring * (actuator - system.base_platen)
    records = []
    for iteration in range(1, max_iterations + 1):
        normal_tangent = wc * kn * active
        stick_tangent = wc * kt * (active & ~sliding)
        slide_tangent = wc * mu * kn * direction * (active & sliding)
        matrix = np.asarray(system.matrix).copy()
        matrix += bn.T @ (normal_tangent[:, None] * bn)
        matrix += bt.T @ (stick_tangent[:, None] * bt)
        matrix += bt.T @ (slide_tangent[:, None] * bn)
        constant = bn.T @ (normal_tangent * system.base_dn)
        constant += bt.T @ (
            stick_tangent * (system.base_ds - state.plastic_slip) + slide_tangent * system.base_dn
        )
        q = solve(matrix, external - constant, check_finite=False)
        if not np.all(np.isfinite(q)):
            raise ValueError("Nonfinite healed static contact root")
        closure = system.base_dn + bn @ q
        tangent = system.base_ds + bt @ q
        pressure = kn * closure
        trial = kt * (tangent - state.plastic_slip)
        new_active = closure > 0
        new_sliding = new_active & (abs(trial) > mu * pressure)
        new_direction = np.where(new_sliding, np.sign(trial), 0.0)
        records.append(
            dict(iteration=iteration, open=int((~new_active).sum()), sliding=int(new_sliding.sum()))
        )
        if (
            np.array_equal(new_active, active)
            and np.array_equal(new_sliding, sliding)
            and np.array_equal(new_direction, direction)
        ):
            break
        active, sliding, direction = new_active, new_sliding, new_direction
    else:
        raise ValueError(f"Healed static contact active set did not converge: {records[-3:]}")
    shear = np.where(sliding, direction * mu * pressure, trial) * active
    plastic = np.where(sliding, tangent - shear / kt, state.plastic_slip)
    corrected = ContactState(plastic, state.cumulative_slip.copy())
    residual, _, response = system.evaluate(
        q, np.zeros(con.size), normal, actuator, corrected, frozen=True, jacobian=False
    )
    error = float(np.max(abs(residual)))
    excess = float(np.max(abs(response.shear) - mu * response.pressure))
    if not np.isfinite(error) or error >= 1e-7 or excess > 1e-7:
        raise ValueError("Healed static contact violates the original force/strength tolerance")
    return (
        q,
        corrected,
        dict(
            policy="Tatva healed Coulomb active set, then retain prescribed weakening memory",
            active_set_iterations=records,
            static_sliding_points=int(sliding.sum()),
            static_open_points=int((~active).sum()),
            original_force_residual_mpa=error,
            maximum_strength_excess_mpa=excess,
            maximum_plastic_offset_change_mm=float(np.max(abs(plastic - state.plastic_slip))),
            input_seed_unchanged=bool(np.array_equal(plastic, state.plastic_slip)),
            weakening_memory_unchanged=bool(
                np.array_equal(corrected.cumulative_slip, state.cumulative_slip)
            ),
            no_dynamic_weakening_during_preload=True,
        ),
    )


def initialize_fem_static_contact(model, quadrature, reduction, state):
    refined = runpy.run_path(str(Path(__file__).with_name("refined_contact.py")))
    system = refined["RefinedContactEquilibrium"](
        model, quadrature, reduction.impedance, base_force=-reduction.effective_load
    )
    q, corrected, diagnostic = solve_healed_static_contact(
        system, model.reference.normal_pressure, model.reference.restart_actuator, state
    )
    # Recheck the root using the independent frozen-state lift, including
    # uncondensed full-body balance and trace constraints at original tolerances.
    lift = refined["initialize_refined_contact"](model, quadrature, reduction, corrected)
    if np.max(abs(lift.retained_displacement - q)) > 1e-8:
        raise ValueError("Healed active-set and original frozen-state roots differ")
    diagnostics = {
        **lift.diagnostics,
        "static_initialization": diagnostic,
        "seed_unchanged": diagnostic["input_seed_unchanged"],
    }
    return replace(lift, diagnostics=diagnostics)

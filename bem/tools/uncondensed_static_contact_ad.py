"""Implicit selected-set parameter sensitivity of the uncondensed BEM preload.

Differentiate the equilibrium root, not active-set iterations. The body
operator, source offset, normal load and geometry stay fixed. Bulk port force
sensitivities use the same measured-body-force projection as the primal.
"""

import runpy
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq

TOOLS = Path(__file__).resolve().parent


def uncondensed_static_parameter_jacobian(model, quadrature, reduction, input_state, lift):
    direct = runpy.run_path(str(TOOLS / "uncondensed_static_contact.py"))
    selected = runpy.run_path(str(TOOLS / "fem_static_contact_ad.py"))
    system = direct["BodyStaticEquilibrium"](model, quadrature, reduction)
    p = model.constraints.prolongation
    root = np.asarray(lift.full_displacement)[np.argmax(p, axis=0)]
    if not np.array_equal(p @ root, lift.full_displacement):
        raise ValueError("Static lift violates original support/platen constraints")
    derivative = selected["healed_static_parameter_jacobian"](
        system,
        model.reference.normal_pressure,
        model.reference.restart_actuator,
        input_state,
        root,
    )
    du = derivative["displacement_jacobian"]
    dq = reduction.jump @ du
    body_force_derivative = reduction.reduced_stiffness @ du
    bulk = lstsq(reduction.jump.T, body_force_derivative, lapack_driver="gelsy")[0]
    return {
        **derivative,
        "displacement_jacobian": dq,
        "body_displacement_jacobian": du,
        "retained_bulk_force_jacobian": bulk,
        "bulk_force_derivative_source": "Projection of Kr*du; fixed bulk operator and normal load",
        "uncondensed_static_equilibrium_differentiated": True,
        "full_transient_initialization_gradient_connected": False,
    }

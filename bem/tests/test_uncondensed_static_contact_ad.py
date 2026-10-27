import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.nodal import load_nodal_model, nodal_body_stiffness

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def test_full_body_static_ad_matches_trace_ad_and_measured_body_force():
    pytest.importorskip("jax").config.update("jax_enable_x64", True)
    names = [
        "refined_contact.py",
        "uncondensed_static_contact.py",
        "fem_static_contact_ad.py",
        "uncondensed_static_contact_ad.py",
    ]
    tools = {name: runpy.run_path(str(TOOLS / name)) for name in names}
    model = load_nodal_model(CASE, max_panel_length=160, contact_panel_length=160)
    refined = tools["refined_contact.py"]
    space = refined["ContactQuadrature"].build(
        model.reference.moving.starts[model.moving_contact, 1], 2
    )
    seed = space.seed_from_bulk_p1(model.restart)
    reduction = refined["reduce_static_trace"](model, nodal_body_stiffness(model, quadrature=4))
    lift = tools["uncondensed_static_contact.py"]["initialize_uncondensed_contact"](
        model, space, reduction, seed
    )
    derivative = tools["uncondensed_static_contact_ad.py"]["uncondensed_static_parameter_jacobian"](
        model, space, reduction, seed, lift
    )
    trace = refined["RefinedContactEquilibrium"](
        model, space, reduction.impedance, base_force=-reduction.effective_load
    )
    expected = tools["fem_static_contact_ad.py"]["healed_static_parameter_jacobian"](
        trace,
        model.reference.normal_pressure,
        model.reference.restart_actuator,
        seed,
        lift.retained_displacement,
    )
    np.testing.assert_allclose(
        derivative["displacement_jacobian"],
        expected["displacement_jacobian"],
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        derivative["plastic_jacobian"], expected["plastic_jacobian"], rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        derivative["retained_bulk_force_jacobian"],
        reduction.impedance @ derivative["displacement_jacobian"],
        rtol=1e-7,
        atol=1e-8,
    )
    assert not derivative["active_set_iterations_differentiated"]
    assert not derivative["full_transient_initialization_gradient_connected"]
    # Independently resolve perturbed NumPy roots, including the measured bulk
    # force. This checks derivatives without reusing the selected-set AD map.
    for column, name in enumerate(["normal_penalty", "tangential_penalty"]):
        law = model.reference.contact
        step = getattr(law, name) * 1e-3
        sides = []
        for sign in [-1, 1]:
            changed = replace(
                model,
                reference=replace(
                    model.reference,
                    contact=replace(law, **{name: getattr(law, name) + sign * step}),
                ),
            )
            sides.append(
                tools["uncondensed_static_contact.py"]["initialize_uncondensed_contact"](
                    changed, space, reduction, seed
                )
            )
        for key, jacobian in [
            ("retained_displacement", "displacement_jacobian"),
            ("retained_bulk_force", "retained_bulk_force_jacobian"),
        ]:
            finite = (getattr(sides[1], key) - getattr(sides[0], key)) / (2 * step)
            np.testing.assert_allclose(
                derivative[jacobian][:, column], finite, rtol=3e-5, atol=1e-8
            )

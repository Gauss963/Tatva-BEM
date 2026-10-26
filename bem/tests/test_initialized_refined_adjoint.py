import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
from finiteblock_bem.friction import ContactState
from finiteblock_bem.jax_backend import JaxContactEquilibrium
from finiteblock_bem.nonlinear import ContactEquilibrium

ROOT = Path(__file__).resolve().parents[1]
SOURCE = runpy.run_path(str(ROOT / "tests/test_refined_contact_adjoint.py"))
STATIC = runpy.run_path(str(ROOT / "tools/fem_static_contact.py"))
STATIC_AD = runpy.run_path(str(ROOT / "tools/fem_static_contact_ad.py"))
AD = runpy.run_path(str(ROOT / "tools/initialized_refined_adjoint.py"))


def problem(parameters, substeps):
    reference, original, history, time, ramp = SOURCE["problem"]()
    names = (
        "normal_penalty",
        "tangential_penalty",
        "static_friction",
        "kinetic_friction",
        "characteristic_slip",
    )
    law = replace(reference.case.contact, **dict(zip(names, parameters, strict=True)))
    reference.case = SimpleNamespace(contact=law)
    bulk = original[:3].sum(axis=0)
    static = SimpleNamespace(**vars(reference))
    static.matrix = bulk.copy()
    static.matrix[-1, -1] += reference.spring
    static.evaluate = lambda *a, **kw: ContactEquilibrium.evaluate(static, *a, **kw)
    m, d = len(reference.base_dn), reference.constraints.size
    old = ContactState(np.zeros(m), np.zeros(m))
    preload = 1.1
    q0, state, diagnostics = STATIC["solve_healed_static_contact"](static, 0.0, preload, old)
    reference.base_dn = reference.base_dn + reference.constraints.normal_jump @ q0
    reference.base_ds = reference.base_ds + reference.constraints.tangential_jump @ q0
    reference.base_platen = q0[-1]
    reference.base_t = reference.base_t + bulk @ q0
    system = (
        JaxContactEquilibrium(reference)
        if substeps == 1
        else SOURCE["SUBSTEP"]["SubsteppedContactEquilibrium"](reference, substeps=substeps)
    )
    q = np.zeros((len(time), d))
    z, c = np.zeros((len(time), m)), np.zeros((len(time), m))
    z[0], c[0] = state.plastic_slip, state.cumulative_slip
    actuator = preload + 0.5 * ramp
    for i in range(1, len(time)):
        past = sum((original[i - j] @ q[j] for j in range(i)), start=np.zeros(d))
        answer = system.solve(
            0.0, actuator[i], ContactState(z[i - 1], c[i - 1]), past=past, guess=q[i - 1]
        )
        assert answer.converged
        q[i], z[i], c[i] = (
            answer.reduced_displacement,
            answer.contact.state.plastic_slip,
            answer.contact.state.cumulative_slip,
        )
    result = SimpleNamespace(time=time, bulk_displacement=q, plastic_slip=z, cumulative_slip=c)
    return reference, history, result, static, old, q0, diagnostics


@pytest.mark.parametrize("substeps", [1, 4])
def test_initialized_complete_contact_gradient_matches_original_z_reinitialization(substeps):
    p = np.array([100.0, 40.0, 0.8, 0.5, 0.2])
    reference, history, result, static, seed, q0, diagnostic = problem(p, substeps)
    assert diagnostic["static_sliding_points"] > 0
    assert result.cumulative_slip[-1].max() > 0
    jacobian = STATIC_AD["healed_static_parameter_jacobian"](static, 0.0, 1.1, seed, q0)
    gradients = [
        AD["initialized_gradient"](
            method, static, jacobian, reference, history, result, constitutive_substeps=substeps
        )
        for method in ("forward_gradient", "reverse_gradient", "batched_forward_gradient")
    ]
    np.testing.assert_allclose(
        gradients[0]["gradient"], gradients[1]["gradient"], rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        gradients[2]["gradient"], gradients[1]["gradient"], rtol=1e-9, atol=1e-10
    )
    wc = reference.constraints.contact_weights
    for j, epsilon in enumerate([1e-3, 1e-3, 1e-5, 1e-5, 1e-5]):
        delta = np.eye(5)[j] * epsilon
        losses = []
        for sign in (-1, 1):
            trajectory = problem(p + sign * delta, substeps)[2]
            losses.append(np.average(trajectory.cumulative_slip[-1], weights=wc))
        finite = (losses[1] - losses[0]) / (2 * epsilon)
        np.testing.assert_allclose(gradients[1]["gradient"][j], finite, rtol=3e-4, atol=3e-7)
    assert gradients[1]["initial_contact_force_history_source_included"]

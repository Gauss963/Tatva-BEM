import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ROOT = Path(__file__).resolve().parents[1]
PRIMAL = runpy.run_path(str(ROOT / "tests/test_fem_static_contact.py"))
AD = runpy.run_path(str(ROOT / "tools/fem_static_contact_ad.py"))


@pytest.mark.parametrize("shear", [-20.0, 0.0, 20.0])
def test_static_root_and_offset_ad_match_independent_active_set_finite_differences(shear):
    from finiteblock_bem.friction import ContactState

    ref = PRIMAL["reference"]()
    ref.base_t[1] = shear
    old = ContactState(np.zeros(1), np.zeros(1))
    root, state, _ = PRIMAL["TOOL"]["solve_healed_static_contact"](ref, 0.0, 0.0, old)
    actual = AD["healed_static_parameter_jacobian"](ref, 0.0, 0.0, old, root)
    np.testing.assert_allclose(actual["plastic_slip"], state.plastic_slip, atol=1e-12)
    law = ref.case.contact
    names = (
        "normal_penalty",
        "tangential_penalty",
        "static_friction",
        "kinetic_friction",
        "characteristic_slip",
    )
    for j, name in enumerate(names):
        step = 1e-5 * getattr(law, name)
        values = []
        for sign in (-1, 1):
            ref.case = SimpleNamespace(
                contact=replace(law, **{name: getattr(law, name) + sign * step})
            )
            q, z, _ = PRIMAL["TOOL"]["solve_healed_static_contact"](ref, 0.0, 0.0, old)
            values.append(np.r_[q, z.plastic_slip])
        finite = (values[1] - values[0]) / (2 * step)
        derivative = np.r_[actual["displacement_jacobian"][:, j], actual["plastic_jacobian"][:, j]]
        np.testing.assert_allclose(derivative, finite, rtol=2e-7, atol=1e-9)
    ref.case = SimpleNamespace(contact=law)
    mapping, p, _ = AD["healed_static_map"](ref, old, root)
    weights = np.arange(1, 5, dtype=float)

    def objective(parameters):
        import jax.numpy as jnp

        q, z = mapping(parameters, 0.0, 0.0)
        return jnp.dot(jnp.concatenate((q, z)), weights)

    reverse = jax.grad(objective)(p)
    expected = weights @ np.vstack((actual["displacement_jacobian"], actual["plastic_jacobian"]))
    np.testing.assert_allclose(reverse, expected, rtol=1e-11, atol=1e-12)
    assert not actual["active_set_iterations_differentiated"]
    assert not actual["full_transient_initialization_gradient_connected"]

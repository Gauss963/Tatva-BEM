"""Independent small constrained model, original identity and complex symmetry."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parents[1] / "tools"
POLICY = runpy.run_path(str(TOOLS / "trace_solve_policy.py"))
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


@pytest.mark.parametrize("general,lazy", [(False, True), (True, False), (True, True)])
def test_actual_complex_body_maps_retain_original_and_full_reconstruction(general, lazy):
    model = load_nodal_model(
        CASE,
        max_panel_length=160,
        contact_panel_length=160,
        contact_end_length=160,
        corner_panel_length=160,
    )
    original = POLICY["TRACE"](model, quadrature=12)
    operators = [p.assemble(175 + 31400j) for p in original.prepared]
    original_method = original._from_reduced.__func__
    candidate = POLICY["transfer_policy"](original, general=general, lazy=lazy)
    expected, reconstruction = original.evaluate_operators(operators, reconstruction=True)
    wrapped = POLICY["layers"](operators, general)
    actual = candidate.evaluate_operators(wrapped)
    if not general:
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    full = candidate.evaluate_operators(wrapped, reconstruction=True)
    np.testing.assert_allclose(full[0], actual, rtol=1e-14, atol=1e-10)
    np.testing.assert_allclose(full[1], reconstruction, rtol=1e-10, atol=1e-10)
    assert original._from_reduced.__func__ is original_method
    assert all(type(a) is POLICY["GalerkinOperators"] for a in operators)
    assert all(a.single_layer is b.single_layer for a, b in zip(wrapped, operators, strict=True))


def test_explicit_general_solver_keeps_bilinear_complex_equation_and_inputs():
    a = np.array([[3 + 2j, 0.5j], [0.5j, 2 - 1j]])
    rhs = np.array([[1 + 2j, 3], [4, 5 - 3j]])
    saved_a, saved_rhs = a.copy(), rhs.copy()
    actual = POLICY["general_solve"](a, rhs, check_finite=False)
    np.testing.assert_allclose(a @ actual, rhs, rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(a, saved_a)
    np.testing.assert_array_equal(rhs, saved_rhs)
    with pytest.raises(ValueError, match="explicit"):
        POLICY["general_solve"](a, rhs, assume_a="her")

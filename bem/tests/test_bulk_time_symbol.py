import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.io import environment_record
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/audit_bulk_time_symbol.py"))


def test_symbol_matches_bdf2_on_a_sampled_exponential():
    s, dt = 200 + 20000j, 2e-6
    # Divide the backward difference of exp(s*t) by the current value.
    difference = (3 - 4 * np.exp(-s * dt) + np.exp(-2 * s * dt)) / (2 * dt)
    assert AUDIT["mapped_parameter"](s, dt) == pytest.approx(difference, rel=1e-13)


def test_fixed_oscillator_transfer_has_second_order_symbol_error():
    s = 20 + 700j
    stiffness, mass = 1e6, 1
    diagonal = np.array([4.0, 5.0, 8.0])

    def response(parameter):
        # A coupled positive symmetric stiffness, with mass only in the bulk.
        k = stiffness * np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        z = k + parameter**2 * mass * np.eye(3)
        return AUDIT["closed_stick_response"](z, diagonal, diagonal[-1])[0]

    exact = response(s)
    errors = [
        np.linalg.norm(response(AUDIT["mapped_parameter"](s, dt)) - exact)
        for dt in [2e-5, 1e-5, 0.5e-5]
    ]
    assert 3.9 < errors[0] / errors[1] < 4.1
    assert 3.9 < errors[1] / errors[2] < 4.1


def test_complex_profile_metric_detects_phase_error_on_the_whole_interface():
    y = np.array([0.0, 1.0, 2.0])
    q = np.r_[np.ones(3), np.ones(3), 1.0].astype(complex)
    actual = q.copy()
    actual[1] = 1j  # Same amplitude and unchanged endpoints.
    result = AUDIT["response_errors"](y, actual, q, 20.0, 10.0)
    assert result["pressure_profile_relative_l2"] == pytest.approx(np.sqrt(2 / 3))
    assert result["signed_shear_profile_relative_l2"] == 0
    assert result["platen_relative_error"] == 0


@pytest.mark.parametrize("s,dt", [(0j, 1e-6), (-1 + 2j, 1e-6), (1 + 2j, 0), (1 + 4j, 1)])
def test_symbol_rejects_invalid_or_aliased_parameters(s, dt):
    with pytest.raises(ValueError):
        AUDIT["mapped_parameter"](s, dt)


def test_response_uses_the_actual_instantaneous_contact_and_spring():
    z = np.array([[10, -2, 1], [-2, 8, -3], [1, -3, 7]], dtype=complex)
    diagonal = np.array([30.0, 40.0, 50.0])
    actual, residual = AUDIT["closed_stick_response"](z, diagonal, 50)
    expected = np.linalg.solve(z + np.diag(diagonal), [0, 0, 50])
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
    assert residual < 1e-14
    with pytest.raises(ValueError):
        AUDIT["closed_stick_response"](z, diagonal, 25)


@pytest.mark.parametrize("method", ["bdf2", "radau"])
def test_streamed_two_body_diagnostic_matches_independent_compliance_path(
    tmp_path, monkeypatch, method
):
    case = (
        Path(__file__).resolve().parents[1]
        / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    )
    options = dict(
        case=str(case),
        h=160,
        contact_h=160,
        contact_end_h=20,
        corner_h=20,
        restart_knot_window=None,
        restart_knot_subdivisions=1,
        quadrature=4,
    )
    metadata = tmp_path / "mesh.json"
    metadata.write_text(
        json.dumps(dict(source_mesh_arguments=options, environment=environment_record(case)))
    )
    output = tmp_path / "diagnostic"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            str(metadata),
            "--output",
            str(output),
            "--frequencies",
            "1000",
            "--dt",
            "2e-6",
            "1e-6",
            "--method",
            method,
        ],
    )
    original = NodalInterfaceTransfer.evaluate
    AUDIT["main"]()
    assert NodalInterfaceTransfer.evaluate is original
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "completed_sampled_bulk_time_symbol_diagnostic"
    assert not summary["complete_nonlinear_time_convergence_verified"]
    assert not list(output.glob("supported-*.npy"))
    model = load_nodal_model(
        case,
        max_panel_length=160,
        contact_panel_length=160,
        contact_end_length=20,
        corner_panel_length=20,
    )
    transfer = NodalInterfaceTransfer(model, quadrature=4)
    w, law = model.constraints.contact_weights, model.reference.contact
    spring = model.reference.spring_stiffness * model.constraints.platen_width
    diagonal = np.r_[w * law.normal_penalty, w * law.tangential_penalty, spring]
    rhs = np.r_[np.zeros(len(diagonal) - 1), spring]
    for index, row in enumerate(summary["records"]):
        if method == "bdf2" or row["dt_s"] is None:
            s = complex(row["s_real_per_s"], row["s_imag_per_s"])
            z = transfer.evaluate(s)
            expected = np.linalg.solve(z + np.diag(diagonal), rhs)
        else:
            s = complex(row["physical_s_real_per_s"], row["physical_s_imag_per_s"])
            dt = row["dt_s"]
            zeta = np.exp(-s * dt)
            a = np.array([[5 / 12, -1 / 12], [3 / 4, 1 / 4]])
            symbol = (
                np.linalg.inv(a + zeta / (1 - zeta) * np.outer(np.ones(2), [3 / 4, 1 / 4])) / dt
            )
            # Build all stage blocks first; solve their coupled root directly.
            zstage = AUDIT["RADAU"]["matrix_function"](transfer.evaluate, symbol)
            forcing = np.exp(s * dt * np.array([-2 / 3, 0]))
            expected = np.linalg.solve(
                zstage + np.kron(np.eye(2), np.diag(diagonal)), np.kron(forcing, rhs)
            )[-len(diagonal) :]
        actual = np.load(output / f"response-{index}.npz")["retained_displacement"]
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)


def test_radau_modes_match_a_coupled_stage_oscillator_root():
    s, dt = 20 + 700j, 2e-4
    k = 1e6 * np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    mass = np.diag([2.0, 3.0, 4.0])
    penalty = np.array([4.0, 5.0, 8.0])
    modes = AUDIT["parameter_modes"](s, dt, "radau")
    actual = sum(
        weight * AUDIT["closed_stick_response"](k + value**2 * mass, penalty, 8)[0]
        for value, weight in modes
    )
    a = np.array([[5 / 12, -1 / 12], [3 / 4, 1 / 4]])
    zeta = np.exp(-s * dt)
    symbol = np.linalg.inv(a + zeta / (1 - zeta) * np.outer(np.ones(2), [3 / 4, 1 / 4])) / dt
    matrix = np.kron(symbol @ symbol, mass) + np.kron(np.eye(2), k + np.diag(penalty))
    stage_force = np.exp(s * dt * np.array([-2 / 3, 0]))
    expected = np.linalg.solve(matrix, np.kron(stage_force, [0, 0, 8]))[-3:]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-16)


def test_radau_integrator_matches_quadrature_and_third_order_endpoint_error():
    s = 20 + 700j
    errors = []
    for dt in [1e-4, 0.5e-4, 0.25e-4]:
        modes = AUDIT["parameter_modes"](s, dt, "radau")
        actual = sum(weight / value for value, weight in modes)
        # Independent interval quadrature followed by the geometric sum.
        quadrature = (
            dt
            * np.dot([3 / 4, 1 / 4], np.exp(s * dt * np.array([-2 / 3, 0])))
            / (-np.expm1(-s * dt))
        )
        assert actual == pytest.approx(quadrature, rel=1e-12)
        errors.append(abs(actual - 1 / s))
        assert sum(weight for _, weight in modes) == pytest.approx(1, abs=1e-14)
        assert all(value.real > 0 for value, _ in modes)
    assert 7.8 < errors[0] / errors[1] < 8.2
    assert 7.8 < errors[1] / errors[2] < 8.2


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        AUDIT["parameter_modes"](1 + 2j, 1e-6, "unknown")

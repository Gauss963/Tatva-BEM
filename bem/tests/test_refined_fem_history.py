import hashlib
import json
import runpy
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

TOOLS = Path(__file__).resolve().parents[1] / "tools"
TOOL = runpy.run_path(str(TOOLS / "compare_refined_fem_history.py"))


def test_tensor_p1_norm_matches_independent_cell_gauss_integration():
    rng = np.random.default_rng(412)
    at, bt = np.array([0, 0.2, 0.7, 1]), np.array([0.1, 0.3, 0.8, 1.1])
    ay, by = np.array([0, 0.4, 2]), np.array([0, 1.1, 1.7, 2])
    av, bv = rng.normal(size=(len(at), len(ay))), rng.normal(size=(len(bt), len(by)))
    ai, bi = RegularGridInterpolator((at, ay), av), RegularGridInterpolator((bt, by), bv)
    tt, yy = np.union1d(at, bt), np.union1d(ay, by)
    tt = np.union1d([0.1, 1], tt[(tt > 0.1) & (tt < 1)])
    x, w = np.polynomial.legendre.leggauss(2)
    expected_error, expected_reference = 0.0, 0.0
    for t0, t1 in pairwise(tt):
        for y0, y1 in pairwise(yy):
            for ti, tw in zip(x, w, strict=True):
                for yi, yw in zip(x, w, strict=True):
                    point = [(t0 + t1) / 2 + (t1 - t0) * ti / 2, (y0 + y1) / 2 + (y1 - y0) * yi / 2]
                    a, b = ai([point])[0], bi([point])[0]
                    weight = tw * yw * (t1 - t0) * (y1 - y0) / 4
                    expected_error += weight * (b - a) ** 2
                    expected_reference += weight * a**2
    for chunk in [1, 3, 128]:
        result = TOOL["compare_saved_p1_fields"]((at, ay, av), (bt, by, bv), row_chunk=chunk)
        assert result["common_time_interval_s"] == [0.1, 1]
        np.testing.assert_allclose(result["error_squared_space_time_integral"], expected_error)
        np.testing.assert_allclose(
            result["reference_squared_space_time_integral"], expected_reference
        )
        assert not result["extrapolation_used"]
        assert result["fitted_time_shift_s"] == 0


def test_bilinear_analytic_integral_and_interior_pulse_are_not_endpoint_checks():
    at, bt, ay, by = map(np.array, ([0, 0.5, 2], [0, 0.1, 0.7, 1.4, 2], [0, 2, 3], [0, 1, 3]))
    a, b = (1 + at[:, None]) * (1 + ay), 2 * (1 + bt[:, None]) * (1 + by)
    result = TOOL["compare_saved_p1_fields"]((at, ay, a), (bt, by, b), row_chunk=2)
    np.testing.assert_allclose(result["error_squared_space_time_integral"], 182)
    np.testing.assert_allclose(result["relative_space_time_p1_l2"], 1)
    t, y = np.arange(5.0), np.arange(3.0)
    a, b = np.ones((5, 3)), np.ones((5, 3))
    b[2, 1] += 4
    result = TOOL["compare_saved_p1_fields"]((t, y, a), (t, y, b))
    assert result["maximum_saved_interpolant_absolute_difference"] == 4
    assert result["relative_space_time_p1_l2"] > 0
    assert result["final_shared_time_relative_p1_l2"] == 0


def test_zero_reference_and_disjoint_domains_are_explicit():
    t, y = [0, 1], [0, 2]
    result = TOOL["compare_saved_p1_fields"]((t, y, np.zeros((2, 2))), (t, y, np.ones((2, 2))))
    assert result["relative_space_time_p1_l2"] is None
    assert result["error_squared_space_time_integral"] == 2
    with pytest.raises(ValueError, match="no time interval"):
        TOOL["compare_saved_p1_fields"]((t, y, np.ones((2, 2))), ([2, 3], y, np.ones((2, 2))))
    with pytest.raises(ValueError, match="same complete contact interval"):
        TOOL["compare_saved_p1_fields"]((t, y, np.ones((2, 2))), (t, [0, 1], np.ones((2, 2))))


def test_integer_fem_clock_preserves_normal_phase_overshoot_and_checks_rounding():
    dt, normal_steps = 1.2492928633545953e-8, 160091
    steps = np.array([0, 1, 401, 801, 3602038])
    absolute = normal_steps * dt + steps * dt
    trace = dict(
        phase_id=np.array([1, 2, 2, 2, 2]),
        step_id=steps,
        history=dict(time=absolute.astype(np.float32).astype(float)),
    )
    summary = dict(summary=dict(dt=dt, pressure_steps=normal_steps))
    config = dict(time_step_override=dt, normal_phase_time=0.002)
    indices, time = TOOL["fem_step_clock"](trace, summary, config)
    np.testing.assert_array_equal(indices, np.arange(1, 5))
    assert time[0] > dt  # The normal phase ends slightly after its nominal duration.
    assert time[-1] > 0.045
    trace["history"]["time"][2] += 1e-5
    with pytest.raises(AssertionError):
        TOOL["fem_step_clock"](trace, summary, config)


def test_cli_reads_resolved_friction_parameters_and_compares_all_saved_fields(
    tmp_path, monkeypatch
):
    import h5py

    bem, fem, output = (tmp_path / name for name in ["bem", "fem", "comparison"])
    bem.mkdir()
    fem.mkdir()
    law = dict(
        normal_penalty=76620.0,
        tangential_penalty=7662.0,
        static_friction=0.8,
        kinetic_friction=0.6,
        characteristic_slip=0.0125959,
    )
    loading = dict(normal_pressure=16.0, spring_stiffness=25.0, actuator_rate=20.0)
    meta = dict(
        status="completed_requested_horizon",
        duration_s=0.045,
        all_history_retained=True,
        completed_steps=3,
        dt_s=0.015,
        case=dict(contact=law, loading=loading, source=dict(revision="test")),
    )
    (bem / "summary.json").write_text(json.dumps(meta))
    bt, y = np.arange(4) * 0.015, np.array([0, 480])
    scalar_keys = [
        "max_cumulative_slip_mm",
        "mean_contact_pressure_mpa",
        "mean_contact_shear_mpa",
        "platen_mm",
        "actuator_mm",
        "applied_shear_mpa",
    ]
    scalar_values = [bt, bt * 0 + 16, bt * 0 + 9, bt * 0 + 3, bt * 0 + 4, bt * 0 + 25]
    np.savetxt(
        bem / "history.csv",
        np.column_stack([bt, *scalar_values]),
        delimiter=",",
        header=",".join(["time_s", *scalar_keys]),
        comments="",
    )
    np.savez(
        bem / "trajectory.npz",
        time=bt,
        contact_y=y,
        plastic_slip=np.repeat((9 + bt)[:, None], 2, axis=1),
        cumulative_slip=np.repeat(bt[:, None], 2, axis=1),
        friction_coefficient=np.full((4, 2), 0.8),
        pressure=np.full((4, 2), 16.0),
    )
    # Actual resolved config has no raw mu_s/mu_k/Dc keys: null overrides
    # inherit the archived case, whose resolved values live in the summary.
    config = dict(
        dtype="float64",
        time_step_override=0.001,
        normal_phase_time=0.002,
        mu_k_override=None,
        critical_slip_override=None,
        friction_law="slip-weakening",
        mu_s_start_fraction=1,
        mu_s_end_fraction=1,
        pw_length=0,
        loading_edge_nucleation_length=0,
        leading_edge_creep_length=0,
        leading_edge_guard_length=0,
        leading_edge_tangential_taper_length=0,
    )
    (fem / "resolved_run_config.json").write_text(json.dumps(config))
    manifest = fem / "reference_manifest.json"
    manifest.write_text(json.dumps(dict(source=dict(revision="test"))))
    resolved = dict(
        dt=0.001,
        pressure_steps=2,
        normal_penalty=76620.0,
        tangential_penalty=7662.0,
        mu_k=0.6,
        critical_slip=0.0125959,
        legacy=dict(friction=dict(mu_s=0.8)),
        normal_stress=16.0,
        shear_loading_stiffness=25.0,
        shear_post_ramp_rate=20.0,
    )
    summary = dict(
        status="completed",
        full_original_horizon=True,
        computational_dtype="float64",
        source_and_protected_input_unchanged=True,
        source_revision="test",
        summary=resolved,
        reference_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    (fem / "summary.json").write_text(json.dumps(summary))
    steps = np.array([1, 20, 45])
    ft = steps * 0.001
    names = [
        "time",
        "max_slip",
        "avg_sigma_n",
        "avg_tau",
        "loading_face_displacement",
        "applied_shear_displacement",
        "applied_shear",
    ]
    values = np.column_stack(
        [ft + 0.002, ft, ft * 0 + 16, ft * 0 + 9, ft * 0 + 3, ft * 0 + 4, ft * 0 + 25]
    ).astype(np.float32)
    with h5py.File(fem / "simulation.h5", "w") as stream:
        stream.attrs["saved_interface_frames"] = 3
        group = stream.create_group("interface_high_rate")
        group["history"], group["history_columns"] = values, np.array(names, dtype="S32")
        group["phase_id"], group["step_id"] = np.full(3, 2), steps
        group["contact_line_y"] = y
        for key, v in [
            ("plastic_slip", 9 + ft),
            ("cumulative_slip", ft),
            ("friction_coefficient", ft * 0 + 0.8),
            ("friction_strength", ft * 0 + 0.8 * 16),
        ]:
            group[key] = np.repeat(v[:, None], 2, axis=1).astype(np.float32)
    monkeypatch.setattr(sys, "argv", ["compare", str(bem), str(fem), "--output", str(output)])
    TOOL["main"]()
    result = json.loads((output / "summary.json").read_text())
    assert result["status"] == "completed_saved_space_time_comparison"
    assert set(result["field_errors"]) == {*TOOL["FIELDS"], "pressure_reconstructed"}
    assert set(result["scalar_errors"]) == set(scalar_keys)
    assert result["field_errors"]["cumulative_slip"]["relative_space_time_p1_l2"] < 1e-7
    assert not result["same_accuracy_fem_advantage_verified"]

import copy
import runpy
from pathlib import Path

import h5py
import numpy as np
import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/compare_fem_mesh.py"))


def test_saved_space_time_norm_matches_independent_tensor_gauss_quadrature():
    time = np.array([0.0, 0.15, 0.9])
    y, yy = np.array([0.0, 0.4, 1]), np.array([0.0, 0.2, 0.75, 1.0])
    a = np.array([[1.0, 2.0, 3.0], [3.0, -1.0, 2.0], [2.0, 4.0, 0.5]])
    b = np.array([[1.5, 2.1, 2.7, 3.2], [2.5, -2.0, 1.0, 1.5], [1.7, 3.8, 2.0, 0.1]])
    result = TOOL["error_report"](y, a, yy, b, time=time)
    points, weights = np.polynomial.legendre.leggauss(3)
    knots = np.union1d(y, yy)
    reference2 = error2 = 0.0
    # Integrate by evaluating the interpolants at independent quadrature
    # points, rather than using the mass-matrix formula under test.
    for i, dt in enumerate(np.diff(time)):
        for j, dx in enumerate(np.diff(knots)):
            x = knots[j] + (points + 1) * dx / 2
            for q, w in zip((points + 1) / 2, weights, strict=True):
                av = np.interp(x, y, (1 - q) * a[i] + q * a[i + 1])
                bv = np.interp(x, yy, (1 - q) * b[i] + q * b[i + 1])
                reference2 += dt * dx / 4 * w * np.dot(weights, av**2)
                error2 += dt * dx / 4 * w * np.dot(weights, (av - bv) ** 2)
    np.testing.assert_allclose(result["absolute_space_time_l2"], np.sqrt(error2), rtol=2e-15)
    np.testing.assert_allclose(
        result["relative_space_time_l2"], np.sqrt(error2 / reference2), rtol=2e-15
    )


def test_p1_comparison_preserves_fields_between_different_mesh_knots():
    y, yy, t = np.array([0.0, 1.0]), np.array([0.0, 0.1, 0.4, 1.0]), np.array([0.0, 0.2, 1.0])
    reference = (1 + t[:, None]) * (1 + y[None, :])
    actual = 1.03 * (1 + t[:, None]) * (1 + yy[None, :])
    report = TOOL["error_report"](y, reference, yy, actual, time=t)
    assert report["relative_space_time_l2"] == pytest.approx(0.03)
    assert report["final_relative_p1_l2"] == pytest.approx(0.03)
    assert report["final_reference_p1_mean"] == pytest.approx(3.0)
    assert report["final_actual_p1_mean"] == pytest.approx(3.09)
    zero = TOOL["error_report"](y, np.zeros((1, 2)), yy, np.ones((1, 4)))
    assert zero["final_relative_p1_l2"] is None
    assert zero["final_absolute_p1_l2"] == pytest.approx(1.0)


@pytest.mark.parametrize("change", ["extrapolation", "unordered", "nan", "time", "interval"])
def test_p1_comparison_rejects_invalid_alignment(change):
    y, yy, t = np.array([0.0, 1.0]), np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0])
    a, b = np.ones((2, 2)), np.ones((2, 3))
    if change == "extrapolation":
        with pytest.raises(ValueError, match="extrapolation"):
            TOOL["interpolate_nodes"](y, a, [-0.1, 1.0])
        return
    if change == "unordered":
        yy[1] = 1.0
    elif change == "nan":
        b[0, 0] = np.nan
    elif change == "time":
        t[1] = 0
    else:
        yy[-1] = 1.01
    with pytest.raises(ValueError):
        TOOL["error_report"](y, a, yy, b, time=t)


def mesh_records():
    config = dict(
        mesh_size=2.0,
        dtype="float64",
        time_step_override=1e-8,
        normal_phase_time=0.002,
        shear_phase_time=0.045,
        normal_penalty=100.0,
        tangential_penalty=10.0,
        restart_from_file="original.h5",
    )
    seed = dict(path="original.h5", seed_sha256="seed")
    manifest = dict(
        source=dict(revision="revision", archive_sha256="archive"),
        protected_input_sha256={"case": "case-sha"},
        source_restart=seed,
        actual_restart=seed,
        restart_projection=None,
    )
    summary = dict(
        status="completed",
        full_original_horizon=True,
        source_and_protected_input_unchanged=True,
        computational_dtype="float64",
        jax_enable_x64=True,
        source_revision="revision",
        source_archive_sha256="archive",
        summary=dict(dt=1e-8, mesh_size=2.0, pressure_time=0.002, shear_time=0.045),
    )
    a = dict(config=config, manifest=manifest, summary=summary)
    b = copy.deepcopy(a)
    b["config"].update(mesh_size=1.0, restart_from_file="projection.h5")
    b["summary"]["summary"]["mesh_size"] = 1.0
    b["manifest"].update(
        actual_restart=dict(path="projection.h5", seed_sha256="projected"),
        restart_projection=dict(source_plastic_dtype="float64", source_restart=seed),
    )
    return a, b


def test_spatial_control_allows_only_mesh_and_verified_projection_path():
    TOOL["validate_mesh_pair"](*mesh_records())


@pytest.mark.parametrize(
    "change", ["dt", "physics", "precision", "source", "seed", "incomplete", "coarser"]
)
def test_nominal_spatial_controls_reject_other_changes(change):
    a, b = mesh_records()
    if change == "dt":
        b["config"]["time_step_override"] = b["summary"]["summary"]["dt"] = 2e-8
    elif change == "physics":
        b["config"]["normal_penalty"] = 101.0
    elif change == "precision":
        b["config"]["dtype"] = b["summary"]["computational_dtype"] = "float32"
    elif change == "source":
        b["manifest"]["source"]["revision"] = b["summary"]["source_revision"] = "other"
    elif change == "seed":
        b["manifest"]["restart_projection"]["source_plastic_dtype"] = "float32"
    elif change == "incomplete":
        b["summary"]["status"] = "running"
    else:
        b["config"]["mesh_size"] = b["summary"]["summary"]["mesh_size"] = 4.0
    with pytest.raises(ValueError):
        TOOL["validate_mesh_pair"](a, b)


def test_float64_endpoint_pressure_uses_geometry_gap_and_correct_body_layout(tmp_path):
    path = tmp_path / "final.h5"
    with h5py.File(path, "w") as stream:
        stream["moving/coords"] = [[1.0, 0.0], [0.8, 1.0]]
        stream["stationary/coords"] = [[1.0, 1.0], [1.0, 0.0]]
        stream["interface/master_nodes"] = [0, 1]
        stream["interface/slave_nodes"] = [1, 0]
        stream["interface/contact_line_y"] = [0.0, 1.0]
        stream["final_state/u"] = [0.03, 0.1, 0.3, 0.2, 0.02, 0.04, 0.01, 0.01]
        stream["final_state/plastic_slip"] = [0.01, 0.02]
        stream["final_state/cumulative_slip"] = [0.02, 0.04]
    y, fields = TOOL["read_final_fields"](path, dict(normal_penalty=100.0, tangential_penalty=10.0))
    np.testing.assert_array_equal(y, [0, 1])
    np.testing.assert_allclose(fields["pressure"], [2.0, 8.0], atol=1e-14)
    np.testing.assert_allclose(fields["elastic_trial_shear"], [0.8, 1.4])
    with h5py.File(path, "a") as stream:
        del stream["final_state/u"]
        stream["final_state/u"] = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="true float64"):
        TOOL["read_final_fields"](path, dict(normal_penalty=100.0, tangential_penalty=10.0))

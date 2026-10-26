import copy
import json
import runpy
import sys
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


def test_legacy_projection_field_is_optional_only_for_the_original_unprojected_seed():
    a, b = mesh_records()
    del a["manifest"]["restart_projection"]
    TOOL["validate_mesh_pair"](a, b)
    a["manifest"]["actual_restart"] = dict(path="other.h5", seed_sha256="other")
    with pytest.raises(ValueError, match="source seed"):
        TOOL["validate_mesh_pair"](a, b)
    a, b = mesh_records()
    del b["manifest"]["restart_projection"]
    with pytest.raises(ValueError, match="source seed"):
        TOOL["validate_mesh_pair"](a, b)


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
        stream["final_state/v_half"] = np.zeros(8)
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


def test_full_mesh_comparison_checks_projection_and_reports_static_initialization(
    tmp_path, monkeypatch
):
    baseline, refined = mesh_records()
    original = tmp_path / "original.h5"
    y0 = np.array([0.0, 2.0])
    plastic0 = np.array([0.01 + 2**-40, 0.02 + 2**-40])
    with h5py.File(original, "w") as stream:
        stream["interface/contact_line_y"] = y0.astype(np.float32)
        stream["final_state/u"] = np.zeros(8)
        stream["final_state/plastic_slip"] = plastic0
        stream["final_state"].attrs["applied_shear_displacement"] = 3.0
    source = TOOL["SEED"]["restart_seed_record"](original)
    case = tmp_path / "case.toml"
    case.write_text("fixture = true\n")
    records = []
    for i, record in enumerate((baseline, refined)):
        path = tmp_path / f"mesh-{i}"
        path.mkdir()
        y = y0 if i == 0 else np.array([0.0, 1.0, 2.0])
        n = len(y)
        expected = np.interp(y, y0, plastic0)
        record["config"]["restart_from_file"] = str(original)
        record["manifest"].update(
            source_restart=source,
            actual_restart=source,
            protected_input_sha256={str(case): TOOL["fingerprint"](case)},
        )
        if i:
            projected = path / "projected.h5"
            with h5py.File(projected, "w") as stream:
                stream["final_state/u"] = np.zeros(4 * n)
                stream["final_state/plastic_slip"] = expected
                stream["final_state"].attrs["applied_shear_displacement"] = 3.0
            _, _, projection = TOOL["SEED"]["project_restart_seed"](original, y, source)
            record["config"]["restart_from_file"] = str(projected)
            record["manifest"].update(
                actual_restart=TOOL["SEED"]["restart_seed_record"](projected),
                restart_projection=projection,
            )
        hdf5 = path / "simulation.h5"
        with h5py.File(hdf5, "w") as stream:
            stream["moving/coords"] = np.c_[np.ones(n), y]
            stream["stationary/coords"] = np.c_[np.ones(n), y]
            stream["interface/master_nodes"] = np.arange(n)
            stream["interface/slave_nodes"] = np.arange(n)
            stream["interface/contact_line_y"] = y
            stream["final_state/u"] = np.r_[np.tile([0.03, 0.1], n), np.tile([0.01, 0.01], n)]
            stream["final_state/v_half"] = np.zeros(4 * n)
            stream["final_state/plastic_slip"] = expected + 0.02
            stream["final_state/cumulative_slip"] = np.full(n, 0.02)
            high = stream.create_group("interface_high_rate")
            high["history_columns"] = np.array([b"time", b"max_slip", b"shear_loading_stopped"])
            high["history"] = [
                [0.0, 0.0, 0.0],
                [0.002, 0.0, 0.0],
                [0.00200001, 0.0, 0.0],
                [0.047, 0.02, 1.0],
            ]
            high["phase_id"] = [0, 1, 2, 2]
            high["step_id"] = [0, 2, 3, 10]
            high["contact_line_y"] = y
            high["plastic_slip"] = np.repeat((expected + 1e-5)[None], 4, axis=0).astype(np.float32)
            high["plastic_slip"][-1] = (expected + 0.02).astype(np.float32)
            high["cumulative_slip"] = np.tile([0.0, 0.0, 0.0, 0.02], (n, 1)).T.astype(np.float32)
            high["friction_coefficient"] = np.full((4, n), 0.8, dtype=np.float32)
            high["friction_strength"] = np.full((4, n), 1.6, dtype=np.float32)
            stream.attrs["saved_interface_frames"] = 4
        for key, name in (
            ("config", "resolved_run_config.json"),
            ("manifest", "reference_manifest.json"),
        ):
            (path / name).write_text(json.dumps(record[key]))
        record["summary"]["reference_manifest_sha256"] = TOOL["fingerprint"](
            path / "reference_manifest.json"
        )
        record["summary"]["timing_seconds"] = dict(simulation_including_outputs=1.0)
        (path / "summary.json").write_text(json.dumps(record["summary"]))
        records.append(path)
    verified = tmp_path / "verification.json"
    trace = TOOL["READER"]["read_fem_trace"](records[0] / "simulation.h5")
    verified.write_text(
        json.dumps(
            dict(
                original_fem_reference_verified=True,
                rerun=trace["provenance"],
                frozen_input_sha256={
                    str(records[0] / "summary.json"): TOOL["fingerprint"](
                        records[0] / "summary.json"
                    )
                },
            )
        )
    )
    output = tmp_path / "outputs/comparison"
    main = TOOL["main"]
    monkeypatch.setitem(main.__globals__, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_fem_mesh.py",
            *map(str, records),
            "--field-verification",
            str(verified),
            "--output",
            str(output),
        ],
    )
    main()
    result = json.loads((output / "summary.json").read_text())
    assert result["exact_same_steps_and_saved_times"]
    assert result["compared_shear_rows"] == 2
    assert not result["fem_mesh_convergence_verified"]
    assert not result["same_accuracy_fem_advantage_verified"]
    assert result["true_float64_final_state"]["pressure"]["final_relative_p1_l2"] == 0
    assert result["true_float64_final_state"]["plastic_slip"]["final_relative_p1_l2"] < 1e-15
    assert all(
        check["first_shear_minus_input_maximum_mm"] > 9.9e-6 for check in result["initialization"]
    )
    assert all(
        check["saved_final_slip_matches_float64_after_storage_rounding"]
        for check in result["endpoint_storage_checks"]
    )
    bad_output = output.with_name("bad")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_fem_mesh.py",
            *map(str, records),
            "--field-verification",
            str(verified),
            "--output",
            str(bad_output),
        ],
    )
    with h5py.File(records[1] / "simulation.h5", "a") as stream:
        stream["interface_high_rate/plastic_slip"][-1, 0] += 0.001
    with pytest.raises(ValueError, match="Saved final slip"):
        main()
    with h5py.File(records[1] / "simulation.h5", "a") as stream:
        stream["interface_high_rate/plastic_slip"][-1] = stream["final_state/plastic_slip"][
            :
        ].astype(np.float32)
    # A changed projected seed is rejected even when it still has float64 storage.
    with h5py.File(records[1] / "projected.h5", "a") as stream:
        stream["final_state/plastic_slip"][1] += 2**-45
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_fem_mesh.py",
            *map(str, records),
            "--field-verification",
            str(verified),
            "--output",
            str(output.with_name("bad")),
        ],
    )
    with pytest.raises(ValueError, match="restart changed"):
        main()

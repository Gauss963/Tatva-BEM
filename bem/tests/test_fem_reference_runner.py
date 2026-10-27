import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/run_fem_reference.py"))


@pytest.fixture
def archived_project(tmp_path):
    project = tmp_path / "project"
    package = project / "tatva"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 3\n")
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(["git", "-C", str(project), "add", "tatva"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
    ).strip()
    return project, revision


def test_fem_source_staging_verifies_cached_bytes_without_rewriting(archived_project, tmp_path):
    project, revision = archived_project
    snapshot = tmp_path / "snapshot"
    first = TOOL["stage_snapshot"](project, snapshot, revision=revision)
    path = snapshot / "tatva/__init__.py"
    before = path.stat().st_mtime_ns
    second = TOOL["stage_snapshot"](project, snapshot, revision=revision)
    assert first == second
    assert path.read_text() == "VALUE = 3\n"
    assert path.stat().st_mtime_ns == before
    assert first["files"]["tatva/__init__.py"]["sha256"] == TOOL["digest"](path)
    assert (project / "tatva/__init__.py").read_text() == "VALUE = 3\n"


@pytest.mark.parametrize("change", ["modified", "extra"])
def test_fem_source_tampering_is_rejected_not_silently_overwritten(
    archived_project, tmp_path, change
):
    project, revision = archived_project
    snapshot = tmp_path / "snapshot"
    TOOL["stage_snapshot"](project, snapshot, revision=revision)
    path = snapshot / ("tatva/__init__.py" if change == "modified" else "tatva/unexpected.py")
    path.write_text("VALUE = 900\n")
    with pytest.raises(ValueError, match="snapshot"):
        TOOL["stage_snapshot"](project, snapshot, revision=revision)
    assert path.read_text() == "VALUE = 900\n"


@pytest.mark.parametrize(
    "arguments,message",
    [(["--interface-shear-frames", "0"], "Frame targets"), (["--shear-duration", "inf"], "finite")],
)
def test_fem_invalid_sampling_rejected_before_loading_original_files(
    tmp_path, monkeypatch, arguments, message
):
    main = TOOL["main"]
    monkeypatch.setitem(main.__globals__, "PROJECT", tmp_path)
    output = tmp_path / "bem/outputs/invalid"
    monkeypatch.setattr(sys, "argv", ["run_fem_reference.py", "--output", str(output), *arguments])
    with pytest.raises(ValueError, match=message):
        main()
    assert not output.exists()


def reference_inputs(tmp_path):
    reference = tmp_path / "reference"
    (reference / "input").mkdir(parents=True)
    (reference / "stats").mkdir()
    authored = reference / "input/case.toml"
    authored.write_text('[solver]\ndtype = "float64"\n')
    resolved = dict(
        git_revision="test-revision",
        input_sha256=TOOL["digest"](authored),
        solver=dict(dtype="float64", normal_penalty=76620.0),
        case=dict(materials={}),
    )
    (reference / "input/resolved.json").write_text(json.dumps(resolved))
    (reference / "stats/summary.json").write_text(json.dumps(dict(summary=dict(dtype="float32"))))
    return reference, resolved


@pytest.mark.parametrize(
    "arguments",
    [
        ["--contact-end-size", "nan"],
        ["--checkpoint-interval", "0"],
        ["--grading-growth", "1"],
    ],
)
def test_invalid_grading_and_checkpoint_controls_do_not_touch_reference(
    tmp_path, monkeypatch, arguments
):
    main = TOOL["main"]
    monkeypatch.setitem(main.__globals__, "PROJECT", tmp_path)
    output = tmp_path / "bem/outputs/invalid"
    monkeypatch.setattr(sys, "argv", ["reference", "--output", str(output), *arguments])
    with pytest.raises(ValueError, match="positive endpoint/checkpoint"):
        main()
    assert not output.exists()


def test_resume_rejects_source_or_configuration_changes_before_mutation(tmp_path):
    arguments = dict(resume=False, prepare_only=False, mesh_size=2, contact_end_size=0.015625)
    manifest = dict(runner_sha256="source", arguments=arguments)
    p = tmp_path / "reference_manifest.json"
    p.write_text(json.dumps(manifest))
    (tmp_path / "checkpoint.npz").write_bytes(b"original checkpoint")
    (tmp_path / "simulation.h5").write_bytes(b"partial hdf5")
    before = {q.name: q.read_bytes() for q in tmp_path.iterdir()}
    assert TOOL["resume_manifest"](tmp_path, dict(arguments, resume=True), "source") == manifest
    with pytest.raises(ValueError, match="argument: contact_end_size"):
        TOOL["resume_manifest"](tmp_path, dict(arguments, contact_end_size=0.03125), "source")
    with pytest.raises(ValueError, match="same FEM runner source"):
        TOOL["resume_manifest"](tmp_path, arguments, "modified")
    assert {q.name: q.read_bytes() for q in tmp_path.iterdir()} == before
    (tmp_path / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="completed FEM run"):
        TOOL["resume_manifest"](tmp_path, arguments, "source")


def test_invocation_accounting_includes_interrupted_work_and_rejects_missing_end(tmp_path):
    event = TOOL["invocation_event"]
    for identity, status in [("first", "checkpointed"), ("second", "completed")]:
        event(tmp_path, dict(event="started", id=identity))
        event(
            tmp_path,
            dict(
                event="ended",
                id=identity,
                status=status,
                preparation_seconds=3,
                simulation_seconds=10,
                simulation_cpu_seconds=25,
                total_seconds=13,
                total_cpu_seconds=28,
            ),
        )
    totals = TOOL["invocation_totals"](tmp_path)
    assert totals["all_invocations_accounted"]
    assert totals["simulation_seconds"] == 20
    assert totals["total_seconds"] == 26
    event(tmp_path, dict(event="started", id="killed_without_end_record"))
    missing = TOOL["invocation_totals"](tmp_path)
    assert not missing["all_invocations_accounted"]
    assert missing["simulation_seconds"] is None


def test_fem_uses_authored_resolved_precision_instead_of_stale_statistics(tmp_path):
    reference, expected = reference_inputs(tmp_path)
    actual = TOOL["load_reference_input"](reference, revision="test-revision")
    assert actual == expected
    assert actual["solver"]["dtype"] == "float64"
    assert (
        json.loads((reference / "stats/summary.json").read_text())["summary"]["dtype"] == "float32"
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ("toml", "fingerprint"),
        ("revision", "revision"),
        ("dtype", "explicitly"),
        ("precision_mismatch", "precision differ"),
    ],
)
def test_fem_rejects_unverified_resolved_inputs_before_building_a_model(tmp_path, change, match):
    reference, resolved = reference_inputs(tmp_path)
    if change == "toml":
        (reference / "input/case.toml").write_text('[solver]\ndtype = "float32"\n')
    elif change == "revision":
        resolved["git_revision"] = "another-revision"
    elif change == "dtype":
        resolved["solver"].pop("dtype")
    else:
        resolved["solver"]["dtype"] = "float32"
    (reference / "input/resolved.json").write_text(json.dumps(resolved))
    with pytest.raises(ValueError, match=match):
        TOOL["load_reference_input"](reference, revision="test-revision")


def test_static_restart_seed_values_are_fingerprinted(tmp_path):
    import h5py

    path = tmp_path / "restart.h5"
    with h5py.File(path, "w") as stream:
        group = stream.create_group("final_state")
        group["plastic_slip"] = [1.0, 2.0]
        group["u"] = [0.0, 0.0, 0.0, 0.0]
        group.attrs["applied_shear_displacement"] = 10.806226665707777
    before = TOOL["restart_seed_record"](path)
    assert before["checked_u_shape"] == [4]
    assert before["plastic_shape"] == [2]
    with h5py.File(path, "r+") as stream:
        stream["final_state/plastic_slip"][0] += 0.01
    after = TOOL["restart_seed_record"](path)
    assert before["seed_sha256"] != after["seed_sha256"]


def test_refined_mesh_seed_uses_original_float64_values_and_actuator(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "seed.h5"
    source = np.array([9.0000000123, 9.123456789, 9.9000000456])
    with h5py.File(path, "w") as stream:
        group = stream.create_group("final_state")
        group["plastic_slip"] = source
        group["u"] = np.zeros(12)
        group.attrs["applied_shear_displacement"] = 10.123456789123
        stream["interface/contact_line_y"] = np.array([0, 2, 4], dtype=np.float32)
    before = path.stat()
    record = TOOL["restart_seed_record"](path)
    values, actuator, projection = TOOL["project_restart_seed"](path, [0, 1, 2, 3, 4], record)
    np.testing.assert_array_equal(values[[0, 2, 4]], source)
    np.testing.assert_allclose(values[[1, 3]], (source[:-1] + source[1:]) / 2, rtol=0, atol=2e-15)
    assert values[0] != float(np.float32(source[0]))
    assert actuator == record["actuator_mm"]
    assert projection["source_plastic_dtype"] == "float64"
    assert projection["target_nodes"] == 5
    assert (before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns)
    for target in [[-1, 2, 4], [0, 2, 5], [0, 2, 2, 4]]:
        with pytest.raises(ValueError, match="contact interval"):
            TOOL["project_restart_seed"](path, target, record)
    with h5py.File(path, "r+") as stream:
        stream["final_state/plastic_slip"][0] += 1e-5
    with pytest.raises(ValueError, match="recorded source"):
        TOOL["project_restart_seed"](path, [0, 1, 2, 3, 4], record)

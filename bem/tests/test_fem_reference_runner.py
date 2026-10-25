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

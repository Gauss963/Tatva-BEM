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

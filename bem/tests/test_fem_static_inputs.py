import runpy
import sys
from pathlib import Path

import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/fem_static_onset.py"))


def test_static_default_uses_verified_authored_precision_and_preserves_historical_control():
    authored = TOOL["static_inputs"]()
    control = TOOL["static_inputs"](historical=True)
    assert authored["solver"]["dtype"] == "float64"
    assert control["solver"]["dtype"] == "float32"
    assert not authored["historical_control"] and control["historical_control"]
    for key in ["moving", "stationary", "materials", "friction", "simulation"]:
        assert authored["case"][key] == control["case"][key]
    assert len(authored["input_sha256"]) == 2
    assert len(control["input_sha256"]) == 1


@pytest.mark.parametrize("meshes", [["0"], ["nan"], ["-1"]])
def test_invalid_static_mesh_rejected_before_loading_fem_dependencies(
    tmp_path, monkeypatch, meshes
):
    main = TOOL["main"]
    monkeypatch.setitem(main.__globals__, "ROOT", tmp_path)
    output = tmp_path / "outputs/new"
    monkeypatch.setattr(sys, "argv", ["static", "--output", str(output), "--meshes", *meshes])
    with pytest.raises(ValueError, match="positive mesh"):
        main()
    assert not output.exists()


def test_existing_static_artifacts_are_not_overwritten(tmp_path, monkeypatch):
    main = TOOL["main"]
    monkeypatch.setitem(main.__globals__, "ROOT", tmp_path)
    output = tmp_path / "outputs/existing"
    output.mkdir(parents=True)
    marker = output / "summary.json"
    marker.write_text("original")
    monkeypatch.setattr(sys, "argv", ["static", "--output", str(output)])
    with pytest.raises(ValueError, match="new empty"):
        main()
    assert marker.read_text() == "original"

import json
import runpy
import sys
from pathlib import Path


def test_primitive_screen_uses_requested_frequency_grid_without_claiming_full_bem(
    tmp_path, monkeypatch
):
    tool = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools/screen_raw_interpolation_degree.py")
    )
    output = tmp_path / "screen"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen",
            "--dt",
            "2e-6",
            "--fft-size",
            "32",
            "--radius",
            ".9",
            "--degrees",
            "8",
            "16",
            "--output",
            str(output),
        ],
    )
    tool["main"]()
    report = json.loads((output / "summary.json").read_text())
    assert report["frequency_grid"] == dict(dt_s=2e-6, fft_size=32, radius=0.9)
    assert not report["full_bem_accuracy_or_speed_verified"]
    assert [r["degree"] for r in report["records"]] == [8, 16]
    assert all(
        r["source_evaluations"] == 17 and r["all_fft_grid_points_emitted"]
        for r in report["records"]
    )

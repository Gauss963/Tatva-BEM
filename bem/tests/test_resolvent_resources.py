import runpy
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ["simulate_resolvent.py", "adjoint_resolvent.py"])
@pytest.mark.parametrize("budget", ["nan", "inf", "0"])
def test_invalid_memory_budget_rejected_before_reading_run(tmp_path, monkeypatch, script, budget):
    if script == "adjoint_resolvent.py":
        pytest.importorskip("jax")
    path = Path(__file__).resolve().parents[1] / "tools" / script
    namespace = runpy.run_path(str(path))
    args = [str(path), "missing-run", "--max-memory-mb", budget, "--fft-cache-mb", "0"]
    if script == "simulate_resolvent.py":
        args.extend(["--output", str(tmp_path / "result")])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="finite positive memory budget"):
        namespace["main"]()
    assert not (tmp_path / "result").exists()

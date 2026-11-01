"""Complete independent CQ/startup control and explicit solver-policy lineage."""

import json
import runpy
import sys
from pathlib import Path
from types import FunctionType

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
TOOLS = Path(__file__).resolve().parents[1] / "tools"
PRODUCER = runpy.run_path(str(TOOLS / "prepare_trace_policy_radau.py"))
FIXTURE = runpy.run_path(str(Path(__file__).with_name("test_raw_radau_producer.py")))


@pytest.mark.parametrize("layout", ["c", "fortran"])
def test_complete_general_lazy_kernel_against_independent_dense_original(
    tmp_path, monkeypatch, layout
):
    private = PRODUCER["private_producer"]()
    original_main = private["main"]

    def durable():
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--durable-build"])
        original_main()

    original = FIXTURE[
        "test_complete_fresh_kernel_weights_and_causal_startup_match_dense_original_bem"
    ]
    run = FunctionType(
        original.__code__, {**original.__globals__, "PRODUCER": {**private, "main": durable}}
    )
    run(tmp_path, monkeypatch, layout, False)
    metadata = json.loads((tmp_path / "fresh/summary.json").read_text())
    assert metadata["jax_backend"]["backend"] == PRODUCER["BACKEND"]
    assert metadata["jax_backend"]["direct_startup_keeps_original_unmodified_transfer"]
    with pytest.raises(ValueError, match="source, runtime or policy"):
        PRODUCER["JAX"]["verify_context"](tmp_path / "fresh")
    PRODUCER["verify_context"](tmp_path / "fresh")

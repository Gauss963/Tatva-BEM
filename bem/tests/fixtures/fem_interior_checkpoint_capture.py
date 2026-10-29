"""Test-only copy of original checkpoints before the source deletes them."""

import runpy
import shutil
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

root = Path(__file__).resolve().parents[3]
output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
original = runpy.run_path


def load(path, *args, **kwargs):
    records = original(path, *args, **kwargs)
    if Path(path).resolve() == root / "bem/tools/fem_interior_observations.py":
        baseline = records["make_observed_runner"]

        def make(function, *positional, **keywords):
            writer = function.__globals__["_write_simulation_checkpoint"]

            def copy_writer(path, carry, metadata):
                writer(path, carry, metadata)
                shutil.copy2(path, output / "retained_checkpoint.npz")

            rebound = FunctionType(
                function.__code__,
                {**function.__globals__, "_write_simulation_checkpoint": copy_writer},
                function.__name__,
                function.__defaults__,
                function.__closure__,
            )
            rebound.__kwdefaults__ = function.__kwdefaults__
            rebound.__annotations__ = function.__annotations__
            return baseline(rebound, *positional, **keywords)

        records = {**records, "make_observed_runner": make}
    return records


runner = original(str(root / "bem/tools/run_fem_interior_reference.py"))
runner["main"].__globals__["runpy"] = SimpleNamespace(run_path=load)
runner["main"]()

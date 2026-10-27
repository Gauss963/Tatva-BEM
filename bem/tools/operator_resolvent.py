"""Private native-kernel bindings shared by experimental bulk operators.

The ordinary producer and numerical package are left unchanged. Operator
sources, representation provenance and extra construction arrays are included
in the same completion/source checks as the original native producer.
"""

import runpy
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

from finiteblock_bem.io import fingerprint


def prepare_operator(factory, definition, source_paths, arguments, *, extra_workspace=None):
    tools = Path(__file__).resolve().parent
    producer_path = tools / "prepare_native_resolvent.py"
    paths = [Path(__file__).resolve(), producer_path, *map(Path, source_paths)]
    frozen = {str(path.resolve()): fingerprint(path) for path in paths}
    producer = runpy.run_path(str(producer_path))

    def check():
        if any(fingerprint(path) != value for path, value in frozen.items()):
            raise ValueError("Experimental operator, wrapper or base producer changed")

    def transfer(model, *, quadrature):
        check()
        return factory(model, quadrature=quadrature)

    original_plan = producer["plan_nodal_history"]

    def plan(model, *args, **kwargs):
        result, estimate = original_plan(model, *args, **kwargs)
        additional = 0 if extra_workspace is None else extra_workspace(model)
        if not isinstance(additional, int) or additional < 0:
            raise ValueError("Experimental array estimate must be a nonnegative integer")
        definition["additional_construction_workspace_bytes"] = additional
        return result, estimate + additional

    def load(path):
        namespace = runpy.run_path(str(path))
        if Path(path).resolve() != tools / "native_spectrum.py":
            return namespace
        original = namespace["native_parsimonious_weights"]

        def weights(*args, **kwargs):
            prior = kwargs.get("source_check")

            def source_check():
                if prior is not None:
                    prior()
                check()

            kwargs["source_check"] = source_check
            kwargs["provenance"] = {
                **kwargs.get("provenance", {}),
                "experimental_operator": dict(definition),
            }
            return original(*args, **kwargs)

        return {**namespace, "native_parsimonious_weights": weights}

    original_write = producer["write_json"]

    def write(path, record):
        check()
        if record.get("status") == "completed_resolvent_kernel_only":
            record.update(experimental_operator=dict(definition))
            record["frozen_inputs_sha256"].update(frozen)
        return original_write(path, record)

    original = producer["main"]
    clone = FunctionType(
        original.__code__,
        {
            **original.__globals__,
            "NodalInterfaceTransfer": transfer,
            "plan_nodal_history": plan,
            "runpy": SimpleNamespace(run_path=load),
            "write_json": write,
        },
        original.__name__,
        original.__defaults__,
    )
    clone.__kwdefaults__ = original.__kwdefaults__
    previous = sys.argv
    try:
        sys.argv = [previous[0], *arguments]
        clone()
    finally:
        sys.argv = previous


def prepare_raw_operator(factory, definition, source_paths, arguments):
    """Bind a Schur factory only in the fresh raw producer's private namespace.

    The original conservative allowance of 32 matrices per worker is retained.
    The supported trace variant uses smaller elimination blocks; no tighter
    RSS or memory estimate is inferred from this binding.
    """
    tools = Path(__file__).resolve().parent
    producer_path = tools / "prepare_raw_layer_resolvent.py"
    paths = [Path(__file__).resolve(), producer_path, *map(Path, source_paths)]
    frozen = {str(path.resolve()): fingerprint(path) for path in paths}
    producer = runpy.run_path(str(producer_path))

    def check():
        if any(fingerprint(path) != value for path, value in frozen.items()):
            raise ValueError("Experimental raw operator, wrapper or base producer changed")

    def transfer(model, *, quadrature):
        check()
        return factory(model, quadrature=quadrature)

    def load(path):
        namespace = runpy.run_path(str(path))
        if Path(path).resolve() == tools / "schur_condensation_experiment.py":
            return {**namespace, "ConstrainedSchurTransfer": transfer}
        return namespace

    original_write = producer["write_json"]

    def write(path, record):
        check()
        keys = [k for k in ["frozen_input_sha256", "frozen_inputs_sha256"] if k in record]
        if keys:
            record.update(experimental_operator=dict(definition))
            for key in keys:
                record[key].update(frozen)
        if record.get("status") == "completed_resolvent_kernel_only":
            record["assembly"] = definition["method"]
        return original_write(path, record)

    original = producer["main"]
    clone = FunctionType(
        original.__code__,
        {**original.__globals__, "runpy": SimpleNamespace(run_path=load), "write_json": write},
        original.__name__,
        original.__defaults__,
    )
    clone.__kwdefaults__ = original.__kwdefaults__
    previous = sys.argv
    try:
        sys.argv = [previous[0], *arguments]
        clone()
        check()
    finally:
        sys.argv = previous

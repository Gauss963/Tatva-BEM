"""Prepare a native resolvent using isolated fixed geometric quadrature.

All ordinary options are forwarded to prepare_native_resolvent.py. A private
factory supplies the experimental bulk operator; package/module globals are
unchanged. Both the frequency representation and final kernel record the
operator definition and frozen source hashes. Full trajectory/AD comparison
is still required before this can replace the original quadrature.
"""

import argparse
import runpy
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

from finiteblock_bem.io import fingerprint


def main():
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--far-order", type=int, default=8)
    parser.add_argument("--separation", type=float, default=2)
    args, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        print(parser.format_help())
    if args.far_order < 4 or args.far_order % 2:
        raise ValueError("Far order must be even and at least 4")
    if not np.isfinite(args.separation) or args.separation <= 0:
        raise ValueError("Finite positive geometric separation required")
    tools = Path(__file__).resolve().parent
    producer_path = tools / "prepare_native_resolvent.py"
    helper_path = tools / "mixed_quadrature_experiment.py"
    frozen = {
        str(path): fingerprint(path)
        for path in [Path(__file__).resolve(), producer_path, helper_path]
    }
    producer = runpy.run_path(str(producer_path))
    helper = runpy.run_path(str(helper_path))
    geometry = []
    definition = dict(
        method="fixed_geometric_galerkin_quadrature",
        far_order=args.far_order,
        separation=args.separation,
        near_order="source mesh arguments quadrature",
        pair_selection_depends_on_frequency=False,
        experimental_operator_sha256=frozen[str(helper_path)],
        experimental_wrapper_sha256=frozen[str(Path(__file__).resolve())],
    )

    def check():
        if any(fingerprint(path) != value for path, value in frozen.items()):
            raise ValueError("Mixed operator, wrapper or base producer changed")

    def transfer(model, *, quadrature):
        check()
        operator = helper["mixed_transfer"](
            model, near_order=quadrature, far_order=args.far_order, separation=args.separation
        )
        definition["near_order"] = quadrature
        geometry.extend(body.statistics() for body in operator.prepared)
        return operator

    def load(path):
        namespace = runpy.run_path(str(path))
        if Path(path).resolve() != tools / "native_spectrum.py":
            return namespace
        original = namespace["native_parsimonious_weights"]

        def weights(*arguments, **kwargs):
            prior = kwargs.get("source_check")

            def source_check():
                if prior is not None:
                    prior()
                check()

            kwargs["source_check"] = source_check
            kwargs["provenance"] = {
                **kwargs.get("provenance", {}),
                "experimental_operator": {**definition, "geometry": geometry},
            }
            return original(*arguments, **kwargs)

        return {**namespace, "native_parsimonious_weights": weights}

    original_write = producer["write_json"]

    def write(path, record):
        check()
        if record.get("status") == "completed_resolvent_kernel_only":
            record.update(experimental_operator={**definition, "geometry": geometry})
            record["frozen_inputs_sha256"].update(frozen)
        return original_write(path, record)

    original_main = producer["main"]
    clone = FunctionType(
        original_main.__code__,
        {
            **original_main.__globals__,
            "NodalInterfaceTransfer": transfer,
            "runpy": SimpleNamespace(run_path=load),
            "write_json": write,
        },
        original_main.__name__,
        original_main.__defaults__,
    )
    clone.__kwdefaults__ = original_main.__kwdefaults__
    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        clone()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()

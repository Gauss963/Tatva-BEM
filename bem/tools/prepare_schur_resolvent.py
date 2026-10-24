"""Build a native resolvent using fixed-jump Schur condensation.

Other arguments are forwarded to prepare_native_resolvent.py. This is an
experimental operator; full trajectory, branch and adjoint checks are required.
"""

import argparse
import runpy
from pathlib import Path

from finiteblock_bem.io import fingerprint


def main():
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--formulation", choices=["joint", "trace"], default="trace")
    args, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        print(parser.format_help())
    tools = Path(__file__).resolve().parent
    helper = tools / (
        "schur_condensation_experiment.py"
        if args.formulation == "joint"
        else "trace_schur_experiment.py"
    )
    paths = {Path(__file__), helper, tools / "schur_condensation_experiment.py"}
    name = "ConstrainedSchurTransfer" if args.formulation == "joint" else "TraceSchurTransfer"
    operator = runpy.run_path(str(helper))[name]
    definition = dict(
        method="fixed_jump_galerkin_schur",
        formulation=args.formulation,
        quadrature="source mesh arguments quadrature",
        basis_depends_on_frequency=False,
        source_sha256={str(path.resolve()): fingerprint(path) for path in paths},
    )

    def factory(model, *, quadrature):
        transfer = operator(model, quadrature=quadrature)
        m, d = transfer.constraints.size, transfer.dofs
        definition.update(quadrature=quadrature, full_reduced_dofs=m, retained_dofs=d)
        if args.formulation == "trace":
            definition.update(
                trace_dofs=len(transfer.trace_indices),
                body_interior_dofs=[len(ids) for ids in transfer.body_interiors],
                common_trace_elimination_dofs=transfer.trace_nullspace.shape[1],
            )
        return transfer

    def extra_workspace(model):
        # Conservative allowance for fixed QR bases, supported matrix,
        # projected products and recovery maps, beyond the original producer.
        m, d = model.constraints.size, 2 * len(model.moving_contact) + 1
        return 16 * (12 * m * m + 8 * m * d)

    runpy.run_path(str(tools / "operator_resolvent.py"))["prepare_operator"](
        factory, definition, paths, remaining, extra_workspace=extra_workspace
    )


if __name__ == "__main__":
    main()

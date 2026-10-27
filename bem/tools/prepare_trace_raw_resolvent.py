"""Fresh raw-layer CQ with two-stage trace Schur elimination.

All CLI arguments are passed to prepare_raw_layer_resolvent.py. Only the
supported body elimination changes; sampling checks, complete weights, startup,
storage and budgets retain the original producer. This is an explicit experiment.
"""

import runpy
import sys
from pathlib import Path

from finiteblock_bem.io import fingerprint


def main():
    tools = Path(__file__).resolve().parent
    helper = tools / "trace_schur_experiment.py"
    paths = [Path(__file__), helper, tools / "schur_condensation_experiment.py"]
    operator = runpy.run_path(str(helper))["TraceSchurTransfer"]
    definition = dict(
        method="raw_layers_before_two_stage_trace_schur",
        formulation="trace",
        basis_depends_on_frequency=False,
        raw_quadrature_or_sampling_changed=False,
        time_history_or_contact_modes_removed=False,
        construction_allowance="Original conservative raw-producer matrix allowance retained",
        source_sha256={str(p.resolve()): fingerprint(p) for p in paths},
    )

    def factory(model, *, quadrature):
        transfer = operator(model, quadrature=quadrature)
        definition.update(
            full_reduced_dofs=transfer.constraints.size,
            retained_dofs=transfer.dofs,
            trace_dofs=len(transfer.trace_indices),
            body_interior_dofs=[len(ids) for ids in transfer.body_interiors],
            common_trace_elimination_dofs=transfer.trace_nullspace.shape[1],
        )
        return transfer

    runpy.run_path(str(tools / "operator_resolvent.py"))["prepare_raw_operator"](
        factory, definition, paths, sys.argv[1:]
    )


if __name__ == "__main__":
    main()

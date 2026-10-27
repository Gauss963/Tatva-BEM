"""Fresh raw-layer CQ with two-stage trace Schur elimination.

All CLI arguments are passed to prepare_raw_layer_resolvent.py. Only the
supported body elimination changes; sampling checks, complete weights, startup,
storage and budgets retain the original producer. This is an explicit experiment.
"""

import argparse
import math
import runpy
from pathlib import Path

from finiteblock_bem.io import fingerprint


def main():
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--primitive-partition", action="store_true")
    parser.add_argument("--partition-distances", type=float, nargs="+")
    args, remaining = parser.parse_known_args()
    if args.partition_distances is not None and not args.primitive_partition:
        raise ValueError("Proxy distances require the explicit primitive-partition experiment")
    if "--help" in remaining or "-h" in remaining:
        print(parser.format_help())
    tools = Path(__file__).resolve().parent
    helper = tools / "trace_schur_experiment.py"
    paths = [Path(__file__), helper, tools / "schur_condensation_experiment.py"]
    seeded = None
    distances = None
    if args.primitive_partition:
        paths.append(tools / "seeded_raw_sampling.py")
        seeded = runpy.run_path(str(paths[-1]))
        distances = list(
            seeded["DEFAULT_DISTANCES"]
            if args.partition_distances is None
            else args.partition_distances
        )
        if not 1 <= len(distances) <= 64 or any(not math.isfinite(v) or v <= 0 for v in distances):
            raise ValueError("Advisory distances must contain 1-64 finite positive values")
    operator = runpy.run_path(str(helper))["TraceSchurTransfer"]
    definition = dict(
        method="raw_layers_before_partitioned_two_stage_trace_schur"
        if args.primitive_partition
        else "raw_layers_before_two_stage_trace_schur",
        formulation="trace",
        basis_depends_on_frequency=False,
        raw_quadrature_changed=False,
        sampling_acceptance_criteria_changed=False,
        initial_sampling_partition_changed=args.primitive_partition,
        initial_partition="advisory_scalar_green"
        if args.primitive_partition
        else "global_bisection",
        proxy_distances_mm=distances,
        time_history_or_contact_modes_removed=False,
        construction_allowance="Original conservative raw-producer matrix allowance retained",
        source_sha256={str(p.resolve()): fingerprint(p) for p in paths},
    )
    material = []

    def factory(model, *, quadrature):
        transfer = operator(model, quadrature=quadrature)
        material.append(model.reference.material)
        definition.update(
            full_reduced_dofs=transfer.constraints.size,
            retained_dofs=transfer.dofs,
            trace_dofs=len(transfer.trace_indices),
            body_interior_dofs=[len(ids) for ids in transfer.body_interiors],
            common_trace_elimination_dofs=transfer.trace_nullspace.shape[1],
        )
        return transfer

    def sampling_factory(original):
        def sample(*positional, **keywords):
            if len(material) != 1:
                raise ValueError("Partition proxy requires the actual constructed body's material")
            record = seeded["sample_with_primitive_partition"](
                original, *positional, material=material[0], distances=distances, **keywords
            )
            definition["proxy_partition_intervals"] = len(record["initial_intervals"])
            definition["proxy_source_evaluations"] = record["primitive_partition_screen"][
                "records"
            ]["source_evaluations"]
            return record

        return sample

    runpy.run_path(str(tools / "operator_resolvent.py"))["prepare_raw_operator"](
        factory,
        definition,
        paths,
        remaining,
        sampling_factory=sampling_factory if args.primitive_partition else None,
    )


if __name__ == "__main__":
    main()

"""Independent BEM endpoint static refinement at unchanged physical parameters."""

import argparse
import json
import resource
import runpy
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.friction import ContactState
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model, nodal_body_stiffness

TOOLS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_kernel", type=Path, help="Physical case and numerical defaults only"
    )
    parser.add_argument("--restart-seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-h", type=float, default=10)
    parser.add_argument("--contact-h", type=float, default=2)
    parser.add_argument("--end-sizes", type=float, nargs="+", default=[1, 0.5, 0.25, 0.125])
    parser.add_argument("--quadrature", type=int)
    parser.add_argument("--fem-studies", type=Path, nargs="*", default=[])
    parser.add_argument("--max-boundary-dofs", type=int, default=5000)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a fresh static output outside protected Tatva")
    if (
        not all(np.isfinite(v) and v > 0 for v in [args.outer_h, args.contact_h, *args.end_sizes])
        or any(v > min(args.outer_h, args.contact_h) for v in args.end_sizes)
        or len(set(args.end_sizes)) != len(args.end_sizes)
        or args.max_boundary_dofs < 1
    ):
        raise ValueError("Require positive mesh sizes and distinct smaller endpoint sizes")
    source_path = args.source_kernel / "summary.json"
    source = json.loads(source_path.read_text())
    if source["status"] != "completed_resolvent_kernel_only":
        raise ValueError("Require completed source metadata; frequency/weight files are not read")
    options = source["source_mesh_arguments"]
    q = options["quadrature"] if args.quadrature is None else args.quadrature
    if q < 4 or q % 2:
        raise ValueError("Require even quadrature of at least four")
    names = [
        "refined_contact.py",
        "fem_static_contact.py",
        "restart_seed_experiment.py",
        "compare_fem_mesh.py",
    ]
    helpers = {n: runpy.run_path(str(TOOLS / n)) for n in names}
    paths = [
        Path(__file__),
        source_path,
        args.restart_seed,
        *[TOOLS / n for n in names],
        *[p / "summary.json" for p in args.fem_studies],
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    environment = environment_record(options["case"])
    for key in ["source_sha256", "case_sha256", "restart_data_sha256"]:
        if source["environment"][key] != environment[key]:
            raise ValueError("Physical case or BEM package changed")
    fem = []
    for folder in args.fem_studies:
        report = json.loads((folder / "summary.json").read_text())
        if report["status"] != "completed_graded_fem_static_controls":
            raise ValueError("Require completed independent FEM static controls")
        if report["frozen_input_sha256"][str(args.restart_seed.resolve())] != fingerprint(
            args.restart_seed
        ):
            raise ValueError("FEM and BEM source restart fields differ")
        for row in report["records"]:
            p = Path(row["snapshot"])
            if fingerprint(p) != row["snapshot_sha256"]:
                raise ValueError("FEM static snapshot changed")
            frozen[str(p.resolve())] = row["snapshot_sha256"]
            fem.append((row, p))
    output.mkdir(parents=True)
    write_json(
        output / "run_manifest.json",
        dict(
            status="prepared_not_completed",
            frozen_input_sha256=frozen,
            arguments={
                k: str(v)
                if isinstance(v, Path)
                else [str(x) for x in v]
                if k == "fem_studies"
                else v
                for k, v in vars(args).items()
            },
            source_role="Physical/numerical metadata only; fresh static BEM assembly",
            original_tatva_readonly=True,
        ),
    )
    begun = perf_counter()
    records = []
    fields = []
    compare = helpers["compare_fem_mesh.py"]["error_report"]
    with threadpool_limits(limits=1, user_api="blas"):
        for fine in sorted(args.end_sizes, reverse=True):
            start = perf_counter()
            effective = dict(
                options,
                h=args.outer_h,
                contact_h=args.contact_h,
                contact_end_h=fine,
                corner_h=fine,
                quadrature=q,
                restart_knot_window=None,
                restart_knot_subdivisions=1,
            )
            model = load_nodal_model(
                options["case"],
                max_panel_length=args.outer_h,
                contact_panel_length=args.contact_h,
                contact_end_length=fine,
                corner_panel_length=fine,
                restart_knot_window=None,
                restart_knot_subdivisions=1,
            )
            if (
                model.reference.moving.dofs + model.reference.stationary.dofs
                > args.max_boundary_dofs
            ):
                raise ValueError("BEM boundary DOFs exceed the requested static study limit")
            model, seed_record = helpers["restart_seed_experiment.py"]["apply_restart_seed"](
                model, args.restart_seed
            )
            y = model.reference.moving.starts[model.moving_contact, 1]
            space = helpers["refined_contact.py"]["ContactQuadrature"].build(y, 1)
            with np.load(args.restart_seed, allow_pickle=False) as seed:
                state = ContactState(
                    np.interp(space.y, seed["y"], seed["plastic_slip"]),
                    np.interp(space.y, seed["y"], seed["cumulative_slip"]),
                )
                projection = compare(
                    seed["y"], seed["plastic_slip"][None], space.y, state.plastic_slip[None]
                )
            stiffness = nodal_body_stiffness(model, quadrature=q)
            reduction = helpers["refined_contact.py"]["reduce_static_trace"](model, stiffness)
            lift = helpers["fem_static_contact.py"]["initialize_fem_static_contact"](
                model, space, reduction, state
            )
            law = model.reference.contact
            comparisons = []
            for row, path in fem:
                if (
                    row["config"]["normal_penalty"] != law.normal_penalty
                    or row["config"]["tangential_penalty"] != law.tangential_penalty
                ):
                    raise ValueError("Physical penalties differ from the FEM reference")
                with np.load(path, allow_pickle=False) as target:
                    comparisons.append(
                        dict(
                            fem_snapshot=str(path),
                            fem_mesh_size=row["mesh_size"],
                            fem_end_size=row["end_size"],
                            pressure=compare(
                                target["y"], target["pressure"][None], space.y, lift.pressure[None]
                            ),
                            shear=compare(
                                target["y"], target["shear"][None], space.y, lift.shear[None]
                            ),
                        )
                    )
            name = f"end-{fine:g}"
            target = output / name
            target.mkdir()
            np.savez(
                target / "static.npz",
                y=space.y,
                pressure=lift.pressure,
                shear=lift.shear,
                plastic_slip=lift.state.plastic_slip,
                cumulative_slip=lift.state.cumulative_slip,
                retained_displacement=lift.retained_displacement,
                full_displacement=lift.full_displacement,
            )
            record = dict(
                name=name,
                source_mesh_arguments=effective,
                environment=environment,
                panels=[model.reference.moving.count, model.reference.stationary.count],
                contact_nodes=len(y),
                initialization=lift.diagnostics,
                restart_seed=seed_record,
                source_p1_projection=projection,
                maximum_fixed_jump_trial_shear_projection_change_mpa=law.tangential_penalty
                * projection["maximum_saved_absolute_difference"],
                maximum_pressure_mpa=float(np.max(lift.pressure)),
                maximum_shear_mpa=float(np.max(lift.shear)),
                comparisons=comparisons,
                static_seconds=perf_counter() - start,
                snapshot=str(target / "static.npz"),
                snapshot_sha256=fingerprint(target / "static.npz"),
                status="completed_static_mesh_control_not_a_dynamic_kernel",
            )
            write_json(target / "summary.json", record)
            records.append(record)
            fields.append((space.y.copy(), lift.pressure.copy(), lift.shear.copy()))
            write_json(output / "progress.json", dict(records=records))
            print(
                json.dumps(
                    dict(
                        name=name,
                        seconds=record["static_seconds"],
                        pressure=record["maximum_pressure_mpa"],
                        shear=record["maximum_shear_mpa"],
                    )
                ),
                flush=True,
            )
            del stiffness, reduction, lift, model, space
    successive = []
    for i in range(1, len(fields)):
        ay, ap, at = fields[i - 1]
        by, bp, bt = fields[i]
        successive.append(
            dict(
                coarser=records[i - 1]["name"],
                finer=records[i]["name"],
                pressure=compare(by, bp[None], ay, ap[None]),
                shear=compare(by, bt[None], ay, at[None]),
            )
        )
    if any(fingerprint(p) != v for p, v in frozen.items()):
        raise ValueError("BEM static study implementation or inputs changed")
    current = environment_record(options["case"])
    if any(
        current[k] != environment[k]
        for k in ["source_sha256", "case_sha256", "restart_data_sha256"]
    ):
        raise ValueError("BEM package or physical inputs changed")
    write_json(
        output / "summary.json",
        dict(
            status="completed_graded_bem_static_controls",
            records=records,
            successive_refinement=successive,
            frozen_input_sha256=frozen,
            total_seconds=perf_counter() - begun,
            peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2 if sys.platform == "darwin" else 1024),
            physical_convergence_verified=False,
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
            scope="Independent static endpoint/corner refinement at fixed contact/outer scale. Source P1 projection errors recorded; no transient or continuum certificate.",
        ),
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

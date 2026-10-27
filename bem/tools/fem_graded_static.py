"""Resolve FEM contact-end spatial error without transient integration.

Uses the pinned original constitutive/static kernels with an explicitly
recorded chamfer mesh adapter, original float64 restart P1 field and healed
Coulomb policy. Results do not certify dynamic convergence or BEM agreement.
"""

import argparse
import gc
import json
import os
import resource
import runpy
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.io import fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart-seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-size", type=float, default=2)
    parser.add_argument("--end-sizes", type=float, nargs="+", default=[1, 0.5, 0.25, 0.125])
    parser.add_argument("--growth", type=float, default=1.2)
    parser.add_argument("--include-unrefined", action="store_true")
    parser.add_argument("--max-dofs", type=int, default=500000)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a fresh static-study output outside protected Tatva")
    if (
        not np.isfinite(args.mesh_size)
        or args.mesh_size <= 0
        or not np.isfinite(args.growth)
        or args.growth <= 1
        or len(set(args.end_sizes)) != len(args.end_sizes)
        or any(not np.isfinite(v) or not 0 < v < args.mesh_size for v in args.end_sizes)
        or args.max_dofs < 1
    ):
        raise ValueError("Require positive mesh sizes, smaller distinct end sizes and growth > 1")
    os.environ["JAX_ENABLE_X64"] = "true"
    sys.dont_write_bytecode = True
    static = runpy.run_path(str(TOOLS / "fem_static_onset.py"))
    inputs = static["static_inputs"]()
    if inputs["solver"]["dtype"] != "float64":
        raise ValueError("Require the authored float64 FEM inputs")
    archive = static["REFERENCE_TOOL"]["stage_snapshot"](
        ROOT.parent, ROOT / "outputs/fem-source" / static["REVISION"]
    )
    snapshot = ROOT / "outputs/fem-source" / static["REVISION"]
    sys.path.insert(0, str(snapshot))
    from tatva.pmma import dynamics, static_normal
    from tatva.pmma.model import (
        BlockSpec,
        FrictionReference,
        LoadingReference,
        Material,
        PMMAModelInput,
    )

    legacy = inputs["case"]
    case = PMMAModelInput(
        BlockSpec(**legacy["moving"]),
        BlockSpec(**legacy["stationary"]),
        {k: Material(**v) for k, v in legacy["materials"].items()},
        FrictionReference(**legacy["friction"]),
        LoadingReference(**legacy["simulation"]),
    )
    adapter = runpy.run_path(str(TOOLS / "chamfer_mesh_refinement.py"))
    builder = adapter["make_chamfer_refined_builder"](dynamics)
    frozen = {
        str(p.resolve()): fingerprint(p)
        for p in [
            Path(__file__),
            TOOLS / "chamfer_mesh_refinement.py",
            TOOLS / "fem_static_onset.py",
            TOOLS / "run_fem_reference.py",
            TOOLS / "compare_fem_mesh.py",
            args.restart_seed,
        ]
    }
    frozen.update(inputs["input_sha256"])
    with np.load(args.restart_seed, allow_pickle=False) as data:
        if set(data.files) != {"y", "plastic_slip", "cumulative_slip", "actuator"}:
            raise ValueError("Require the captured complete interface seed schema")
        source = {k: data[k] for k in data.files}
    if any(v.dtype != np.float64 or not np.all(np.isfinite(v)) for v in source.values()):
        raise ValueError("Restart input must be finite float64")
    y, z, c, actuator = (source[k] for k in ["y", "plastic_slip", "cumulative_slip", "actuator"])
    if (
        y.ndim != 1
        or len(y) < 2
        or z.shape != y.shape
        or c.shape != y.shape
        or np.any(np.diff(y) <= 0)
        or np.any(c != 0)
        or actuator.shape != ()
    ):
        raise ValueError("Require an ordered P1 plastic field and zero healed memory")
    output.mkdir(parents=True)
    write_json(
        output / "run_manifest.json",
        dict(
            status="prepared_not_completed",
            frozen_input_sha256=frozen,
            source_archive=archive,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            scope="Original FEM kernels with explicit mesh-only adapter; no transient solve",
            original_tatva_readonly=True,
        ),
    )
    begun = perf_counter()
    records, fields = [], []
    original_builder = dynamics.build_case_model
    for fine in ([None] if args.include_unrefined else []) + sorted(args.end_sizes, reverse=True):
        start = perf_counter()
        name = "unrefined" if fine is None else f"end-{fine:g}"
        config = replace(
            dynamics.RunConfig(**inputs["solver"]),
            mesh_size=args.mesh_size,
            restart_from_file=None,
            time_step_override=None,
            fault_end_refinement_size=fine,
            fault_end_refinement_growth=args.growth,
            output_prefix=str(output / name),
        )
        model = builder(case, config)
        total = 2 * (model["moving"].n_nodes + model["stationary"].n_nodes)
        if total > args.max_dofs:
            raise ValueError(f"FEM model has {total} DOFs, above the requested static study limit")
        coords = np.asarray(model["moving"].mesh.coords)
        yy = coords[np.asarray(model["master_nodes"]), 1]
        if yy[0] < y[0] or yy[-1] > y[-1] or len(yy) < 2 or np.any(np.diff(yy) <= 0):
            raise ValueError("Refined FEM contact does not align with the original seed interval")
        interpolated = np.interp(yy, y, z)
        preparation = perf_counter() - start
        answer = static_normal.solve_static_normal_phase(
            model, config, actuator_displacement=float(actuator), plastic_slip=interpolated
        )
        if answer["relative_residual"] > 1e-8 or any(
            not np.all(np.isfinite(answer[key])) for key in ["u", "sigma_n", "tau", "plastic_slip"]
        ):
            np.savez(
                output / "failed_static.npz",
                y=yy,
                full_displacement=answer["u"],
                pressure=answer["sigma_n"],
                shear=answer["tau"],
                plastic_slip=answer["plastic_slip"],
            )
            write_json(
                output / "failure.json",
                dict(
                    status="failed_original_fem_static_acceptance_gate",
                    mesh_size=args.mesh_size,
                    end_size=fine,
                    dofs=total,
                    contact_nodes=len(yy),
                    relative_force_residual=answer["relative_residual"],
                    required_relative_force_residual=1e-8,
                    snapshot=str(output / "failed_static.npz"),
                    snapshot_sha256=fingerprint(output / "failed_static.npz"),
                    frozen_input_sha256=frozen,
                    original_tolerance_relaxed=False,
                ),
            )
            raise ValueError("Original FEM static equilibrium did not pass its residual check")
        if dynamics.build_case_model is not original_builder:
            raise RuntimeError("Original FEM builder binding changed")
        target = output / name
        target.mkdir()
        # Tatva resets weakening memory at shear start; static plastic offsets
        # may change through its original healed Coulomb active set.
        np.savez(
            target / "static.npz",
            y=yy,
            pressure=answer["sigma_n"],
            shear=answer["tau"],
            plastic_slip=answer["plastic_slip"],
            cumulative_slip=np.zeros_like(yy),
            full_displacement=answer["u"],
            contact_weights=np.asarray(model["interface_weights"]),
        )
        record = dict(
            name=name,
            mesh_size=args.mesh_size,
            end_size=fine,
            dofs=total,
            contact_nodes=len(yy),
            config=asdict(config),
            relative_force_residual=answer["relative_residual"],
            static_sliding_nodes=answer["sliding_nodes"],
            static_open_nodes=answer["open_nodes"],
            maximum_plastic_offset_change_mm=float(
                np.max(abs(answer["plastic_slip"] - interpolated))
            ),
            minimum_contact_spacing_mm=float(np.min(np.diff(yy))),
            first_contact_spacing_mm=float(yy[1] - yy[0]),
            last_contact_spacing_mm=float(yy[-1] - yy[-2]),
            maximum_pressure_mpa=float(np.max(answer["sigma_n"])),
            maximum_shear_mpa=float(np.max(answer["tau"])),
            model_automatic_dt_s=float(model["dt"]),
            mesh_and_model_seconds=preparation,
            total_static_seconds=perf_counter() - start,
            snapshot=str(target / "static.npz"),
            snapshot_sha256=fingerprint(target / "static.npz"),
            source_builder_binding_unchanged=True,
            healed_memory_reset_to_zero=True,
        )
        write_json(target / "summary.json", record)
        records.append(record)
        fields.append((yy.copy(), answer["sigma_n"].copy(), answer["tau"].copy()))
        write_json(output / "progress.json", dict(records=records))
        print(json.dumps(record), flush=True)
        del model, answer, coords
        gc.collect()
    compare = runpy.run_path(str(TOOLS / "compare_fem_mesh.py"))["error_report"]
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
    static["REFERENCE_TOOL"]["verify_snapshot"](snapshot, archive["files"])
    if any(fingerprint(p) != v for p, v in frozen.items()):
        raise ValueError("FEM study input or mesh adapter changed")
    write_json(
        output / "summary.json",
        dict(
            status="completed_graded_fem_static_controls",
            records=records,
            successive_refinement=successive,
            frozen_input_sha256=frozen,
            source_revision=static["REVISION"],
            source_archive_sha256=archive["archive_sha256"],
            total_seconds=perf_counter() - begun,
            peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2 if sys.platform == "darwin" else 1024),
            physical_convergence_verified=False,
            same_accuracy_fem_advantage_verified=False,
            original_tatva_readonly=True,
            scope="True float64 static roots using original constitutive/contact kernels and a recorded mesh adapter; no dynamic convergence or speed comparison",
        ),
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

"""Compare advisory intervals on actual coarse finite-body Galerkin matrices.

The full requested FFT circle is emitted in both runs. Fresh direct points
check both spectra; only operator evaluation counts are a performance screen.
This produces no time weights and does not certify the fine nonlinear event.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import CQPlan, bdf2_symbol
from finiteblock_bem.galerkin_fast import prepared_geometry_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--h", type=float, default=160)
    parser.add_argument("--contact-h", type=float, default=160)
    parser.add_argument("--contact-end-h", type=float, default=160)
    parser.add_argument("--corner-h", type=float, default=160)
    parser.add_argument("--quadrature", type=int, default=4)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--max-memory-mb", type=float, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a fresh screen outside protected Tatva")
    if not np.isfinite(args.max_memory_mb) or args.max_memory_mb <= 0:
        raise ValueError("Require a positive finite diagnostic memory budget")
    paths = [
        Path(__file__),
        args.case,
        *[
            TOOLS / name
            for name in [
                "seeded_raw_sampling.py",
                "raw_layer_sampling.py",
                "trace_schur_experiment.py",
                "schur_condensation_experiment.py",
                "build_resolvent_history.py",
                "replay_resolvent.py",
            ]
        ],
    ]
    if args.plan is not None:
        paths.append(args.plan)
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}
    environment = environment_record(args.case)
    frequency = (
        dict(dt=2e-6, fft_size=65536, radius=0.999648715112665)
        if args.plan is None
        else json.loads(args.plan.read_text())["plan"]
    )
    frequency = {k: frequency[k] for k in ["dt", "fft_size", "radius"]}
    if (
        not np.isfinite(frequency["dt"])
        or frequency["dt"] <= 0
        or not np.isfinite(frequency["radius"])
        or not 0 < frequency["radius"] < 1
        or not isinstance(frequency["fft_size"], int)
        or frequency["fft_size"] < 2
        or frequency["fft_size"] & (frequency["fft_size"] - 1)
    ):
        raise ValueError("Require a valid complete FFT frequency contour")
    raw_tool = runpy.run_path(str(TOOLS / "raw_layer_sampling.py"))
    seeded = runpy.run_path(str(TOOLS / "seeded_raw_sampling.py"))
    options = raw_tool["RawSamplingOptions"](degree=args.degree)
    model = load_nodal_model(
        args.case,
        max_panel_length=args.h,
        contact_panel_length=args.contact_h,
        contact_end_length=args.contact_end_h,
        corner_panel_length=args.corner_h,
    )
    d = 2 * len(model.moving_contact) + 1
    plan = CQPlan(frequency["dt"], 1, d, frequency["fft_size"], frequency["radius"], 0, 0)
    count = plan.fft_size // 2 + 1
    bodies = [model.reference.moving, model.reference.stationary]
    packet = sum(3 * mesh.dofs**2 * 16 for mesh in bodies)
    interpolation = (4 * (options.degree + 1) + 3 * options.check_points + 6) * packet
    estimate = sum(prepared_geometry_bytes(mesh, args.quadrature) for mesh in bodies)
    estimate += (
        interpolation + 32 * (model.constraints.size**2 + sum(m.dofs**2 for m in bodies)) * 16
    )
    estimate += 6 * count * d * d * 16 + 128 * 1024**2
    if estimate > args.max_memory_mb * 1024**2:
        raise ValueError("Complete paired diagnostic spectra exceed the array budget")
    factory = runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"]
    compliance = runpy.run_path(str(TOOLS / "build_resolvent_history.py"))[
        "compliance_in_work_coordinates"
    ]
    digest = runpy.run_path(str(TOOLS / "replay_resolvent.py"))["digest_file"]
    line = np.r_[
        model.constraints.contact_weights,
        model.constraints.contact_weights,
        model.constraints.platen_width,
    ]
    law = model.reference.contact
    tangent = np.diag(
        np.r_[
            model.constraints.contact_weights * law.normal_penalty,
            model.constraints.contact_weights * law.tangential_penalty,
            0,
        ]
    )
    tangent[-1, -1] = model.reference.spring_stiffness * model.constraints.platen_width
    output.mkdir(parents=True)
    begun = perf_counter()
    rows, spectra = [], []
    with threadpool_limits(limits=1, user_api="blas"):
        transfer = factory(model, quadrature=args.quadrature)

        def raw(s):
            return raw_tool["RawLayerPacket"].from_operators(
                p.assemble(s) for p in transfer.prepared
            )

        def transform(packet, s):
            return compliance(transfer.evaluate_operators(packet.operators()), tangent, line)

        for advisory in [False, True]:
            values = np.empty((count, d, d), complex)

            def consume(index, s, value):
                values[index] = value

            start = perf_counter()
            if advisory:
                row = seeded["sample_with_primitive_partition"](
                    raw_tool["sample_raw_layers"],
                    raw,
                    transform,
                    plan,
                    consume,
                    material=model.reference.material,
                    options=options,
                    max_workspace_mb=min(args.max_memory_mb, 512),
                    workers=1,
                )
            else:
                row = raw_tool["sample_raw_layers"](
                    raw,
                    transform,
                    plan,
                    consume,
                    options=options,
                    max_workspace_mb=min(args.max_memory_mb, 512),
                    workers=1,
                )
            row.update(advisory_partition=advisory, diagnostic_seconds=perf_counter() - start)
            rows.append(row)
            spectra.append(values)
            write_json(output / "progress.json", dict(records=rows))
            print(
                json.dumps(
                    dict(
                        advisory=advisory,
                        source_evaluations=row["source_evaluations"],
                        complete=row["all_fft_grid_points_emitted"],
                    )
                ),
                flush=True,
            )
        normalized = np.linalg.norm(spectra[1] - spectra[0], axis=(-2, -1)) / (
            options.output_atol + options.output_rtol * np.linalg.norm(spectra[0], axis=(-2, -1))
        )
        selected = sorted(
            {
                i
                for i in [
                    0,
                    1,
                    11,
                    101,
                    500,
                    1000,
                    (count - 1) // 8,
                    (count - 1) // 4,
                    (count - 1) // 2,
                    count - 1,
                ]
                if 0 <= i < count
            }
        )
        audits = []
        for index in selected:
            s = bdf2_symbol(plan.radius * np.exp(-1j * np.pi * index / (count - 1))) / plan.dt
            exact = transform(raw(s), s)
            scale = options.output_atol + options.output_rtol * np.linalg.norm(exact)
            audits.append(
                dict(
                    index=index,
                    normalized_errors=[
                        float(np.linalg.norm(a[index] - exact) / scale) for a in spectra
                    ],
                )
            )
    passed = float(normalized.max()) <= 1 and all(max(r["normalized_errors"]) <= 1 for r in audits)
    fewer = rows[1]["source_evaluations"] < rows[0]["source_evaluations"]
    current = environment_record(args.case)
    if any(
        current[k] != environment[k]
        for k in ["source_sha256", "case_sha256", "restart_data_sha256"]
    ) or any(fingerprint(p) != value for p, value in frozen.items()):
        raise ValueError("Raw partition screen source or physical input changed")
    np.savez(output / "spectra.npz", baseline=spectra[0], advisory=spectra[1])
    write_json(
        output / "summary.json",
        dict(
            status="passed_seeded_raw_matrix_screen"
            if passed and fewer
            else "failed_seeded_raw_matrix_screen",
            all_fft_grid_points_compared=count,
            records=rows,
            maximum_normalized_spectral_difference=float(normalized.max()),
            independent_fresh_frequency_audits=audits,
            accuracy_screen_passed=passed,
            actual_matrix_evaluations_reduced=fewer,
            matrix_source_evaluation_ratio=rows[1]["source_evaluations"]
            / rows[0]["source_evaluations"],
            frequency_grid=frequency,
            raw_sampling_options=vars(options),
            actual_mesh_arguments=dict(
                h=args.h,
                contact_h=args.contact_h,
                contact_end_h=args.contact_end_h,
                corner_h=args.corner_h,
                quadrature=args.quadrature,
            ),
            retained_dofs=d,
            environment=environment,
            frozen_input_sha256=frozen,
            spectra_sha256=digest(output / "spectra.npz"),
            estimated_arrays_mib=estimate / 1024**2,
            diagnostic_seconds=perf_counter() - begun,
            scope="Actual coarse finite bodies and whole FFT circle; no fine-kernel or nonlinear-event certificate",
            timing_scope="Diagnostic may overlap the owned two-worker build; compare evaluation counts, not a solver wall-time ratio",
            complete_time_weights_or_fem_advantage_verified=False,
            original_tatva_readonly=True,
        ),
    )
    if not passed or not fewer:
        raise ValueError("Advisory partition did not pass accuracy and matrix-evaluation screens")
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()

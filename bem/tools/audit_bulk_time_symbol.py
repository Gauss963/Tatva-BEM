"""Measure CQ transfer error on the same spatially discrete finite bodies.

Compare closed-stick actuator responses with continuous time. BDF2 uses its
scalar symbol; Radau IIA uses both modes and the actual stage forcing.
This isolates the bulk time symbol without a frequency interpolant, IFFT,
nonlinear contact trajectory, or a second FEM run. It is a sampled linear
diagnostic, not an event error bound or a time-convergence certificate.
"""

import argparse
import json
import os
import runpy
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import bdf2_symbol
from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators, prepared_geometry_bytes
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
PROFILE = runpy.run_path(str(TOOLS / "compare_linear_bulk_response.py"))["profile_relative_error"]
TRANSFER = runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"]
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))


def mapped_parameter(s, dt):
    """The original bulk CQ symbol, evaluated at an exponential z-transform point."""
    if not np.isfinite(s) or np.real(s) <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Require a right-half-plane parameter and positive time step")
    if abs(np.imag(s)) * dt >= np.pi:
        raise ValueError("The diagnostic frequency must be below the time-grid Nyquist limit")
    return complex(bdf2_symbol(np.exp(-complex(s) * dt)) / dt)


def parameter_modes(s, dt, method):
    """Endpoint transfer weights for the sampled exponential's stage data.

    For Radau, v_i=exp(s*dt*(c_i-1)) is normalized to the interval endpoint.
    Diagonalize Delta(exp(-s*dt))/dt=P*diag(lambda)*P^-1. The endpoint
    response is sum_l P[-1,l]*(P^-1*v)[l]*T(lambda_l). Both modes and the
    complex bilinear projectors are required, even for the endpoint alone.
    """
    if method not in ["bdf2", "radau"]:
        raise ValueError("Unknown time-symbol method")
    mapped = mapped_parameter(s, dt)
    if method == "bdf2":
        return [(mapped, 1 + 0j)]
    symbol = RADAU["radau_symbol"](np.exp(-complex(s) * dt)) / dt
    values, vectors = np.linalg.eig(symbol)
    if not np.isfinite(np.linalg.cond(vectors)) or np.linalg.cond(vectors) > 1e8:
        raise ValueError("Near-defective Radau symbol requires a different evaluator")
    order = np.argsort(abs(values))
    values, vectors = values[order], vectors[:, order]
    forcing = np.exp(complex(s) * dt * (RADAU["C"] - 1))
    weights = vectors[-1] * np.linalg.solve(vectors, forcing)
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weights))
        or np.any(values.real <= 0)
        or abs(np.sum(weights) - 1) > 1e-12
    ):
        raise ValueError("Invalid Radau modes or endpoint projectors")
    return [
        (complex(value), complex(weight)) for value, weight in zip(values, weights, strict=True)
    ]


def closed_stick_response(impedance, penalty, spring):
    """Unit actuator response; contact and spring are instantaneous in both cases."""
    z, diagonal = np.asarray(impedance), np.asarray(penalty)
    if (
        z.ndim != 2
        or z.shape != (len(diagonal), len(diagonal))
        or len(diagonal) < 3
        or not np.all(np.isfinite(z))
        or not np.all(np.isfinite(diagonal))
        or np.any(diagonal <= 0)
        or not np.isfinite(spring)
        or spring <= 0
        or diagonal[-1] != spring
    ):
        raise ValueError("Require finite aligned impedance, positive penalties and platen spring")
    matrix = z + np.diag(diagonal)
    rhs = np.zeros(len(diagonal))
    rhs[-1] = spring
    response = solve(matrix, rhs, check_finite=False)
    residual = float(np.linalg.norm(matrix @ response - rhs) / spring)
    if not np.all(np.isfinite(response)) or not np.isfinite(residual) or residual > 1e-9:
        raise ValueError("Closed-stick response failed its original linear equation")
    return response, residual


def response_errors(y, actual, reference, normal_penalty, tangential_penalty):
    n = len(y)
    if actual.shape != reference.shape or actual.shape != (2 * n + 1,):
        raise ValueError("Responses must align with the complete retained contact/platen trace")
    errors = dict(
        pressure_profile_relative_l2=PROFILE(
            y, normal_penalty * actual[:n], y, normal_penalty * reference[:n]
        ),
        signed_shear_profile_relative_l2=PROFILE(
            y, tangential_penalty * actual[n:-1], y, tangential_penalty * reference[n:-1]
        ),
        platen_relative_error=float(abs(actual[-1] - reference[-1]) / abs(reference[-1])),
    )
    if not all(np.isfinite(v) for v in errors.values()):
        raise ValueError("A zero/nonfinite reference response cannot define relative error")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Small JSON with source_mesh_arguments")
    parser.add_argument("--dt", type=float, nargs="+", default=[2e-6, 1e-6, 0.5e-6])
    parser.add_argument("--frequencies", type=float, nargs="+", default=[0, 1000, 10000])
    parser.add_argument("--laplace-real", type=float, default=200)
    parser.add_argument("--method", choices=["bdf2", "radau"], default="bdf2")
    parser.add_argument("--max-memory-mb", type=float, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a fresh diagnostic folder outside protected Tatva")
    if (
        not args.dt
        or len(set(args.dt)) != len(args.dt)
        or any(not np.isfinite(t) or t <= 0 for t in args.dt)
        or not args.frequencies
        or len(set(args.frequencies)) != len(args.frequencies)
        or any(not np.isfinite(f) or f < 0 for f in args.frequencies)
        or not np.isfinite(args.laplace_real)
        or args.laplace_real <= 0
        or not np.isfinite(args.max_memory_mb)
        or args.max_memory_mb <= 0
    ):
        raise ValueError("Require distinct valid time steps/frequencies and positive budgets")
    for f in args.frequencies:
        for dt in args.dt:
            parameter_modes(complex(args.laplace_real, 2 * np.pi * f), dt, args.method)
    begun = perf_counter()
    source = json.loads(args.source.read_text())
    options = source["source_mesh_arguments"]
    environment = environment_record(options["case"])
    if any(
        source["environment"][key] != environment[key]
        for key in ["source_sha256", "case_sha256", "restart_data_sha256"]
    ):
        raise ValueError("Source mesh metadata and current numerical package/case differ")
    paths = [
        args.source,
        Path(__file__),
        TOOLS / "compare_linear_bulk_response.py",
        TOOLS / "trace_schur_experiment.py",
        TOOLS / "schur_condensation_experiment.py",
    ]
    if args.method == "radau":
        paths.append(TOOLS / "radau_experiment.py")
    frozen = {str(p.resolve()): fingerprint(p) for p in paths}

    def check_source():
        current = environment_record(options["case"])
        if any(fingerprint(p) != sha for p, sha in frozen.items()) or any(
            current[key] != environment[key]
            for key in ["source_sha256", "case_sha256", "restart_data_sha256"]
        ):
            raise ValueError("Numerical source or mesh metadata changed during diagnostic")

    model = load_nodal_model(
        options["case"],
        **{
            dest: options.get(key)
            for dest, key in [
                ("max_panel_length", "h"),
                ("contact_panel_length", "contact_h"),
                ("contact_end_length", "contact_end_h"),
                ("corner_panel_length", "corner_h"),
                ("restart_knot_window", "restart_knot_window"),
                ("restart_knot_subdivisions", "restart_knot_subdivisions"),
            ]
        },
    )
    bodies = [model.reference.moving, model.reference.stationary]
    m, d = model.constraints.size, 2 * len(model.moving_contact) + 1
    # Keep one body's geometry at a time. Store supported matrices on disk,
    # then release that geometry before preparing the other body.
    estimate = max(prepared_geometry_bytes(mesh, options["quadrature"]) for mesh in bodies)
    estimate += 24 * max(mesh.dofs**2 for mesh in bodies) * 16 + 20 * m**2 * 16 + 128 * 1024**2
    if estimate > args.max_memory_mb * 1024**2:
        raise ValueError(f"Diagnostic arrays need about {estimate / 1024**2:.1f} MiB")
    samples, groups = [], []
    for f in args.frequencies:
        s = complex(args.laplace_real, 2 * np.pi * f)
        for dt in [None, *args.dt]:
            modes = [(s, 1 + 0j)] if dt is None else parameter_modes(s, dt, args.method)
            indices = list(range(len(samples), len(samples) + len(modes)))
            samples.extend((f, dt, value, weight) for value, weight in modes)
            groups.append((f, dt, s, indices))
    if shutil.disk_usage(output.parent).free < len(samples) * m**2 * 16 + 64 * 1024**2:
        raise ValueError("Insufficient disk space for the diagnostic supported matrices")
    output.mkdir(parents=True)
    common = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_mesh_arguments=options,
        environment=environment,
        frozen_input_sha256=frozen,
        estimated_array_mib=estimate / 1024**2,
        supported_matrix_disk_bytes=len(samples) * m**2 * 16,
        original_tatva_readonly=True,
        worker_count=1,
        blas_threads=1,
        transfer_method=args.method,
        cores_environment=os.environ.get("FDBEM_CORES"),
        complete_nonlinear_time_convergence_verified=False,
        same_accuracy_fem_advantage_verified=False,
        scope="Sampled closed-stick bulk time-symbol error on a fixed spatial mesh; no nonlinear event, startup, interpolation, IFFT or FEM error certificate",
    )
    write_json(output / "run_manifest.json", dict(status="prepared_not_completed", **common))
    matrices = [output / f"supported-{i}.npy" for i in range(len(samples))]
    with threadpool_limits(limits=1, user_api="blas"):
        transfer = TRANSFER(model, quadrature=options["quadrature"], prepared=False)
        first = 0
        for body, mesh in enumerate(bodies):
            check_source()
            prepared = PreparedGalerkinOperators(
                mesh, model.reference.material, quadrature=options["quadrature"]
            )
            p = model.constraints.prolongation[first : first + mesh.dofs]
            columns = np.flatnonzero(np.any(p != 0, axis=0))
            p = p[:, columns]
            for index, (frequency, dt, s, _) in enumerate(samples):
                tick = perf_counter()
                operator = prepared.assemble(s)
                block = p.T @ operator.force_stiffness() @ p
                stored = np.lib.format.open_memmap(
                    matrices[index], mode="w+" if body == 0 else "r+", dtype=complex, shape=(m, m)
                )
                if body == 0:
                    stored[:] = 0
                stored[np.ix_(columns, columns)] += block
                stored.flush()
                del stored, block, operator
                write_json(
                    output / "progress.json",
                    dict(body=body, sample=index, frequency_hz=frequency, dt_s=dt),
                )
                print(
                    f"body={body}, sample={index}, seconds={perf_counter() - tick:.3f}", flush=True
                )
            del prepared
            first += mesh.dofs
        y = model.reference.moving.starts[model.moving_contact, 1]
        law = model.reference.contact
        weights = model.constraints.contact_weights
        spring = model.reference.spring_stiffness * model.constraints.platen_width
        diagonal = np.r_[weights * law.normal_penalty, weights * law.tangential_penalty, spring]
        solutions, diagnostics = [], []
        for index, (_, _, s, weight) in enumerate(samples):
            matrix = np.load(matrices[index], mmap_mode="r")
            z, recovery, _ = transfer.condense(matrix)
            defect = matrix @ recovery - transfer.jump.T @ z
            balance = float(np.linalg.norm(defect) / max(np.linalg.norm(transfer.jump.T @ z), 1))
            trace = float(np.linalg.norm(transfer.jump @ recovery - np.eye(d), ord=np.inf))
            if (
                not np.isfinite(balance)
                or not np.isfinite(trace)
                or balance > 1e-9
                or trace > 1e-10
            ):
                raise ValueError(
                    "Condensed bulk failed full supported equilibrium or jump identity"
                )
            q, residual = closed_stick_response(z, diagonal, spring)
            solutions.append(q)
            diagnostics.append(
                dict(
                    s_real_per_s=s.real,
                    s_imag_per_s=s.imag,
                    endpoint_weight_real=weight.real,
                    endpoint_weight_imag=weight.imag,
                    weighted_response_l2_norm=float(np.linalg.norm(weight * q)),
                    original_supported_balance_relative_residual=balance,
                    trace_identity_inf_error=trace,
                    closed_stick_equation_relative_residual=residual,
                )
            )
            del matrix, z, recovery, defect
            matrices[index].unlink()
        reference, rows = {}, []
        for index, (frequency, dt, physical_s, indices) in enumerate(groups):
            q = sum(samples[j][3] * solutions[j] for j in indices)
            modes = [diagnostics[j] for j in indices]
            errors = None
            if dt is None:
                reference[frequency] = q
            else:
                errors = response_errors(
                    y, q, reference[frequency], law.normal_penalty, law.tangential_penalty
                )
            rows.append(
                dict(
                    frequency_hz=frequency,
                    dt_s=dt,
                    method="continuous" if dt is None else args.method,
                    physical_s_real_per_s=physical_s.real,
                    physical_s_imag_per_s=physical_s.imag,
                    s_real_per_s=modes[0]["s_real_per_s"] if len(modes) == 1 else None,
                    s_imag_per_s=modes[0]["s_imag_per_s"] if len(modes) == 1 else None,
                    modes=modes,
                    errors_vs_same_mesh_continuous_time=errors,
                    original_supported_balance_relative_residual=max(
                        m["original_supported_balance_relative_residual"] for m in modes
                    ),
                    trace_identity_inf_error=max(m["trace_identity_inf_error"] for m in modes),
                    closed_stick_equation_relative_residual=max(
                        m["closed_stick_equation_relative_residual"] for m in modes
                    ),
                )
            )
            np.savez(
                output / f"response-{index}.npz",
                y=y,
                retained_displacement=q,
                modal_retained_displacements=np.stack([solutions[j] for j in indices]),
                modal_parameters=np.array([samples[j][2] for j in indices]),
                endpoint_weights=np.array([samples[j][3] for j in indices]),
            )
            write_json(output / "responses.json", dict(records=rows))
    check_source()
    write_json(
        output / "summary.json",
        dict(
            status="completed_sampled_bulk_time_symbol_diagnostic",
            records=rows,
            **common,
            seconds_before_reporting=perf_counter() - begun,
            timing_scope="Diagnostic with possible concurrent work; not a controlled speed benchmark",
        ),
    )


if __name__ == "__main__":
    main()

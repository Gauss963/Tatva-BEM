"""Experimental elastic-resolvent replay from a completed BEM half spectrum.

For A=Z+C+Kspring and L=A^-1, write
q=q_el+L*(f_hold-chi), chi=Fc(q,state)-Fc0-C*q.
The full elastic startup is in q_el. Correction memory starts only when its
input becomes nonzero; it is never truncated after that. The original ramp,
coverage hold and local previous-state law are retained. This is a validation
prototype, not a production cache migration or a model-order approximation.
"""

import argparse
import ast
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from scipy.fft import irfft, rfft
from scipy.linalg import solve

from finiteblock_bem.cq import BoundaryHistory, CQPlan, OnlineConvolution, invert_spectrum, plan_cq
from finiteblock_bem.cq_storage import DiskSpectrum
from finiteblock_bem.dynamics import LoadingSchedule, event16_schedule
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.nodal import (
    NodalInterfaceEquilibrium,
    initialize_nodal_restart,
    load_nodal_model,
)
from finiteblock_bem.nodal_dynamics import nodal_cache_key
from finiteblock_bem.sampling import SamplingOptions

ROOT = Path(__file__).resolve().parents[1]


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SavedDiskSpectrum(DiskSpectrum):
    """Read a completed spectrum through the same bounded mappings, without writes."""

    def __init__(self, path, plan):
        self.path = Path(path)
        self.shape = (plan.fft_size // 2 + 1, plan.dofs, plan.dofs)
        self.dtype = np.dtype("complex128")
        self.written = np.ones(self.shape[0], dtype=bool)
        self.pending_writes = 0
        with self.path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            reader = {
                (1, 0): np.lib.format.read_array_header_1_0,
                (2, 0): np.lib.format.read_array_header_2_0,
            }.get(version)
            if reader is None:
                raise ValueError("Unsupported saved spectrum format")
            shape, fortran, dtype = reader(stream)
            self.offset = stream.tell()
        if shape != self.shape or fortran or dtype != self.dtype:
            raise ValueError("Saved spectrum does not match the source contour")

    def __setitem__(self, key, value):
        # invert_spectrum enforces real endpoint matrices. An already real
        # saved endpoint needs no write; all actual mutations remain rejected.
        if isinstance(key, (int, np.integer)) and key in [0, self.shape[0] - 1]:
            if np.array_equal(self[key], value):
                return
        raise ValueError("Saved spectrum is read-only")


def causal_elastic_baseline(column_spectrum, plan, force):
    """Invert CQ coefficients, then apply a zero-padded CAUSAL convolution.

    Multiplying the damped frequency samples by a finite forcing FFT directly
    introduces a small periodic response from future forcing. This two-step
    operation implements the same finite causal sum as the online CQ solver.
    """
    impulse = irfft(column_spectrum, n=plan.fft_size, axis=0)[: plan.steps + 1]
    impulse /= plan.radius ** np.arange(plan.steps + 1)[:, None]
    return irfft(
        rfft(impulse, n=plan.fft_size, axis=0) * rfft(force, n=plan.fft_size)[:, None],
        n=plan.fft_size,
        axis=0,
    )[: plan.steps + 1]


def reuse_elastic_spectrum(record, storage, plan, tangent, environment):
    previous = json.loads(Path(record).read_text())
    if previous["source_spectrum_sha256"] != storage["sha256"]:
        raise ValueError("Resolvent was built from a different source spectrum")
    for name in ["source_sha256", "case_sha256", "restart_data_sha256"]:
        if previous["environment"][name] != environment[name]:
            raise ValueError(f"Saved resolvent differs in {name}")
    saved = previous["resolvent_storage"]
    if "coordinates" in saved:
        raise ValueError("Use --prepared-kernel for work-coordinate spectra")
    if not saved["all_frequency_matrices_written"] or digest_file(saved["path"]) != saved["sha256"]:
        raise ValueError("Saved resolvent is incomplete or its digest differs")
    source = SavedDiskSpectrum(storage["path"], plan)
    target = SavedDiskSpectrum(saved["path"], plan)
    maximum = 0.0
    # Verify the inverse identity at EVERY stored frequency. This also verifies
    # C/Kspring compatibility independently of the old prototype's source text.
    for first in range(0, target.shape[0], 64):
        last = min(first + 64, target.shape[0])
        error = (source[first:last] + tangent) @ target[first:last] - np.eye(plan.dofs)
        maximum = max(maximum, float(np.max(np.sum(abs(error), axis=2))))
        if maximum > 1e-8:
            raise ValueError(f"Saved inverse failed its frequency residual: {maximum}")
    return target, target[:, :, -1], maximum


def invert_elastic_spectrum(path, source_plan, output, tangent, forcing):
    """Read 64 complete matrices at a time; retain the full output on disk."""
    target = DiskSpectrum(output, source_plan)
    count, d = source_plan.fft_size // 2 + 1, source_plan.dofs
    response = np.empty((count, d), complex)
    with Path(path).open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        header_reader = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }.get(version)
        if header_reader is None:
            raise ValueError("Unsupported spectrum format")
        shape, fortran, dtype = header_reader(stream)
        if shape != (count, d, d) or fortran or dtype != np.complex128:
            raise ValueError("Incorrect source half spectrum")
        for first in range(0, count, 64):
            last = min(first + 64, count)
            block = np.fromfile(stream, dtype=complex, count=(last - first) * d * d)
            block = block.reshape(last - first, d, d)
            values = np.stack([solve(z + tangent, np.eye(d), check_finite=False) for z in block])
            target[first:last] = values
            response[first:last] = values[:, :, -1] * forcing[first:last, None]
    return target, response


class CorrectionHistory:
    """Rebase an initially zero causal source without discarding any nonzero term."""

    def __init__(self, history):
        self.history = history
        self.start = None
        self.online = None

    def past(self, index):
        if self.start is None:
            return np.zeros(self.history.plan.dofs)
        if index - self.start > self.history.plan.steps:
            raise ValueError("Correction started too early for the requested history horizon")
        return self.online.past(index - self.start)

    def append(self, index, source, *, active):
        if self.start is None:
            if not active:
                np.testing.assert_allclose(source, 0, rtol=0, atol=1e-7)
                return
            self.start = index
            self.online = OnlineConvolution(self.history)
        self.online.append(source)


def main():
    from finiteblock_bem.jax_backend import JaxContactEquilibrium

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/resolvent-replay")
    parser.add_argument(
        "--history-steps", type=int, help="Maximum duration AFTER the first correction"
    )
    parser.add_argument("--max-memory-mb", type=float, default=2048)
    parser.add_argument(
        "--prepared-kernel", type=Path, help="Directory produced by build_resolvent_history.py"
    )
    parser.add_argument(
        "--reuse-resolvent",
        type=Path,
        help="Previous summary; all frequency inverse residuals are checked",
    )
    args = parser.parse_args()
    if args.prepared_kernel is not None and args.reuse_resolvent is not None:
        raise ValueError("Select either a prepared kernel or a saved physical resolvent")
    started = perf_counter()
    folder = args.run.resolve()
    summary = json.loads((folder / "summary.json").read_text())
    options = json.loads((folder / "run_manifest.json").read_text())["arguments"]
    if summary["status"] != "completed_requested_horizon" or summary["source_changed_during_run"]:
        raise ValueError("A complete source-frozen BEM run is required")
    environment = environment_record(options["case"])
    for name in ["source_sha256", "case_sha256", "restart_data_sha256"]:
        if environment[name] != summary["environment"][name]:
            raise ValueError(f"{name} differs from the source trajectory")
    storage = summary["spectrum_storage"]
    prepared = None
    if args.prepared_kernel is not None:
        prepared = json.loads((args.prepared_kernel / "summary.json").read_text())
        if prepared["status"] != "completed_resolvent_kernel_only" or prepared["source_run"] != str(
            folder
        ):
            raise ValueError("Prepared kernel must reference this completed source run")
        if (
            prepared["dt_s"] != summary["dt_s"]
            or prepared["full_steps"] != summary["completed_steps"]
        ):
            raise ValueError("Replay validation requires the same timestep and horizon")
        for name in ["source_sha256", "case_sha256", "restart_data_sha256"]:
            if prepared["environment"][name] != environment[name]:
                raise ValueError(f"Prepared kernel differs in {name}")
        math_node = next(
            node
            for node in ast.parse(Path(__file__).read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "causal_elastic_baseline"
        )
        math_digest = hashlib.sha256(
            ast.dump(math_node, include_attributes=False).encode()
        ).hexdigest()
        if prepared["helper_math_sha256"] != math_digest:
            raise ValueError("Prepared elastic baseline used different CQ math")
        original_plan = plan_cq(
            summary["history_dofs"][0],
            summary["dt_s"],
            summary["completed_steps"],
            max_memory_mb=65536,
            disk_spectrum=True,
        )
    else:
        if (
            storage["backend"] != "disk_half_spectrum"
            or not storage["all_frequency_matrices_written"]
        ):
            raise ValueError("A completed disk half spectrum is required")
        original_plan = CQPlan(**storage["cq_plan"])
    steps, dt, d = original_plan.steps, original_plan.dt, original_plan.dofs
    history_steps = prepared["correction_steps"] if prepared is not None else steps
    if args.history_steps is not None:
        if prepared is not None and args.history_steps != history_steps:
            raise ValueError("Prepared correction horizon cannot be changed at replay")
        history_steps = args.history_steps
    if not 1 <= history_steps <= steps:
        raise ValueError("Correction history horizon must be between 1 and total steps")
    planned = plan_cq(d, dt, history_steps, max_memory_mb=args.max_memory_mb, disk_spectrum=True)
    # Source contour stays unchanged even when only a prefix of L weights is stored.
    extra = 8 * (original_plan.fft_size - planned.fft_size) * min(d, 4) * d * 8
    extra += 2 * 64 * d * d * 16 + 8 * original_plan.fft_size * d * 16
    peak = planned.estimated_peak_bytes + extra
    if peak > args.max_memory_mb * 1024**2:
        raise ValueError(f"Resolvent arrays need about {peak / 1024**2:.1f} MiB")
    plan = replace(planned, fft_size=original_plan.fft_size, radius=original_plan.radius)
    if prepared is None and digest_file(storage["path"]) != storage["sha256"]:
        raise ValueError("Completed source spectrum SHA-256 differs")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    sampling = (
        SamplingOptions(relative_tolerance=options["sampling_rtol"])
        if options["sampling"] == "adaptive"
        else None
    )
    cache_directory = Path(options["cache"])
    if not cache_directory.is_absolute():
        cache_directory = ROOT.parent / cache_directory
    cache = (
        cache_directory
        / f"galerkin-{nodal_cache_key(model, original_plan, options['quadrature'], sampling)}.npy"
    )
    if not cache.is_file() or not cache.with_suffix(".json").is_file():
        raise ValueError("Matching original CQ cache is required")
    original_weights = np.load(cache, mmap_mode="r", allow_pickle=False)
    w0 = np.array(original_weights[0], copy=True)
    del original_weights
    lift = initialize_nodal_restart(model, quadrature=options["quadrature"])
    reference = NodalInterfaceEquilibrium(model, lift, w0)
    n = len(model.moving_contact)
    weights = model.constraints.contact_weights
    cdiag = np.r_[
        weights * model.reference.contact.normal_penalty,
        weights * model.reference.contact.tangential_penalty,
        0,
    ]
    tangent = np.diag(cdiag)
    tangent[-1, -1] += reference.spring
    a0 = w0 + tangent
    time = np.arange(steps + 1) * dt
    schedule = event16_schedule(model.reference)
    diagnostic = options.get("diagnostic_actuator_increment")
    if diagnostic is not None:
        schedule = LoadingSchedule(
            lambda t: model.reference.normal_pressure,
            lambda t: model.reference.restart_actuator + diagnostic * t / (steps * dt),
            "resolvent-accelerated-diagnostic",
        )
    ramp = np.array([schedule.actuator(float(t)) for t in time])
    force = reference.spring * (ramp - ramp[0])
    args.output.mkdir(parents=True, exist_ok=True)
    inverse_residual = None
    if prepared is not None:
        for filename, key in [("weights.npy", "weights_sha256"), ("elastic.npz", "elastic_sha256")]:
            if digest_file(args.prepared_kernel / filename) != prepared[key]:
                raise ValueError(f"Prepared data differs: {filename}")
        prepared_plan = CQPlan(**prepared["plan"])
        if (
            prepared_plan.dt,
            prepared_plan.steps,
            prepared_plan.dofs,
            prepared_plan.fft_size,
            prepared_plan.radius,
        ) != (plan.dt, plan.steps, plan.dofs, plan.fft_size, plan.radius):
            raise ValueError("Prepared contour differs from original CQ discretization")
        prepared_weights = np.load(
            args.prepared_kernel / "weights.npy", mmap_mode="r", allow_pickle=False
        )
        if (
            prepared_weights.shape != (history_steps + 1, d, d)
            or prepared_weights.dtype != np.float64
        ):
            raise ValueError("Invalid physical resolvent weights")
        history = BoundaryHistory(plan, prepared_weights)
        with np.load(args.prepared_kernel / "elastic.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["time"], time)
            elastic = data["elastic"]
        if elastic.shape != (steps + 1, d):
            raise ValueError("Invalid elastic startup shape")
        spectrum_record = prepared["spectrum_storage"]
    elif args.reuse_resolvent is not None:
        spectrum, elastic_spectrum, inverse_residual = reuse_elastic_spectrum(
            args.reuse_resolvent, storage, original_plan, tangent, environment
        )
    else:
        disk_required = (original_plan.fft_size // 2 + 1) * d * d * 16 + 64 * 1024**2
        if shutil.disk_usage(args.output).free < disk_required:
            raise ValueError("Insufficient disk space for the complete resolvent half spectrum")
        spectrum, elastic_spectrum = invert_elastic_spectrum(
            storage["path"],
            original_plan,
            args.output / f"resolvent-{uuid4().hex}.npy",
            tangent,
            np.ones(original_plan.fft_size // 2 + 1),
        )
    if prepared is None:
        elastic = causal_elastic_baseline(elastic_spectrum, original_plan, force)
        del elastic_spectrum
        history = BoundaryHistory(plan, invert_spectrum(spectrum, plan))
        spectrum_record = spectrum.record()
    composition_error = float(np.linalg.norm(history.instantaneous @ a0 - np.eye(d), ord=np.inf))
    if composition_error > 1e-6:
        raise ValueError(f"CQ inverse composition error is too large: {composition_error}")
    linear_old = model.reference.contact.evaluate(reference.base_dn, reference.base_ds, lift.state)
    fc0 = np.r_[linear_old.pressure * weights, linear_old.shear * weights, 0]
    system = JaxContactEquilibrium(reference)
    memory = CorrectionHistory(history)
    q = np.zeros((steps + 1, d))
    p, tau, z, c = [np.zeros((steps + 1, n)) for _ in range(4)]
    p[0], tau[0], z[0], c[0] = (
        linear_old.pressure,
        linear_old.shear,
        lift.state.plastic_slip,
        lift.state.cumulative_slip,
    )
    old, held, hold_time = lift.state, None, None
    actuator, residual = ramp.copy(), np.zeros(steps + 1)
    kernel_seconds = perf_counter() - started
    for index in range(1, steps + 1):
        if held is not None:
            actuator[index] = held
        past = np.zeros(d)
        past[-1] = force[index]
        past -= a0 @ (elastic[index] + memory.past(index))
        answer = system.solve(
            model.reference.normal_pressure, actuator[index], old, past=past, guess=q[index - 1]
        )
        if not answer.converged:
            raise ValueError(f"Resolvent replay failed at {index}: {answer.residual_mpa}")
        q[index], contact = answer.reduced_displacement, answer.contact
        p[index], tau[index], z[index], c[index] = (
            contact.pressure,
            contact.shear,
            contact.state.plastic_slip,
            contact.state.cumulative_slip,
        )
        residual[index] = answer.residual_mpa
        chi = np.r_[contact.pressure * weights, contact.shear * weights, 0] - fc0 - cdiag * q[index]
        hold_force = reference.spring * (actuator[index] - ramp[index])
        source = -chi
        source[-1] += hold_force
        active = bool(
            hold_force != 0
            or np.any(contact.pressure <= 0)
            or np.any(contact.state.plastic_slip != old.plastic_slip)
        )
        memory.append(index, source, active=active)
        old = contact.state
        if diagnostic is None and held is None:
            fraction = np.average(
                c[index] >= model.reference.contact.characteristic_slip, weights=weights
            )
            if fraction >= model.reference.payload["loading"]["stop_coverage"]:
                held, hold_time = actuator[index], time[index]
    fields = dict(
        pressure=p,
        shear=tau,
        plastic_slip=z,
        cumulative_slip=c,
        actuator=actuator,
        platen=reference.base_platen + q[:, -1],
        contact_jumps=q[:, : 2 * n] + np.r_[reference.base_dn, reference.base_ds],
    )
    differences = {}
    with np.load(folder / "trajectory.npz", allow_pickle=False) as saved:
        for name, value in fields.items():
            differences[name] = float(np.max(abs(value - saved[name])))
    np.savez(args.output / "trajectory.npz", time=time, **fields)
    record = dict(
        maximum_absolute_field_difference=differences,
        max_cumulative_slip_mm=float(c.max()),
        maximum_residual_mpa=float(residual.max()),
        inverse_cq_composition_inf_error=composition_error,
        elastic_baseline="Causal CQ coefficients followed by zero-padded forcing convolution",
        reused_inverse_frequency_max_inf_residual=inverse_residual,
        first_correction_step=memory.start,
        correction_history_steps=history_steps,
        total_steps=steps,
        actuator_hold_time_s=hold_time,
        estimated_array_peak_mib=peak / 1024**2,
        kernel_and_initialization_seconds=kernel_seconds,
        total_seconds=perf_counter() - started,
        resolvent_storage=spectrum_record,
        source_spectrum_sha256=storage.get("sha256"),
        prepared_kernel=None
        if args.prepared_kernel is None
        else str(args.prepared_kernel.resolve()),
        completed_requested_horizon=True,
        strict_field_comparison_passed=max(differences.values()) <= 1e-5,
        environment=environment,
        prototype_sha256=digest_file(Path(__file__)),
        scope="Equivalent elastic-resolvent validation with full elastic startup, original loading policy and complete correction memory after activation; no rational fit",
    )
    write_json(args.output / "summary.json", record)
    print(json.dumps(record, indent=2))
    if max(differences.values()) > 1e-5:
        raise ValueError("Resolvent replay differs from original fields by more than 1e-5")


if __name__ == "__main__":
    main()

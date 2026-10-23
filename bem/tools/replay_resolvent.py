"""Experimental elastic-resolvent replay from a completed BEM half spectrum.

For A=Z+C+Kspring and L=A^-1, write
q=q_el+L*(f_hold-chi), chi=Fc(q,state)-Fc0-C*q.
The full elastic startup is in q_el. Correction memory starts only when its
input becomes nonzero; it is never truncated after that. The original ramp,
coverage hold and local previous-state law are retained. This is a validation
prototype, not a production cache migration or a model-order approximation.
"""

import argparse
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
    args = parser.parse_args()
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
    if storage["backend"] != "disk_half_spectrum" or not storage["all_frequency_matrices_written"]:
        raise ValueError("A completed disk half spectrum is required")
    original_plan = CQPlan(**storage["cq_plan"])
    steps, dt, d = original_plan.steps, original_plan.dt, original_plan.dofs
    history_steps = steps if args.history_steps is None else args.history_steps
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
    if digest_file(storage["path"]) != storage["sha256"]:
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
    transformed_force = rfft(
        force * original_plan.radius ** np.arange(steps + 1), n=original_plan.fft_size
    )
    args.output.mkdir(parents=True, exist_ok=True)
    disk_required = (original_plan.fft_size // 2 + 1) * d * d * 16 + 64 * 1024**2
    if shutil.disk_usage(args.output).free < disk_required:
        raise ValueError("Insufficient disk space for the complete resolvent half spectrum")
    spectrum, elastic_spectrum = invert_elastic_spectrum(
        storage["path"],
        original_plan,
        args.output / f"resolvent-{uuid4().hex}.npy",
        tangent,
        transformed_force,
    )
    elastic = irfft(elastic_spectrum, n=original_plan.fft_size, axis=0)[: steps + 1]
    elastic /= original_plan.radius ** np.arange(steps + 1)[:, None]
    del elastic_spectrum
    history = BoundaryHistory(plan, invert_spectrum(spectrum, plan))
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
        first_correction_step=memory.start,
        correction_history_steps=history_steps,
        total_steps=steps,
        actuator_hold_time_s=hold_time,
        estimated_array_peak_mib=peak / 1024**2,
        kernel_and_initialization_seconds=kernel_seconds,
        total_seconds=perf_counter() - started,
        resolvent_storage=spectrum.record(),
        source_spectrum_sha256=storage["sha256"],
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

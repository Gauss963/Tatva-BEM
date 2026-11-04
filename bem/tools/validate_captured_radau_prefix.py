"""Validate all accepted fields and restore full future CQ state before resume.

No nonlinear root is re-solved and no candidate is committed. The original
accepted prefix is immutable. Residuals are independently re-evaluated; the
missing historical Newton counts are recorded as unavailable, never zero.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.cq import BoundaryHistory, CQPlan
from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
STREAM = runpy.run_path(str(TOOLS / "stream_initial_residual.py"))
RESTORE = runpy.run_path(str(TOOLS / "restore_compact_history.py"))
CACHE = runpy.run_path(str(TOOLS / "history_prefix_cache.py"))


def contact_force(values, weights):
    return np.stack(
        [np.r_[weights * values[:, a, 0], weights * values[:, a, 1], 0.0] for a in range(2)]
    )


def reconstruct_prefix(
    reference, state, q, values, stage_time, ramp, *, stop_coverage=1.0, diagnostic=False
):
    """Reconstruct the exact original correction sources and loading stop state."""
    d, n = reference.constraints.size, len(reference.constraints.contact_weights)
    count = len(q)
    if (
        count < 1
        or q.shape != (count, 2, d)
        or values.shape != (count, n, 2, 7)
        or ramp.shape != stage_time.shape
        or ramp.shape[1:] != (2,)
        or len(ramp) < count
    ):
        raise ValueError("Accepted prefix shapes or time/loading grid differ")
    if not 0 < stop_coverage <= 1 or not all(
        np.all(np.isfinite(a)) for a in [q, values, stage_time, ramp]
    ):
        raise ValueError("Require finite prefix and positive coverage threshold")
    law, weights = reference.case.contact, reference.constraints.contact_weights
    initial = law.evaluate(reference.base_dn, reference.base_ds, state)
    start = np.column_stack(
        (
            initial.pressure,
            initial.shear,
            state.plastic_slip,
            state.cumulative_slip,
            initial.friction_coefficient,
            np.zeros(n),
            np.zeros(n),
        )
    )
    np.testing.assert_array_equal(q[0], 0)
    np.testing.assert_array_equal(values[0], np.repeat(start[:, None, :], 2, axis=1))
    fc0 = np.r_[weights * initial.pressure, weights * initial.shear, 0.0]
    cdiag = np.r_[weights * law.normal_penalty, weights * law.tangential_penalty, 0.0]
    sources, active = np.zeros((count, 2 * d)), np.zeros(count, dtype=bool)
    actuator = ramp[:count].copy()
    held, hold_time = None, None
    for k in range(1, count):
        if held is not None:
            actuator[k] = held
        hold_force = np.zeros((2, d))
        hold_force[:, -1] = reference.spring * (actuator[k] - ramp[k])
        force = contact_force(values[k], weights)
        sources[k] = (np.tile(fc0, (2, 1)) + cdiag[None, :] * q[k] - force + hold_force).ravel()
        old_z, old_c = values[k - 1, :, 1, 2], values[k - 1, :, 1, 3]
        active[k] = bool(
            np.any(hold_force != 0)
            or np.any(values[k, :, :, 0] <= 0)
            or np.any(values[k, :, :, 2] != old_z[:, None])
            or np.any(values[k, :, :, 3] != old_c[:, None])
        )
        if (
            not diagnostic
            and held is None
            and np.average(values[k, :, 1, 3] >= law.characteristic_slip, weights=weights)
            >= stop_coverage
        ):
            held, hold_time = float(actuator[k, 1]), float(stage_time[k, 1])
    return dict(
        sources=sources,
        active=active,
        actuator=actuator,
        held=held,
        hold_time=hold_time,
        fc0=fc0,
        cdiag=cdiag,
    )


def forcing_at(index, reference, matrix, prefix, memory, elastic, affine, ramp, initial_force):
    d = reference.constraints.size
    a0 = matrix + np.diag(np.tile(prefix["cdiag"], 2))
    prediction = elastic[index].ravel() + memory.past(index)
    prediction -= affine[index].ravel()
    actuator = ramp[index] if prefix["held"] is None else np.full(2, prefix["held"])
    hold_force = np.zeros((2, d))
    hold_force[:, -1] = reference.spring * (actuator - ramp[index])
    return (
        -np.tile(prefix["fc0"], 2)
        + np.tile(initial_force, 2)
        - a0 @ prediction
        - hold_force.ravel()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("affine_cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture, cache, output = (
        args.capture.resolve(),
        args.affine_cache.resolve(),
        args.output.resolve(),
    )
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh owned validation directory")
    failure = json.loads(capture.with_name("failure.json").read_text())
    if STREAM["digest_file"](capture) != failure["failed_equilibrium_sha256"]:
        raise ValueError("Original accepted capture changed")
    affine_meta = json.loads((cache / "summary.json").read_text())
    if affine_meta["status"] != "completed_original_formula_column_streamed_primal_affine_cache":
        raise ValueError("Require the completed affine preparation")
    kernel = Path(failure["kernel"])
    metadata = json.loads((kernel / "summary.json").read_text())
    seed = Path(affine_meta["seed"]["path"])
    lift_path = capture.with_name("initial_lift.npz")
    paths = [
        Path(__file__),
        TOOLS / "restore_compact_history.py",
        TOOLS / "stream_initial_residual.py",
        TOOLS / "compact_radau_history.py",
        TOOLS / "history_prefix_cache.py",
        kernel / "summary.json",
        cache / "summary.json",
        lift_path,
        seed,
    ]
    frozen = {str(p): fingerprint(p) for p in paths}
    output.mkdir(parents=True)
    begun = perf_counter()
    model, lift, reference, force, _ = STREAM["load_lifted_reference"](metadata, lift_path, seed)
    np.testing.assert_array_equal(
        force, np.load(cache / "initial_residual_force.npy", allow_pickle=False)
    )
    if STREAM["digest_file"](cache / "affine.npy") != affine_meta["affine_sha256"]:
        raise ValueError("Complete affine history changed")
    affine = np.load(cache / "affine.npy", mmap_mode="r", allow_pickle=False)
    data = {}
    with np.load(capture, allow_pickle=False) as source:
        for key in source.files:
            data[key] = source[key]
    q, values = data.pop("accepted_displacement"), data.pop("accepted_response")
    count, d = len(q), reference.constraints.size
    if count != failure["failed_step"]:
        raise ValueError("Accepted prefix does not end immediately before the failure")
    for expected, actual in [
        (data["base_dn"], reference.base_dn),
        (data["base_ds"], reference.base_ds),
        (data["contact_weights"], reference.constraints.contact_weights),
        (data["reduced_weights"], reference.constraints.reduced_weights),
        (data["previous_displacement"], q[-1, 1]),
        (data["plastic_slip"], values[-1, :, 1, 2]),
        (data["cumulative_slip"], values[-1, :, 1, 3]),
    ]:
        np.testing.assert_array_equal(expected, actual)
    matrix = np.load(kernel / "initial_impedance.npy", allow_pickle=False)
    matrix[[d - 1, 2 * d - 1], [d - 1, 2 * d - 1]] += reference.spring
    np.testing.assert_array_equal(matrix, data["matrix"])
    with np.load(kernel / "elastic.npz", allow_pickle=False) as saved:
        stage_time, elastic, ramp = [
            saved[k] for k in ["stage_time", "elastic", "uninterrupted_actuator"]
        ]
    np.testing.assert_array_equal(stage_time[count], data["stage_time"])
    if affine.shape != elastic.shape:
        raise ValueError("Affine cache does not cover the original full stage grid")
    plan = CQPlan(**metadata["plan"])
    maps = []
    for i, record in enumerate(metadata["family_weights"]):
        write_json(
            output / "progress.json",
            dict(
                phase="verifying_complete_original_weights",
                family=i,
                seconds=perf_counter() - begun,
            ),
        )
        if STREAM["digest_file"](record["path"]) != record["sha256"]:
            raise ValueError("Original kernel time weights changed")
        maps.append(np.load(record["path"], mmap_mode="r", allow_pickle=False))
    histories, cache_record = CACHE["cache_history_prefixes"](
        [BoundaryHistory(plan, a) for a in maps], capacity_bytes=144 * 1024**2, max_lags=32
    )
    wrap = np.load(kernel / "shift_wrap.npy", allow_pickle=False)
    prefix = reconstruct_prefix(
        reference,
        lift.state,
        q,
        values,
        stage_time,
        ramp,
        stop_coverage=model.reference.payload["loading"]["stop_coverage"],
        diagnostic=metadata["source_mesh_arguments"].get("diagnostic_actuator_increment")
        is not None,
    )
    factory = RESTORE["HISTORY"]["CompactCorrectionHistory"]
    restored = factory(*histories, wrap, cache_mb=8, fft_rows=4)
    restore_record = RESTORE["restore_correction_prefix"](
        restored, prefix["sources"], prefix["active"]
    )
    replay = factory(*histories, wrap, cache_mb=8, fft_rows=4)
    replay.append(0, prefix["sources"][0], active=prefix["active"][0])
    residuals = np.zeros(count)
    residuals[0] = np.max(abs(force / reference.constraints.reduced_weights))
    a0 = matrix + np.diag(np.tile(prefix["cdiag"], 2))
    weights = reference.constraints.contact_weights
    wr = np.tile(reference.constraints.reduced_weights, 2)
    for k in range(1, count):
        prediction = elastic[k].ravel() + replay.past(k)
        prediction -= affine[k].ravel()
        hold_force = np.zeros((2, d))
        hold_force[:, -1] = reference.spring * (prefix["actuator"][k] - ramp[k])
        forcing = (
            -np.tile(prefix["fc0"], 2) + np.tile(force, 2) - a0 @ prediction - hold_force.ravel()
        )
        residuals[k] = np.max(
            abs((matrix @ q[k].ravel() + forcing + contact_force(values[k], weights).ravel()) / wr)
        )
        if not np.isfinite(residuals[k]) or residuals[k] >= 1e-7:
            raise ValueError(f"Accepted-prefix full equilibrium failed at {k}: {residuals[k]}")
        replay.append(k, prefix["sources"][k], active=prefix["active"][k])
        if k % 128 == 0:
            write_json(
                output / "progress.json",
                dict(
                    phase="replaying_original_accepted_prefix",
                    accepted_steps_checked=k,
                    prefix_steps=count,
                    seconds=perf_counter() - begun,
                ),
            )
    assert restored.start == replay.start and restored.next_index == replay.next_index == count
    assert restored.maximum_inactive_source_force == replay.maximum_inactive_source_force
    if restored.online is not None:
        np.testing.assert_array_equal(restored.online.previous_f1, replay.online.previous_f1)
        for a, b in zip(restored.online.streams, replay.online.streams, strict=True):
            assert a.next_step == b.next_step
            np.testing.assert_array_equal(a.inputs, b.inputs)
            np.testing.assert_array_equal(a.pending[a.next_step :], b.pending[b.next_step :])
    forcing = forcing_at(count, reference, matrix, prefix, restored, elastic, affine, ramp, force)
    mismatch = float(np.max(abs((forcing - data["forcing"]) / wr)))
    if not np.isfinite(mismatch) or mismatch > 1e-9:
        raise ValueError(f"Original failed-equation forcing differs: {mismatch} MPa")
    payload = dict(
        prefix_residual_mpa=residuals,
        prefix_newton_iterations=np.full(count, -1, dtype=int),
        actuator_prefix=prefix["actuator"],
        initial_contact_force=prefix["fc0"],
        initial_residual_force=force,
    )
    if restored.online is not None:
        online = restored.online
        np.testing.assert_array_equal(online.streams[0].inputs, online.streams[1].inputs)
        payload["active_inputs"] = online.streams[0].inputs[: online.next_step]
        payload["previous_f1"] = online.previous_f1
        stop = max(
            online.next_step,
            max(
                (
                    int(np.max(np.flatnonzero(np.any(s.pending != 0, axis=(1, 2))))) + 1
                    if np.any(s.pending != 0)
                    else online.next_step
                )
                for s in online.streams
            ),
        )
        for i, stream in enumerate(online.streams):
            payload[f"future_pending_{i}"] = stream.pending[online.next_step : stop]
    else:
        stop = 0
    np.savez(output / "history_state.npz", **payload)
    if any(fingerprint(p) != h for p, h in frozen.items()):
        raise ValueError("Checkpoint validation sources changed")
    write_json(
        output / "summary.json",
        dict(
            status="validated_original_fine_prefix_and_full_future_history",
            source_sha256=frozen,
            capture=str(capture),
            capture_sha256=failure["failed_equilibrium_sha256"],
            affine_cache=str(cache),
            affine_sha256=affine_meta["affine_sha256"],
            history_state_sha256=STREAM["digest_file"](output / "history_state.npz"),
            kernel=str(kernel),
            next_step=count,
            first_correction_step=restored.start,
            local_history_next_step=None if restored.online is None else restored.online.next_step,
            pending_stop=stop,
            maximum_inactive_source_force=restored.maximum_inactive_source_force,
            held_actuator=prefix["held"],
            hold_time_s=prefix["hold_time"],
            maximum_prefix_residual_mpa=float(residuals.max()),
            failed_forcing_normalized_difference_mpa=mismatch,
            failed_forcing_bitwise_equal=bool(np.array_equal(forcing, data["forcing"])),
            replay_and_restored_complete_future_bitwise_equal=True,
            history_prefix_cache=cache_record,
            restore_record=restore_record,
            seconds=perf_counter() - begun,
            unavailable_original_newton_count_sentinel=-1,
            complete_event_resumed=False,
            scope="Independent full-prefix force balance and exact future-history restoration only; no new root committed or time step advanced.",
        ),
    )


if __name__ == "__main__":
    main()

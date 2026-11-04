"""Stream the original affine initialization history in physical column order.

The mathematical formula and every lag are unchanged. This is a primal
cache keyed by the exact initial force and kernel; it must not be treated
as a constant when differentiating a parameter-dependent initialization.
"""

import argparse
import hashlib
import json
import runpy
from pathlib import Path
from time import perf_counter, process_time
from types import SimpleNamespace

import numpy as np

from finiteblock_bem.friction import ContactState
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))
REFINED = runpy.run_path(str(TOOLS / "refined_contact.py"))


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_initial_residual(first, second, force, steps, *, column_hook=None, rows=8):
    """Project complete time columns before applying the original cumulative sum.

    Only force contractions change their summation layout. The lag-zero
    subtraction still calls the original row-tiled matrix-vector operation.
    No term is discarded even when a force component is very small or zero.
    """
    force = np.asarray(force)
    shape = np.shape(first)
    if (
        len(shape) != 3
        or shape != np.shape(second)
        or shape[1] != shape[2]
        or force.shape != (shape[1],)
        or force.dtype != np.dtype("float64")
        or first.dtype != np.dtype("float64")
        or second.dtype != np.dtype("float64")
        or not np.all(np.isfinite(force))
        or not isinstance(steps, int)
        or isinstance(steps, bool)
        or not 1 <= steps < shape[0]
        or not isinstance(rows, int)
        or isinstance(rows, bool)
        or rows < 1
    ):
        raise ValueError("Require matching float64 weights, finite force and valid horizon/rows")
    d = shape[1]
    first_projected = np.zeros((steps - 1, d), dtype=np.float64, order="F")
    second_projected = np.zeros_like(first_projected)
    work = np.empty_like(first_projected)
    for column in range(d):
        if column_hook is not None:
            column_hook(column)
        np.multiply(first[1:steps, :, column], force[column], out=work)
        np.add(first_projected, work, out=first_projected)
        np.multiply(second[1:steps, :, column], force[column], out=work)
        np.add(second_projected, work, out=second_projected)
    np.cumsum(first_projected, axis=0, out=first_projected)
    for begin in range(0, d, rows):
        end = min(begin + rows, d)
        second_projected[:, begin:end] -= np.asarray(second[0, begin:end]) @ force
    if not np.all(np.isfinite(first_projected)) or not np.all(np.isfinite(second_projected)):
        raise ValueError("Nonfinite complete affine initialization history")
    answer = np.zeros((steps + 1, 2, d), dtype=np.float64)
    coefficient = np.linalg.inv(RADAU["A"]) @ np.ones(2)
    for stage in range(2):
        np.multiply(second_projected, coefficient[stage], out=work)
        np.add(first_projected, work, out=answer[2:, stage])
    return answer


def load_lifted_reference(metadata, lift_path, seed_path):
    """Rehydrate the exact saved body lift through the original reference builder."""
    options = metadata["source_mesh_arguments"]
    current = environment_record(options["case"])
    keys = ["source_sha256", "case_sha256", "restart_data_sha256", "numpy", "scipy"]
    if any(current[key] != metadata["environment"][key] for key in keys):
        raise ValueError("Prepared physical inputs, complete package or runtime changed")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    model, seed_record = runpy.run_path(str(TOOLS / "restart_seed_experiment.py"))[
        "apply_restart_seed"
    ](model, seed_path)
    with np.load(lift_path, allow_pickle=False) as saved:
        lift = SimpleNamespace(**{k: saved[k] for k in saved.files})
    lift.state = ContactState(lift.plastic_slip, lift.cumulative_slip)
    np.testing.assert_array_equal(lift.state.plastic_slip, model.restart.plastic_slip)
    np.testing.assert_array_equal(lift.state.cumulative_slip, model.restart.cumulative_slip)
    space = REFINED["ContactQuadrature"].build(
        model.reference.moving.starts[model.moving_contact, 1], 1
    )
    reference = REFINED["dynamic_refined_reference"](
        model, space, lift, np.zeros((space.dofs, space.dofs))
    )
    initial, _, response = reference.evaluate(
        np.zeros(space.dofs),
        np.zeros(space.dofs),
        model.reference.normal_pressure,
        model.reference.restart_actuator,
        lift.state,
        jacobian=False,
    )
    np.testing.assert_array_equal(response.pressure, lift.pressure)
    np.testing.assert_array_equal(response.shear, lift.shear)
    if not np.all(np.isfinite(initial)) or np.max(abs(initial)) >= 1e-7:
        raise ValueError("Saved lift fails the original static-equilibrium gate")
    return model, lift, reference, initial * reference.constraints.reduced_weights, seed_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("initial_lift", type=Path)
    parser.add_argument("--restart-seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, kernel = args.output.resolve(), args.kernel.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Choose a fresh owned affine-cache directory")
    metadata_path = kernel / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["status"] != "completed_compact_radau_resolvent_kernel_only":
        raise ValueError("Require the completed original compact kernel")
    paths = [
        Path(__file__),
        metadata_path,
        args.initial_lift.resolve(),
        args.restart_seed.resolve(),
        TOOLS / "radau_initialization.py",
        TOOLS / "radau_experiment.py",
        TOOLS / "refined_contact.py",
        TOOLS / "restart_seed_experiment.py",
    ]
    frozen = {str(p): fingerprint(p) for p in paths}
    output.mkdir(parents=True)
    begun, cpu = perf_counter(), process_time()
    _model, _lift, reference, force, seed_record = load_lifted_reference(
        metadata, args.initial_lift, args.restart_seed
    )
    np.save(output / "initial_residual_force.npy", force, allow_pickle=False)
    arrays, hashers, stats = [], [], []
    for record in metadata["family_weights"]:
        path = Path(record["path"])
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            a.dtype != np.float64
            or a.shape
            != (metadata["full_steps"] + 1, reference.constraints.size, reference.constraints.size)
            or not a.flags.f_contiguous
        ):
            raise ValueError("Require the original physical Fortran float64 time weights")
        if path.stat().st_size != a.offset + a.nbytes:
            raise ValueError("Unexpected bytes outside the time-weight array")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            hasher.update(stream.read(a.offset))
        arrays.append(a)
        hashers.append(hasher)
        stats.append(path.stat())

    def column_hook(column):
        for array, hasher in zip(arrays, hashers, strict=True):
            flat = array[:, :, column].ravel(order="F")
            if not np.shares_memory(flat, array) or not flat.flags.c_contiguous:
                raise ValueError("Physical column must be a view, never a full-array copy")
            hasher.update(memoryview(flat))
        if column % 16 == 0:
            write_json(
                output / "progress.json",
                dict(
                    phase="streaming_complete_columns",
                    completed_columns=column,
                    total_columns=reference.constraints.size,
                    seconds=perf_counter() - begun,
                    cpu_seconds=process_time() - cpu,
                ),
            )

    answer = stream_initial_residual(
        *arrays, force, metadata["full_steps"], column_hook=column_hook
    )
    for record, array, hasher, before in zip(
        metadata["family_weights"], arrays, hashers, stats, strict=True
    ):
        after = Path(record["path"]).stat()
        if hasher.hexdigest() != record["sha256"] or (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("Original complete time weights changed")
        array._mmap.close()
    if any(fingerprint(p) != h for p, h in frozen.items()):
        raise ValueError("Affine-cache source or initial lift changed")
    path = output / "affine.npy"
    np.save(path, answer, allow_pickle=False)
    write_json(
        output / "summary.json",
        dict(
            status="completed_original_formula_column_streamed_primal_affine_cache",
            source_sha256=frozen,
            initial_lift_sha256=fingerprint(args.initial_lift),
            seed=seed_record,
            initial_residual_force_sha256=fingerprint(output / "initial_residual_force.npy"),
            affine_sha256=digest_file(path),
            complete_kernel_weight_hashes_verified=True,
            all_lags_retained=True,
            steps=metadata["full_steps"],
            shape=list(answer.shape),
            seconds=perf_counter() - begun,
            cpu_seconds=process_time() - cpu,
            peak_projection_payload_bytes=answer.nbytes
            + 3 * (metadata["full_steps"] - 1) * reference.constraints.size * 8,
            original_full_event_reproduced=False,
            pde_resume_forcing_not_yet_validated=True,
            scope="Primal affine cache for this exact initial force. Different summation layout; requires original failure-forcing and prefix validation before resume. Parameter-dependent initialization AD must retain derivatives through this operator.",
        ),
    )


if __name__ == "__main__":
    main()

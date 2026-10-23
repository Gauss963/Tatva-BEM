"""Exact disk-backed half-spectrum prototype, outside production numerical code.

The original invert_spectrum function consumes bounded row copies from a
file-backed spectrum. Public mmap mappings close after each operation so
resident spectrum pages do not accumulate. Weights remain dense in RAM.
"""

import argparse
import hashlib
import json
import mmap
import os
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from types import FunctionType

import numpy as np

from finiteblock_bem.cq import convolution_weights, plan_cq
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.sampling import SamplingOptions, parsimonious_weights

ROOT = Path(__file__).resolve().parents[1]


class DiskSpectrum:
    """Minimal spectrum storage: first-axis writes and copied arbitrary reads."""

    def __init__(self, path, plan):
        self.path = path
        self.shape = (plan.fft_size // 2 + 1, plan.dofs, plan.dofs)
        self.dtype = np.dtype("complex128")
        self.written = np.zeros(self.shape[0], dtype=bool)
        self.pending_writes = 0
        with path.open("w+b") as stream:
            np.lib.format.write_array_header_2_0(
                stream,
                dict(descr=self.dtype.str, fortran_order=False, shape=self.shape),
            )
            self.offset = stream.tell()
            stream.truncate(self.offset + int(np.prod(self.shape)) * self.dtype.itemsize)

    def sync(self):
        if self.pending_writes:
            with self.path.open("r+b") as stream:
                os.fsync(stream.fileno())
            self.pending_writes = 0

    def _mapped(self, operation, first, last, *, write=False):
        matrix_bytes = self.shape[1] * self.shape[2] * self.dtype.itemsize
        offset = self.offset + first * matrix_bytes
        aligned = offset // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
        length = offset - aligned + (last - first) * matrix_bytes
        with self.path.open("r+b" if write else "rb") as stream:
            mapping = mmap.mmap(
                stream.fileno(),
                length,
                offset=aligned,
                access=mmap.ACCESS_WRITE if write else mmap.ACCESS_READ,
            )
            array = np.ndarray(
                (last - first, *self.shape[1:]), self.dtype, buffer=mapping, offset=offset - aligned
            )
            try:
                # Read operations MUST copy their result; no array views escape.
                result = operation(array)
            finally:
                del array
                mapping.close()
        return result

    def __setitem__(self, key, value):
        if not isinstance(key, (int, slice)):
            raise ValueError("Spectrum writes must contain complete matrices")
        if isinstance(key, int):
            if not -self.shape[0] <= key < self.shape[0]:
                raise IndexError("Frequency index outside spectrum")
            key %= self.shape[0]
        first, last = (key, key + 1) if isinstance(key, int) else key.indices(self.shape[0])[:2]
        if isinstance(key, slice) and key.step not in [None, 1]:
            raise ValueError("Only unit-stride spectrum writes supported")
        if first < last:
            self._mapped(
                lambda array: array.__setitem__(0 if isinstance(key, int) else slice(None), value),
                first,
                last,
                write=True,
            )
        self.written[key] = True
        self.pending_writes += last - first
        if self.pending_writes >= 64:
            self.sync()

    def __getitem__(self, key):
        if not np.all(self.written):
            raise ValueError("Complete the entire half spectrum before inversion")
        self.sync()
        key = key if isinstance(key, tuple) else (key,)
        first_axis, tail = key[0], key[1:]
        if isinstance(first_axis, int):
            if not -self.shape[0] <= first_axis < self.shape[0]:
                raise IndexError("Frequency index outside spectrum")
            first = first_axis % self.shape[0]
            return self._mapped(
                lambda array: np.array(array[(0, *tail)], copy=True), first, first + 1
            )
        first, last, stride = first_axis.indices(self.shape[0])
        if stride != 1:
            raise ValueError("Only unit-stride frequency reads supported")
        # Reading all frequency rows through one huge map lets OS fault-around
        # pull unrelated spectrum pages into RSS. Map at most 64 matrices.
        shape = np.broadcast_to(np.empty((), self.dtype), self.shape[1:])[tail].shape
        result = np.empty((max(last - first, 0), *shape), self.dtype)
        for begin in range(first, last, 64):
            end = min(begin + 64, last)
            result[begin - first : end - first] = self._mapped(
                lambda array: np.array(array[(slice(None), *tail)], copy=True), begin, end
            )
        return result


def disk_weights(transfer, plan, path, *, adaptive=False):
    spectrum = DiskSpectrum(path, plan)

    class ArrayAllocation:
        # This isolated prototype substitutes only the spectrum allocation in
        # a private function copy. It does not monkeypatch NumPy or a module.
        def __getattr__(self, name):
            return getattr(np, name)

        def empty(self, shape, dtype=float):
            if shape != spectrum.shape or np.dtype(dtype) != spectrum.dtype:
                raise ValueError("Unexpected allocation in the experimental sampler")
            return spectrum

    original = parsimonious_weights if adaptive else convolution_weights
    bindings = dict(original.__globals__, np=ArrayAllocation())
    sampler = FunctionType(original.__code__, bindings, original.__name__, original.__defaults__)
    sampler.__kwdefaults__ = original.__kwdefaults__
    # All sampler code, checks and the original inversion remain unchanged.
    # A production API would make this storage injection explicit.
    return sampler(transfer, plan, **({"options": SamplingOptions()} if adaptive else {}))


def worker(args):
    start = perf_counter()
    model = None
    if args.bem:
        from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

        model = load_nodal_model(
            ROOT / "src/finiteblock_bem/resources/cases/pmma_event16.toml",
            max_panel_length=args.bem_h,
        )
        prepared = NodalInterfaceTransfer(model, quadrature=12)
        transfer = prepared.evaluate
        plan = plan_cq(2 * len(model.moving_contact) + 1, 8e-6, args.steps, max_memory_mb=2048)
    else:
        plan = plan_cq(args.dofs, 0.01, args.steps, max_memory_mb=2048)
        rng = np.random.default_rng(53)
        a, b, c, d = rng.normal(size=(4, args.dofs, args.dofs)) / args.dofs

        def transfer(s):
            return s * a + b + c / (s + 2) + d / (s + 7)

    if args.variant == "reference":
        weights = (parsimonious_weights if args.adaptive else convolution_weights)(transfer, plan)
    else:
        weights = disk_weights(
            transfer, plan, args.output / "half-spectrum.npy", adaptive=args.adaptive
        )
    metadata = None
    if args.adaptive:
        weights, metadata = weights
    np.save(args.output / f"{args.variant}.npy", weights, allow_pickle=False)
    record = dict(
        variant=args.variant,
        steps=args.steps,
        dofs=plan.dofs,
        seconds=perf_counter() - start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        spectrum_mib=(plan.fft_size // 2 + 1) * plan.dofs**2 * 16 / 1024**2,
        weights_mib=weights.nbytes / 1024**2,
        sampling=metadata,
    )
    if model is not None:
        record["dynamic_diagnostic"] = dynamic_check(model, plan, weights, args)
    write_json(args.output / f"{args.variant}.json", record)
    print(json.dumps(record), flush=True)


def dynamic_check(model, plan, weights, args):
    from finiteblock_bem import nodal_dynamics
    from finiteblock_bem.adjoint import trajectory_adjoint
    from finiteblock_bem.cq import BoundaryHistory
    from finiteblock_bem.dynamics import LoadingSchedule
    from finiteblock_bem.nodal import NodalInterfaceEquilibrium, initialize_nodal_restart

    history = BoundaryHistory(plan, weights)
    lift = initialize_nodal_restart(model, quadrature=12)
    schedule = LoadingSchedule(
        lambda _: model.reference.normal_pressure,
        lambda t: model.reference.restart_actuator + 2 * t / (plan.steps * plan.dt),
        "disk_storage_accelerated_diagnostic",
    )
    original = nodal_dynamics.build_nodal_history
    try:
        nodal_dynamics.build_nodal_history = lambda *a, **kw: history
        result = nodal_dynamics.simulate_nodal(
            model, plan.dt, plan.steps, lift=lift, schedule=schedule, backend="jax", quadrature=12
        )
    finally:
        nodal_dynamics.build_nodal_history = original
    if result.diagnostics["status"] != "completed_requested_horizon":
        write_json(args.output / f"{args.variant}_failure.json", result.diagnostics)
        raise RuntimeError("Diagnostic forward did not finish")
    equilibrium = NodalInterfaceEquilibrium(model, lift, history.instantaneous)
    answer = trajectory_adjoint(
        model.contact_case(), lift, history, result, equilibrium=equilibrium
    )
    fields = {
        key: getattr(result, key)
        for key in [
            "pressure",
            "shear",
            "contact_jumps",
            "platen",
            "plastic_slip",
            "cumulative_slip",
        ]
    }
    fields.update(
        {
            key: answer[key]
            for key in [
                "gradient",
                "actuator_gradient",
                "initial_plastic_slip_gradient",
                "initial_cumulative_slip_gradient",
            ]
        }
    )
    np.savez(args.output / f"{args.variant}-dynamic.npz", **fields)
    value = float(np.max(result.cumulative_slip))
    if value <= 0.1:
        raise AssertionError("The storage diagnostic must exercise nonlinear sliding")
    return dict(max_cumulative_slip_mm=value, objective_mm=answer["value_mm"])


def small_checks(output):
    plan = plan_cq(3, 0.01, 4)
    spectrum = DiskSpectrum(output / "mapping-check.npy", plan)
    values = np.arange(np.prod(spectrum.shape)).reshape(spectrum.shape) * (1 + 2j)
    spectrum[:2] = values[:2]
    try:
        spectrum[0]
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete spectrum was accepted")
    spectrum[2:] = values[2:]
    np.testing.assert_array_equal(spectrum[:, :2, 1:], values[:, :2, 1:])
    np.testing.assert_array_equal(np.load(spectrum.path), values)
    copy = spectrum[0]
    copy[:] = 0
    np.testing.assert_array_equal(spectrum[0], values[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--dofs", type=int, default=64)
    parser.add_argument("--variant", choices=["reference", "disk"])
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument(
        "--bem", action="store_true", help="Use BEM kernels and a short sliding/AD diagnostic"
    )
    parser.add_argument(
        "--bem-h", type=float, default=80, help="Panel upper length for BEM diagnostic"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/disk-spectrum-experiment")
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 32 if args.bem else 4097
    args.output.mkdir(parents=True, exist_ok=True)
    if args.variant:
        worker(args)
        return
    small_checks(args.output)
    for variant in ["reference", "disk"]:
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--steps",
            str(args.steps),
            "--dofs",
            str(args.dofs),
            "--output",
            str(args.output),
            "--variant",
            variant,
        ]
        environment = os.environ.copy()
        if args.adaptive:
            command.append("--adaptive")
        if args.bem:
            command.extend(["--bem", "--bem-h", str(args.bem_h)])
        for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
            environment[name] = "1"
        subprocess.run(command, check=True, env=environment)
    a = np.load(args.output / "reference.npy", mmap_mode="r")
    b = np.load(args.output / "disk.npy", mmap_mode="r")
    maximum, bitwise = 0.0, True
    for start in range(0, len(a), 32):
        maximum = max(maximum, float(np.max(abs(a[start : start + 32] - b[start : start + 32]))))
        bitwise &= np.array_equal(a[start : start + 32], b[start : start + 32])
        np.testing.assert_allclose(
            b[start : start + 32], a[start : start + 32], rtol=1e-10, atol=1e-10
        )
    fields = {}
    if args.bem:
        with (
            np.load(args.output / "reference-dynamic.npz") as first,
            np.load(args.output / "disk-dynamic.npz") as second,
        ):
            for key in first.files:
                fields[key] = float(np.max(abs(first[key] - second[key])))
                np.testing.assert_allclose(second[key], first[key], rtol=1e-8, atol=1e-10)
    records = [json.loads((args.output / f"{v}.json").read_text()) for v in ["reference", "disk"]]
    if records[0]["sampling"] != records[1]["sampling"]:
        raise AssertionError("The storage change altered adaptive-sampling decisions")
    write_json(
        args.output / "summary.json",
        dict(
            records=records,
            max_weight_difference=maximum,
            weights_bitwise_identical=bool(bitwise),
            small_mapping_checks_passed=True,
            dynamic_field_differences=fields,
            environment=environment_record(),
            prototype_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope=(
                "Exact storage prototype with short accelerated BEM sliding/adjoint; not event-16"
                if args.bem
                else "Exact storage prototype with rational transfer; no BEM trajectory or speed claim"
            ),
        ),
    )


if __name__ == "__main__":
    main()

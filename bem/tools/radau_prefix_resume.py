"""Restore a validated prefix before the unchanged original Radau time loop.

The private function adaptation inserts one restore call and changes only
the loop's start and optional initial guess. All physical step equations,
acceptance checks, endpoint state commits and stopping rules stay original.
"""

import ast
import inspect
import json
import runpy
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from finiteblock_bem.friction import ContactState
from finiteblock_bem.io import fingerprint

TOOLS = Path(__file__).resolve().parent
HISTORY = runpy.run_path(str(TOOLS / "compact_radau_history.py"))
STREAM = runpy.run_path(str(TOOLS / "stream_initial_residual.py"))


def restore_saved_history(memory, record, payload):
    """Fill fresh original buffers with the validated, serialized future state."""
    if (
        memory.start is not None
        or memory.online is not None
        or memory.next_index not in [0, 1]
        or memory.maximum_inactive_source_force
    ):
        raise ValueError("Restore requires an empty original correction history")
    count, start = record["next_step"], record["first_correction_step"]
    if not isinstance(count, int) or count < 1:
        raise ValueError("Invalid accepted prefix length")
    memory.maximum_inactive_source_force = record["maximum_inactive_source_force"]
    if start is not None:
        length, stop = record["local_history_next_step"], record["pending_stop"]
        d = memory.families[0].plan.dofs
        if (
            not isinstance(start, int)
            or not 0 <= start < count
            or length != count - start
            or not length <= stop <= memory.families[0].plan.steps + 1
        ):
            raise ValueError("Saved history indices do not align with the original CQ horizon")
        shapes = dict(
            active_inputs=(length, d, 2),
            previous_f1=(d, 2),
            future_pending_0=(stop - length, d, 2),
            future_pending_1=(stop - length, d, 2),
        )
        for key, shape in shapes.items():
            a = payload[key]
            if a.shape != shape or a.dtype != np.dtype("float64") or not np.all(np.isfinite(a)):
                raise ValueError(f"Invalid saved history field: {key}")
        online = HISTORY["CompactRadauConvolution"](
            *memory.families, memory.wrap, cache_mb=memory.cache_mb, fft_rows=memory.fft_rows
        )
        for i, stream in enumerate(online.streams):
            stream.inputs[:length] = payload["active_inputs"]
            stream.pending[length:stop] = payload[f"future_pending_{i}"]
            stream.next_step = length
        online.previous_f1[:] = payload["previous_f1"]
        memory.start, memory.online = start, online
    elif record["local_history_next_step"] is not None or record["pending_stop"]:
        raise ValueError("Inactive prefix cannot contain a started convolution")
    memory.next_index = count
    return memory


@dataclass
class PrefixSnapshot:
    record: dict
    capture: dict
    payload: dict
    root_seed: np.ndarray | None = None
    applied: bool = False
    boundary_forcing_error_mpa: float | None = None

    @classmethod
    def load(cls, directory, *, root_path=None, root_key="random_hybrid"):
        directory = Path(directory).resolve()
        record = json.loads((directory / "summary.json").read_text())
        if record["status"] != "validated_original_fine_prefix_and_full_future_history":
            raise ValueError("Require the successful complete-prefix validation receipt")
        if any(fingerprint(p) != h for p, h in record["source_sha256"].items()):
            raise ValueError("Validated checkpoint source or protected physical input changed")
        if STREAM["digest_file"](directory / "history_state.npz") != record["history_state_sha256"]:
            raise ValueError("Saved full future history changed")
        if STREAM["digest_file"](record["capture"]) != record["capture_sha256"]:
            raise ValueError("Original accepted prefix changed")
        with np.load(record["capture"], allow_pickle=False) as data:
            capture = {k: data[k] for k in data.files}
        with np.load(directory / "history_state.npz", allow_pickle=False) as data:
            payload = {k: data[k] for k in data.files}
        root_seed = None
        if root_path is not None:
            with np.load(root_path, allow_pickle=False) as data:
                root_seed = data[root_key]
        return cls(record, capture, payload, root_seed)

    def restore(self, frame):
        """Copy accepted fields and return the original loop's restored locals."""
        if self.applied:
            raise ValueError("A checkpoint cannot be applied twice")
        count = self.record["next_step"]
        q, values = self.capture["accepted_displacement"], self.capture["accepted_response"]
        reference, state = frame["reference"], frame["state"]
        d = reference.constraints.size
        if (
            not 1 <= count <= frame["steps"] + 1
            or q.shape != frame["q"][:count].shape
            or values.shape != frame["values"][:count].shape
        ):
            raise ValueError("Checkpoint fields or horizon differ from the current simulation")
        for a, b in [
            (q[0], frame["q"][0]),
            (values[0], frame["values"][0]),
            (self.capture["matrix"], frame["matrix"]),
            (self.capture["base_dn"], reference.base_dn),
            (self.capture["base_ds"], reference.base_ds),
            (self.capture["contact_weights"], reference.constraints.contact_weights),
            (self.capture["reduced_weights"], reference.constraints.reduced_weights),
            (self.capture["parameters"], np.asarray(frame["system"].parameters)),
            (self.payload["initial_contact_force"], frame["fc0"]),
            (np.tile(self.payload["initial_residual_force"], 2), frame["residual_force"]),
            (values[0, :, 1, 2], state.plastic_slip),
            (values[0, :, 1, 3], state.cumulative_slip),
        ]:
            np.testing.assert_array_equal(a, b)
        if (
            self.record["maximum_prefix_residual_mpa"] >= 1e-7
            or frame["actuator_plateau"] is not None
        ):
            raise ValueError("Unaccepted prefix or unsupported prescribed loading change")
        frame["q"][:count], frame["values"][:count] = q, values
        for name, key in [
            ("actuator", "actuator_prefix"),
            ("residual", "prefix_residual_mpa"),
            ("iterations", "prefix_newton_iterations"),
        ]:
            if frame[name][:count].shape != self.payload[key].shape:
                raise ValueError("Saved endpoint diagnostics have inconsistent shapes")
            frame[name][:count] = self.payload[key]
        memory = restore_saved_history(frame["memory"], self.record, self.payload)
        held = self.record["held_actuator"]
        old = ContactState(frame["values"][count - 1, :, 1, 2], frame["values"][count - 1, :, 1, 3])
        if count <= frame["steps"]:
            prediction = frame["elastic"][count].ravel() + memory.past(count)
            if frame["affine"] is not None:
                prediction -= frame["affine"][count].ravel()
            actuator = frame["ramp"][count] if held is None else np.full(2, held)
            hold_force = np.zeros((2, d))
            hold_force[:, -1] = reference.spring * (actuator - frame["ramp"][count])
            forcing = (
                -np.tile(frame["fc0"], 2)
                + frame["residual_force"]
                - frame["a0"] @ prediction
                - hold_force.ravel()
            )
            error = float(
                np.max(
                    abs(
                        (forcing - self.capture["forcing"])
                        / np.tile(reference.constraints.reduced_weights, 2)
                    )
                )
            )
            if not np.isfinite(error) or error > 1e-9:
                raise ValueError(f"Checkpoint boundary forcing differs: {error} MPa")
            self.boundary_forcing_error_mpa = error
        self.applied = True
        # All arrays were copied into the original integrator's own storage.
        # Release the large duplicate prefix before advancing physical time.
        del self.capture["accepted_displacement"], self.capture["accepted_response"]
        return old, held, self.record["hold_time_s"], memory, count

    def guess(self, step, previous):
        if self.root_seed is not None and step == self.record["next_step"]:
            seed = np.asarray(self.root_seed)
            if seed.shape != (2 * len(previous),) or not np.all(np.isfinite(seed)):
                raise ValueError("Invalid recovered-root initial guess")
            return seed
        return np.tile(previous, 2)


def resumable_simulator(original, snapshot):
    """Make exactly three explicit private AST adaptations to the original loop."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    loops = [
        v
        for v in function.body
        if isinstance(v, ast.For) and isinstance(v.target, ast.Name) and v.target.id == "k"
    ]
    if len(loops) != 1 or ast.dump(loops[0].iter) != ast.dump(
        ast.parse("range(1, steps + 1)", mode="eval").body
    ):
        raise ValueError("Original physical loop changed; re-audit prefix adaptation")
    loop = loops[0]
    calls = [
        v
        for v in ast.walk(loop)
        if isinstance(v, ast.Call)
        and isinstance(v.func, ast.Attribute)
        and isinstance(v.func.value, ast.Name)
        and v.func.value.id == "system"
        and v.func.attr == "solve"
    ]
    expected = ast.parse("np.tile(q[k - 1, 1], 2)", mode="eval").body
    if (
        len(calls) != 1
        or len(calls[0].args) != 4
        or ast.dump(calls[0].args[2]) != ast.dump(expected)
    ):
        raise ValueError("Original root call changed; re-audit prefix adaptation")
    insertion = ast.parse(
        "old, held, hold_time, memory, _resume_start = _restore_radau_prefix(locals())"
    ).body[0]
    function.body.insert(function.body.index(loop), insertion)
    loop.iter.args[0] = ast.Name(id="_resume_start", ctx=ast.Load())
    calls[0].args[2] = ast.parse("_resume_radau_guess(k, q[k - 1, 1])", mode="eval").body
    namespace = {
        **original.__globals__,
        "_restore_radau_prefix": snapshot.restore,
        "_resume_radau_guess": snapshot.guess,
    }
    exec(
        compile(ast.fix_missing_locations(tree), str(TOOLS / "<resumed-original-loop>"), "exec"),
        namespace,
    )
    return namespace[original.__name__]

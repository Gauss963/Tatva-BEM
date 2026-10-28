"""Verified, bounded disk checkpoints for one raw Radau kernel construction.

Only fsynced consecutive matrices are published. Existing committed matrices
are verified before they become readable; uncommitted suffixes are recomputed.
No checkpoint changes a raw/operator acceptance test or truncates causal history.
"""

import ast
import fcntl
import hashlib
import inspect
import json
import os
import runpy
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from finiteblock_bem.cq_storage import DiskSpectrum, spectrum_bytes

TOOLS = Path(__file__).resolve().parent
BUFFER = runpy.run_path(str(TOOLS / "buffered_spectrum.py"))["BufferedSpectrumWriter"]
PARTITION = runpy.run_path(str(TOOLS / "seeded_raw_sampling.py"))["validate_partition"]
CHANNELS = tuple(f"{kind}-family{i}.npy" for kind in ["work", "compact"] for i in range(2))


def atomic_json(path, record):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest_range(path, offset=0, count=None):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        while count is None or count:
            data = stream.read(4 * 1024**2 if count is None else min(count, 4 * 1024**2))
            if not data:
                if count:
                    raise ValueError("Truncated committed spectrum prefix")
                break
            value.update(data)
            if count is not None:
                count -= len(data)
    return value.hexdigest()


def header(path, plan):
    with Path(path).open("rb") as stream:
        if np.lib.format.read_magic(stream) != (2, 0):
            raise ValueError("Require the original version-2 disk spectrum header")
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        offset = stream.tell()
    if (
        shape != (plan.fft_size // 2 + 1, plan.dofs, plan.dofs)
        or fortran
        or dtype != np.dtype("complex128")
        or Path(path).stat().st_size != offset + spectrum_bytes(plan)
    ):
        raise ValueError("Saved spectrum shape, dtype, layout or length changed")
    return dict(offset=offset, shape=list(shape), dtype=dtype.str)


def exclusive_lease(folder):
    stream = (Path(folder) / "construction.lock").open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise RuntimeError("Another process owns this kernel construction") from None
    return stream


def validate_stack(stack, count, start):
    first = start
    if not stack or not 0 <= start < count:
        raise ValueError("Invalid checkpoint sampler stack")
    for left, right in reversed(stack):
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, int)
            or not isinstance(right, int)
            or left != first
            or not left < right <= count
        ):
            raise ValueError("Checkpoint sampler stack omits or overlaps frequencies")
        first = right
    if first != count:
        raise ValueError("Checkpoint sampler stack does not cover the remaining grid")


def checkpointed_sampler(original, stack, count, start, interval):
    """Restore bookkeeping and add an output-only interval callback; guards are unchanged."""
    validate_stack(stack, count, start)
    expected = TOOLS / "raw_layer_sampling.py"
    if Path(original.__code__.co_filename).resolve() != expected.resolve():
        raise ValueError("Require the original raw-layer sampler")
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    baseline = ast.dump(tree, include_attributes=False)
    assignments = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "stack"
    ]
    counters = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.keyword) and n.arg == "transformed_grid_points"
    ]
    loops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.While) and isinstance(n.test, ast.Name) and n.test.id == "stack"
    ]
    if (
        len(assignments) != 1
        or len(counters) != 1
        or len(loops) != 1
        or ast.dump(assignments[0].value, include_attributes=False)
        != ast.dump(ast.parse("[(0, half + 1)]", mode="eval").body, include_attributes=False)
        or not isinstance(counters[0].value, ast.Constant)
        or counters[0].value.value != 0
    ):
        raise ValueError("Original sampler initialization changed")
    original_stack_node, original_counter_node = assignments[0].value, counters[0].value
    assignments[0].value = ast.Call(
        func=ast.Name(id="list", ctx=ast.Load()),
        args=[ast.Name(id="_restored_initial_stack", ctx=ast.Load())],
        keywords=[],
    )
    counters[0].value = ast.Name(id="_restored_emitted_prefix", ctx=ast.Load())
    # The first original loop statement pops (left, right); all numerical work follows it.
    if ast.dump(loops[0].body[0], include_attributes=False) != ast.dump(
        ast.parse("left, right = stack.pop()").body[0], include_attributes=False
    ):
        raise ValueError("Original sampler interval entry changed")
    loops[0].body.insert(1, ast.parse("_checkpoint_interval(left, right, stack)").body[0])
    new_stack, new_counter = assignments[0].value, counters[0].value
    callback = loops[0].body.pop(1)
    assignments[0].value, counters[0].value = original_stack_node, original_counter_node
    if ast.dump(tree, include_attributes=False) != baseline:
        raise RuntimeError("Resume adapter changed another sampler statement")
    assignments[0].value, counters[0].value = new_stack, new_counter
    loops[0].body.insert(1, callback)
    namespace = {
        **original.__globals__,
        "_restored_initial_stack": list(map(tuple, stack)),
        "_restored_emitted_prefix": start,
        "_checkpoint_interval": interval,
    }
    exec(
        compile(ast.fix_missing_locations(tree), str(expected) + "|restored-prefix", "exec"),
        namespace,
    )
    return namespace[original.__name__]


class SamplingJournal:
    def __init__(self, folder, plan, *, resume=False, receipt=None):
        self.folder, self.plan = Path(folder).resolve(), plan
        self.path = self.folder / "sampling-journal.json"
        self.preflight = digest_range(self.folder / "preflight.json")
        self.begun, self.cpu = perf_counter(), process_time()
        self.attempt = None
        if self.path.is_file():
            if not resume:
                raise ValueError("A sampling journal already exists")
            self.data = json.loads(self.path.read_text())
            if self.data["schema_version"] != 1 or self.data["preflight_sha256"] != self.preflight:
                raise ValueError("Sampling journal configuration changed")
        else:
            self.data = dict(
                schema_version=1,
                preflight_sha256=self.preflight,
                phase="raw",
                channels={},
                branch_records={},
                attempts=[],
            )
            if resume:
                if receipt is None:
                    raise ValueError(
                        "Legacy raw spectra require an explicit trusted prefix receipt"
                    )
                proof = json.loads(Path(receipt).read_text())
                if proof["source_preflight_sha256"] != self.preflight or Path(
                    proof["path"]
                ).resolve() != self._path("work-family0.npy"):
                    raise ValueError("Legacy receipt identifies a different build")
                info = header(self._path("work-family0.npy"), plan)
                prefix = proof["committed_matrices"]
                if (
                    not isinstance(prefix, int)
                    or isinstance(prefix, bool)
                    or not 1 <= prefix <= info["shape"][0]
                    or proof["offset"] != info["offset"]
                    or proof["shape"] != info["shape"]
                    or proof["dtype"] != "complex128"
                    or proof["fortran"]
                    or proof["committed_bytes"] != prefix * plan.dofs**2 * 16
                ):
                    raise ValueError("Legacy receipt prefix/header changed")
                self.data["channels"]["work-family0.npy"] = dict(
                    **info,
                    chunks=[dict(first=0, last=prefix, sha256=proof["committed_bytes_sha256"])],
                )
                self.data["channels"]["work-family1.npy"] = dict(
                    **header(self._path("work-family1.npy"), plan), chunks=[]
                )
                self.data["legacy_receipt"] = dict(
                    path=str(Path(receipt).resolve()), sha256=digest_range(receipt), record=proof
                )
        for name, entry in self.data["channels"].items():
            self.verify_channel(name, entry)

    def _path(self, name):
        path = (self.folder / name).resolve()
        if name not in CHANNELS or path.parent != self.folder:
            raise ValueError("Checkpoint channel is outside this owned build")
        return path

    def verify_channel(self, name, entry):
        path = self._path(name)
        actual = header(path, self.plan)
        if any(entry[k] != v for k, v in actual.items()):
            raise ValueError("Checkpoint channel header changed")
        prefix = 0
        for chunk in entry["chunks"]:
            first, last = chunk["first"], chunk["last"]
            if (
                not isinstance(first, int)
                or not isinstance(last, int)
                or isinstance(first, bool)
                or isinstance(last, bool)
                or first != prefix
                or not first < last <= actual["shape"][0]
            ):
                raise ValueError("Checkpoint chunks omit or overlap frequency matrices")
            offset = actual["offset"] + first * self.plan.dofs**2 * 16
            if (
                digest_range(path, offset, (last - first) * self.plan.dofs**2 * 16)
                != chunk["sha256"]
            ):
                raise ValueError("Committed spectrum bytes changed")
            prefix = last

    def publish(self):
        self.data["utc_update"] = datetime.now(UTC).isoformat()
        if self.attempt is not None:
            self.attempt.update(
                last_observed_wall_seconds=perf_counter() - self.begun,
                last_observed_cpu_seconds=process_time() - self.cpu,
            )
        atomic_json(self.path, self.data)

    def begin(self):
        self.attempt = dict(
            pid=os.getpid(),
            utc_start=datetime.now(UTC).isoformat(),
            status="running",
            timing_scope="After model/resource preflight; use external invocation intervals for total costs",
            inherited_unfinished_attempts=sum(
                a["status"] == "running" for a in self.data["attempts"]
            ),
        )
        self.data["attempts"].append(self.attempt)
        self.publish()

    def end(self, status):
        if self.attempt is not None:
            self.attempt.update(status=status, utc_end=datetime.now(UTC).isoformat())
            self.publish()

    def prefix(self, name):
        chunks = self.data["channels"].get(name, {}).get("chunks", [])
        return 0 if not chunks else chunks[-1]["last"]

    def spectrum(self, name):
        path = self._path(name)
        if not path.exists():
            spectrum = DiskSpectrum(path, self.plan)
        else:
            info = header(path, self.plan)
            spectrum = DiskSpectrum.__new__(DiskSpectrum)
            spectrum.path, spectrum.offset = path, info["offset"]
            spectrum.shape, spectrum.dtype = tuple(info["shape"]), np.dtype(info["dtype"])
            spectrum.written = np.zeros(spectrum.shape[0], dtype=bool)
            spectrum.pending_writes = 0
        if name not in self.data["channels"]:
            self.data["channels"][name] = dict(**header(path, self.plan), chunks=[])
            self.publish()
        spectrum.written[: self.prefix(name)] = True
        return spectrum

    def commit(self, name, first, last, sha256):
        if first != self.prefix(name) or not first < last <= self.plan.fft_size // 2 + 1:
            raise ValueError("Committed spectrum matrices are not consecutive")
        self.data["channels"][name]["chunks"].append(dict(first=first, last=last, sha256=sha256))
        self.publish()

    def mark(self, phase, **extra):
        self.data.update(phase=phase, **extra)
        self.publish()

    def sampler_stack(self, name, intervals):
        count, prefix = self.plan.fft_size // 2 + 1, self.prefix(name)
        saved = self.data.get("sampler_states", {}).get(name)
        if saved is None:
            intervals = PARTITION(intervals, count)
            suffix = [pair for pair in intervals if pair[1] > prefix]
            if not suffix:
                raise ValueError("A completed raw branch does not require sampling")
            stack, start = list(reversed(suffix)), suffix[0][0]
        else:
            stack, start = saved["stack"], saved["start"]
        validate_stack(stack, count, start)
        if not start <= prefix <= count:
            raise ValueError("Committed prefix is outside its saved active interval")
        return stack, start

    def save_sampler_state(self, name, left, right, stack, writer):
        # Complete the preceding interval before publishing the next stack.
        writer.flush()
        restored = [*map(tuple, stack), (left, right)]
        validate_stack(restored, self.plan.fft_size // 2 + 1, left)
        if not left <= writer.committed <= self.plan.fft_size // 2 + 1:
            raise ValueError("Sampler interval and durable matrix prefix disagree")
        self.data.setdefault("sampler_states", {})[name] = dict(stack=restored, start=left)
        self.publish()


class JournaledSpectrumWriter(BUFFER):
    def __init__(self, journal, spectrum, *, matrices=64):
        super().__init__(spectrum, matrices=matrices)
        self.journal, self.name = journal, spectrum.path.name
        self.emitted = self.committed = journal.prefix(self.name)
        self.restored_prefix = self.committed

    def accept(self, index, value):
        if (
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or not 0 <= index < self.spectrum.shape[0]
        ):
            raise ValueError("Invalid checkpoint frequency index")
        if index < self.restored_prefix:
            actual = np.ascontiguousarray(value, dtype=self.spectrum.dtype)
            saved = self.spectrum._mapped(lambda a: np.array(a[0], copy=True), index, index + 1)
            if actual.shape != saved.shape or not np.array_equal(
                actual.view(np.uint64), saved.view(np.uint64)
            ):
                raise ValueError("Recomputed preserved matrix is not bitwise identical")
            return
        self.append(index, value)

    def flush(self):
        if not self.pending:
            return
        first, last = self.committed, self.emitted
        values = self.buffer[: self.pending]
        if not values.flags.c_contiguous:
            raise ValueError("Journal digest requires the original contiguous buffer")
        checksum = hashlib.sha256(memoryview(values).cast("B")).hexdigest()
        super().flush()
        self.spectrum.sync()
        self.journal.commit(self.name, first, last, checksum)


def discard_derived_weights(folder, journal):
    """Regenerate only derived files, after all retained source matrices were verified."""
    count = journal.plan.fft_size // 2 + 1
    if any(journal.prefix(f"compact-family{i}.npy") != count for i in range(2)):
        raise ValueError(
            "Complete verified compact sources are required before discarding derived weights"
        )
    if (Path(folder) / "summary.json").exists():
        raise ValueError("Do not discard a completed kernel's weight files")
    for name in ["weights0.npy", "weights1.npy", "weights0-fortran.npy", "weights1-fortran.npy"]:
        path = (Path(folder) / name).resolve()
        if path.parent != Path(folder).resolve():
            raise ValueError("Derived weight path is outside this owned build")
        if path.exists():
            with path.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                reader = {
                    (1, 0): np.lib.format.read_array_header_1_0,
                    (2, 0): np.lib.format.read_array_header_2_0,
                }.get(version)
                if reader is None:
                    raise ValueError("Unknown derived weight header; preserve it for inspection")
                shape, _, dtype = reader(stream)
            if (
                shape != (journal.plan.steps + 1, journal.plan.dofs, journal.plan.dofs)
                or dtype != np.float64
            ):
                raise ValueError("Unexpected derived weight file; preserve it for inspection")
            path.unlink()

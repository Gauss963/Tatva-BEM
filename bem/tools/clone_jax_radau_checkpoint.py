"""Copy verified original raw prefixes into a fresh, separately owned JAX build.

The source must have no active constructor and partial branches must be exactly
at a saved sampler boundary. Source files are opened read-only. Only committed
matrix ranges are inherited; future JAX writes cannot change the original.
No interpolation, constitutive or bitwise replay gate is relaxed.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import runpy
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from finiteblock_bem.cq import CQPlan

TOOLS = Path(__file__).resolve().parent
JAX = runpy.run_path(str(TOOLS / "prepare_jax_radau_resolvent.py"))
BASE = runpy.run_path(str(TOOLS / "prepare_raw_radau_resolvent.py"))
CHECK = BASE["CHECKPOINT"]
DIGEST = CHECK["digest_range"]
CONTROL_ARGUMENTS = {
    "output",
    "workers",
    "max_memory_mb",
    "max_workspace_mb",
    "preflight_only",
    "resume",
    "resume_receipt",
    "durable_build",
    "verify_resume_only",
}


@contextmanager
def readonly_source_lease(folder):
    path = folder / "construction.lock"
    if not path.is_file() or path.is_symlink():
        raise ValueError("Require the original regular construction lock file")
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("The original constructor still owns its lease") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verify_problem(parent, template):
    for key in ["plan", "source_mesh_arguments"]:
        if parent[key] != template[key]:
            raise ValueError("Handoff changes the physical problem or CQ plan: " + key)
    for key in ["source_sha256", "case_sha256", "restart_data_sha256", "numpy", "scipy"]:
        if parent["environment"][key] != template["environment"][key]:
            raise ValueError("Handoff changes a numerical input/runtime: " + key)
    first, second = parent["arguments"], template["arguments"]
    for key in (first.keys() | second.keys()) - CONTROL_ARGUMENTS:
        a, b = first.get(key), second.get(key)
        if key == "source":
            a, b = Path(a).resolve(), Path(b).resolve()
        if a != b:
            raise ValueError("Handoff changes a numerical construction argument: " + key)
    for path, expected in parent["frozen_input_sha256"].items():
        actual = DIGEST(path)
        if Path(path).resolve() == (TOOLS / "prepare_raw_radau_resolvent.py").resolve():
            if expected not in {actual, BASE["LEGACY_PRODUCER_SHA256"]}:
                raise ValueError("Unknown original producer migration")
        elif actual != expected:
            raise ValueError("Original numerical source/input changed: " + path)
    for path, expected in template["frozen_input_sha256"].items():
        if DIGEST(path) != expected:
            raise ValueError("JAX preflight input changed: " + path)


def aligned_prefixes(journal):
    count = journal.plan.fft_size // 2 + 1
    if journal.data["phase"] != "raw" or set(journal.data["channels"]) != {
        "work-family0.npy",
        "work-family1.npy",
    }:
        raise ValueError("Only an original raw construction can be handed off")
    prefixes = {}
    for name in journal.data["channels"]:
        prefix = journal.prefix(name)
        state = journal.data.get("sampler_states", {}).get(name)
        if 0 < prefix < count:
            if state is None or state["start"] != prefix:
                raise ValueError("Partial prefix is not at its saved sampler boundary")
            CHECK["validate_stack"](state["stack"], count, prefix)
        elif prefix == 0 and state is not None:
            if state["start"] != 0:
                raise ValueError("An empty branch has a nonzero saved start")
            CHECK["validate_stack"](state["stack"], count, 0)
        prefixes[name] = prefix
    return prefixes


def copy_committed_channel(parent, target, name, entry, plan):
    source = parent / name
    destination = target / name
    info = CHECK["header"](source, plan)
    with source.open("rb") as reader, destination.open("xb") as writer:
        writer.write(reader.read(info["offset"]))
        writer.truncate(source.stat().st_size)
        for chunk in entry["chunks"]:
            digest = hashlib.sha256()
            offset = info["offset"] + chunk["first"] * plan.dofs**2 * 16
            left = (chunk["last"] - chunk["first"]) * plan.dofs**2 * 16
            reader.seek(offset)
            writer.seek(offset)
            while left:
                data = reader.read(min(left, 4 * 1024**2))
                if not data:
                    raise ValueError("Original committed matrix range was truncated")
                writer.write(data)
                digest.update(data)
                left -= len(data)
            if digest.hexdigest() != chunk["sha256"]:
                raise ValueError("Original committed chunk changed while copying")
        writer.flush()
        os.fsync(writer.fileno())
    if CHECK["header"](destination, plan) != info:
        raise ValueError("Copied spectrum header changed")


def clone_checkpoint(parent, target):
    begun = perf_counter()
    parent, target = Path(parent).resolve(), Path(target).resolve()
    protected = Path("/Volumes/Gauss-T7/tatva").resolve()
    if (
        parent == target
        or target.is_relative_to(parent)
        or parent.is_relative_to(target)
        or target.is_relative_to(protected)
    ):
        raise ValueError("Choose a separate owned target outside protected Tatva")
    expected = {"preflight.json", JAX["CONTEXT"]}
    if not target.is_dir() or {p.name for p in target.iterdir()} != expected:
        raise ValueError("Target must contain only a fresh JAX preflight and backend context")
    template_text = (target / "preflight.json").read_bytes()
    template = json.loads(template_text)
    if not template["arguments"].get("preflight_only") or not template["arguments"].get(
        "durable_build"
    ):
        raise ValueError("Generate the target with --durable-build --preflight-only")
    JAX["verify_context"](target)
    with readonly_source_lease(parent):
        parent_preflight_text = (parent / "preflight.json").read_bytes()
        parent_journal_text = (parent / "sampling-journal.json").read_bytes()
        original = json.loads(parent_preflight_text)
        if (
            Path(original["arguments"]["output"]).resolve() != parent
            or Path(template["arguments"]["output"]).resolve() != target
        ):
            raise ValueError("Checkpoint preflights identify different output folders")
        if "jax_backend" in original or (parent / "summary.json").exists():
            raise ValueError("Require an incomplete original-only raw build")
        verify_problem(original, template)
        plan = CQPlan(**template["plan"])
        journal = CHECK["SamplingJournal"](parent, plan, resume=True)
        prefixes = aligned_prefixes(journal)
        if template["required_disk_bytes"] > shutil.disk_usage(target).free:
            raise ValueError("Insufficient disk for the complete target construction")
        lease = CHECK["exclusive_lease"](target)
        try:
            for name, entry in journal.data["channels"].items():
                copy_committed_channel(parent, target, name, entry, plan)
            if (parent / "preflight.json").read_bytes() != parent_preflight_text or (
                parent / "sampling-journal.json"
            ).read_bytes() != parent_journal_text:
                raise ValueError("Original checkpoint metadata changed during the copy")
            snapshots = {
                "parent-preflight.json": parent_preflight_text,
                "parent-sampling-journal.json": parent_journal_text,
                "template-preflight.json": template_text,
            }
            for name, data in snapshots.items():
                with (target / name).open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            manifest = dict(
                schema_version=1,
                utc=datetime.now(UTC).isoformat(),
                parent=str(parent),
                original_backend="original_scipy_galerkin",
                target_backend=JAX["BACKEND"],
                source_readonly=True,
                same_problem_verified=True,
                prefixes=prefixes,
                original_channel_chunks=copy.deepcopy(journal.data["channels"]),
                snapshots_sha256={name: DIGEST(target / name) for name in snapshots},
                copy_source_sha256=DIGEST(Path(__file__)),
                inherited_attempts_preserved=True,
                bitwise_prefix_replay_gate_unchanged=True,
                uncommitted_suffix_inherited=False,
            )
            manifest_path = target / "handoff.json"
            CHECK["atomic_json"](manifest_path, manifest)
            context = JAX["verify_context"](target)
            context["handoff"] = dict(path=manifest_path.name, sha256=DIGEST(manifest_path))
            CHECK["atomic_json"](target / JAX["CONTEXT"], context)
            updated = copy.deepcopy(template)
            updated["status"] = "inherited_raw_prefix_ready_for_jax_resume"
            updated["arguments"].update(
                preflight_only=False,
                durable_build=True,
                resume=False,
                resume_receipt=None,
                verify_resume_only=False,
            )
            updated["source_role"] = (
                "Verified original raw prefix inherited at sampler boundaries; future raw values use the frozen JAX backend."
            )
            updated["jax_backend"] = context
            updated["inherited_raw_prefix"] = context["handoff"] | {"prefixes": prefixes}
            for path in [
                target / JAX["CONTEXT"],
                manifest_path,
                *[target / name for name in snapshots],
                Path(__file__).resolve(),
            ]:
                updated["frozen_input_sha256"][str(path.resolve())] = DIGEST(path)
            allocated = sum((target / n).stat().st_blocks * 512 for n in journal.data["channels"])
            updated["additional_disk_bytes"] = max(0, updated["required_disk_bytes"] - allocated)
            CHECK["atomic_json"](target / "preflight.json", updated)
            copied = copy.deepcopy(journal.data)
            copied["preflight_sha256"] = DIGEST(target / "preflight.json")
            copied["backend_handoff"] = context["handoff"]
            CHECK["atomic_json"](target / "sampling-journal.json", copied)
            verified = CHECK["SamplingJournal"](target, plan, resume=True)
            if {n: verified.prefix(n) for n in verified.data["channels"]} != prefixes:
                raise ValueError("Copied journal prefixes changed")
            JAX["verify_context"](target)
            receipt = dict(
                status="verified_original_prefix_copied_for_jax_resume",
                prefixes=prefixes,
                parent=str(parent),
                target=str(target),
                source_files_opened_readonly=True,
                all_committed_chunks_verified_on_source_and_target=True,
                sampler_boundaries_exact=True,
                original_cache_modified=False,
                seconds=perf_counter() - begun,
                handoff_sha256=DIGEST(manifest_path),
                target_preflight_sha256=DIGEST(target / "preflight.json"),
                target_journal_sha256=DIGEST(target / "sampling-journal.json"),
                full_original_event_reproduced=False,
                same_accuracy_fem_advantage_verified=False,
            )
            CHECK["atomic_json"](target / "handoff-receipt.json", receipt)
            return receipt
        finally:
            lease.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    print(json.dumps(clone_checkpoint(args.parent, args.target), indent=2), flush=True)


if __name__ == "__main__":
    main()

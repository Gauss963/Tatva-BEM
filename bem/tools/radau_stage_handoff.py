"""Atomic-rename ownership transfer from audited raw producer to new staging."""

import json
import runpy
from dataclasses import asdict
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
TX = runpy.run_path(str(TOOLS / "inplace_radau_compaction.py"))
DIGEST = TX["CHECKPOINT"]["digest_range"]


def handoff_raw(source, target, records, plan, journal_path, *, resume=False, progress=None):
    source, target, journal_path = map(lambda p: Path(p).resolve(), [source, target, journal_path])
    protected = Path("/Volumes/Gauss-T7/tatva").resolve()
    if (
        source.is_relative_to(protected)
        or target.is_relative_to(protected)
        or journal_path.is_relative_to(protected)
        or source.is_relative_to(target)
        or target.is_relative_to(source)
        or not source.is_dir()
        or len(records) != 2
    ):
        raise ValueError("Require distinct owned staging directories outside protected Tatva")
    if resume != journal_path.exists():
        raise ValueError("Ownership handoff resume must match an existing journal")
    if not target.exists():
        target.mkdir()
    if source.stat().st_dev != target.stat().st_dev:
        raise ValueError("Ownership handoff requires atomic renames on the same filesystem")
    if any(p.name not in [*TX["NAMES"], "construction.lock"] for p in target.iterdir()):
        raise ValueError("Ownership target already contains non-handoff data")
    context = dict(
        source=str(source),
        target=str(target),
        plan=asdict(plan),
        raw_records=records,
        source_sha256={
            str(p): DIGEST(p)
            for p in [Path(__file__).resolve(), TOOLS / "inplace_radau_compaction.py"]
        },
    )
    leases = []
    try:
        for folder in [source, target]:
            leases.append(TX["CHECKPOINT"]["exclusive_lease"](folder))
        state = (
            json.loads(journal_path.read_text())
            if resume
            else dict(
                schema_version=1,
                context=context,
                phase="transferring",
                identities=[],
                moved=[],
            )
        )
        if (
            state["schema_version"] != 1
            or state["context"] != context
            or state["phase"] not in ["transferring", "completed"]
        ):
            raise ValueError("Ownership handoff context changed")
        locations = []
        for i, name in enumerate(TX["NAMES"]):
            before, after = source / name, target / name
            if before.exists() == after.exists() or (not resume and after.exists()):
                raise ValueError("Raw handoff must have exactly one owned copy per branch")
            path = before if before.exists() else after
            stat = path.stat()
            info = TX["CHECKPOINT"]["header"](path, plan)
            r = records[i]
            if (
                path.is_symlink()
                or stat.st_nlink != 1
                or Path(r["path"]).resolve() != before
                or r.get("role") != TX["ROLE"]
                or r.get("mode_branch") != i
                or r.get("all_frequency_matrices_written") is not True
                or r.get("shape") != info["shape"]
                or r.get("bytes") != stat.st_size
                or DIGEST(path) != r.get("sha256")
            ):
                raise ValueError("Owned audited raw handoff bytes or role changed")
            identity = [stat.st_dev, stat.st_ino]
            if resume:
                if identity != state["identities"][i]:
                    raise ValueError("Raw handoff inode changed")
            else:
                state["identities"].append(identity)
            locations.append(path)
        # Both complete files are verified before the first mutation.
        if not resume:
            TX["publish"](journal_path, state)
        for i, path in enumerate(locations):
            after = target / TX["NAMES"][i]
            if path != after:
                path.rename(after)
                TX["sync_directory"](source)
                TX["sync_directory"](target)
                if progress is not None:
                    progress(dict(phase="renamed", branch=i))
            state["moved"] = list(range(i + 1))
            TX["publish"](journal_path, state)
        state["phase"] = "completed"
        state["staging_raw_records"] = [
            {**r, "path": str(target / TX["NAMES"][i])} for i, r in enumerate(records)
        ]
        TX["publish"](journal_path, state)
        return state["staging_raw_records"], state
    finally:
        for lease in reversed(leases):
            lease.close()

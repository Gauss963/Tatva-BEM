"""Original raw Radau sampling/audits, stopped before storage postprocessing.

AST adaptation changes only the disk estimate and replaces the postprocessing
tail with an audited raw-stage receipt. The original file is never edited.
Durable sampling and its original resume/acceptance rules are required.
"""

import ast
import inspect
import os
import runpy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.cq_storage import spectrum_bytes

TOOLS = Path(__file__).resolve().parent
ORIGINAL = TOOLS / "prepare_raw_radau_resolvent.py"
TX = runpy.run_path(str(TOOLS / "inplace_radau_compaction.py"))


def raw_required_bytes(plan):
    return 2 * spectrum_bytes(plan) + 32 * plan.dofs**2 * 16 + 64 * 1024**2


def completed(values):
    values["check_source"]()
    output, plan = values["output"], values["plan"]
    journal = values["journal"]
    if journal is None or journal.data["phase"] != "raw_complete":
        raise ValueError("Raw-stage completion requires durable complete raw branches")
    records = []
    for i, spectrum in enumerate(values["spectra"]):
        if journal.prefix(spectrum.path.name) != plan.fft_size // 2 + 1:
            raise ValueError("Incomplete raw branch at the storage handoff")
        records.append(
            dict(
                path=str(spectrum.path.resolve()),
                shape=list(spectrum.shape),
                bytes=spectrum.path.stat().st_size,
                sha256=values["branch_digests"][i]["pre_combination_sha256"],
                all_frequency_matrices_written=True,
                role=TX["ROLE"],
                mode_branch=i,
            )
        )
    np.save(output / "initial_impedance.npy", values["impedance"].real, allow_pickle=False)
    np.savez(output / "stage-audit.npz", indices=values["indices"], stages=values["fresh_stages"])
    for name in ["initial_impedance.npy", "stage-audit.npz"]:
        with (output / name).open("rb") as stream:
            os.fsync(stream.fileno())
    digest = TX["CHECKPOINT"]["digest_range"]
    TX["publish"](
        output / "summary.json",
        dict(
            status="completed_independently_audited_raw_radau_stage",
            arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(values["args"]).items()
            },
            source_mesh_arguments=values["options"],
            plan=asdict(plan),
            environment=values["environment"],
            raw_records=records,
            frequency_sampling=values["records"],
            sampling_options=asdict(values["sampling"]),
            sampled_fresh_bem_mode_checks=values["checks"],
            stage_audit_sha256=digest(output / "stage-audit.npz"),
            initial_impedance_sha256=digest(output / "initial_impedance.npy"),
            frozen_inputs_sha256=values["frozen"],
            adapter_sha256=digest(Path(__file__)),
            original_producer_sha256=digest(ORIGINAL),
            durable_construction=dict(attempts=journal.data["attempts"]),
            timing_seconds=dict(
                raw_sampling_and_branch_hashes=values["sampling_seconds"],
                independent_operator_audit=values["audit_seconds"],
                total_before_reporting=perf_counter() - values["begun"],
            ),
            postprocessing_performed=False,
            original_tatva_readonly=True,
            full_original_event_reproduced=False,
            same_accuracy_fem_advantage_verified=False,
        ),
    )
    print(output / "summary.json", flush=True)


def adapted_namespace():
    original = runpy.run_path(str(ORIGINAL))
    tree = ast.parse(inspect.getsource(original["_main"]))
    function = tree.body[0]
    baseline = ast.dump(tree, include_attributes=False)

    def assigned(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        )

    start = [i for i, n in enumerate(function.body) if assigned(n, "weight_bytes")]
    tail = [i for i, n in enumerate(function.body) if assigned(n, "combination_start")]
    if len(start) != 1 or len(tail) != 1:
        raise ValueError("Original producer preflight/postprocessing boundaries changed")
    a, b = start[0], tail[0]
    storage = function.body[a : a + 4]
    expected = ast.parse(
        "weight_bytes = (steps + 1) * d * d * 8\n"
        "required = 2 * spectrum_bytes(plan) + 2 * weight_bytes + (steps + 1) * 8 * d * 8 + 1024**2\n"
        "if args.durable_build: required += 2 * spectrum_bytes(plan)\n"
        'if args.weight_layout == "fortran": required += weight_bytes + 8192\n'
    ).body
    if [ast.dump(n) for n in storage] != [ast.dump(n) for n in expected] or not a + 4 < b:
        raise ValueError("Original producer disk preflight changed")
    saved_tail = function.body[b:]
    function.body = function.body[:b]
    function.body[a : a + 4] = ast.parse("required = _raw_required_bytes(plan)").body
    function.body += ast.parse("_raw_stage_complete(locals())\nreturn").body
    adapted = list(function.body)
    # Restore exactly the two changed regions and prove all remaining AST equal.
    function.body = adapted[:-2]
    function.body[a : a + 1] = storage
    function.body += saved_tail
    if ast.dump(tree, include_attributes=False) != baseline:
        raise RuntimeError("Raw-stage adapter changed a sampling/audit statement")
    function.body = adapted
    namespace = {
        **original["_main"].__globals__,
        "_raw_required_bytes": raw_required_bytes,
        "_raw_stage_complete": completed,
    }
    checkpoint = dict(namespace["CHECKPOINT"])
    atomic_json = checkpoint["atomic_json"]

    def publish(path, value):
        if Path(path).name == "preflight.json":
            value["frozen_input_sha256"].update(
                {
                    str(p): TX["CHECKPOINT"]["digest_range"](p)
                    for p in [Path(__file__).resolve(), TOOLS / "inplace_radau_compaction.py"]
                }
            )
        atomic_json(path, value)

    checkpoint["atomic_json"] = publish
    namespace["CHECKPOINT"] = checkpoint
    main_tree = ast.parse(inspect.getsource(original["main"]))
    for part in [tree, main_tree]:
        exec(
            compile(ast.fix_missing_locations(part), str(ORIGINAL) + "|raw-stage", "exec"),
            namespace,
        )
    return namespace


def main():
    import sys

    if "--durable-build" not in sys.argv:
        raise ValueError("The independent raw stage requires --durable-build")
    adapted_namespace()["main"]()


if __name__ == "__main__":
    main()

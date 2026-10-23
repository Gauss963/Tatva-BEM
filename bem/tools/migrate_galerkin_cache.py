"""Explicitly migrate a pre-streaming cache after checking its numerical provenance.

Only a cache from a completed, source-frozen run is eligible. All original
operator files must match the recorded Git source. The CQ weight-generating
functions must have the same AST after the single, checked spectrum-allocation
hook is expanded; memory planning/online stepping may differ.
No transfer matrix is approximated or recomputed by this tool.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from finiteblock_bem.cq import cq_math_signature, plan_cq
from finiteblock_bem.io import fingerprint, write_json
from finiteblock_bem.nodal import load_nodal_model
from finiteblock_bem.nodal_dynamics import nodal_cache_key
from finiteblock_bem.sampling import SamplingOptions

ROOT = Path(__file__).resolve().parents[1]
OPERATOR_FILES = [
    "galerkin.py",
    "galerkin_fast.py",
    "nodal.py",
    "cq.py",
    "sampling.py",
    "kernels.py",
    "material.py",
    "mesh.py",
]

BUFFER_SOURCE = '''
def spectrum_buffer(plan, provided=None):
    """Allocate a RAM spectrum, or validate an explicitly supplied storage object."""
    shape = (plan.fft_size // 2 + 1, plan.dofs, plan.dofs)
    if provided is None:
        return np.empty(shape, dtype=complex)
    if provided.shape != shape or provided.dtype != np.dtype("complex128"):
        raise ValueError("Incorrect half-spectrum shape or dtype")
    return provided
'''


class StorageHook(ast.NodeTransformer):
    """Expand only the approved optional allocation hook, preserving all math."""

    def visit_FunctionDef(self, node):
        if node.name in ["convolution_weights", "parsimonious_weights"]:
            for index, argument in enumerate(node.args.kwonlyargs):
                if argument.arg == "spectrum":
                    default = node.args.kw_defaults[index]
                    if not isinstance(default, ast.Constant) or default.value is not None:
                        raise ValueError("Unrecognized spectrum default")
                    del node.args.kwonlyargs[index]
                    del node.args.kw_defaults[index]
                    break
        return self.generic_visit(node)

    def visit_Assign(self, node):
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "spectrum_buffer"
        ):
            if (
                ast.dump(node.targets[0], include_attributes=False)
                != ast.dump(ast.Name(id="spectrum", ctx=ast.Store()), include_attributes=False)
                or len(node.targets) != 1
            ):
                raise ValueError("Unrecognized spectrum assignment")
            expected = ast.parse("spectrum_buffer(plan, spectrum)", mode="eval").body
            if ast.dump(call, include_attributes=False) != ast.dump(
                expected, include_attributes=False
            ):
                raise ValueError("Unrecognized spectrum hook arguments")
            node.value = ast.parse(
                "np.empty((half + 1, plan.dofs, plan.dofs), dtype=complex)", mode="eval"
            ).body
        return node

    def visit_ImportFrom(self, node):
        if node.level == 1 and node.module == "cq":
            node.names = [name for name in node.names if name.name != "spectrum_buffer"]
        return node


def approved_storage_only(old_cq, new_cq, old_sampling, new_sampling):
    current = ast.parse(new_cq)
    buffers = [
        node
        for node in current.body
        if isinstance(node, ast.FunctionDef) and node.name == "spectrum_buffer"
    ]
    if len(buffers) != 1 or ast.dump(buffers[0], include_attributes=False) != ast.dump(
        ast.parse(BUFFER_SOURCE).body[0], include_attributes=False
    ):
        raise ValueError("Allocation helper changed; review before migrating weights")
    names = ["bdf2_symbol", "convolution_weights", "invert_spectrum"]
    for name in names:
        nodes = []
        for source in [old_cq, new_cq]:
            candidates = [
                node
                for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == name
            ]
            if len(candidates) != 1:
                raise ValueError("Incomplete CQ source")
            nodes.append(ast.dump(StorageHook().visit(candidates[0]), include_attributes=False))
        if nodes[0] != nodes[1]:
            raise ValueError(f"CQ mathematics changed in {name}; rebuild weights")
    normalized = [
        ast.dump(StorageHook().visit(ast.parse(source)), include_attributes=False)
        for source in [old_sampling, new_sampling]
    ]
    if normalized[0] != normalized[1]:
        raise ValueError("Adaptive-sampling mathematics changed; rebuild weights")


def weight_fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_plan(source, dofs, dt, steps):
    tree = ast.parse(source)
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ["CQPlan", "ResourceBudgetError", "plan_cq"]
    ]
    # This is reviewed repository code, obtained using git show and checked
    # against the original run's SHA-256. No external run text is executed.
    namespace = {"__name__": __name__, "dataclass": dataclass, "np": np, "math": math}
    exec(compile(tree, "<verified historical CQ planner>", "exec"), namespace)
    return namespace["plan_cq"](dofs, dt, steps, max_memory_mb=65536)


def migrate(run, git_ref, directory):
    summary = json.loads((run / "summary.json").read_text())
    manifest = json.loads((run / "run_manifest.json").read_text())
    args, environment = manifest["arguments"], manifest["environment"]
    if summary["status"] != "completed_requested_horizon" or summary["source_changed_during_run"]:
        raise ValueError("A complete, source-frozen forward run is required")
    if summary["environment"]["source_sha256"] != environment["source_sha256"]:
        raise ValueError("Summary and manifest disagree about numerical source")
    revision = subprocess.check_output(["git", "rev-parse", git_ref], cwd=ROOT).decode().strip()
    sources = {}
    for name in OPERATOR_FILES:
        sources[name] = subprocess.check_output(
            ["git", "show", f"{revision}:bem/src/finiteblock_bem/{name}"], cwd=ROOT
        )
        if hashlib.sha256(sources[name]).hexdigest() != environment["source_sha256"][name]:
            raise ValueError(f"{name}: Git revision differs from recorded source")
        if (
            name not in ["cq.py", "sampling.py"]
            and sources[name] != (ROOT / "src/finiteblock_bem" / name).read_bytes()
        ):
            raise ValueError(f"{name}: operator changed; cache cannot be migrated")
    math_signature = cq_math_signature(sources["cq.py"].decode())
    approved_storage_only(
        sources["cq.py"].decode(),
        (ROOT / "src/finiteblock_bem/cq.py").read_text(),
        sources["sampling.py"].decode(),
        (ROOT / "src/finiteblock_bem/sampling.py").read_text(),
    )
    nodal_source = subprocess.check_output(
        ["git", "show", f"{revision}:bem/src/finiteblock_bem/nodal_dynamics.py"], cwd=ROOT
    )
    if (
        hashlib.sha256(nodal_source).hexdigest()
        != environment["source_sha256"]["nodal_dynamics.py"]
    ):
        raise ValueError("Historical cache builder differs from recorded source")
    case = Path(args["case"])
    if fingerprint(case) != environment["case_sha256"]:
        raise ValueError("Original case input is required")
    model = load_nodal_model(
        case,
        max_panel_length=args["h"],
        contact_panel_length=args["contact_h"],
        contact_end_length=args["contact_end_h"],
        corner_panel_length=args["corner_h"],
        restart_knot_window=args.get("restart_knot_window"),
        restart_knot_subdivisions=args.get("restart_knot_subdivisions", 1),
    )
    dofs, dt, steps = 2 * len(model.moving_contact) + 1, summary["dt_s"], summary["completed_steps"]
    plan = historical_plan(sources["cq.py"].decode(), dofs, dt, steps)
    sampling = (
        SamplingOptions(relative_tolerance=args["sampling_rtol"])
        if args["sampling"] == "adaptive"
        else None
    )
    canonical = "def nodal_cache_key" in nodal_source.decode()
    contour = (plan.dt, plan.steps, plan.dofs, plan.fft_size, plan.radius) if canonical else plan
    digest = hashlib.sha256(
        repr((contour, model.reference.material, args["quadrature"], sampling)).encode()
    )
    for mesh in [model.reference.moving, model.reference.stationary]:
        for data in [mesh.starts, mesh.ends, mesh.labels]:
            digest.update(data.tobytes())
    digest.update(model.constraints.prolongation.tobytes())
    for name in OPERATOR_FILES:
        digest.update(math_signature.encode() if name == "cq.py" and canonical else sources[name])
    directory = Path(args["cache"]) if directory is None else directory
    old = directory / f"galerkin-{digest.hexdigest()}.npy"
    weights = np.load(old, mmap_mode="r", allow_pickle=False)
    if weights.shape != (steps + 1, dofs, dofs) or weights.dtype != np.float64:
        raise ValueError("Original cache has the wrong shape or dtype")
    metadata = json.loads(old.with_suffix(".json").read_text())
    original_sampling = {
        key: value for key, value in summary["frequency_sampling"].items() if key != "cache_hit"
    }
    if metadata != original_sampling:
        raise ValueError("Original cache metadata differs from completed run")
    current_plan = plan_cq(dofs, dt, steps, max_memory_mb=65536)
    key = nodal_cache_key(model, current_plan, args["quadrature"], sampling)
    target = directory / f"galerkin-{key}.npy"
    provenance = dict(
        old_cache=old.name,
        source_git_revision=revision,
        source_manifest_sha256=fingerprint(run / "run_manifest.json"),
        weight_sha256=weight_fingerprint(old),
        historical_cq_math_signature=math_signature,
        cq_math_signature=cq_math_signature(),
        predecessor_cache_migration=metadata.get("cache_migration"),
        scope="Identical operators, contour and mathematical AST after checked spectrum-allocation expansion",
    )
    if target.exists():
        if weight_fingerprint(target) != provenance["weight_sha256"]:
            raise ValueError("Destination cache already exists with different weights")
    else:
        os.link(old, target)
    write_json(target.with_suffix(".json"), dict(metadata, cache_migration=provenance))
    return dict(target=str(target.resolve()), **provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--git-ref", default="595bb54aeb58887060180cbc42fe2c5a9b689888")
    parser.add_argument("--cache-directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(migrate(args.run, args.git_ref, args.cache_directory), indent=2))


if __name__ == "__main__":
    main()

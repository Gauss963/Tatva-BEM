"""Optional JAX batch backend for the original durable Radau construction.

No original source/global is patched. The original sampler, acceptance tests,
checkpoint writer, transformations, weights and startup are retained. Backend
sources/runtime are additionally frozen. An original-only cache requires an
explicit separately owned handoff; this wrapper cannot silently resume it.
"""

import ast
import copy
import inspect
import json
import os
import platform
import runpy
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from types import FunctionType

import jax
import jaxlib

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.io import fingerprint
from finiteblock_bem.nodal import NodalInterfaceTransfer

TOOLS = Path(__file__).resolve().parent
JIT = runpy.run_path(str(TOOLS / "jax_ray_galerkin.py"))
CONTEXT = "jax-backend-context.json"
BACKEND = "jax_x64_ray_batch_original_equations"


def backend_sources():
    names = [
        Path(__file__).name,
        "jax_ray_galerkin.py",
        "ray_bessel_galerkin.py",
        "factored_galerkin.py",
    ]
    return {str((TOOLS / n).resolve()): fingerprint(TOOLS / n) for n in names}


def backend_runtime():
    devices = jax.devices()
    if not jax.config.x64_enabled or any(d.platform != "cpu" for d in devices):
        raise ValueError("The durable JAX backend requires x64 CPU execution")
    return dict(
        jax=jax.__version__,
        jaxlib=jaxlib.__version__,
        x64=jax.config.x64_enabled,
        devices=[str(d) for d in devices],
        platform=platform.platform(),
        thread_environment={
            k: os.environ.get(k)
            for k in [
                "FDBEM_CORES",
                "XLA_FLAGS",
                "JAX_ENABLE_X64",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "TF_NUM_INTRAOP_THREADS",
                "TF_NUM_INTEROP_THREADS",
            ]
        },
    )


def backend_record():
    return dict(
        schema_version=1,
        backend=BACKEND,
        runtime=backend_runtime(),
        sources_sha256=backend_sources(),
        original_audit_uses_scipy=True,
        memory_policy="Add max(512 MiB, min(2 GiB, geometry/2)) to base array peak; estimate, not a hard RSS cap.",
    )


def initialize_context(output, frozen):
    path = Path(output) / CONTEXT
    if path.exists():
        raise ValueError("A fresh backend context must not overwrite an existing file")
    record = backend_record() | {"utc_creation": datetime.now(UTC).isoformat()}
    checkpoint = runpy.run_path(str(TOOLS / "radau_sampling_checkpoint.py"))
    checkpoint["atomic_json"](path, record)
    frozen[str(path.resolve())] = fingerprint(path)


def verify_context(output, frozen=None):
    path = Path(output) / CONTEXT
    if not path.is_file():
        raise ValueError("Missing JAX backend context; original caches require an explicit handoff")
    saved = json.loads(path.read_text())
    if any(saved.get(k) != v for k, v in backend_record().items()):
        raise ValueError("JAX backend source, runtime or policy changed")
    if frozen is not None:
        preflight = Path(output) / "preflight.json"
        if preflight.is_file():
            # A copied-build handoff freezes its immutable parent snapshots and
            # manifest too. Carry these into ongoing interval source checks,
            # without overwriting the base producer's current source digest.
            inherited = json.loads(preflight.read_text())["frozen_input_sha256"]
            for source, digest in inherited.items():
                frozen.setdefault(source, digest)
        frozen[str(path.resolve())] = fingerprint(path)
    return saved


class OriginalAuditTransfer(NodalInterfaceTransfer):
    """Keep the producer's independent body-compliance audit on SciPy kernels."""

    @property
    def prepared(self):
        return self._original_prepared

    @prepared.setter
    def prepared(self, value):
        if value is None:
            self._original_prepared = None
            return
        original = []
        for item in value:
            if isinstance(item, JIT["JaxRayGalerkin"]):
                item = item.prepared
            if type(item) is not PreparedGalerkinOperators:
                raise ValueError("Independent audit requires original prepared geometry")
            original.append(item)
        self._original_prepared = original


def private_producer():
    """Rebind only backend/preflight plumbing in a fresh private namespace."""
    original = runpy.run_path(str(TOOLS / "prepare_raw_radau_resolvent.py"))

    class JaxTraceTransfer(original["TRANSFER"]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prepared = [JIT["JaxRayGalerkin"](a) for a in self.prepared]

    def construction_arrays(*args, **kwargs):
        values = original["construction_arrays"](*args, **kwargs)
        extra = max(512 * 1024**2, min(2 * 1024**3, values["geometry"] // 2))
        return {
            **values,
            "peak": values["peak"] + extra,
            "jax_runtime_and_compiler_allowance": extra,
        }

    tree = ast.parse(textwrap.dedent(inspect.getsource(original["_main"])))
    function = tree.body[0]
    body = function.body
    positions = [
        i
        for i, n in enumerate(body)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and ast.unparse(n.targets[0]) == "frozen"
    ]
    fresh = [n for n in body if isinstance(n, ast.If) and ast.unparse(n.test) == "not args.resume"]
    checks = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == "check_source"]
    parsed = [
        i
        for i, n in enumerate(body)
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and ast.unparse(n.targets[0]) == "args"
    ]
    if len(positions) != 1 or len(fresh) != 1 or len(checks) != 1 or len(parsed) != 1:
        raise ValueError("Original producer layout changed; re-audit the JAX adapter")
    if ast.unparse(fresh[0].body[0]) != "output.mkdir()":
        raise ValueError("Original producer output boundary changed")
    # Save node identities before insertion so no arithmetic/sampling statement
    # needs to be copied or reconstructed by this adapter.
    frozen_node, args_node = body[positions[0]], body[parsed[0]]
    body[body.index(frozen_node) + 1 : body.index(frozen_node) + 1] = ast.parse(
        "frozen.update(_backend_sources())\nif args.resume:\n    _verify_backend(output, frozen)"
    ).body
    body[body.index(args_node) + 1 : body.index(args_node) + 1] = ast.parse(
        "if not args.durable_build and not args.preflight_only:\n"
        "    raise ValueError('JAX construction requires --durable-build')"
    ).body
    fresh[0].body.insert(1, ast.parse("_initialize_backend(output, frozen)").body[0])
    checks[0].body.insert(0, ast.parse("_verify_backend(output)").body[0])
    statuses = {
        "preflight_only_not_a_complete_kernel",
        "completed_compact_radau_resolvent_kernel_only",
    }
    annotated = 0
    for node in ast.walk(function):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "dict"
        ):
            continue
        if any(
            k.arg == "status" and isinstance(k.value, ast.Constant) and k.value.value in statuses
            for k in node.keywords
        ):
            node.keywords.append(
                ast.keyword(
                    arg="jax_backend", value=ast.parse("_verify_backend(output)", mode="eval").body
                )
            )
            annotated += 1
    if annotated != 2:
        raise ValueError("Original producer reporting boundaries changed")
    namespace = dict(original)
    namespace.update(
        TRANSFER=JaxTraceTransfer,
        NodalInterfaceTransfer=OriginalAuditTransfer,
        construction_arrays=construction_arrays,
        _backend_sources=backend_sources,
        _initialize_backend=initialize_context,
        _verify_backend=verify_context,
    )
    exec(
        compile(
            ast.fix_missing_locations(copy.deepcopy(tree)),
            str(TOOLS / "prepare_raw_radau_resolvent.py") + "|jax-backend",
            "exec",
        ),
        namespace,
    )
    # Both functions share ACTIVE_JOURNAL/ACTIVE_LEASE; error handling and
    # signal cleanup therefore operate on this construction, never a module global.
    namespace["main"] = FunctionType(
        original["main"].__code__,
        namespace,
        original["main"].__name__,
        original["main"].__defaults__,
    )
    return namespace


def main():
    private_producer()["main"]()


if __name__ == "__main__":
    main()

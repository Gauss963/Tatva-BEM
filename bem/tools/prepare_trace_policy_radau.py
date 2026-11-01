"""Optional JAX raw assembly plus general-LU/lazy Trace-Schur construction.

Reuses the existing durable JAX adapter. Only private transfer/work bindings
and explicit backend context differ. Direct startup and independent original
audits retain their original SciPy evaluation. Existing original/JAX-only
caches cannot silently adopt this policy.
"""

import runpy
from pathlib import Path
from types import FunctionType, MethodType

from finiteblock_bem.io import fingerprint

TOOLS = Path(__file__).resolve().parent
JAX = runpy.run_path(str(TOOLS / "prepare_jax_radau_resolvent.py"))
POLICY = runpy.run_path(str(TOOLS / "trace_solve_policy.py"))
BACKEND = "jax_x64_raw_general_lu_lazy_trace_original_startup"


def backend_sources():
    return {
        **JAX["backend_sources"](),
        **{
            str((TOOLS / name).resolve()): fingerprint(TOOLS / name)
            for name in [Path(__file__).name, "trace_solve_policy.py"]
        },
    }


def backend_record():
    return {
        **JAX["backend_record"](),
        "backend": BACKEND,
        "sources_sha256": backend_sources(),
        "dynamic_linear_solve_assumption": "gen",
        "unused_reconstruction_skipped_for_impedance_only": True,
        "direct_startup_keeps_original_unmodified_transfer": True,
    }


def context_function(name):
    original = JAX[name]
    return FunctionType(
        original.__code__,
        {**original.__globals__, "backend_record": backend_record},
        original.__name__,
        original.__defaults__,
    )


initialize_context = context_function("initialize_context")
verify_context = context_function("verify_context")


def private_producer():
    namespace = JAX["private_producer"]()
    original_transfer = namespace["TRANSFER"]

    def transfer(*args, **kwargs):
        original = original_transfer(*args, **kwargs)
        candidate = POLICY["transfer_policy"](original, general=True, lazy=True)
        original_assemble = original.assemble_reduced.__func__

        def assemble(self, s, *, operators=None):
            if operators is None:
                operators = [p.assemble(s) for p in self.prepared]
            return original_assemble(self, s, operators=POLICY["layers"](operators, True))

        candidate.assemble_reduced = MethodType(assemble, candidate)
        # The direct method is used for two lag-zero modes. Binding it to the
        # unmodified original object preserves every startup solve and bit.
        candidate.evaluate = original.evaluate
        return candidate

    namespace.update(
        TRANSFER=transfer,
        WORK=POLICY["work_policy"](True),
        _backend_sources=backend_sources,
        _initialize_backend=initialize_context,
        _verify_backend=verify_context,
    )
    return namespace


def main():
    private_producer()["main"]()


if __name__ == "__main__":
    main()

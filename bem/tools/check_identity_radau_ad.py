"""Run the original complete initialized RK AD check with exact nodal tangents.

The original checker still verifies the saved primal, causal operator audit,
initial static sensitivities and every forward/reverse history source. Only
the identity-map contact Jacobian assembly changes. Newton iterations and
cached checkpoint states are not treated as a differentiable trajectory.
"""

import hashlib
import runpy
from pathlib import Path
from types import FunctionType, SimpleNamespace

TOOLS = Path(__file__).resolve().parent
POLICY = runpy.run_path(str(TOOLS / "identity_stage_tangent.py"))


def bind(function, **bindings):
    """Give a function private globals without mutating its original module."""
    result = FunctionType(
        function.__code__,
        {**function.__globals__, **bindings},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def identity_context(module, factory):
    context = module["context"]
    return bind(
        context,
        CONTACT={**context.__globals__["CONTACT"], "RadauContactEquilibrium": factory},
    )


def adapt_adjoint(module, factory):
    """Keep original complete forward/reverse loops, using the chosen tangent."""
    context = identity_context(module, factory)
    return {
        **module,
        **{
            name: bind(module[name], context=context)
            for name in ["forward_directions", "reverse_gradient"]
        },
    }


def adapt_initialized(module, factory):
    """Retain measured initial displacement, plastic and force derivatives."""
    base = module["BASE"]
    context = bind(
        module["initialized_context"],
        BASE={**base, "context": identity_context(base, factory)},
    )
    return {
        **module,
        "initialized_gradient": bind(module["initialized_gradient"], initialized_context=context),
    }


def main():
    checker = runpy.run_path(str(TOOLS / "check_compact_radau_ad.py"))
    paths = [Path(__file__), TOOLS / "identity_stage_tangent.py"]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    policies = []

    def verify():
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != v for p, v in frozen.items()):
            raise ValueError("Optional RK tangent assembly source changed")

    def factory(*args, **kwargs):
        verify()
        system = POLICY["stage_equilibrium"](*args, **kwargs)
        policies.append(system.tangent_policy)
        return system

    def load(path, *args, **kwargs):
        module = runpy.run_path(path, *args, **kwargs)
        if Path(path).name == "radau_contact_adjoint.py":
            return adapt_adjoint(module, factory)
        if Path(path).name == "initialized_radau_adjoint.py":
            return adapt_initialized(module, factory)
        return module

    def write(path, record):
        verify()
        if Path(path).name in ["preflight.json", "summary.json"]:
            record = {
                **record,
                "optional_tangent_assembly": dict(
                    policy="exact_nodal_contact_scatter_with_original_local_jax_ad",
                    source_sha256=frozen,
                    systems=policies.copy(),
                    original_full_equations_and_ad_checks_retained=True,
                    original_initialization_gradient_mode_retained=True,
                    every_history_source_retained=True,
                    general_contact_maps_use_original_tangent=True,
                    isolated_runtime_advantage_verified=False,
                ),
            }
        checker["write_json"](path, record)

    execute = bind(checker["main"], runpy=SimpleNamespace(run_path=load), write_json=write)
    execute()
    verify()


if __name__ == "__main__":
    main()

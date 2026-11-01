"""Private linear-solver and unused-reconstruction controls for Trace-Schur.

All original matrices, constraints and bilinear transposes are retained.
General LU changes floating-point solves. Lazy reconstruction removes only
work whose result is unused when impedance alone is requested. Neither policy
patches a module global or modifies a prepared operator.
"""

import ast
import copy
import inspect
import runpy
import textwrap
from pathlib import Path
from types import FunctionType, MethodType

from scipy.linalg import solve

from finiteblock_bem.galerkin import GalerkinOperators

TOOLS = Path(__file__).resolve().parent
TRACE = runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"]
WORK = runpy.run_path(str(TOOLS / "build_resolvent_history.py"))["compliance_in_work_coordinates"]


def general_solve(*args, **kwargs):
    if "assume_a" in kwargs:
        raise ValueError("Do not silently replace an explicit original matrix assumption")
    return solve(*args, **kwargs, assume_a="gen")


def general_function(function):
    return FunctionType(
        function.__code__,
        {**function.__globals__, "solve": general_solve},
        function.__name__,
        function.__defaults__,
    )


class GeneralLayers(GalerkinOperators):
    force_stiffness = general_function(GalerkinOperators.force_stiffness)


def layers(operators, general):
    if not general:
        return operators
    return [
        GeneralLayers(a.single_layer, a.trace_double_layer, a.hypersingular, a.mixed_mass)
        for a in operators
    ]


def impedance_only(general=False):
    """Delete exactly the three unused recovery statements and its return product."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(TRACE.condense)))
    expected = {
        "recover = np.zeros((self.constraints.size, len(trace)), dtype=stiffness.dtype)",
        "recover[trace] = np.eye(len(trace))",
        "recover[ids] = -response",
    }
    removed = set()

    class Remove(ast.NodeTransformer):
        def visit_Assign(self, node):
            source = ast.unparse(node)
            if source in expected:
                removed.add(source)
                return None
            return self.generic_visit(node)

    tree = Remove().visit(tree)
    function = tree.body[0]
    if (
        removed != expected
        or ast.unparse(function.body[-1])
        != "return (impedance, recover @ trace_response, eliminated)"
    ):
        raise ValueError("Original Trace-Schur recovery boundary changed")
    function.name = "impedance_only"
    function.body[-1] = ast.Return(value=ast.Name(id="impedance", ctx=ast.Load()))
    namespace = {**TRACE.condense.__globals__, "solve": general_solve if general else solve}
    exec(compile(ast.fix_missing_locations(tree), __file__ + "|impedance-only", "exec"), namespace)
    return namespace["impedance_only"]


def transfer_policy(original, *, general=False, lazy=False):
    candidate = copy.copy(original)
    if general:
        candidate.condense = MethodType(general_function(TRACE.condense), candidate)
    if lazy:
        evaluate = impedance_only(general)
        original_from_reduced = TRACE._from_reduced

        def from_reduced(self, stiffness, *, reconstruction=False):
            if not reconstruction:
                return evaluate(self, stiffness)
            return original_from_reduced(self, stiffness, reconstruction=True)

        candidate._from_reduced = MethodType(from_reduced, candidate)
    return candidate


def work_policy(general=False):
    return general_function(WORK) if general else WORK

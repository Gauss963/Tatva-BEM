"""Experimental CPU-JIT batch assembly of the unchanged Galerkin form.

The quadrature and singular corrections are extracted from the original
assembler, not reimplemented. Frequency and ray coefficients are dynamic JIT
inputs. All bins must pass the existing empirical table checks; otherwise the
entire frequency uses the original factored NumPy assembler. No production
module or paid cache is modified. This is not a material/geometry AD interface.
"""

import ast
import copy
import inspect
import runpy
import textwrap
from collections import namedtuple
from functools import partial
from pathlib import Path
from types import FunctionType

import jax
import jax.numpy as jnp
import numpy as np

from finiteblock_bem import kernels
from finiteblock_bem.galerkin import ROTATION, GalerkinOperators
from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators

TOOLS = Path(__file__).resolve().parent
RAY = runpy.run_path(str(TOOLS / "ray_bessel_galerkin.py"))
FACTORED = runpy.run_path(str(TOOLS / "factored_galerkin.py"))
FIELDS = "vector measure px py dx dy nx ny angular log integral log_moment".split()
Batch = namedtuple("Batch", FIELDS)
Material = namedtuple("Material", "mu lam density cs cp")


def _jax_copy(original):
    return FunctionType(original.__code__, {**original.__globals__, "np": jnp}, original.__name__)


_series = _jax_copy(kernels._difference_derivatives_series)
_tensor = _jax_copy(kernels._tensor_from_radial)
_traction = _jax_copy(kernels.source_traction)


def _pair(z, coefficients, first_exponent):
    """All arguments are on the validated ray and inside accepted table bins."""
    mantissa, exponent = jnp.frexp(jnp.abs(z))
    ids = exponent - 1 - first_exponent
    x = 4 * mantissa - 3
    zero = jnp.zeros((2, *z.shape), dtype=jnp.complex128)

    def step(index, carry):
        a, b = carry
        k = coefficients.shape[-1] - 1 - index
        return coefficients[:, ids, k] + 2 * x * a - b, a

    a, b = jax.lax.fori_loop(0, coefficients.shape[-1] - 1, step, (zero, zero))
    value = (coefficients[:, ids, 0] + x * a - b) * jnp.exp(-z)[None]
    return value[0], value[1]


def _radial_code(original):
    """Retain radial algebra; translate the data-dependent series assignment."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    expected = {
        "ks, kp = complex(s) / material.cs, complex(s) / material.cp": "ks, kp = s / material.cs, s / material.cp",
        "k0s, k1s, k0p, k1p = kv(0, zs), kv(1, zs), kv(0, zp), kv(1, zp)": "k0s, k1s = _pair(zs, coefficients, first_exponent)\n"
        "k0p, k1p = _pair(zp, coefficients, first_exponent)",
        "if np.any(small):\n"
        "    d1, d2, d3 = _difference_derivatives_series(r[small], ks, kp)\n"
        "    f1[small], f2[small], f3[small] = d1, d2, d3":
        # The inactive auxiliary input avoids large-power overflow. Actual
        # separations and all active series inputs remain unchanged.
        "auxiliary_r = np.where(small, r, 0.1 / np.abs(ks))\n"
        "d1, d2, d3 = _series(auxiliary_r, ks, kp)\n"
        "f1 = np.where(small, d1, f1)\n"
        "f2 = np.where(small, d2, f2)\n"
        "f3 = np.where(small, d3, f3)",
    }
    body = tree.body[0].body
    for before, after in expected.items():
        target = ast.dump(ast.parse(before).body[0], include_attributes=False)
        positions = [
            i for i, n in enumerate(body) if ast.dump(n, include_attributes=False) == target
        ]
        if len(positions) != 1:
            raise ValueError("Original radial source changed; re-audit the JAX adapter")
        index = positions[0]
        body[index : index + 1] = ast.parse(after).body
    function = tree.body[0]
    function.args.args.extend([ast.arg(arg="coefficients"), ast.arg(arg="first_exponent")])
    namespace = dict(np=jnp, _pair=_pair, _series=_series)
    exec(compile(ast.fix_missing_locations(tree), __file__ + "|radial", "exec"), namespace)
    return namespace[original.__name__]


_radial = _radial_code(kernels._radial)


def _batch_code(original):
    """Extract the original per-batch equations, stopping before global scatter."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    loops = [n for n in tree.body[0].body if isinstance(n, ast.For)]
    if len(loops) != 1 or ast.unparse(loops[0].target) != "b":
        raise ValueError("Original Galerkin loop changed; re-audit the JAX adapter")
    body = []
    for node in loops[0].body:
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith("np.add.at("):
            break
        if isinstance(node, ast.If) and ast.unparse(node.test) == "s is None":
            body.extend(copy.deepcopy(node.orelse))
        else:
            body.append(copy.deepcopy(node))
    if (
        not body
        or not isinstance(body[-1], ast.Assign)
        or ast.unparse(body[-1].targets[0]) != "local_k"
    ):
        raise ValueError("Original local block boundary changed; re-audit the JAX adapter")
    # Retrieve the reverse double-layer integral from its original scatter block.
    reverse = [
        n
        for n in ast.walk(loops[0])
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "local_reverse"
    ]
    if len(reverse) != 1:
        raise ValueError("Original reverse block changed; re-audit the JAX adapter")
    header = ast.parse(
        "mu, lam = material.mu, material.lam\n"
        "log_green = -(lam + 3 * mu) / (4 * np.pi * mu * (lam + 2 * mu))\n"
        "log_derivative = 4 * mu**2 * (log_green + 1 / (2 * np.pi * (lam + 2 * mu)))\n"
        "effective_s = s\n"
    ).body

    # The extracted call receives the dynamic frequency-local coefficients.
    class Adapt(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if ast.unparse(node) == "b.self_pair":
                return ast.copy_location(ast.Name(id="self_pair", ctx=ast.Load()), node)
            return self.generic_visit(node)

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "_radial":
                node.args.extend(
                    [
                        ast.Name(id="coefficients", ctx=ast.Load()),
                        ast.Name(id="first_exponent", ctx=ast.Load()),
                    ]
                )
            return node

    footer = ast.parse("local_reverse = np.zeros_like(local_k)\nif not self_pair:\n    pass\n").body
    footer[-1].body = [copy.deepcopy(reverse[0])]
    function = ast.parse(
        "def batch(b, s, material, coefficients, first_exponent, self_pair):\n    pass\n"
    ).body[0]
    function.body = (
        header
        + body
        + footer
        + ast.parse("return local_v, local_k, local_d.reshape(p, 4, 4), local_reverse").body
    )
    module = Adapt().visit(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(
        np=jnp,
        _radial=_radial,
        _tensor_from_radial=_tensor,
        source_traction=_traction,
        ROTATION=jnp.asarray(ROTATION),
    )
    exec(compile(ast.fix_missing_locations(module), __file__ + "|batch", "exec"), namespace)
    return namespace["batch"]


_original_batch = _batch_code(PreparedGalerkinOperators.assemble)
COMPILED_SIGNATURES = []


@partial(jax.jit, static_argnames=("self_pair",))
def _local_blocks(b, s, material, coefficients, first_exponent, self_pair):
    # Python executes this during tracing only; it never retains device arrays.
    COMPILED_SIGNATURES.append(
        dict(
            measure_shape=list(b.measure.shape),
            coefficients_shape=list(coefficients.shape),
            self_pair=self_pair,
        )
    )
    return _original_batch(b, s, material, coefficients, first_exponent, self_pair)


class JaxRayGalerkin(RAY["RayBesselGalerkin"]):
    """Bounded per-batch JIT work, original host scatter, explicit fallback."""

    def prepare(self, s):
        if not jax.config.x64_enabled:
            raise ValueError("The JAX Galerkin experiment requires JAX_ENABLE_X64=true")
        _, table = super().prepare(s)
        if not np.all(table.accepted):
            return (
                lambda: FACTORED["FactoredGalerkinOperators"].share(self.prepared).assemble(s),
                table,
            )
        material = Material(*(getattr(self.prepared.material, k) for k in Material._fields))
        # Uploaded once per frequency, not once per quadrature batch. np.asarray
        # below synchronizes every result, including first-call compilation.
        coefficients = jnp.asarray(table.coefficients)
        frequency = jnp.asarray(complex(s), dtype=jnp.complex128)
        first = jnp.asarray(table.first_exponent, dtype=jnp.int32)

        def evaluate():
            n = self.prepared.mesh.dofs
            vmat = np.zeros((n, n), complex)
            hmat, dmat = vmat.copy(), vmat.copy()
            for b in self.prepared.batches:
                args = Batch(*(getattr(b, k) for k in FIELDS))
                blocks = _local_blocks(args, frequency, material, coefficients, first, b.self_pair)
                local_v, local_k, local_d, local_reverse = [np.asarray(a) for a in blocks]
                np.add.at(vmat.ravel(), b.vindices, local_v)
                np.add.at(hmat.ravel(), b.hindices, local_k)
                np.add.at(dmat.ravel(), b.dindices, local_d)
                if not b.self_pair:
                    np.add.at(vmat.ravel(), (b.vindices % n) * n + b.vindices // n, local_v)
                    np.add.at(dmat.ravel(), (b.dindices % n) * n + b.dindices // n, local_d)
                    np.add.at(hmat.ravel(), b.reverse_hindices, local_reverse)
            hmat += 0.5 * self.prepared.mass
            return GalerkinOperators(vmat, hmat, dmat, self.prepared.mass)

        return evaluate, table

    def assemble(self, s=None):
        if s is None:
            return self.prepared.assemble(None)
        evaluate, _ = self.prepare(s)
        return evaluate()

    def statistics(self):
        return dict(
            backend="jax_x64_ray_batch_original_equations",
            whole_frequency_fallback_if_any_bin_rejected=True,
            original_host_scatter=True,
            original_small_argument_series_retained=True,
            dynamic_frequency_and_coefficients=True,
            fixed_table_material_or_geometry_derivatives_supported=False,
            uniform_error_certificate=False,
            production_backend_replaced=False,
        )

"""Evaluate both scaled-Bessel orders together with factored contractions.

The existing ray table and its empirical acceptance/fallback policy are
unchanged. K0/K1 share coordinates, masks and exp(-z), and only a private
copy of the original radial function and assembler uses the pair evaluator.
This experimental backend does not modify a running kernel producer.
"""

import ast
import inspect
import runpy
import textwrap
from pathlib import Path
from types import FunctionType

import numpy as np
from scipy.special import kv

from finiteblock_bem import kernels
from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators

TOOLS = Path(__file__).resolve().parent
RAY = runpy.run_path(str(TOOLS / "ray_bessel_galerkin.py"))
FACTORED = runpy.run_path(str(TOOLS / "factored_galerkin.py"))


def bessel_pair(table, z):
    """Same two ray interpolants; no last-call cache or shared mutable state."""
    original = np.asarray(z)
    if not np.iscomplexobj(original):
        return kv(0, z), kv(1, z)
    array = np.asarray(z, complex)
    flat = array.reshape(-1)
    use, bins, x = table._coordinates(flat)
    result = np.empty((2, len(flat)), complex)
    result[0, ~use], result[1, ~use] = kv(0, flat[~use]), kv(1, flat[~use])
    if np.any(use):
        ids, coordinate = bins[use], x[use][None, :]
        c = table.coefficients
        a = np.zeros((2, len(ids)), complex)
        b = np.zeros_like(a)
        for k in range(c.shape[-1] - 1, 0, -1):
            a, b = c[:, ids, k] + 2 * coordinate * a - b, a
        result[:, use] = (c[:, ids, 0] + coordinate * a - b) * np.exp(-flat[use])[None, :]
    result = result.reshape((2, *array.shape))
    return result[0], result[1]


def paired_radial_code(original):
    """Replace just the four explicit Bessel calls; retain the radial algebra."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    expected = ast.parse("k0s, k1s, k0p, k1p = kv(0, zs), kv(1, zs), kv(0, zp), kv(1, zp)").body[0]
    positions = [
        i
        for i, node in enumerate(function.body)
        if ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False)
    ]
    if len(positions) != 1 or original.__code__.co_freevars:
        raise ValueError("Original radial Bessel assignment changed; re-audit the adapter")
    index = positions[0]
    function.body[index : index + 1] = ast.parse(
        "k0s, k1s = _bessel_pair(zs)\nk0p, k1p = _bessel_pair(zp)"
    ).body
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), __file__ + "|radial", "exec"), namespace)
    return namespace[original.__name__].__code__


RADIAL_CODE = paired_radial_code(kernels._radial)


class PairedRayGalerkin(RAY["RayBesselGalerkin"]):
    """Read-only shared geometry, frequency-local table, private contractions."""

    def prepare(self, s):
        _, table = super().prepare(s)
        original = kernels._radial
        radial = FunctionType(
            RADIAL_CODE,
            {**original.__globals__, "_bessel_pair": lambda z: bessel_pair(table, z)},
            original.__name__,
            original.__defaults__,
        )
        radial.__kwdefaults__ = original.__kwdefaults__
        original = PreparedGalerkinOperators.assemble
        assemble = FunctionType(
            original.__code__,
            {**original.__globals__, "_radial": radial, "np": FACTORED["_PrivateNumpy"]()},
            original.__name__,
            original.__defaults__,
        )

        def evaluate():
            return assemble(self.prepared, s)

        return evaluate, table

    def assemble(self, s=None):
        if s is None:
            return FACTORED["FactoredGalerkinOperators"].share(self.prepared).assemble(None)
        evaluate, _ = self.prepare(s)
        return evaluate()

    def statistics(self):
        return dict(
            paired_orders_share_coordinates_and_exponential=True,
            original_ray_bin_policy_retained=True,
            original_small_argument_series_retained=True,
            factored_contractions=FACTORED["FactoredGalerkinOperators"]
            .share(self.prepared)
            .statistics(),
            uniform_error_certificate=False,
            production_backend_replaced=False,
        )

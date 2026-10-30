"""Factor quadrature contractions without changing geometry or Green kernels.

Only four einsum contractions differ in a private copy of the reference
assembler. Factors independent of the quadrature index move outside its sum.
This changes floating-point reduction order, not the Galerkin bilinear form.
No original module, NumPy global, mesh, quadrature or correction is modified.
"""

from types import FunctionType

import numpy as np

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators

SIGNATURES = (
    "pq,pqab,pqi,pqj->piajb",
    "pq,pqab,pi,pj->piajb",
    "pq,pqab,pqi,pj->piajb",
    "pq,pqab,pi,pqj->piajb",
)


def factored_einsum(signature, *arrays, **kwargs):
    if signature not in SIGNATURES or kwargs:
        return np.einsum(signature, *arrays, **kwargs)
    measure, tensor, left, right = arrays
    if signature == SIGNATURES[0]:
        basis = measure[..., None, None] * left[..., :, None] * right[..., None, :]
        return np.einsum("pqij,pqab->piajb", basis, tensor)
    if signature == SIGNATURES[1]:
        integral = np.einsum("pq,pqab->pab", measure, tensor)
        return np.einsum("pab,pi,pj->piajb", integral, left, right)
    if signature == SIGNATURES[2]:
        integral = np.einsum("pq,pqab,pqi->piab", measure, tensor, left)
        return np.einsum("piab,pj->piajb", integral, right)
    integral = np.einsum("pq,pqab,pqj->pajb", measure, tensor, right)
    return np.einsum("pajb,pi->piajb", integral, left)


class _PrivateNumpy:
    einsum = staticmethod(factored_einsum)

    def __getattr__(self, name):
        return getattr(np, name)


_original = PreparedGalerkinOperators.assemble
_ASSEMBLE = FunctionType(
    _original.__code__,
    {**_original.__globals__, "np": _PrivateNumpy()},
    _original.__name__,
    _original.__defaults__,
)


class FactoredGalerkinOperators(PreparedGalerkinOperators):
    @classmethod
    def share(cls, prepared):
        """Share the original immutable geometry, without a second preparation."""
        if type(prepared) is not PreparedGalerkinOperators:
            raise ValueError("Require the original prepared Galerkin assembler")
        if (
            any(
                a.flags.writeable
                for batch in prepared.batches
                for a in vars(batch).values()
                if isinstance(a, np.ndarray)
            )
            or prepared.mass.flags.writeable
        ):
            raise ValueError("Shared Galerkin geometry must be immutable")
        result = cls.__new__(cls)
        result.__dict__.update(prepared.__dict__)
        return result

    def assemble(self, s=None):
        return _ASSEMBLE(self, s)

    def statistics(self):
        # The first real product (2*p*q) coexists with the final basis (4*p*q).
        # These and the small complex integrated blocks are
        # bounded by one existing batch. This is additional array workspace,
        # excluding the unchanged reference kernels, geometry and solver.
        extra = max(6 * b.measure.size * 8 + len(b.i) * (4 + 8) * 16 for b in self.batches)
        return dict(
            contractions=list(SIGNATURES),
            additional_workspace_estimate_bytes=extra,
            prepared_geometry_shared=True,
            reduction_order_changed=True,
            quadrature_or_kernel_approximation=False,
        )

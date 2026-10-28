"""Exact real transpose of the fixed-kernel interior CQ reference.

The artificial initial block is fixed at zero. Return cotangents at every
physical input stage, keeping the complete future dependence. Kernel/material,
geometry and separately reconstructed static-lift derivatives are not included.
No forward trajectory or differentiation through frequency factorizations is
needed for this linear pullback. This dense reference is not a large-event
bounded-history backend.
"""

import numpy as np


def interior_pullback(weights, cotangent):
    weights, cotangent = np.asarray(weights), np.asarray(cotangent)
    if (
        weights.ndim != 3
        or cotangent.shape != weights.shape[:2]
        or weights.shape[1] % 2
        or weights.shape[2] % 2
        or np.iscomplexobj(weights)
        or np.iscomplexobj(cotangent)
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(cotangent))
    ):
        raise ValueError("Require complete finite real CQ weights and output stage cotangents")
    answer = np.zeros((len(weights), weights.shape[2]), dtype=np.result_type(weights, cotangent))
    for j in range(1, len(weights)):
        answer[j] = np.einsum("lij,li->j", weights[: len(weights) - j], cotangent[j:])
    return answer

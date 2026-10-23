import runpy
from pathlib import Path

import numpy as np
import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/audit_replay_residual.py"))


@pytest.mark.parametrize("steps,dofs", [(7, 3), (37, 7)])
def test_original_force_reconstruction_matches_every_direct_nonsymmetric_sum(steps, dofs):
    rng = np.random.default_rng(165)
    weights = rng.normal(size=(steps + 1, dofs, dofs))
    q = rng.normal(size=(steps + 1, dofs))
    actual = TOOL["original_bulk_force"](weights, q)
    expected = np.array(
        [np.einsum("kij,kj->i", weights[: n + 1], q[n::-1]) for n in range(steps + 1)]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)

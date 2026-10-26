import runpy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
ROOT = Path(__file__).resolve().parents[1]
SOURCE = runpy.run_path(str(ROOT / "tests/test_refined_contact_adjoint.py"))
TOOL = runpy.run_path(str(ROOT / "tools/batched_refined_adjoint.py"))


@pytest.mark.parametrize("substeps", [1, 4])
def test_batched_complete_history_gradient_matches_scalar_jvp_and_reverse(substeps):
    reference, original, history, time, actuator = SOURCE["problem"]()
    result = SOURCE["direct_trajectory"](reference, original, time, actuator, substeps)
    actual = TOOL["forward_gradient_batched"](
        reference, history, result, constitutive_substeps=substeps, cache_mb=0
    )
    for method in ("forward_gradient", "reverse_gradient"):
        expected = SOURCE["AD"][method](reference, history, result, constitutive_substeps=substeps)
        np.testing.assert_allclose(actual["gradient"], expected["gradient"], rtol=1e-10, atol=1e-11)
    assert actual["history"]["kernel_fft_shared_across_right_hand_sides"]

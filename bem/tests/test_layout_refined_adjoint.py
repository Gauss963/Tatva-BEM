import runpy
from functools import partial
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
ROOT = Path(__file__).resolve().parents[1]
SOURCE = runpy.run_path(str(ROOT / "tests/test_refined_contact_adjoint.py"))
LAYOUT = runpy.run_path(str(ROOT / "tools/layout_aware_history.py"))[
    "LayoutAwareBoundedConvolution"
]


@pytest.mark.parametrize("substeps", [1, 4])
def test_layout_reverse_full_history_gradients_match_scalar_jvp_and_row_reference(substeps):
    reference, original, history, time, actuator = SOURCE["problem"]()
    result = SOURCE["direct_trajectory"](reference, original, time, actuator, substeps)
    actual = SOURCE["AD"]["reverse_gradient"](
        reference,
        history,
        result,
        constitutive_substeps=substeps,
        convolution_factory=partial(LAYOUT, cache_mb=0, fft_rows=3, direct_block=2),
    )
    expected = SOURCE["AD"]["reverse_gradient"](
        reference, history, result, constitutive_substeps=substeps
    )
    forward = SOURCE["AD"]["forward_gradient"](
        reference, history, result, constitutive_substeps=substeps
    )
    for target in [expected, forward]:
        np.testing.assert_allclose(actual["gradient"], target["gradient"], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(
        actual["actuator_gradient"], expected["actuator_gradient"], rtol=1e-10, atol=1e-11
    )

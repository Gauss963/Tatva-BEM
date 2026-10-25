import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy.fft import rfft

from finiteblock_bem.cq import plan_cq

pytest.importorskip("jax")
TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/forward_resolvent_experiment.py")
)


@pytest.mark.parametrize("work_coordinates", [False, True])
def test_prefix_forward_action_retains_all_lags_and_discards_wrapped_early_outputs(
    work_coordinates,
):
    rng = np.random.default_rng(41)
    plan = plan_cq(7, 0.02, 37, max_memory_mb=32)
    weights = rng.normal(size=(38, 7, 7))
    source = rng.normal(size=(21, 7, 2))
    scale = np.geomspace(0.25, 20, 7) if work_coordinates else np.ones(7)
    spectrum = rfft(
        weights
        * scale[None, :, None]
        * scale[None, None, :]
        * plan.radius ** np.arange(38)[:, None, None],
        n=plan.fft_size,
        axis=0,
    )
    actual = TOOL["prefix_forward_response"](spectrum, plan, source, 37, coordinate_scale=scale)
    expected = np.array([sum(weights[i - j] @ source[j] for j in range(21)) for i in range(20, 38)])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

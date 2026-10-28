"""Original RK audit includes both modes, all lags and finite-circle wrap."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import invert_spectrum, plan_cq

TOOLS = Path(__file__).resolve().parents[1] / "tools"
AUDIT = runpy.run_path(str(TOOLS / "radau_original_history_audit.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))


def spectra(plan):
    mass = np.array([[2.0, 0.4], [0.4, 1.0]])
    stiffness = np.array([[0.7, -0.2], [-0.2, 0.5]])
    memory = np.array([[0.4, 0.1], [0.1, 0.3]])

    def original(s):
        return s * mass + stiffness + memory / (s + 0.2)

    def compliance(s):
        return np.linalg.inv(original(s))

    nodes = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    compact = [RADAU["compact_matrix_function"](compliance, z, plan.dt) for z in nodes]
    dense = [RADAU["matrix_function"](original, RADAU["radau_symbol"](z) / plan.dt) for z in nodes]
    return np.array([v[0] for v in compact]), np.array([v[1] for v in compact]), np.array(dense)


@pytest.mark.parametrize("workers,rows", [(1, 1), (2, 2)])
def test_reconstructed_causal_force_matches_dense_original_and_excludes_future(workers, rows):
    plan = replace(plan_cq(2, 0.1, 12, max_memory_mb=32), fft_size=64, radius=0.05 ** (1 / 64))
    f0, f1, dense = spectra(plan)
    g0, g1 = np.empty_like(f0), np.empty_like(f1)
    record = AUDIT["invert_compact_spectra"](f0, f1, (g0, g1), plan, workers=workers)
    assert record["maximum_inverse_composition_inf_error"] < 1e-12
    np.testing.assert_array_equal(f0.shape, g0.shape)
    line = np.array([2.0, 7.0])
    q = np.random.default_rng(83).normal(size=(plan.steps + 1, 2, plan.dofs, 2))
    action, info = AUDIT["causal_compact_original_action"](
        g0, g1, plan, q, line, fft_rows=rows, max_workspace_mb=32
    )
    assert info["maximum_absolute_work_family_shift_wrap"] > 1e-9
    assert info["finite_dft_shift_wrap_retained"]
    weights = invert_spectrum(dense, replace(plan, dofs=2 * plan.dofs))
    expanded_scale = np.tile(np.sqrt(line), 2)
    weights = weights * expanded_scale[None, :, None] * expanded_scale[None, None, :]
    flattened = q.reshape(len(q), 2 * plan.dofs, 2)
    expected = np.array(
        [np.einsum("kij,kjp->ip", weights[: k + 1], flattened[k::-1]) for k in range(len(q))]
    ).reshape(q.shape)
    np.testing.assert_allclose(action, expected, rtol=2e-11, atol=2e-11)
    changed = q.copy()
    changed[8:] += 1e5
    other, _ = AUDIT["causal_compact_original_action"](
        g0, g1, plan, changed, line, fft_rows=rows, max_workspace_mb=32
    )
    np.testing.assert_allclose(other[:8], action[:8], rtol=0, atol=1e-8)


def test_bad_budget_or_nonfinite_spectrum_is_rejected():
    plan = plan_cq(2, 0.1, 8, max_memory_mb=32)
    f0, f1, _ = spectra(plan)
    q = np.zeros((plan.steps + 1, 2, 2))
    with pytest.raises(ValueError, match="array workspace"):
        AUDIT["causal_compact_original_action"](f0, f1, plan, q, np.ones(2), max_workspace_mb=1e-6)
    f0[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite source"):
        AUDIT["invert_compact_spectra"](f0, f1, (np.empty_like(f0), np.empty_like(f1)), plan)

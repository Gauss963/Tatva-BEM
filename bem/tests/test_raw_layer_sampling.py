import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import solve

from finiteblock_bem.cq import bdf2_symbol, invert_spectrum, plan_cq
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOLS = Path(__file__).resolve().parents[1] / "tools"
RAW = runpy.run_path(str(TOOLS / "raw_layer_sampling.py"))
SCHUR = runpy.run_path(str(TOOLS / "schur_condensation_experiment.py"))["ConstrainedSchurTransfer"]
CASE = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def collect(raw, transform, plan, **kwargs):
    values = np.zeros((plan.fft_size // 2 + 1, plan.dofs, plan.dofs), complex)
    visited = []

    def consume(index, s, value):
        visited.append(index)
        values[index] = value

    record = RAW["sample_raw_layers"](raw, transform, plan, consume, **kwargs)
    assert sorted(visited) == list(range(len(values)))
    assert len(visited) == len(set(visited))
    return values, record


def test_polynomial_raw_layers_recover_every_resonant_condensed_frequency_and_lag():
    plan = plan_cq(2, 0.01, 256, max_memory_mb=32)
    eye = np.eye(2)

    def raw(s):
        v = (1 + 0.01 * s) * eye
        h = np.array([[1.0, 0.1 * s], [0, 1]], complex)
        d = s * s * np.diag([1.0, 1.3]) + 0.08 * s * eye + 400 * np.array([[1.0, 0.1], [0.1, 1.0]])
        return RAW["RawLayerPacket"]((v, h, d), (eye,))

    def transform(packet, s):
        return solve(packet.operators()[0].force_stiffness() + 0.2 * eye, eye)

    actual, record = collect(raw, transform, plan, workers=2)
    theta = np.linspace(0, np.pi, len(actual))
    exact = np.array(
        [transform(raw(s), s) for s in bdf2_symbol(plan.radius * np.exp(-1j * theta)) / plan.dt]
    )
    relative = np.linalg.norm(actual - exact, axis=(1, 2)) / np.linalg.norm(exact, axis=(1, 2))
    assert relative.max() < 1e-7
    np.testing.assert_allclose(
        invert_spectrum(actual, plan), invert_spectrum(exact, plan), rtol=1e-6, atol=1e-10
    )
    assert record["accepted_intervals"] > 0
    assert record["source_evaluations"] < len(actual)
    assert record["all_fft_grid_points_emitted"]


def test_independent_output_check_detects_small_raw_family_mode_amplification():
    plan = plan_cq(2, 0.01, 256, max_memory_mb=32)
    eye = np.eye(2)

    def raw(s):
        z = 2 - np.sqrt(1 + 2 * s * plan.dt)
        theta = -np.angle(z)
        d = np.diag([1e12, 1 + 0.02 * np.cos(25 * theta)])
        return RAW["RawLayerPacket"]((eye, np.zeros((2, 2)), d), (eye,))

    def transform(packet, s):
        return solve(packet.families[2], eye)

    options = RAW["RawSamplingOptions"](raw_rtol=1e-10, output_rtol=1e-9)
    actual, record = collect(raw, transform, plan, options=options)
    angles = np.linspace(0, np.pi, len(actual))
    exact = np.array(
        [transform(raw(s), s) for s in bdf2_symbol(plan.radius * np.exp(-1j * angles)) / plan.dt]
    )
    np.testing.assert_allclose(actual, exact, rtol=1e-7, atol=1e-12)
    assert record["rejected_intervals"] > 0


def test_real_finite_blocks_match_fresh_bem_at_every_frequency_and_cq_lag():
    model = load_nodal_model(CASE, max_panel_length=250)
    original = NodalInterfaceTransfer(model, quadrature=4)
    reduced = SCHUR(model, quadrature=4, prepared=False)
    plan = plan_cq(original.dofs, 1e-4, 128, max_memory_mb=64)
    weights = model.constraints.contact_weights
    law = model.reference.contact
    diagonal = np.r_[weights * law.normal_penalty, weights * law.tangential_penalty, 0]
    diagonal[-1] += model.reference.spring_stiffness * model.constraints.platen_width
    line = np.r_[weights, weights, model.constraints.platen_width]
    scale = np.sqrt(line)

    def raw(s):
        return RAW["RawLayerPacket"].from_operators(p.assemble(s) for p in original.prepared)

    def compliance(z):
        return scale[:, None] * solve(z + np.diag(diagonal), np.diag(scale))

    def transform(packet, s):
        return compliance(reduced.evaluate_operators(packet.operators()))

    actual, record = collect(raw, transform, plan, workers=2, max_workspace_mb=64)
    exact = np.array(
        [
            compliance(original.evaluate(s))
            for s in bdf2_symbol(plan.radius * np.exp(-1j * np.linspace(0, np.pi, len(actual))))
            / plan.dt
        ]
    )
    relative = np.linalg.norm(actual - exact, axis=(1, 2)) / np.linalg.norm(exact, axis=(1, 2))
    assert relative.max() < 1e-7
    np.testing.assert_allclose(
        invert_spectrum(actual, plan), invert_spectrum(exact, plan), rtol=1e-5, atol=1e-10
    )
    assert record["all_fft_grid_points_emitted"]


def test_workspace_guard_rejects_before_any_grid_output_and_fixed_mass_cannot_change():
    plan = plan_cq(2, 0.01, 64, max_memory_mb=16)
    eye = np.eye(2)

    def raw(s):
        return RAW["RawLayerPacket"]((eye, eye, eye), (eye,))

    with pytest.raises(ValueError, match="interpolation arrays"):
        RAW["sample_raw_layers"](
            raw,
            lambda p, s: eye,
            plan,
            lambda *a: pytest.fail("Output before workspace rejection"),
            max_workspace_mb=1e-6,
        )

    def changed(s):
        return RAW["RawLayerPacket"]((eye, eye, eye), (eye * (1 + abs(s)),))

    with pytest.raises(ValueError, match="mixed mass changed"):
        collect(changed, lambda p, s: eye, plan)

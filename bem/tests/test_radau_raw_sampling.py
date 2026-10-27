import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import bdf2_symbol, plan_cq
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CHECK = runpy.run_path(str(TOOLS / "radau_raw_sampling.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))
SCHUR = runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"]
WORK = runpy.run_path(str(TOOLS / "build_resolvent_history.py"))["compliance_in_work_coordinates"]


def test_continuous_half_circle_labels_match_the_full_stage_symbol():
    zeta = 0.93 * np.exp(-1j * np.linspace(0, np.pi, 301))
    modes = CHECK["lower_half_radau_modes"](zeta, 2e-6)
    for z, values in zip(zeta, modes, strict=True):
        symbol = RADAU["radau_symbol"](z) / 2e-6
        for value in values:
            assert (
                abs(np.linalg.det(symbol - value * np.eye(2))) < 1e-12 * np.linalg.norm(symbol) ** 2
            )
    assert np.max(abs(np.diff(modes, axis=0))) < 0.02 * np.max(abs(modes))
    assert modes[-1, 0] == pytest.approx(modes[-1, 1].conjugate(), rel=1e-14)
    assert abs(modes[-1, 0].imag) > 1e6


def test_bdf2_labels_are_only_coordinates_and_every_radau_index_is_emitted():
    plan = plan_cq(2, 2e-6, 128, disk_spectrum=True, max_memory_mb=2048)
    original = CHECK["RAW"]["sample_raw_layers"]
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    modes = CHECK["lower_half_radau_modes"](zeta, plan.dt)

    def raw(s):
        a = np.eye(2) * s
        return CHECK["RAW"]["RawLayerPacket"]((a, 2 * a, 3 * a), (np.eye(2),))

    for branch in [0, 1]:
        emitted = []
        records = CHECK["sample_radau_branch"](
            raw,
            lambda packet, s: packet.families[0],
            plan,
            branch,
            lambda i, s, value: emitted.append((i, s, value)),
            max_workspace_mb=16,
        )
        assert records["all_fft_grid_points_emitted"]
        assert records["branch_spectrum_is_not_a_real_kernel"]
        assert not records["source_frequency_values_reused"]
        assert [i for i, _, _ in emitted] == list(range(len(zeta)))
        np.testing.assert_allclose([s for _, s, _ in emitted], modes[:, branch], rtol=1e-13)
        np.testing.assert_allclose(
            [a for _, _, a in emitted],
            modes[:, branch, None, None] * np.eye(2),
            rtol=1e-11,
            atol=1e-7,
        )
    assert CHECK["RAW"]["sample_raw_layers"] is original


def test_dimensionless_coefficients_preserve_derivative_scaling_and_endpoint_reality():
    zeta = 0.9 * np.exp(-1j * np.array([0.0, 0.2, np.pi]))
    dt = 2e-6
    modes = CHECK["lower_half_radau_modes"](zeta, dt)
    first, second = [modes[:, i, None, None] * np.eye(2) for i in [0, 1]]
    f0, f1 = CHECK["compact_coefficients"](zeta, first, second)
    np.testing.assert_allclose(f0, 0, atol=1e-9)
    np.testing.assert_allclose(f1, np.broadcast_to(np.eye(2) / dt, f1.shape), rtol=1e-14, atol=1e-9)
    np.testing.assert_allclose(f1[[0, -1]].imag, 0, atol=1e-10)


def test_slow_mode_remains_accurate_near_the_causal_limit():
    epsilon, dt = 2.0**-30, 2e-6
    slow, _ = CHECK["lower_half_radau_modes"](1 - epsilon, dt)
    # Agreement with -log(z)/dt through cubic order; a direct subtraction
    # of the two roots loses about seven digits at this node.
    expected = -np.log1p(-epsilon) / dt
    assert slow == pytest.approx(expected, rel=3e-15)


def test_fresh_two_body_branch_weights_match_dense_original_stage_assembly():
    case = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    model = load_nodal_model(case, max_panel_length=160, contact_panel_length=160)
    transfer = SCHUR(model, quadrature=4)
    direct = NodalInterfaceTransfer(model, quadrature=4)
    wc, law = model.constraints.contact_weights, model.reference.contact
    line = np.r_[wc, wc, model.constraints.platen_width]
    tangent = np.diag(
        np.r_[
            wc * law.normal_penalty,
            wc * law.tangential_penalty,
            model.reference.spring_stiffness * model.constraints.platen_width,
        ]
    )
    plan = plan_cq(transfer.dofs, 2e-6, 16, max_memory_mb=2048)
    branches = [np.empty((plan.fft_size // 2 + 1, plan.dofs, plan.dofs), complex) for _ in range(2)]

    def raw(s):
        return CHECK["RAW"]["RawLayerPacket"].from_operators(
            p.assemble(s) for p in transfer.prepared
        )

    for branch in [0, 1]:
        CHECK["sample_radau_branch"](
            raw,
            lambda packet, s: WORK(transfer.evaluate_operators(packet.operators()), tangent, line),
            plan,
            branch,
            lambda i, s, value: branches[branch].__setitem__(i, value),
            max_workspace_mb=128,
        )
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    f0, f1 = CHECK["compact_coefficients"](zeta, *branches)
    w0, w1, wrap = RADAU["invert_compact_spectrum"](f0, f1, plan)
    actual = RADAU["expand_compact_weights"](w0, w1, wrap)
    stage_plan = plan_cq(2 * plan.dofs, plan.dt, plan.steps, max_memory_mb=2048)
    expected = RADAU["radau_weights"](lambda s: WORK(direct.evaluate(s), tangent, line), stage_plan)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)


def test_invalid_branch_and_nonmatching_sampler_coordinates_are_rejected():
    plan = plan_cq(2, 2e-6, 16, max_memory_mb=2048)
    with pytest.raises(ValueError):
        CHECK["sample_radau_branch"](None, None, plan, 2, None)
    with pytest.raises(ValueError):
        CHECK["node_from_sampler_parameter"](bdf2_symbol(0.5) / plan.dt, plan)
    with pytest.raises(ValueError):
        CHECK["lower_half_radau_modes"](0.5 + 0.2j, plan.dt)

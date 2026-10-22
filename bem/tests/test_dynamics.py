from pathlib import Path

import numpy as np

from finiteblock_bem import cli
from finiteblock_bem.case import load_case
from finiteblock_bem.constraints import build_constraints
from finiteblock_bem.dynamics import event16_schedule, initialize_restart, simulate

CASE = Path(__file__).resolve().parents[1] / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def test_static_lift_and_two_domain_transient_without_fabricated_slip():
    case = load_case(CASE, max_panel_length=80)
    lift = initialize_restart(case)
    assert lift.diagnostics["status"] == "admissible_static_lift"
    assert lift.diagnostics["introduced_cumulative_slip_mm"] == 0
    result = simulate(case, 4e-6, 16, schedule=event16_schedule(case), lift=lift)
    assert result.diagnostics["status"] == "completed_requested_horizon"
    assert result.diagnostics["all_history_retained"]
    assert result.diagnostics["max_contact_strength_ratio"] <= 1 + 1e-10
    assert result.diagnostics["max_boundary_contact_residual_mpa"] < 1e-6
    assert result.diagnostics["full_original_event_reproduced"] is False
    con = build_constraints(case)
    np.testing.assert_allclose(result.displacement[:, con.fixed_dofs], 0, atol=1e-12)
    platen = 2 * case.moving.indices("platen") + 1
    np.testing.assert_allclose(
        result.displacement[:, platen],
        np.repeat(result.platen[:, None], len(platen), axis=1),
        atol=1e-12,
    )
    np.testing.assert_array_equal(result.cumulative_slip, 0)
    assert np.max(abs(result.displacement[-1] - result.displacement[0])) > 1e-6


def test_original_fem_resolution_is_rejected_before_dense_kernel_allocation(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT", tmp_path)
    status = cli.main(
        [
            "preflight",
            "--case",
            str(CASE),
            "--h",
            "2",
            "--dt",
            "1.2492928633545953e-8",
            "--duration",
            ".045",
            "--name",
            "memory-guard",
        ]
    )
    assert status == 2
    assert (tmp_path / "outputs/memory-guard/summary.json").exists()

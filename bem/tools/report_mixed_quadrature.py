"""Curate measured quadrature, trajectory and conditional-adjoint records."""

import json
from pathlib import Path

from finiteblock_bem.io import fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = {
        "fine2_contour": "mixed-quadrature-fine2-q12-near12/summary.json",
        "coarse4_contour": "mixed-quadrature-coarse4-near8/summary.json",
        "h80_contour": "mixed-quadrature-h80-dt2-smoke/summary.json",
        "sampled_bulk_passivity": "mixed-quadrature-sampled-passivity.json",
        "short_kernel": "mixed8-native64-kernel/summary.json",
        "short_rollout": "mixed8-native64-rollout/summary.json",
        "short_adjoint_and_field_guard": "mixed8-native64-rollout/resolvent_adjoint.json",
        "short_original_equation_balance": "mixed8-native64-rollout/original_residual.json",
    }
    record = dict(
        goal_achieved=False,
        same_accuracy_speed_advantage_over_fem_demonstrated=False,
        reason="Best completed fine BEM still has about 25% maximum-slip discrepancy from archived FEM; quadrature equivalence alone does not remove it",
        records={},
    )
    for name, relative in paths.items():
        path = ROOT / "outputs" / relative
        record["records"][name] = dict(
            path=relative,
            sha256=fingerprint(path),
            data=json.loads(path.read_text()),
        )
    full = ROOT / "outputs/mixed6-near8-native-coarse4-rollout"
    record["full_coarse_experiment"] = dict(
        output=str(full),
        status="not_yet_validated",
        scope="Equivalence to the same coarse Q8 problem, not convergence to FEM or the fine Q12 trajectory",
    )
    for filename in ["summary.json", "resolvent_adjoint.json", "original_residual.json"]:
        path = full / filename
        if path.is_file():
            record["full_coarse_experiment"][filename] = dict(
                sha256=fingerprint(path), data=json.loads(path.read_text())
            )
    workflow = ROOT / "outputs/mixed-coarse4-workflow-20261024.json"
    if workflow.is_file():
        state = json.loads(workflow.read_text())
        record["full_coarse_experiment"].update(
            status=state["phase"], workflow=state, workflow_sha256=fingerprint(workflow)
        )
    write_json(ROOT / "docs/mixed_quadrature_results.json", record)
    print(ROOT / "docs/mixed_quadrature_results.json")


if __name__ == "__main__":
    main()

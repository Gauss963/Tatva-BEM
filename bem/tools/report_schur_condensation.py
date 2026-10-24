"""Curate the measured Schur diagnostics and real-BEM integration checks."""

import json
from pathlib import Path

from finiteblock_bem.io import fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = {
        "coarse": "schur-condensation-h80/summary.json",
        "fine": "schur-condensation-fine2/summary.json",
        "trace_fine": "trace-schur-condensation-fine2/summary.json",
        "trace_geometry": "trace-schur-basis-shape.json",
        "trace_kernel": "trace-schur-native64-kernel/summary.json",
        "trace_rollout": "trace-schur-native64-rollout/summary.json",
        "trace_adjoint": "trace-schur-native64-rollout/resolvent_adjoint.json",
        "trace_original_balance": "trace-schur-native64-rollout/original_residual.json",
    }
    records = {}
    for name, relative in paths.items():
        path = ROOT / "outputs" / relative
        records[name] = dict(
            path=str(path), sha256=fingerprint(path), data=json.loads(path.read_text())
        )
    write_json(
        ROOT / "docs/schur_condensation_results.json",
        dict(
            goal_achieved=False,
            full_original_event_reproduced=False,
            same_accuracy_speed_advantage_over_fem_demonstrated=False,
            scope="Sampled condensation and a short accelerated BEM sliding/conditional-AD diagnostic; not a complete 45ms reproduction or controlled speed benchmark",
            joint_audit_source_commit="740e4d5",
            records=records,
        ),
    )
    print(ROOT / "docs/schur_condensation_results.json")


if __name__ == "__main__":
    main()

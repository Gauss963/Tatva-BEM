"""Publish the completed 2 us / Q12 event without overwriting older studies."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def record(relative):
    path = ROOT / "outputs" / relative
    raw = path.read_bytes()
    data = json.loads(raw)
    actuator = data.pop("actuator_gradient", None)
    if actuator is not None:
        data["actuator_gradient_summary"] = dict(
            size=len(actuator),
            minimum=min(actuator),
            maximum=max(actuator),
            scope="Full vector retained in fingerprinted raw JSON; not independently verified",
        )
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest(), data=data)


def main():
    records = {
        name: record(path)
        for name, path in {
            "comparison": "galerkin-fine2-q12-comparison/event16_results.json",
            "kernel": "direct-resolvent-fine2-q12/summary.json",
            "execution_notes": "direct-resolvent-fine2-q12/execution_notes.json",
            "rollout": "galerkin-fine2-q12-resolvent/summary.json",
            "conditional_adjoint": "galerkin-fine2-q12-resolvent/resolvent_adjoint.json",
            "fronts": "front-fine2-q12-comparison/summary.json",
            "weakening_resolution": "weakening-fine2-q12-comparison/summary.json",
            "workflow": "resolvent-research-pipeline-20261024.json",
        }.items()
    }
    comparison = records["comparison"]["data"]
    fine = next(r for r in comparison["bem"] if r["name"] == "galerkin-fine2-q12-resolvent")
    if fine["status"] != "completed_requested_horizon" or fine["source_changed_during_run"]:
        raise ValueError("A completed run with unchanged numerical source is required")
    slip = fine["endpoint"]["max_cumulative_slip_mm"]
    result = dict(
        goal_achieved=False,
        full_original_event_reproduced=False,
        same_accuracy_speed_advantage_over_fem_demonstrated=False,
        maximum_slip_relative_difference=slip["difference"] / slip["fem_at_same_time"],
        scope="Completed full 45ms BEM; the 4us/Q6 to 2us/Q12 comparison changes both time grid and quadrature",
        independent_fine2_gradient_comparison=False,
        controlled_cold_timing=False,
        records=records,
    )
    target = ROOT / "docs/fine2_event_results.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    figures = ROOT / "docs/figures"
    for source, destination in {
        "galerkin-fine2-q12-comparison/figures/event16_comparison.png": "galerkin_fine2_event16.png",
        "galerkin-fine2-q12-comparison/figures/event16_profiles.png": "galerkin_fine2_profiles.png",
        "front-fine2-q12-comparison/fronts.png": "galerkin_fine2_fronts.png",
    }.items():
        shutil.copyfile(ROOT / "outputs" / source, figures / destination)
    print(target)


if __name__ == "__main__":
    main()

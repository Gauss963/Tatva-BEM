import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
TOOL = runpy.run_path(str(TOOLS / "compare_refined_primal.py"))


def test_complete_field_gate_detects_an_interior_history_error():
    fields = {k: np.zeros((17, 5)) for k in TOOL["TOLERANCES"]}
    fields.update(
        time=np.arange(17) * 0.01,
        contact_y=np.arange(5),
        contact_weights=np.ones(5),
        contact_interpolation=np.eye(5),
        retained_contact_y=np.arange(5),
    )
    candidate = {k: v.copy() for k, v in fields.items()}
    assert all(v["passed"] for v in TOOL["compare_fields"](fields, candidate).values())
    candidate["pressure"][7, 2] = 0.01
    result = TOOL["compare_fields"](fields, candidate)
    assert not result["pressure"]["passed"]
    assert result["pressure"]["maximum_absolute_difference"] == 0.01
    candidate["time"][3] += 0.001
    with pytest.raises(AssertionError):
        TOOL["compare_fields"](fields, candidate)


def constitutive_fields():
    time = np.arange(17) * (0.045 / 16)
    y = np.arange(5, dtype=float)
    values = dict(
        time=time,
        contact_y=y,
        contact_weights=np.ones(5),
        contact_interpolation=np.eye(5),
        retained_contact_y=y,
        initial_retained_displacement=np.zeros(11),
        initial_retained_bulk_force=np.zeros(11),
        platen=10.4 + 19 * time,
        actuator=10.8 + 20 * time,
    )
    for key, value in dict(
        pressure=10.0,
        shear=-3.0,
        plastic_slip=0.2,
        cumulative_slip=0.0,
        friction_coefficient=0.8,
    ).items():
        values[key] = np.full((len(time), len(y)), value)
    return values


def test_constitutive_screen_detects_interior_error_despite_identical_final_fields():
    reference = constitutive_fields()
    candidate = deepcopy(reference)
    control = TOOL["compare_constitutive_refinement"](reference, candidate)
    assert control["all_critical_fields_passed"]
    candidate["pressure"][7] += 2
    result = TOOL["compare_constitutive_refinement"](reference, candidate)
    assert result["fields"]["pressure"]["final_shared_time_relative_p1_l2"] == 0
    assert result["fields"]["pressure"]["relative_space_time_p1_l2"] > 0.01
    assert not result["checks"]["pressure"]
    assert not result["all_critical_fields_passed"]
    assert result["scalar_errors"]["actuator_increment_mm"]["relative_space_time_p1_l2"] == 0


def test_constitutive_zero_memory_requires_zero_error_and_initial_trace_is_fixed():
    reference = constitutive_fields()
    candidate = deepcopy(reference)
    candidate["cumulative_slip"][5, 2] = 1e-12
    result = TOOL["compare_constitutive_refinement"](reference, candidate)
    assert result["fields"]["cumulative_slip"]["relative_space_time_p1_l2"] is None
    assert not result["checks"]["cumulative_slip"]
    candidate["initial_retained_bulk_force"][2] = 2e-7
    with pytest.raises(AssertionError):
        TOOL["compare_constitutive_refinement"](reference, candidate)


def test_constitutive_screen_rejects_rounded_fields_and_changed_grid():
    reference = constitutive_fields()
    candidate = deepcopy(reference)
    candidate["shear"] = candidate["shear"].astype(np.float32)
    with pytest.raises(ValueError, match="true float64"):
        TOOL["compare_constitutive_refinement"](reference, candidate)
    candidate = deepcopy(reference)
    candidate["contact_y"][2] += 0.001
    with pytest.raises(AssertionError):
        TOOL["compare_constitutive_refinement"](reference, candidate)


def test_constitutive_mode_only_varies_local_substeps_and_uses_a_finer_reference():
    fine = dict(
        status="completed_requested_horizon",
        all_history_retained=True,
        correction_start=1,
        max_boundary_contact_residual_mpa=1e-9,
        history_representation="elastic_resolvent_refined_contact",
        duration_s=0.045,
        dt_s=2e-6,
        schedule="closed-loop",
        panels=10,
        case={},
        subdivisions=1,
        seed_interpolation="original-p1",
        static_initialization="fem-body-active-set",
        bulk_contact_nodes=5,
        contact_nodes=5,
        completed_steps=22500,
        constitutive_substeps=160,
        kernel_summary_sha256="same kernel",
        restart_seed_override=dict(sha256="same seed"),
        environment=dict(
            source_sha256="same source", case_sha256="same case", restart_data_sha256="same data"
        ),
    )
    coarse = deepcopy(fine)
    coarse["constitutive_substeps"] = 64
    TOOL["validate_records"](fine, coarse, mode="constitutive-refinement")
    with pytest.raises(ValueError, match="constitutive_substeps"):
        TOOL["validate_records"](fine, coarse)
    with pytest.raises(ValueError, match="more local substeps"):
        TOOL["validate_records"](coarse, fine, mode="constitutive-refinement")
    for key, value in [("dt_s", 1e-6), ("kernel_summary_sha256", "other kernel")]:
        changed = {**coarse, key: value}
        with pytest.raises(ValueError):
            TOOL["validate_records"](fine, changed, mode="constitutive-refinement")


def test_constitutive_invocations_reject_another_solver_option_or_source_change():
    arguments = dict(
        kernel="same kernel",
        restart_seed="same seed",
        subdivisions=1,
        constitutive_substeps=160,
        seed_interpolation="original-p1",
        static_initialization="fem-body-active-set",
        kinetic_root_seed=True,
        output="fine run",
    )
    manifests = [dict(arguments=arguments, frozen_input_sha256={"solver": "same hash"})]
    coarse = deepcopy(manifests[0])
    coarse["arguments"].update(constitutive_substeps=64, output="coarse run")
    manifests.append(coarse)
    records = [dict(constitutive_substeps=n) for n in [160, 64]]
    TOOL["validate_constitutive_arguments"](manifests, records)
    changed = deepcopy(manifests)
    changed[1]["arguments"]["kinetic_root_seed"] = False
    with pytest.raises(ValueError, match="kinetic_root_seed"):
        TOOL["validate_constitutive_arguments"](changed, records)
    changed = deepcopy(manifests)
    changed[1]["frozen_input_sha256"]["solver"] = "other hash"
    with pytest.raises(ValueError, match="different numerical source"):
        TOOL["validate_constitutive_arguments"](changed, records)
    with pytest.raises(ValueError, match="completed result"):
        TOOL["validate_constitutive_arguments"](manifests, records[::-1])

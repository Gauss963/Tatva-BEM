import runpy
from pathlib import Path

import numpy as np
import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/fem_energy_budget.py"))


def history(a, face):
    a, face = np.asarray(a, float), np.asarray(face, float)
    zero = np.zeros_like(a)
    return dict(
        time=np.arange(len(a), dtype=float),
        applied_shear_displacement=a,
        loading_face_displacement=face,
        normal_external_force=zero.copy(),
        normal_loading_coordinate=zero.copy(),
        elastic_energy=zero.copy(),
        interface_energy=zero.copy(),
        kinetic_energy=zero.copy(),
    )


def test_fixed_face_spring_charging_is_storage_instead_of_dissipation():
    h = history([0, 0.2, 0.7, 1], [0, 0, 0, 0])
    for stride in [1, 2, 8]:
        record = TOOL["mechanical_budget"](h, 4, stride=stride)
        assert record["actuator_work"] == pytest.approx(2)
        assert record["spring_energy_change"] == pytest.approx(2)
        assert abs(record["friction_and_numerical_remainder"]) < 1e-14


def test_constant_normal_force_work_becomes_particle_kinetic_energy():
    h = history([0, 0, 0], [0, 0, 0])
    h["normal_external_force"][:] = 3
    h["normal_loading_coordinate"] = np.array([0, 0.2, 1.0])
    h["kinetic_energy"] = 3 * h["normal_loading_coordinate"]
    record = TOOL["mechanical_budget"](h, 4)
    assert record["normal_external_work"] == pytest.approx(3)
    assert abs(record["friction_and_numerical_remainder"]) < 1e-14


def test_massless_coulomb_slider_work_is_dissipated_at_constant_spring_extension():
    a = np.array([1.0, 1.2, 1.7])
    record = TOOL["mechanical_budget"](history(a, a - 0.5), 4)
    assert record["spring_energy_change"] == pytest.approx(0)
    assert record["friction_and_numerical_remainder"] == pytest.approx(2 * (a[-1] - a[0]))


def test_nonfinite_or_unordered_saved_samples_are_rejected():
    h = history([0, 1], [0, 0])
    h["time"][:] = 0
    with pytest.raises(ValueError, match="increase strictly"):
        TOOL["mechanical_budget"](h, 4)
    h["time"][1] = 1
    h["elastic_energy"][0] = np.nan
    with pytest.raises(ValueError, match="aligned finite"):
        TOOL["mechanical_budget"](h, 4)

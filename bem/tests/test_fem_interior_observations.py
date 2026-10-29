import runpy
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/fem_interior_observations.py")
)


def source_state():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    elements = np.array([[0, 1, 2, 3]])
    moving = SimpleNamespace(mesh=SimpleNamespace(coords=coords, elements=elements))
    stationary = SimpleNamespace(
        mesh=SimpleNamespace(coords=coords + np.array([2.0, 0.0]), elements=elements)
    )
    model = dict(
        dimension=2,
        moving=moving,
        stationary=stationary,
        moving_offset=8,
        moving_material=SimpleNamespace(lmbda=2.0, mu=3.0),
        stationary_material=SimpleNamespace(lmbda=4.0, mu=5.0),
        dt=0.125000000000003,
        pressure_steps=3,
    )
    gradient = np.array([[0.2, 0.3], [-0.1, -0.4]])
    velocity_gradient = np.array([[0.5, -0.2], [0.3, 0.7]])
    all_coords = np.vstack([coords, coords + np.array([2.0, 0.0])])
    u = (all_coords @ gradient.T + [1.2, -0.8]).ravel()
    v = (all_coords @ velocity_gradient.T + [-0.3, 0.6]).ravel()
    u.flags.writeable = v.flags.writeable = False
    plan = dict(moving=[[0.7, 0.3]], stationary=[[2.8, 0.6]])
    return model, (u, v), plan, gradient, velocity_gradient


def make_high(stream, count):
    group = stream.create_group("interface_high_rate")
    group.create_dataset("history", shape=(count, 17), dtype="f4")
    group.create_dataset("phase_id", shape=(count,), dtype="i4")
    group.create_dataset("step_id", shape=(count,), dtype="i8")


def observe(observer, stream, index, carry, phase, step, model):
    stream["interface_high_rate/phase_id"][index] = phase
    stream["interface_high_rate/step_id"][index] = step
    observer(stream, index, carry, phase, step, model, None)


def test_q1_affine_patch_and_staggered_velocity_preserve_all_source_arrays(tmp_path):
    model, carry, plan, gradient, velocity_gradient = source_state()
    before = [x.copy() for x in carry]
    with h5py.File(tmp_path / "probe.h5", "w") as stream:
        make_high(stream, 2)
        observer = TOOL["InteriorObserver"](plan)
        observe(observer, stream, 0, carry, 0, 0, model)
        observe(observer, stream, 1, carry, 2, 4, model)
        stream.attrs["saved_interface_frames"] = 2
        record = TOOL["validate_probe_group"](stream)
        assert record["complete"]
        group = stream["interior_observations"]
        assert group["time_s"][1] == 7 * model["dt"]
        assert group["velocity_time_s"][1] == 7.5 * model["dt"]
        assert group["velocity_time_s"][0] == 0.5 * model["dt"]
        for body in ["moving", "stationary"]:
            points = np.array(plan[body])
            saved = group[body]
            material = model[body + "_material"]
            np.testing.assert_allclose(
                saved["displacement"][1], points @ gradient.T + [1.2, -0.8], atol=1e-14
            )
            np.testing.assert_allclose(
                saved["velocity_half_step"][1],
                points @ velocity_gradient.T + [-0.3, 0.6],
                atol=1e-14,
            )
            np.testing.assert_allclose(saved["gradient"][1], gradient[None], atol=1e-14)
            sigma = material.lmbda * np.trace(gradient) * np.eye(2) + material.mu * (
                gradient + gradient.T
            )
            np.testing.assert_allclose(
                saved["stress"][1],
                [[sigma[0, 0], sigma[1, 1], sigma[0, 1], material.lmbda * np.trace(gradient)]],
                atol=1e-14,
            )
    for a, b in zip(before, carry, strict=True):
        np.testing.assert_array_equal(a, b)


def test_distorted_q1_inverse_map_and_point_gradient_match_physical_finite_differences():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.5, 1.3], [-0.2, 0.9]])
    natural = np.array([0.23, -0.31])
    point = TOOL["q1_basis"](natural)[0] @ coords
    values = np.random.default_rng(8).normal(size=(4, 2))

    def sampled(p):
        return TOOL["Q1ProbeSampler"](coords, np.array([[0, 1, 2, 3]]), [p]).evaluate(
            values, np.zeros_like(values), 2.0, 3.0
        )

    actual = sampled(point)
    recovered = TOOL["Q1ProbeSampler"](coords, np.array([[0, 1, 2, 3]]), [point])
    np.testing.assert_allclose(recovered.reference_coordinates[0], natural, atol=1e-13)
    for k in range(2):
        shift = np.eye(2)[k] * 1e-5
        derivative = (
            sampled(point + shift)["displacement"] - sampled(point - shift)["displacement"]
        ) / (2e-5)
        np.testing.assert_allclose(actual["gradient"][:, :, k], derivative, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("point", [[0.0, 0.5], [2.1, 0.5], [2.0, 0.5]])
def test_exterior_and_element_edges_are_rejected_instead_of_averaging_stress(point):
    with pytest.raises(ValueError, match="strictly inside"):
        TOOL["Q1ProbeSampler"](
            np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]),
            np.array([[0, 1, 2, 3]]),
            [point],
        )


@pytest.mark.parametrize("which", ["displacement", "velocity"])
def test_float32_source_is_rejected_even_when_the_other_field_is_float64(tmp_path, which):
    model, carry, plan, _, _ = source_state()
    carry = list(carry)
    carry[0 if which == "displacement" else 1] = carry[0 if which == "displacement" else 1].astype(
        np.float32
    )
    with h5py.File(tmp_path / "rounded.h5", "w") as stream:
        make_high(stream, 1)
        with pytest.raises(ValueError, match="true float64"):
            TOOL["InteriorObserver"](plan)(stream, 0, carry, 0, 0, model, None)
        assert "interior_observations" not in stream


def test_resume_rewinds_only_uncommitted_probe_rows_and_requires_existing_prefix(tmp_path):
    model, carry, plan, _, _ = source_state()
    with h5py.File(tmp_path / "resume.h5", "w") as stream:
        make_high(stream, 3)
        observer = TOOL["InteriorObserver"](plan)
        for k in range(3):
            observe(observer, stream, k, carry, 2, k + 1, model)
        before = stream["interior_observations/moving/displacement"][:]
        stream.attrs.update(resumed_from_checkpoint=1, saved_interface_frames=1)
        resumed = TOOL["InteriorObserver"](plan)
        observe(resumed, stream, 1, carry, 2, 2, model)
        observe(resumed, stream, 2, carry, 2, 3, model)
        np.testing.assert_array_equal(
            before, stream["interior_observations/moving/displacement"][:]
        )
        with pytest.raises(ValueError, match="duplicated"):
            observe(resumed, stream, 2, carry, 2, 3, model)
        del stream["interior_observations"]
        with pytest.raises(ValueError, match="Cannot invent"):
            observe(TOOL["InteriorObserver"](plan), stream, 1, carry, 2, 2, model)


@pytest.mark.parametrize("corruption", ["clock", "float32", "point", "incomplete"])
def test_validator_reads_actual_clock_storage_and_physical_plan(tmp_path, corruption):
    model, carry, plan, _, _ = source_state()
    with h5py.File(tmp_path / "corrupt.h5", "w") as stream:
        make_high(stream, 2)
        observer = TOOL["InteriorObserver"](plan)
        observe(observer, stream, 0, carry, 0, 0, model)
        observe(observer, stream, 1, carry, 2, 4, model)
        stream.attrs["saved_interface_frames"] = 2
        group = stream["interior_observations"]
        if corruption == "clock":
            group["velocity_time_s"][1] = group["time_s"][1]
        elif corruption == "float32":
            values = group["moving/stress"][:]
            del group["moving/stress"]
            group["moving/stress"] = values.astype(np.float32)
        elif corruption == "point":
            group["moving/points"][0, 0] += 0.01
        else:
            group.attrs["written_frames"] = 1
        with pytest.raises(ValueError):
            TOOL["validate_probe_group"](stream)


def test_original_source_guard_rejects_other_dynamics_function():
    with pytest.raises(ValueError, match="pinned"):
        TOOL["make_observed_runner"](source_state, {}, {})

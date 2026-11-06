"""Audit projected startup and reciprocal storage against dense stage sums."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import solve

from finiteblock_bem.cq import invert_spectrum, plan_cq

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def test_projected_startup_and_packed_branches_preserve_full_causal_response(tmp_path):
    projected = runpy.run_path(str(TOOLS / "projected_radau_startup.py"))
    reciprocal = runpy.run_path(str(TOOLS / "reciprocal_radau_storage.py"))
    radau, raw = projected["RADAU"], projected["RAW"]
    rng = np.random.default_rng(11)
    factor = rng.normal(size=(6, 6))
    stiffness = factor.T @ factor + np.eye(6)
    mass = np.diag([0.05, 0.06, 0.07, 0.1, 0.13, 0.17])
    jump = rng.normal(size=(3, 6))
    tangent = np.array([10.0, 3.0, 2.0])
    forces = np.column_stack((np.eye(3)[:, -1], [1e-8, -2e-8, 2.5e-8]))
    line = np.array([0.31, 0.84, 2.0])
    scale = np.sqrt(line[:, None] * line[None, :])
    plan = plan_cq(3, 0.009, 31)
    nodes = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    modes = raw["lower_half_radau_modes"](nodes, plan.dt)

    def original(s):
        compliance = jump @ solve(stiffness + s**2 * mass, jump.T)
        impedance = solve(compliance, np.eye(3))
        return solve(impedance + np.diag(tangent), np.eye(3))

    actions = np.array(
        [
            [
                projected["projected_resolvent"](stiffness + s**2 * mass, jump, tangent, forces)
                for s in pair
            ]
            for pair in modes
        ]
    )
    dense = np.array(
        [radau["matrix_function"](original, radau["radau_symbol"](z) / plan.dt) for z in nodes]
    )
    weights = invert_spectrum(dense, replace(plan, dofs=6))
    ramp = np.zeros((plan.steps + 1, 2))
    ramp[1:] = (np.arange(plan.steps)[:, None] + radau["C"]) * plan.dt * 1.5
    load = (ramp[:, :, None] * forces[:, 0]).reshape(-1, 6)
    expected_ramp = np.stack(
        [np.einsum("kij,kj->i", weights[: k + 1], load[k::-1]) for k in range(len(load))]
    ).reshape(-1, 2, 3)
    expected_affine = np.zeros_like(expected_ramp)
    expected_affine[2:] = np.cumsum(
        weights[1 : plan.steps] @ np.tile(forces[:, 1], 2), axis=0
    ).reshape(-1, 2, 3)
    startup, past = projected["projected_histories"](plan, actions, ramp, rows=2)
    np.testing.assert_allclose(startup, expected_ramp, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(past, expected_affine, rtol=1e-10, atol=1e-18)

    paths = [tmp_path / f"branch{j}.bin" for j in range(2)]
    for j, path in enumerate(paths):
        spectrum = np.array([original(s) * scale for s in modes[:, j]])
        with reciprocal["ReciprocalBranch"](path, plan, block=2) as writer:
            writer.append(spectrum[:3])
            writer.append(spectrum[3:])
        assert path.stat().st_size < spectrum.nbytes
    out = tmp_path / "weights"
    wrap = reciprocal["invert_reciprocal_branches"](paths, plan, line, out, block=2)
    families = [np.load(out / f"weights{i}.npy", allow_pickle=False) for i in range(2)]
    assert all(array.flags.f_contiguous for array in families)
    np.testing.assert_allclose(
        radau["expand_compact_weights"](*families, wrap), weights, rtol=1e-10, atol=1e-12
    )


def test_reciprocal_writer_rejects_asymmetric_or_incomplete_source(tmp_path):
    writer_type = runpy.run_path(str(TOOLS / "reciprocal_radau_storage.py"))["ReciprocalBranch"]
    plan = plan_cq(3, 0.009, 7)
    asymmetric = np.eye(3, dtype=complex)[None]
    asymmetric[0, 1, 0] = 0.1j
    first, second = tmp_path / "first.bin", tmp_path / "second.bin"
    with (
        pytest.raises(ValueError, match="bilinear reciprocity"),
        writer_type(first, plan) as writer,
    ):
        writer.append(asymmetric)
    with pytest.raises(ValueError, match="Incomplete reciprocal branch"), writer_type(second, plan):
        pass
    assert not first.exists()
    assert not second.exists()

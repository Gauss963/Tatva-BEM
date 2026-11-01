"""Saved-kernel normalization, local path resolution and corruption rejection."""

import json
import runpy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import plan_cq
from finiteblock_bem.io import fingerprint

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CHECK = runpy.run_path(str(TOOLS / "check_mixed_radau_nonlinear.py"))


@pytest.fixture
def saved(tmp_path):
    plan = plan_cq(2, 0.01, 4, max_memory_mb=512)
    line = np.array([2.0, 7.0])
    zeta = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    matrix = np.array([[2.0, 0.3], [0.3, 5.0]])
    physical = [
        matrix[None] / (2 - zeta[:, None, None]),
        0.2 * matrix[None] / (3 - zeta[:, None, None]),
    ]
    weights = CHECK["RADAU"]["invert_compact_spectrum"](*physical, plan)
    spectra, records = [], [[], []]
    for i in range(2):
        spectra.append(physical[i] * np.sqrt(line[:, None] * line[None, :]))
        for group, name, values in [
            (0, f"compact-family{i}.npy", spectra[-1]),
            (1, f"weights{i}.npy", weights[i]),
        ]:
            np.save(tmp_path / name, values)
            records[group].append(
                dict(
                    path="/nonexistent/original-location/" + name,
                    sha256=fingerprint(tmp_path / name),
                )
            )
    np.save(tmp_path / "shift_wrap.npy", weights[2])
    np.save(tmp_path / "initial_impedance.npy", np.eye(4))
    np.savez(tmp_path / "elastic.npz", unused=np.zeros(1))
    environment = {
        k: "pinned"
        for k in ["source_sha256", "case_sha256", "restart_data_sha256", "numpy", "scipy"]
    }
    metadata = dict(
        status="completed_compact_radau_resolvent_kernel_only",
        plan=asdict(plan),
        full_steps=4,
        correction_steps=4,
        family_spectra_coordinates="diag(sqrt(w)) * L * diag(sqrt(w))",
        environment=environment,
        family_spectra=records[0],
        family_weights=records[1],
        **{
            key: fingerprint(tmp_path / name)
            for name, key in [
                ("shift_wrap.npy", "shift_wrap_sha256"),
                ("initial_impedance.npy", "initial_impedance_sha256"),
                ("elastic.npz", "elastic_sha256"),
            ]
        },
    )
    (tmp_path / "summary.json").write_text(json.dumps(metadata))
    return tmp_path, plan, line, environment, physical, weights, metadata


def test_reads_only_owned_copies_and_keeps_actual_saved_weight_bits(saved):
    folder, plan, line, environment, physical, weights, _ = saved
    _, actual, stored, _ = CHECK["verified_arrays"](folder, plan, line, environment)
    for a, b in zip(actual, physical, strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-16)
    for a, b in zip(stored, weights, strict=True):
        np.testing.assert_array_equal(a, b)


def test_workspace_estimate_does_not_change_the_exact_mathematical_contour(saved):
    folder, plan, line, environment, _, weights, metadata = saved
    # The producer uses disk_spectrum=True; the small primal uses in-memory
    # histories. Workspace estimates differ although all mathematical data agree.
    metadata["plan"]["estimated_peak_bytes"] += 1024
    (folder / "summary.json").write_text(json.dumps(metadata))
    _, _, actual, _ = CHECK["verified_arrays"](folder, plan, line, environment)
    for a, b in zip(actual, weights, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("damage", ["bytes", "nonfinite", "plan", "runtime"])
def test_rejects_changed_or_nonfinite_inputs_without_writing(saved, damage):
    folder, plan, line, environment, _, _, metadata = saved
    path = folder / "weights0.npy"
    if damage == "bytes":
        value = bytearray(path.read_bytes())
        value[-1] ^= 1
        path.write_bytes(value)
    elif damage == "nonfinite":
        value = np.load(path)
        value[0, 0, 0] = np.nan
        np.save(path, value)
        metadata["family_weights"][0]["sha256"] = fingerprint(path)
    elif damage == "plan":
        metadata["plan"]["dt"] *= 2
    else:
        metadata["environment"]["numpy"] = "changed"
        environment = {**environment, "numpy": "pinned"}
    (folder / "summary.json").write_text(json.dumps(metadata))
    before = {p.name: fingerprint(p) for p in folder.iterdir()}
    with pytest.raises(ValueError):
        CHECK["verified_arrays"](folder, plan, line, environment)
    assert {p.name: fingerprint(p) for p in folder.iterdir()} == before

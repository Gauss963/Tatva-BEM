import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import ResourceBudgetError, convolution_weights, plan_cq
from finiteblock_bem.cq_storage import DiskSpectrum
from finiteblock_bem.nodal import load_nodal_model
from finiteblock_bem.nodal_dynamics import build_nodal_history
from finiteblock_bem.sampling import parsimonious_weights


@pytest.mark.parametrize("steps", [9, 80, 257])
@pytest.mark.parametrize("adaptive", [False, True])
def test_disk_and_ram_keep_identical_weights_and_sampling(tmp_path, steps, adaptive):
    plan = plan_cq(3, 0.01, steps)
    a = np.array([[1, 2, 3], [-2, 5, 6], [8, 0, 1]])
    b = np.array([[0, -1, 2], [3, 4, -5], [0, 6, 7]])

    def transfer(s):
        return s * a + b / (s + 2)

    spectrum = DiskSpectrum(tmp_path / "spectrum.npy", plan)
    sampler = parsimonious_weights if adaptive else convolution_weights
    reference = sampler(transfer, plan)
    actual = sampler(transfer, plan, spectrum=spectrum, workers=2)
    if adaptive:
        reference, reference_info = reference
        actual, actual_info = actual
        assert actual_info == reference_info
    np.testing.assert_array_equal(actual, reference)
    record = spectrum.record()
    with spectrum.path.open("rb") as stream:
        assert record["sha256"] == hashlib.file_digest(stream, "sha256").hexdigest()
    assert record["all_frequency_matrices_written"]


def test_storage_roundtrip_bounds_ownership_and_incomplete_guard(tmp_path):
    plan = plan_cq(3, 0.01, 137)
    spectrum = DiskSpectrum(tmp_path / "spectrum.npy", plan)
    data = np.arange(np.prod(spectrum.shape)).reshape(spectrum.shape) * (1 + 2j)
    spectrum[:67] = data[:67]
    with pytest.raises(ValueError, match="entire"):
        spectrum[:]
    spectrum[67:] = data[67:]
    np.testing.assert_array_equal(np.load(spectrum.path), data)
    np.testing.assert_array_equal(spectrum[:, 1:, :2], data[:, 1:, :2])
    copied = spectrum[-1]
    copied[:] = 0
    np.testing.assert_array_equal(spectrum[-1], data[-1])
    assert spectrum[10:2].shape == (0, 3, 3)
    with pytest.raises(IndexError):
        spectrum[len(data)]
    with pytest.raises(ValueError, match="shape"):
        spectrum[0] = np.zeros(2)
    with pytest.raises(FileExistsError):
        DiskSpectrum(spectrum.path, plan)


def test_disk_plan_removes_spectrum_residency_but_guards_weights_and_history():
    with pytest.raises(ResourceBudgetError):
        plan_cq(275, 4e-6, 11250, max_memory_mb=20480)
    disk = plan_cq(275, 4e-6, 11250, max_memory_mb=20480, disk_spectrum=True)
    assert disk.weight_bytes < disk.estimated_peak_bytes < 20480 * 1024**2
    with pytest.raises(ResourceBudgetError):
        plan_cq(275, 4e-6, 1000000, max_memory_mb=20480, disk_spectrum=True)


def test_nodal_disk_cache_and_storage_signature(tmp_path, monkeypatch):
    from finiteblock_bem import nodal_dynamics

    case = (
        Path(__file__).resolve().parents[1]
        / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    )
    model = load_nodal_model(case, max_panel_length=160)
    info = {}
    first = build_nodal_history(
        model,
        8e-6,
        4,
        quadrature=6,
        cache_directory=tmp_path / "cache",
        spectrum_directory=tmp_path / "spectra",
        diagnostics=info,
    )
    assert not info["cache_hit"]
    assert info["spectrum_storage"]["all_frequency_matrices_written"]

    def forbid(*args, **kwargs):
        raise RuntimeError("Unexpected transfer rebuild")

    monkeypatch.setattr(nodal_dynamics, "NodalInterfaceTransfer", forbid)
    second = build_nodal_history(
        model, 8e-6, 4, quadrature=6, cache_directory=tmp_path / "cache", diagnostics=info
    )
    np.testing.assert_array_equal(first.weights, second.weights)
    assert info["cache_hit"]
    metadata = next((tmp_path / "cache").glob("*.json"))
    bad = json.loads(metadata.read_text())
    bad["spectrum_storage"]["implementation_sha256"] = "changed"
    metadata.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="rebuild"):
        build_nodal_history(model, 8e-6, 4, quadrature=6, cache_directory=tmp_path / "cache")


def test_ram_cache_can_satisfy_disk_request_without_rebuilding(tmp_path, monkeypatch):
    from finiteblock_bem import nodal_dynamics

    case = (
        Path(__file__).resolve().parents[1]
        / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    )
    model = load_nodal_model(case, max_panel_length=160)
    first = build_nodal_history(model, 8e-6, 4, quadrature=6, cache_directory=tmp_path / "cache")

    def forbid(*args, **kwargs):
        raise RuntimeError("A valid RAM cache should be reused")

    monkeypatch.setattr(nodal_dynamics, "NodalInterfaceTransfer", forbid)
    info = {}
    second = build_nodal_history(
        model,
        8e-6,
        4,
        quadrature=6,
        cache_directory=tmp_path / "cache",
        spectrum_directory=tmp_path / "unused",
        diagnostics=info,
    )
    np.testing.assert_array_equal(first.weights, second.weights)
    assert info["cache_hit"]
    assert not (tmp_path / "unused").exists()


def test_disk_guard_fails_before_allocating_spectrum(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from finiteblock_bem import nodal_dynamics

    case = (
        Path(__file__).resolve().parents[1]
        / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    )
    model = load_nodal_model(case, max_panel_length=160)
    monkeypatch.setattr(nodal_dynamics.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ResourceBudgetError, match="free disk"):
        build_nodal_history(model, 8e-6, 4, spectrum_directory=tmp_path)
    assert not list(tmp_path.glob("*.npy"))

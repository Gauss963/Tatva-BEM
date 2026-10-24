import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from finiteblock_bem.cq import plan_cq
from finiteblock_bem.sampling import SamplingOptions, parsimonious_weights

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/native_spectrum.py"))


@pytest.mark.parametrize("workers", [1, 4])
def test_native_sampling_matches_original_grid_weights_and_call_count(tmp_path, workers):
    plan = plan_cq(4, 0.01, 255, max_memory_mb=32)
    a = np.array([[4.0, 0.3, 0, 0.4], [-0.2, 6, 0.1, 0], [0, 0, 3, 0.2], [0.1, 0.2, 0, 5.0]])

    def transfer(s):
        return np.linalg.inv(s * np.eye(4) + a)

    expected_grid = np.empty((plan.fft_size // 2 + 1, 4, 4), complex)
    options = SamplingOptions(relative_tolerance=1e-9, absolute_tolerance=1e-14)
    expected, record = parsimonious_weights(
        transfer, plan, options=options, workers=workers, spectrum=expected_grid
    )
    original_globals = dict(parsimonious_weights.__globals__)
    actual, spectrum, metadata = TOOL["native_parsimonious_weights"](
        transfer, plan, tmp_path / "data", options=options, workers=workers
    )
    assert metadata == record
    assert any(piece["kind"] == "barycentric" for piece in spectrum.pieces)
    np.testing.assert_allclose(spectrum[:], expected_grid, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        spectrum[:, 1:3, :], expected_grid[:, 1:3, :], rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
    for name, value in original_globals.items():
        assert parsimonious_weights.__globals__[name] is value


def test_failed_native_source_validation_cannot_open_as_completed(tmp_path):
    plan = plan_cq(2, 0.01, 32, max_memory_mb=32)

    def changed():
        raise ValueError("source changed")

    with pytest.raises(ValueError, match="source changed"):
        TOOL["native_parsimonious_weights"](
            lambda s: np.eye(2) / (s + 1), plan, tmp_path / "data", source_check=changed
        )
    reader = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools/polynomial_spectrum.py")
    )["PolynomialSpectrum"]
    with pytest.raises(ValueError, match="completed"):
        reader(tmp_path / "data")


@pytest.mark.parametrize("rows", [1, 4])
def test_disk_time_weights_keep_every_lag_and_original_row_ifft(tmp_path, rows):
    plan = plan_cq(3, 0.01, 127, max_memory_mb=32)
    a = np.array([[4.0, 0.3, 0.2], [-0.2, 6, 0.1], [0.4, -0.1, 3]])

    def transfer(s):
        return np.linalg.inv(s * np.eye(3) + a)

    expected, _ = parsimonious_weights(transfer, plan)
    actual, spectrum, _ = TOOL["native_parsimonious_weights"](
        transfer,
        plan,
        tmp_path / "data",
        weight_path=tmp_path / "weights.npy",
        fft_rows=rows,
        max_workspace_mb=32,
        reserved_disk_bytes=65536,
    )
    assert isinstance(actual, np.memmap)
    assert not actual.flags.writeable
    assert spectrum.metadata["time_weights_storage"] == "disk_mmap"
    assert spectrum.metadata["reserved_other_output_bytes"] == 65536
    np.testing.assert_array_equal(actual, expected)


def test_disk_ifft_never_overwrites_and_checks_workspace_before_writing(tmp_path):
    plan = plan_cq(2, 0.01, 32, max_memory_mb=32)
    source = np.ones((plan.fft_size // 2 + 1, 2, 2), complex)
    target = tmp_path / "weights.npy"
    target.write_bytes(b"keep existing kernel")
    with pytest.raises(FileExistsError):
        TOOL["invert_spectrum_to_file"](source, plan, target)
    assert target.read_bytes() == b"keep existing kernel"
    new = tmp_path / "new.npy"
    with pytest.raises(ValueError, match="workspace"):
        TOOL["invert_spectrum_to_file"](source, plan, new, max_workspace_mb=1e-6)
    assert not new.exists()


def test_native_budget_and_incomplete_or_overlapping_grid_are_rejected(tmp_path):
    plan = plan_cq(2, 0.01, 32, max_memory_mb=32)
    with pytest.raises(ValueError, match="storage budget"):
        TOOL["native_parsimonious_weights"](
            lambda s: np.eye(2) / (s + 1), plan, tmp_path / "small", max_storage_mb=1e-6
        )
    assert not (tmp_path / "small" / "summary.json").exists()
    sink = TOOL["NativeSpectrumSink"](
        tmp_path / "partial", plan, {}, None, max_storage_mb=32, rtol=1e-13, atol=1e-18
    )
    sink[:2] = np.zeros((2, 2, 2), complex)
    with pytest.raises(ValueError, match="exactly once"):
        sink[1:3] = np.zeros((2, 2, 2), complex)
    with pytest.raises(ValueError, match="incomplete"):
        sink.seal(None)
    assert not (tmp_path / "partial" / "summary.json").exists()


def test_frequency_storage_protects_space_for_pending_time_output(tmp_path, monkeypatch):
    plan = plan_cq(2, 0.01, 32, max_memory_mb=32)
    monkeypatch.setattr(TOOL["shutil"], "disk_usage", lambda _: SimpleNamespace(free=20000))
    with pytest.raises(ValueError, match="storage budget"):
        TOOL["native_parsimonious_weights"](
            lambda s: np.eye(2) / (s + 1),
            plan,
            tmp_path / "reserved",
            reserved_disk_bytes=20000,
        )
    assert not (tmp_path / "reserved" / "summary.json").exists()
    with pytest.raises(ValueError, match="Reserved disk"):
        TOOL["native_parsimonious_weights"](
            lambda s: np.eye(2), plan, tmp_path / "negative", reserved_disk_bytes=-1
        )

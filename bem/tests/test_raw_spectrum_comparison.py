import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import plan_cq

TOOLS = Path(__file__).resolve().parents[1] / "tools"
TOOL = runpy.run_path(str(TOOLS / "compare_raw_kernel_spectrum.py"))
READ = runpy.run_path(str(TOOLS / "replay_resolvent.py"))["SavedDiskSpectrum"]


def test_complete_comparison_hashes_actual_npy_bytes_and_finds_hidden_weak_direction(tmp_path):
    plan = plan_cq(3, 0.01, 10, max_memory_mb=16)
    shape = (plan.fft_size // 2 + 1, 3, 3)
    old = np.broadcast_to(np.diag([1.0, 2.0, 1e-8]), shape).astype(complex).copy()
    new = old.copy()
    new[7, 2, 2] *= 1.0001
    inverse = np.linalg.inv(old)
    paths = [tmp_path / n for n in ["old.npy", "new.npy", "inverse.npy"]]
    for p, a in zip(paths, [old, new, inverse], strict=True):
        np.save(p, a)
    result = TOOL["compare_complete_spectra"](*[READ(p, plan) for p in paths], workers=2, block=7)
    assert result["maximum_relative_work_frobenius_difference"] < 1e-7
    assert not result["passed"]
    assert result["worst_inverse_composition"]["index"] == 7
    assert result["maximum_original_inverse_composition_inf"] > 1e-6
    assert result["streamed_npy_sha256"] == [
        hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    ]
    np.save(paths[1], old)
    result = TOOL["compare_complete_spectra"](*[READ(p, plan) for p in paths], workers=1, block=5)
    assert result["passed"]
    assert result["frequency_matrices"] == shape[0]


def test_emitted_prefix_detects_weak_direction_without_claiming_unwritten_tail(tmp_path):
    old = np.broadcast_to(np.diag([1.0, 2.0, 1e-8]), (12, 3, 3)).astype(complex).copy()
    new = old.copy()
    new[3, 2, 2] *= 1.0001
    old[-1] = new[-1] = np.nan
    paths = [tmp_path / name for name in ["old.npy", "new.npy"]]
    for path, value in zip(paths, [old, new], strict=True):
        np.save(path, value)
    result = TOOL["compare_emitted_prefix"](
        *paths, count=7, reference_written=8, candidate_written=7, block=3, candidate_live=True
    )
    assert result["maximum_normalized_prefix_difference"] < 1
    assert not result["passed"]
    assert result["worst_reference_solve_index"] == 3
    assert result["maximum_reference_solve_composition_inf"] > 1e-6
    assert not result["whole_kernel_verified"]
    assert not result["original_operator_audit_verified"]
    for path, digest in zip(paths, result["streamed_npy_prefix_sha256"], strict=True):
        with path.open("rb") as f:
            np.lib.format.read_magic(f)
            np.lib.format.read_array_header_1_0(f)
            offset = f.tell()
        assert digest == hashlib.sha256(path.read_bytes()[: offset + 7 * 3 * 3 * 16]).hexdigest()
    with pytest.raises(ValueError, match="proven emitted prefix"):
        TOOL["compare_emitted_prefix"](*paths, count=8, reference_written=8, candidate_written=7)


def test_emitted_prefix_only_passes_its_scope_and_rejects_nonfinite_extension(tmp_path):
    data = np.broadcast_to(np.eye(2, dtype=complex), (10, 2, 2)).copy()
    data[-1] = np.nan
    paths = [tmp_path / name for name in ["reference.npy", "candidate.npy"]]
    for path in paths:
        np.save(path, data)
    result = TOOL["compare_emitted_prefix"](
        *paths, count=9, reference_written=9, candidate_written=9, block=4
    )
    assert result["passed"] and result["maximum_reference_solve_composition_inf"] == 0
    assert result["frequency_prefix_matrices"] == 9
    assert not result["whole_kernel_verified"]
    with pytest.raises(ValueError, match="Nonfinite"):
        TOOL["compare_emitted_prefix"](*paths, count=10, reference_written=10, candidate_written=10)

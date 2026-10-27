import hashlib
import runpy
from pathlib import Path

import numpy as np

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

import runpy
from pathlib import Path

import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/migrate_galerkin_cache.py"))
CHECK = TOOL["approved_storage_only"]

OLD_CQ = """
def bdf2_symbol(zeta):
    return 1.5 - 2*zeta + 0.5*zeta*zeta
def convolution_weights(transfer, plan, *, workers=1):
    half = plan.fft_size // 2
    spectrum = np.empty((half + 1, plan.dofs, plan.dofs), dtype=complex)
    spectrum[0] = transfer(bdf2_symbol(0.5))
    return invert_spectrum(spectrum, plan)
def invert_spectrum(spectrum, plan):
    return irfft(spectrum, n=plan.fft_size, axis=0)
"""
OLD_SAMPLING = """
from .cq import bdf2_symbol, invert_spectrum
def parsimonious_weights(transfer, plan, *, workers=1):
    half = plan.fft_size // 2
    spectrum = np.empty((half + 1, plan.dofs, plan.dofs), dtype=complex)
    spectrum[0] = transfer(bdf2_symbol(0.5))
    return invert_spectrum(spectrum, plan)
"""


def updated(source):
    return source.replace("workers=1):", "workers=1, spectrum=None):").replace(
        "np.empty((half + 1, plan.dofs, plan.dofs), dtype=complex)",
        "spectrum_buffer(plan, spectrum)",
    )


NEW_CQ = TOOL["BUFFER_SOURCE"] + updated(OLD_CQ)
NEW_SAMPLING = updated(OLD_SAMPLING).replace(
    "from .cq import bdf2_symbol, invert_spectrum",
    "from .cq import bdf2_symbol, invert_spectrum, spectrum_buffer",
)


def test_only_approved_allocation_change_can_reuse_weights():
    CHECK(OLD_CQ, NEW_CQ, OLD_SAMPLING, NEW_SAMPLING)


@pytest.mark.parametrize(
    ("cq", "sampling"),
    [
        (NEW_CQ.replace("1.5 - 2*zeta", "1.6 - 2*zeta"), NEW_SAMPLING),
        (NEW_CQ.replace("complex128", "complex64"), NEW_SAMPLING),
        (NEW_CQ, NEW_SAMPLING.replace("bdf2_symbol(0.5)", "bdf2_symbol(0.6)")),
        (NEW_CQ.replace("spectrum=None", "spectrum=False"), NEW_SAMPLING),
    ],
)
def test_migration_rejects_math_dtype_and_default_changes(cq, sampling):
    with pytest.raises(ValueError):
        CHECK(OLD_CQ, cq, OLD_SAMPLING, sampling)

import hashlib
import json
import os
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.fft import irfft, rfft

from finiteblock_bem.cq import plan_cq

TOOLS = Path(__file__).resolve().parents[1] / "tools"
BOUNDED = runpy.run_path(str(TOOLS / "rectangular_cq_history.py"))
DENSE = runpy.run_path(str(TOOLS / "interior_radau.py"))
TRANSPOSE = runpy.run_path(str(TOOLS / "interior_radau_adjoint.py"))["interior_pullback"]


def example(inputs, outputs, steps):
    plan = plan_cq(max(inputs, 2), 0.02, steps, max_memory_mb=1024)
    rng = np.random.default_rng(13)
    shape = (plan.fft_size // 2 + 1, outputs, inputs)
    families = [rng.normal(size=shape) + 1j * rng.normal(size=shape) for _ in range(2)]
    for f in families:
        f[0] = f[0].real
        f[-1] = f[-1].real
    expanded = []
    for k in range(shape[0]):
        z = plan.radius * np.exp(-2j * np.pi * k / plan.fft_size)
        expanded.append(
            np.kron(np.eye(2), families[0][k])
            + np.kron(DENSE["RK"]["radau_symbol"](z), families[1][k])
        )
    weights = irfft(expanded, n=plan.fft_size, axis=0)[: steps + 1]
    weights *= plan.radius ** (-np.arange(steps + 1))[:, None, None]
    return plan, families, weights, rng


@pytest.mark.parametrize("inputs,outputs,steps,rows", [(2, 3, 7, 1), (5, 2, 33, 2), (3, 7, 65, 3)])
def test_bounded_forward_and_complete_future_transpose_match_dense_independent_stage_ifft(
    inputs, outputs, steps, rows
):
    plan, f, w, rng = example(inputs, outputs, steps)
    q = rng.normal(size=(steps + 1, 2, inputs))
    q[0] = 0
    g = rng.normal(size=(steps + 1, 2, outputs))
    actual, meta = BOUNDED["apply_spectra"](*f, plan, q, output_rows=rows, max_workspace_mb=128)
    expected = DENSE["apply_weights"](w, q.reshape(steps + 1, -1)).reshape(actual.shape)
    np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-10)
    pullback, _ = BOUNDED["apply_spectra"](
        *f, plan, g, output_rows=rows, max_workspace_mb=128, transpose=True
    )
    expected = TRANSPOSE(w, g.reshape(steps + 1, -1)).reshape(pullback.shape)
    np.testing.assert_allclose(pullback, expected, rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(np.vdot(actual, g), np.vdot(q, pullback), rtol=3e-12, atol=3e-10)
    assert np.count_nonzero(actual[0]) == np.count_nonzero(pullback[0]) == 0
    assert meta["complete_time_weight_files"] == 0 and meta["every_requested_lag_retained"]


def test_last_step_impulse_has_no_future_response_from_contour_periodization():
    plan = replace(plan_cq(2, 0.1, 9, max_memory_mb=128), radius=0.8)
    k = np.arange(plan.fft_size // 2 + 1)
    z = plan.radius * np.exp(-2j * np.pi * k / plan.fft_size)
    first = (1 / (1 - 0.9 * z))[:, None, None]
    first[[0, -1]] = first[[0, -1]].real
    source = np.zeros((10, 2, 1))
    source[-1, 0, 0] = 1
    actual, _ = BOUNDED["apply_spectra"](
        first, np.zeros_like(first), plan, source, max_workspace_mb=128
    )
    np.testing.assert_allclose(actual[:-1], 0, atol=1e-12)
    damping = plan.radius ** np.arange(10)
    periodic = (
        irfft(first[:, 0, 0] * rfft(source[:, 0, 0] * damping, n=plan.fft_size), n=plan.fft_size)[
            :10
        ]
        / damping
    )
    assert np.max(abs(periodic[1:-1])) > 1e-6


def records(tmp_path, families):
    result = []
    for i, f in enumerate(families):
        p = tmp_path / f"family{i}.npy"
        m = np.lib.format.open_memmap(
            p, mode="w+", dtype=np.complex128, shape=f.shape, version=(2, 0)
        )
        m[:] = f
        m.flush()
        m._mmap.close()
        result.append(
            dict(
                path=str(p),
                shape=list(f.shape),
                bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                all_frequency_matrices_written=True,
            )
        )
    return result


def test_verified_file_operator_rejects_mutated_weights_and_incomplete_receipts(tmp_path):
    plan, f, _, rng = example(2, 3, 7)
    receipt = records(tmp_path, f)
    operator = BOUNDED["FixedRectangularCQ"](receipt, plan, max_workspace_mb=128)
    q = rng.normal(size=(8, 2, 2))
    q[0] = 0
    operator.apply(q)
    p = Path(receipt[0]["path"])
    data = bytearray(p.read_bytes())
    data[-1] ^= 1
    p.write_bytes(data)
    with pytest.raises(ValueError, match="changed"):
        operator.apply(q)
    operator.close()
    receipt[0]["all_frequency_matrices_written"] = False
    with pytest.raises(ValueError, match="complete"):
        BOUNDED["FixedRectangularCQ"](receipt, plan, max_workspace_mb=128)


def test_workspace_and_fixed_initial_stage_guards_precede_fft(monkeypatch):
    plan, f, _, _ = example(2, 3, 7)

    def forbidden(*a, **k):
        raise AssertionError("FFT before workspace preflight")

    monkeypatch.setitem(BOUNDED["apply_spectra"].__globals__, "rfft", forbidden)
    with pytest.raises(ValueError, match="workspace"):
        BOUNDED["apply_spectra"](*f, plan, np.zeros((8, 2, 2)), max_workspace_mb=1)
    with pytest.raises(ValueError, match="fixed-zero"):
        BOUNDED["apply_spectra"](*f, plan, np.ones((8, 2, 2)), max_workspace_mb=128)
    with pytest.raises(ValueError, match="padded CQ"):
        BOUNDED["apply_spectra"](
            *f, replace(plan, fft_size=8), np.zeros((8, 2, 2)), max_workspace_mb=128
        )


def test_concurrent_initial_file_change_is_rejected_after_both_receipts_are_read(
    tmp_path, monkeypatch
):
    plan, f, _, _ = example(2, 3, 7)
    receipt = records(tmp_path, f)
    namespace = BOUNDED["FixedRectangularCQ"].__init__.__globals__
    original = namespace["DIGEST"]

    def changing(path):
        answer = original(path)
        if Path(path) == Path(receipt[1]["path"]):
            p = Path(receipt[0]["path"])
            data = bytearray(p.read_bytes())
            data[-1] ^= 1
            p.write_bytes(data)
        return answer

    monkeypatch.setitem(namespace, "DIGEST", changing)
    with pytest.raises(ValueError, match="initial verification"):
        BOUNDED["FixedRectangularCQ"](receipt, plan, max_workspace_mb=128)


def test_jax_custom_vjp_matches_complete_transpose_and_independent_linear_jvp(tmp_path):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    plan, f, _, rng = example(2, 3, 17)
    operator = BOUNDED["FixedRectangularCQ"](records(tmp_path, f), plan, max_workspace_mb=128)
    q = rng.normal(size=(18, 2, 2))
    q[0] = 0
    direction = rng.normal(size=q.shape)
    direction[0] = 0
    try:
        with jax.enable_x64(True):
            observed = operator.as_jax()
            value, gradient = jax.jit(
                jax.value_and_grad(lambda x: 0.5 * jnp.sum(observed(x) ** 2))
            )(jnp.asarray(q))
            actual = operator.apply(q)
            expected = operator.apply(actual, transpose=True)
            np.testing.assert_allclose(gradient, expected, rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(value, 0.5 * np.vdot(actual, actual), rtol=1e-12)
            tangent = operator.apply(direction)
            np.testing.assert_allclose(
                np.vdot(gradient, direction), np.vdot(actual, tangent), rtol=1e-12
            )
            batch = jax.jit(jax.vmap(observed))(jnp.asarray(np.stack([q, 2 * q])))
            np.testing.assert_allclose(
                batch, np.stack([actual, 2 * actual]), rtol=1e-12, atol=1e-10
            )
            operator.verify()
    finally:
        operator.close()


def test_real_two_block_interior_driver_reconstructs_static_lift_and_complete_stage_observations(
    tmp_path,
):
    case = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    seed = TOOLS.parent / "src/finiteblock_bem/resources/data/ev16_restart_float64.npz"
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "FDBEM_CORES": "1",
        "JAX_ENABLE_X64": "true",
    }
    kernel, trajectory, observations = [
        tmp_path / k for k in ["kernel", "trajectory", "observations"]
    ]
    commands = [
        [
            "prepare_raw_radau_resolvent.py",
            str(case),
            "--output",
            str(kernel),
            "--h",
            "160",
            "--contact-h",
            "160",
            "--contact-end-h",
            "160",
            "--corner-h",
            "160",
            "--dt",
            "2e-6",
            "--duration",
            "1.6e-5",
            "--quadrature",
            "12",
            "--workers",
            "1",
            "--fft-rows",
            "2",
            "--combine-matrices",
            "3",
            "--startup-rows",
            "1",
            "--max-workspace-mb",
            "512",
            "--max-memory-mb",
            "1024",
            "--durable-build",
        ],
        [
            "simulate_compact_radau.py",
            str(kernel),
            "--output",
            str(trajectory),
            "--restart-seed",
            str(seed),
            "--static-initialization",
            "fem-body-active-set",
            "--segment-substeps",
            "2",
            "4",
            "--contact-root-recovery",
            "--kinetic-root-seed",
            "--fft-cache-mb",
            "0",
            "--max-memory-mb",
            "2048",
        ],
        [
            "reconstruct_interior_reference.py",
            str(trajectory),
            "--output",
            str(observations),
            "--max-workspace-mb",
            "512",
            "--dense-check",
            "--check-ad",
        ],
    ]
    for name, *args in commands:
        result = subprocess.run(
            [sys.executable, "-B", str(TOOLS / name), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((observations / "summary.json").read_text())
    assert summary["checks"]["dense_relative_l2"] < 1e-7
    assert summary["checks"]["jax_fixed_map_vjp_relative_error"] < 1e-9
    assert summary["source_perturbation_and_initial_residual_unmodified"]
    with np.load(observations / "observations.npz", allow_pickle=False) as d:
        assert d["value"].shape == (9, 2, 12, 8)
        np.testing.assert_array_equal(d["value"][0], np.repeat(d["static_lift"][None], 2, axis=0))
        np.testing.assert_array_equal(d["value"][0, :, :, 2:4], 0)

"""An analytic coupled ODE independently checks the complete compact producer."""

import json
import runpy
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from finiteblock_bem.cq import invert_spectrum, plan_cq
from finiteblock_bem.friction import SlipWeakeningContact
from finiteblock_bem.io import fingerprint

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def test_complete_compact_producer_matches_dense_stage_weights_and_startup(tmp_path):
    producer = runpy.run_path(str(TOOLS / "prepare_compact_radau_resolvent.py"))
    radau = runpy.run_path(str(TOOLS / "radau_experiment.py"))
    cauchy = runpy.run_path(str(TOOLS / "cauchy_reuse_experiment.py"))
    audit_tool = runpy.run_path(str(TOOLS / "audit_cauchy_reuse.py"))
    environment = dict(source_sha256="test", case_sha256="test", restart_data_sha256="test")
    model = SimpleNamespace(
        constraints=SimpleNamespace(contact_weights=np.ones(1), platen_width=1.0),
        reference=SimpleNamespace(
            contact=SlipWeakeningContact(100, 40, 0.8, 0.5, 0.2),
            spring_stiffness=2.0,
            actuator_rate=15.0,
            restart_actuator=0.0,
        ),
    )
    mass = np.array([[0.3, 0.02, 0.0], [0.02, 0.2, 0.01], [0, 0.01, 0.4]])
    stiffness = np.array([[30.0, 0.2, 0.1], [0.2, 30, -20], [0.1, -20, 30]])
    tangent = np.diag([100.0, 40, 2])

    def zfun(s):
        return s * mass + stiffness

    def hfun(s):
        return np.linalg.inv(zfun(s) + tangent)

    source_plan = plan_cq(3, 0.02, 40, max_memory_mb=32)
    source_plan = replace(source_plan, fft_size=512, radius=1e-10 ** (1 / 512))
    plan = plan_cq(3, 0.08, 10, max_memory_mb=32)
    circle = cauchy["CircleCauchy"](source_plan.fft_size, source_plan.radius)
    source_values = np.array(
        [hfun((1.5 - 2 * z + z * z / 2) / source_plan.dt) for z in circle.sources[:257]]
    )
    source = tmp_path / "source"
    source.mkdir()
    source_path = source / "work-resolvent.npy"
    np.save(source_path, source_values, allow_pickle=False)
    options = dict(case="test", h=1, contact_h=1, contact_end_h=1, corner_h=1, quadrature=12)
    (source / "summary.json").write_text(
        json.dumps(
            dict(
                status="completed_resolvent_kernel_only",
                source_mesh_arguments=options,
                environment=environment,
                full_steps=40,
                correction_steps=40,
                plan=asdict(source_plan),
                spectrum_storage=dict(
                    path=str(source_path),
                    sha256=fingerprint(source_path),
                    coordinates="diag(sqrt(w)) * L * diag(sqrt(w))",
                ),
            )
        )
    )
    nodes = plan.radius * np.exp(-2j * np.pi * np.arange(plan.fft_size // 2 + 1) / plan.fft_size)
    modes, _ = audit_tool["contour_modes"](nodes, plan.dt, "radau2")
    dense = np.array(
        [radau["matrix_function"](hfun, radau["radau_symbol"](z) / plan.dt) for z in nodes]
    )
    audit = tmp_path / "audit"
    audit.mkdir()
    indices = [0, 1, 4, plan.fft_size // 2]
    np.savez(
        audit / "operators.npz",
        fresh=dense[indices],
        reused=dense[indices],
        mode_frequencies=modes[indices],
    )
    (audit / "summary.json").write_text(
        json.dumps(
            dict(
                status="passed_sampled_operator_comparison",
                target_time_method="radau2",
                relative_tolerance=1e-6,
                source_kernel=str(source),
                source_spectrum_sha256=fingerprint(source_path),
                target_dt_s=plan.dt,
                target_fft_size=plan.fft_size,
                environment=environment,
                operator_arrays_sha256=fingerprint(audit / "operators.npz"),
                indices=indices,
            )
        )
    )
    library = tmp_path / "fake-library-test-only"
    library.write_text("The test uses a direct sum, not a native library ABI.")
    output, storage = tmp_path / "metadata", tmp_path / "large-arrays"
    arguments = [
        "prepare",
        str(source),
        "--operator-audit",
        str(audit),
        "--fmm-library",
        str(library),
        "--dt",
        str(plan.dt),
        "--output",
        str(output),
        "--storage-directory",
        str(storage),
    ]
    replacements = dict(
        environment_record=lambda _: environment,
        load_nodal_model=lambda *a, **kw: model,
        NodalInterfaceTransfer=lambda *a, **kw: SimpleNamespace(evaluate=zfun),
        CAUCHY={**producer["CAUCHY"], "fortran_fmm2d_backend": lambda _: cauchy["_direct"]},
    )
    read_bytes = Path.read_bytes

    def guarded_read(path):
        if path.name.startswith(("work-", "weights")) and path.suffix == ".npy":
            raise AssertionError("Large numerical files require bounded streaming hashes")
        return read_bytes(path)

    with (
        patch.dict(producer["main"].__globals__, replacements),
        patch.object(sys, "argv", arguments),
        patch.object(Path, "read_bytes", guarded_read),
    ):
        producer["main"]()
    manifest = json.loads((output / "summary.json").read_text())
    assert manifest["status"] == "completed_compact_radau_resolvent_kernel_only"
    assert manifest["initial_composition_inf_error"] < 1e-6
    assert not manifest["inversion"]["expanded_stage_time_weights_stored"]
    weights = [np.load(storage / f"weights{i}.npy", allow_pickle=False) for i in range(2)]
    wrap = np.load(output / "shift_wrap.npy", allow_pickle=False)
    actual = radau["expand_compact_weights"](*weights, wrap)
    expected = invert_spectrum(dense, replace(plan, dofs=6))
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
    with np.load(output / "elastic.npz", allow_pickle=False) as data:
        force = np.zeros((plan.steps + 1, 6))
        force[:, [2, 5]] = 2 * 15 * data["stage_time"]
        for k in range(plan.steps + 1):
            value = np.einsum("kij,kj->i", expected[: k + 1], force[k::-1])
            np.testing.assert_allclose(data["elastic"][k].ravel(), value, rtol=1e-8, atol=1e-10)
    # A BDF2 receipt cannot authorize a Radau build even with the same dt/FFT.
    record = json.loads((audit / "summary.json").read_text())
    record["target_time_method"] = "bdf2"
    (audit / "summary.json").write_text(json.dumps(record))
    arguments[arguments.index("--output") + 1] = str(tmp_path / "rejected")
    arguments[-1] = str(tmp_path / "rejected-arrays")
    with (
        patch.dict(producer["main"].__globals__, replacements),
        patch.object(sys, "argv", arguments),
    ):
        with pytest.raises(ValueError, match="independent Radau operator audit"):
            producer["main"]()
    assert not (tmp_path / "rejected").exists()
    assert not (tmp_path / "rejected-arrays").exists()


def test_protected_original_tatva_output_is_rejected():
    tool = runpy.run_path(str(TOOLS / "prepare_compact_radau_resolvent.py"))
    with pytest.raises(ValueError, match="read-only"):
        tool["safe_output"](Path("/Volumes/Gauss-T7/tatva/never-write-here"))

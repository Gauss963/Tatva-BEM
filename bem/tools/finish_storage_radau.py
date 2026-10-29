"""Original complete stage audit, lag-zero inverse check and causal startup."""

import os
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np

from finiteblock_bem.cq import CQPlan
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parent
TX = runpy.run_path(str(TOOLS / "inplace_radau_compaction.py"))
DIRECT = runpy.run_path(str(TOOLS / "direct_fortran_radau.py"))
RK = runpy.run_path(str(TOOLS / "radau_experiment.py"))
STORAGE = runpy.run_path(str(TOOLS / "compact_radau_storage.py"))
DIGEST = TX["CHECKPOINT"]["digest_range"]


def finish(raw, compact, output, *, fft_rows, startup_rows, max_workspace_mb):
    output = Path(output)
    plan = CQPlan(**raw["plan"])
    options = raw["source_mesh_arguments"]
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    con, ref, law = model.constraints, model.reference, model.reference.contact
    line = np.r_[con.contact_weights, con.contact_weights, con.platen_width]
    spring = ref.spring_stiffness * con.platen_width
    tangent = np.diag(
        np.r_[
            con.contact_weights * law.normal_penalty,
            con.contact_weights * law.tangential_penalty,
            spring,
        ]
    )
    raw_folder = output / "raw-stage"
    for name, key in [
        ("stage-audit.npz", "stage_audit_sha256"),
        ("initial_impedance.npy", "initial_impedance_sha256"),
    ]:
        if DIGEST(raw_folder / name) != raw[key]:
            raise ValueError("Original independent raw audit artifact changed")
    with np.load(raw_folder / "stage-audit.npz", allow_pickle=False) as saved:
        indices, fresh_stages = saved["indices"], saved["stages"]
    spectra = [np.load(r["path"], mmap_mode="r", allow_pickle=False) for r in compact]
    errors = []
    try:
        for index, fresh in zip(indices, fresh_stages, strict=True):
            index = int(index)
            zeta = plan.radius * np.exp(-2j * np.pi * index / plan.fft_size)
            actual = np.kron(np.eye(2), spectra[0][index]) + np.kron(
                RK["radau_symbol"](zeta), spectra[1][index]
            )
            error = float(np.linalg.norm(actual - fresh) / np.linalg.norm(fresh))
            if not np.isfinite(error) or error > 1e-6:
                raise ValueError("Combined families failed the independent complete stage audit")
            errors.append(dict(index=int(index), relative_frobenius=error))
        weights = output / "weights-stage"
        paths, wrap, inverse = DIRECT["invert_physical_fortran"](
            compact,
            plan,
            line,
            weights,
            fft_rows=fft_rows,
            max_workspace_mb=max_workspace_mb,
            resume=weights.exists(),
        )
        maps = [np.load(p, mmap_mode="r", allow_pickle=False) for p in paths]
        try:
            instantaneous = RK["expand_compact_weights"](maps[0][:1], maps[1][:1], wrap)[0]
        finally:
            for m in maps:
                m._mmap.close()
        impedance = np.load(raw_folder / "initial_impedance.npy", allow_pickle=False)
        composition = float(
            np.linalg.norm(
                instantaneous @ (impedance + np.kron(np.eye(2), tangent)) - np.eye(2 * plan.dofs),
                ord=np.inf,
            )
        )
        if not np.isfinite(composition) or composition > 1e-6:
            raise ValueError("Fresh Radau lag-zero inverse composition failed")
        stage_time = np.zeros((plan.steps + 1, 2))
        stage_time[1:] = (np.arange(plan.steps)[:, None] + RK["C"]) * plan.dt
        increment = options.get("diagnostic_actuator_increment")
        rate = ref.actuator_rate if increment is None else increment / raw["arguments"]["duration"]
        elastic = np.empty((plan.steps + 1, 2, plan.dofs))
        for first in range(0, plan.dofs, startup_rows):
            last = min(first + startup_rows, plan.dofs)
            columns = [s[:, first:last, -1] / np.sqrt(line[first:last] * line[-1]) for s in spectra]
            elastic[:, :, first:last] = STORAGE["causal_compact_startup"](
                *columns,
                wrap[first:last, -1],
                replace(plan, dofs=last - first),
                spring * rate * stage_time,
            )
    finally:
        for s in spectra:
            s._mmap.close()
    # Only reproducible postprocessing outputs in this owned folder are replaced.
    # Publish each complete file atomically; partial temporary files are not inputs.
    for name, value in [("initial_impedance.npy", impedance), ("shift_wrap.npy", wrap)]:
        tmp = output / (name + ".tmp")
        with tmp.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        tmp.replace(output / name)
    tmp = output / "elastic.npz.tmp"
    with tmp.open("wb") as stream:
        np.savez(
            stream,
            time=np.arange(plan.steps + 1) * plan.dt,
            stage_time=stage_time,
            elastic=elastic,
            uninterrupted_actuator=ref.restart_actuator + rate * stage_time,
        )
    tmp.replace(output / "elastic.npz")
    for name in ["initial_impedance.npy", "shift_wrap.npy", "elastic.npz"]:
        with (output / name).open("rb") as stream:
            os.fsync(stream.fileno())
    TX["sync_directory"](output)
    return dict(
        family_weights=inverse["files"][:2],
        sampled_fresh_stage_checks=errors,
        initial_composition_inf_error=composition,
        shift_wrap_sha256=DIGEST(output / "shift_wrap.npy"),
        initial_impedance_sha256=DIGEST(output / "initial_impedance.npy"),
        elastic_sha256=DIGEST(output / "elastic.npz"),
        inversion=dict(
            fft_rows=fft_rows, resources=inverse["resources"], attempts=inverse["attempts"]
        ),
        time_weights_layout="fortran",
        storage_conversion=[],
    )

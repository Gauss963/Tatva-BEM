"""Independent plane-wave controls for interior displacement, velocity and stress.

Analytic P/S waves on a unit square separate this control from the event-16
contact solve. Temporal controls use a prescribed causal compact pulse, not a
fitted phase. They do not prove finite-block reflected-wave or FEM agreement.
"""

import argparse
import json
import runpy
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem import ElasticMaterial, polygon_boundary
from finiteblock_bem.io import fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
FIELD = runpy.run_path(str(TOOLS / "interior_fields.py"))
CQ = runpy.run_path(str(TOOLS / "interior_radau.py"))
MATERIAL = ElasticMaterial(3.0, 0.23, 1.2)


def pulse(time):
    """C5 compact pulse on [0.2,0.6], with its analytic time derivative."""
    z = (np.asarray(time) - 0.2) / 0.4
    inside = (z > 0) & (z < 1)
    q = np.where(inside, 4 * z * (1 - z), 0)
    return q**6, np.where(inside, 24 / 0.4 * (1 - 2 * z) * q**5, 0)


def prepare_wave(h, kind, points):
    mesh = polygon_boundary([[0, 0], [1, 0], [1, 1], [0, 1]], ["free"] * 4, h)
    potential = FIELD["InteriorPotentials"](mesh, MATERIAL, points, quadrature=12)
    polarization = np.eye(2)[0 if kind == "P" else 1]
    speed = MATERIAL.cp if kind == "P" else MATERIAL.cs

    def evaluate(s):
        k = s / speed
        u = np.exp(-k * mesh.starts[:, 0])[:, None] * polarization
        boundary_gradient = np.zeros((mesh.count, 2, 2), complex)
        boundary_gradient[:, :, 0] = -k * np.exp(-k * mesh.centers[:, 0])[:, None] * polarization
        sigma = FIELD["plane_strain_stress"](boundary_gradient, MATERIAL)
        density = np.column_stack(
            (
                sigma[:, 0] * mesh.normals[:, 0] + sigma[:, 2] * mesh.normals[:, 1],
                sigma[:, 2] * mesh.normals[:, 0] + sigma[:, 1] * mesh.normals[:, 1],
            )
        )
        response = potential.evaluate(u.ravel(), density.ravel(), s)
        response["velocity"] = s * response["displacement"]
        return response

    return evaluate, speed, polarization, mesh.count


def temporal_wave(h, dt, kind, *, points=None):
    points = np.array([[0.4, 0.45]]) if points is None else np.asarray(points)
    evaluate, speed, polarization, panels = prepare_wave(h, kind, points)
    steps = round(1.2 / dt)

    def transfer(s):
        field = evaluate(s)
        return np.concatenate(
            [field[k] for k in ["displacement", "velocity", "stress"]], axis=1
        ).reshape(-1, 1)

    begun = perf_counter()
    weights, metadata = CQ["rectangular_weights"](transfer, dt, steps, 1, 8 * len(points))
    stage_time = np.zeros((steps + 1, 2))
    stage_time[1:] = (np.arange(steps)[:, None] + CQ["RK"]["C"]) * dt
    forcing = pulse(stage_time)[0]
    forcing[0] = 0
    value = CQ["apply_weights"](weights, forcing).reshape(steps + 1, 2, len(points), 8)[:, 1]
    time = np.arange(steps + 1) * dt
    f, derivative = pulse(time[:, None] - points[None, :, 0] / speed)
    exact_u = f[..., None] * polarization
    exact_v = derivative[..., None] * polarization
    gradient = np.zeros((steps + 1, len(points), 2, 2))
    gradient[..., :, 0] = -exact_v / speed
    exact_sigma = FIELD["plane_strain_stress"](gradient, MATERIAL)
    exact = np.concatenate([exact_u, exact_v, exact_sigma], axis=-1)
    errors = {}
    for key, selected in [
        ("displacement", slice(0, 2)),
        ("velocity", slice(2, 4)),
        ("stress", slice(4, 8)),
    ]:
        reference, actual = exact[..., selected], value[..., selected]
        errors[key] = dict(
            relative_saved_endpoint_l2=float(
                np.linalg.norm(actual - reference) / np.linalg.norm(reference)
            ),
            maximum_absolute_difference=float(np.max(abs(actual - reference))),
        )
    return dict(
        kind=kind,
        max_panel_length=h,
        panels=panels,
        points=points.tolist(),
        speed=speed,
        cq=metadata,
        errors=errors,
        sampling_and_apply_seconds=perf_counter() - begun,
        fitted_time_shift_s=0.0,
    ), dict(time=time, value=value, exact=exact)


def spatial_wave(h, kind):
    points, s = np.array([[0.3, 0.4], [0.7, 0.6]]), 1.4 + 0.9j
    evaluate, speed, polarization, panels = prepare_wave(h, kind, points)
    actual = evaluate(s)
    exact_u = np.exp(-s * points[:, 0] / speed)[:, None] * polarization
    gradient = np.zeros((len(points), 2, 2), complex)
    gradient[:, :, 0] = -s * exact_u / speed
    expected = dict(
        displacement=exact_u,
        velocity=s * exact_u,
        gradient=gradient,
        stress=FIELD["plane_strain_stress"](gradient, MATERIAL),
    )
    return dict(
        kind=kind,
        max_panel_length=h,
        panels=panels,
        frequency_s_inverse=[s.real, s.imag],
        relative_errors={
            key: float(np.linalg.norm(actual[key] - v) / np.linalg.norm(v))
            for key, v in expected.items()
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva")):
        raise ValueError("Choose a fresh output outside protected Tatva")
    sources = [
        Path(__file__),
        TOOLS / "interior_fields.py",
        TOOLS / "interior_radau.py",
        TOOLS / "radau_experiment.py",
    ]
    sources += [
        TOOLS.parent / "src/finiteblock_bem" / name
        for name in [
            "kernels.py",
            "galerkin.py",
            "galerkin_fast.py",
            "nodal.py",
            "cq.py",
            "mesh.py",
            "material.py",
        ]
    ]
    frozen = {str(p.resolve()): fingerprint(p) for p in sources}
    begun = perf_counter()
    output.mkdir(parents=True)
    spatial = [spatial_wave(h, kind) for kind in ["P", "S"] for h in [0.125, 0.0625]]
    temporal, datasets = [], {}
    for kind in ["P", "S"]:
        for h, dt in [
            (0.03125, 0.04),
            (0.03125, 0.02),
            (0.03125, 0.01),
            (0.015625, 0.01),
            (0.015625, 0.005),
        ]:
            record, arrays = temporal_wave(h, dt, kind)
            temporal.append(record)
            dataset = output / f"{kind}-h{h:g}-dt{dt:g}.npz"
            np.savez(dataset, **arrays)
            datasets[dataset.name] = fingerprint(dataset)
            print(json.dumps(record), flush=True)
    if any(fingerprint(p) != v for p, v in frozen.items()):
        raise ValueError("Interior verification source changed during execution")
    write_json(
        output / "summary.json",
        dict(
            status="completed_manufactured_interior_controls",
            frozen_input_sha256=frozen,
            spatial=spatial,
            temporal=temporal,
            dataset_sha256=datasets,
            seconds=perf_counter() - begun,
            scope="Unit-square plane P/S waves, analytic causal pulse and no fitted shift; no finite-body reflection, contact, original-event or FEM proof",
            same_accuracy_fem_advantage_verified=False,
            full_original_event_reproduced=False,
        ),
    )


if __name__ == "__main__":
    main()

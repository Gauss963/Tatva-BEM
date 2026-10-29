"""Read-only float64 Q1 interior observations for the pinned Tatva FEM state.

Displacement/stress use the integer endpoint clock. Source carry v_half is
saved unchanged with its nominal +dt/2 clock, never relabelled as endpoint
velocity. Q1 point stress is not the original bulk Gauss-average stress.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_SHA256 = "5f549e3cbd017030cac3db1cbc957ab1998583f868134fe8b69a718c32c13309"


def canonical_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {"moving", "stationary"}:
        raise ValueError("Probe plan needs moving and stationary physical coordinates")
    result = {}
    for name in ["moving", "stationary"]:
        points = np.asarray(plan[name], dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not len(points)
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("Each body needs nonempty finite (probes,2) coordinates")
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("Repeated physical probe positions are not allowed")
        result[name] = points.tolist()
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    return result, hashlib.sha256(encoded.encode()).hexdigest()


def q1_basis(xi):
    """Original Quad4 order: (-1,-1),(1,-1),(1,1),(-1,1)."""
    r, s = xi
    values = 0.25 * np.array(
        [(1 - r) * (1 - s), (1 + r) * (1 - s), (1 + r) * (1 + s), (1 - r) * (1 + s)]
    )
    derivative = 0.25 * np.array(
        [[-(1 - s), 1 - s, 1 + s, -(1 + s)], [-(1 - r), -(1 + r), 1 + r, 1 - r]]
    )
    return values, derivative


class Q1ProbeSampler:
    def __init__(self, coords, elements, points):
        coords, elements, points = (
            np.asarray(coords),
            np.asarray(elements),
            np.asarray(points, dtype=float),
        )
        if (
            coords.dtype != np.float64
            or coords.ndim != 2
            or coords.shape[1] != 2
            or elements.ndim != 2
            or elements.shape[1] != 4
            or elements.dtype.kind not in "iu"
            or not len(elements)
            or np.any(elements < 0)
            or np.any(elements >= len(coords))
            or not np.all(np.isfinite(coords))
            or points.ndim != 2
            or points.shape[1] != 2
            or not len(points)
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(
                "Require true float64 coordinates and valid 2D Quad4 connectivity/probes"
            )
        vertices = coords[elements]
        lower, upper = vertices.min(axis=1), vertices.max(axis=1)
        tol = 128 * np.finfo(float).eps * max(1.0, float(abs(coords).max()))
        self.n_nodes = len(coords)
        self.points = points.copy()
        ids, nodes, natural, shapes, gradients = [], [], [], [], []
        for point in points:
            candidates = np.flatnonzero(
                np.all((lower - tol <= point) & (point <= upper + tol), axis=1)
            )
            hits = []
            edge_hit = False
            for element in candidates:
                x = vertices[element]
                for corner in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]:
                    jac = q1_basis(corner)[1] @ x
                    if np.linalg.det(jac) <= np.finfo(float).eps * np.linalg.norm(jac) ** 2:
                        raise ValueError("Candidate Q1 element is folded, degenerate or reversed")
                xi = np.zeros(2)
                converged = False
                for _ in range(25):
                    shape, derivative = q1_basis(xi)
                    residual = shape @ x - point
                    if np.linalg.norm(residual) <= tol:
                        converged = True
                        break
                    xi -= np.linalg.solve((derivative @ x).T, residual)
                    if not np.all(np.isfinite(xi)) or np.max(abs(xi)) > 1e6:
                        break
                if not converged or np.max(abs(xi)) > 1 + 1e-10:
                    continue
                if np.max(abs(xi)) >= 1 - 1e-10:
                    edge_hit = True
                    continue
                shape, derivative = q1_basis(xi)
                physical = np.linalg.solve(derivative @ x, derivative).T
                hits.append((element, xi.copy(), shape, physical))
            if edge_hit or len(hits) != 1:
                raise ValueError(
                    "Probe must lie strictly inside one Q1 element; edges/exterior/overlaps are ambiguous"
                )
            element, xi, shape, physical = hits[0]
            ids.append(element)
            nodes.append(elements[element])
            natural.append(xi)
            shapes.append(shape)
            gradients.append(physical)
        self.element_ids = np.asarray(ids, dtype=np.int64)
        self.nodes = np.asarray(nodes, dtype=np.int64)
        self.reference_coordinates = np.asarray(natural)
        self.shape_values = np.asarray(shapes)
        self.shape_gradients = np.asarray(gradients)
        digest = hashlib.sha256(coords.tobytes())
        digest.update(elements.astype(np.int64).tobytes())
        self.mesh_sha256 = digest.hexdigest()

    def evaluate(self, displacement, velocity, lam, mu):
        displacement, velocity = np.asarray(displacement), np.asarray(velocity)
        if (
            displacement.dtype != np.float64
            or velocity.dtype != np.float64
            or displacement.shape != (self.n_nodes, 2)
            or velocity.shape != displacement.shape
            or not np.all(np.isfinite(displacement))
            or not np.all(np.isfinite(velocity))
            or not np.isfinite(lam)
            or not np.isfinite(mu)
            or mu <= 0
        ):
            raise ValueError(
                "Require finite true float64 source displacement and half-step velocity"
            )
        u, v = displacement[self.nodes], velocity[self.nodes]
        value = np.einsum("pn,pnc->pc", self.shape_values, u)
        speed = np.einsum("pn,pnc->pc", self.shape_values, v)
        gradient = np.einsum("pnd,pnc->pcd", self.shape_gradients, u)
        strain = 0.5 * (gradient + gradient.swapaxes(-1, -2))
        trace = np.trace(strain, axis1=-2, axis2=-1)
        stress = 2 * mu * strain + lam * trace[:, None, None] * np.eye(2)
        stress = np.column_stack((stress[:, 0, 0], stress[:, 1, 1], stress[:, 0, 1], lam * trace))
        return dict(displacement=value, velocity_half_step=speed, gradient=gradient, stress=stress)


class InteriorObserver:
    def __init__(self, plan):
        self.plan, self.plan_sha256 = canonical_plan(plan)
        self.samplers = None
        self.model_id = None
        self.seen_files = set()

    def __call__(self, h5, index, carry, phase, step, model, mask):
        if int(model["dimension"]) != 2:
            raise ValueError("Interior observations require the pinned two-dimensional model")
        u, v = np.asarray(carry[0]), np.asarray(carry[1])
        offset = int(model["moving_offset"])
        if self.samplers is None or self.model_id != id(model):
            self.samplers = {
                body: Q1ProbeSampler(
                    np.asarray(model[body].mesh.coords),
                    np.asarray(model[body].mesh.elements),
                    self.plan[body],
                )
                for body in ["moving", "stationary"]
            }
            self.model_id = id(model)
        if (
            u.ndim != 1
            or v.shape != u.shape
            or offset != 2 * self.samplers["moving"].n_nodes
            or len(u) != offset + 2 * self.samplers["stationary"].n_nodes
        ):
            raise ValueError("Probe meshes and original source state layout differ")
        fields = {}
        for body, sl in [("moving", slice(None, offset)), ("stationary", slice(offset, None))]:
            material = model[body + "_material"]
            fields[body] = self.samplers[body].evaluate(
                u[sl].reshape(-1, 2),
                v[sl].reshape(-1, 2),
                float(material.lmbda),
                float(material.mu),
            )
        count = h5["interface_high_rate/history"].shape[0]
        dt, normal = float(model["dt"]), int(model["pressure_steps"])
        if (
            phase not in (0, 1, 2)
            or step < 0
            or not np.isfinite(dt)
            or dt <= 0
            or not 0 <= index < count
        ):
            raise ValueError("Invalid original FEM observation clock/index")
        if "interior_observations" not in h5:
            if index != 0:
                raise ValueError("Cannot invent missing interior observations after a checkpoint")
            group = h5.create_group("interior_observations")
            group.attrs.update(
                schema_version=1,
                written_frames=0,
                probe_plan_sha256=self.plan_sha256,
                dt=dt,
                normal_steps=normal,
                velocity_definition="Source carry v_half, nominal endpoint+dt/2; original relaxation/constraints retained",
                stress_definition="Pointwise Q1 plane-strain elastic stress (xx,yy,xy,zz), not Gauss-average bulk stress",
            )
            for name, dtype in [
                ("phase_id", "i4"),
                ("step_id", "i8"),
                ("absolute_step", "i8"),
                ("time_s", "f8"),
                ("velocity_time_s", "f8"),
            ]:
                group.create_dataset(name, shape=(count,), dtype=dtype)
            for body, sampler in self.samplers.items():
                saved = group.create_group(body)
                material = model[body + "_material"]
                saved.attrs.update(
                    mesh_sha256=sampler.mesh_sha256,
                    lam=float(material.lmbda),
                    mu=float(material.mu),
                )
                for name in [
                    "points",
                    "element_ids",
                    "nodes",
                    "reference_coordinates",
                    "shape_values",
                    "shape_gradients",
                ]:
                    saved.create_dataset(name, data=getattr(sampler, name))
                for name, value in fields[body].items():
                    saved.create_dataset(
                        name,
                        shape=(count, *value.shape),
                        dtype="f8",
                        chunks=(1, *value.shape),
                        compression="lzf",
                    )
        group = h5["interior_observations"]
        if (
            group.attrs["schema_version"] != 1
            or group.attrs["probe_plan_sha256"] != self.plan_sha256
            or group.attrs["dt"] != dt
            or group.attrs["normal_steps"] != normal
            or group["step_id"].shape != (count,)
        ):
            raise ValueError("Interior observation schema, plan or source clock changed")
        for body, sampler in self.samplers.items():
            saved = group[body]
            material = model[body + "_material"]
            if (
                saved.attrs["mesh_sha256"] != sampler.mesh_sha256
                or saved.attrs["lam"] != float(material.lmbda)
                or saved.attrs["mu"] != float(material.mu)
            ):
                raise ValueError("Interior probe mesh/material changed")
            for name in [
                "points",
                "element_ids",
                "nodes",
                "reference_coordinates",
                "shape_values",
                "shape_gradients",
            ]:
                if not np.array_equal(saved[name][:], getattr(sampler, name)):
                    raise ValueError("Interior physical probe mapping changed")
        if h5.filename not in self.seen_files:
            if h5.attrs.get("resumed_from_checkpoint", 0):
                checkpoint_frames = int(h5.attrs["saved_interface_frames"])
                if index != checkpoint_frames or checkpoint_frames > int(
                    group.attrs["written_frames"]
                ):
                    raise ValueError("Interior observations do not contain the checkpoint prefix")
                group.attrs["written_frames"] = checkpoint_frames
            self.seen_files.add(h5.filename)
        if index != int(group.attrs["written_frames"]):
            raise ValueError("Interior observation is duplicated or out of order")
        for body, values in fields.items():
            for name, value in values.items():
                if group[body][name].dtype != np.float64 or group[body][name].shape != (
                    count,
                    *value.shape,
                ):
                    raise ValueError("Interior observations must retain true float64 fields")
                group[body][name][index] = value
        absolute = int(step) + (normal if phase == 2 else 0)
        for name, value in dict(
            phase_id=phase,
            step_id=step,
            absolute_step=absolute,
            time_s=absolute * dt,
            velocity_time_s=(absolute + 0.5) * dt,
        ).items():
            group[name][index] = value
        group.attrs["written_frames"] = index + 1


def make_observed_runner(function, plan, contact_adapter):
    """Compose two output observers; original integration AST is untouched."""
    path = Path(function.__code__.co_filename)
    if (
        hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256
        or function.__name__ != "run_simulation_dumped"
    ):
        raise ValueError("Interior observer requires the pinned 933ae1f FEM source")
    contact, interior = contact_adapter["ContactObserver"](), InteriorObserver(plan)

    def observe(*args):
        contact(*args)
        interior(*args)

    return contact_adapter["_instrument_writer"](function, observe)


def validate_probe_group(h5, *, count=None, plan_sha256=None):
    """Read actual stored values; count selects an existing checkpoint prefix."""
    group, high = h5["interior_observations"], h5["interface_high_rate"]
    complete = count is None
    count = int(h5.attrs["saved_interface_frames"]) if complete else int(count)
    written = int(group.attrs["written_frames"])
    if (
        group.attrs["schema_version"] != 1
        or count < 1
        or written < count
        or (
            complete
            and (written != count or count != len(group["time_s"]) or count != len(high["history"]))
        )
        or count > len(group["time_s"])
    ):
        raise ValueError("Incomplete interior observation prefix")
    if plan_sha256 is not None and group.attrs["probe_plan_sha256"] != plan_sha256:
        raise ValueError("Interior observation probe plan changed")
    if any(group[k].dtype.kind not in "iu" for k in ["phase_id", "step_id", "absolute_step"]):
        raise ValueError("Interior observations require actual integer step clocks")
    phase, step, absolute = (group[k][:count] for k in ["phase_id", "step_id", "absolute_step"])
    expected = step + np.where(phase == 2, int(group.attrs["normal_steps"]), 0)
    if (
        not np.array_equal(phase, high["phase_id"][:count])
        or not np.array_equal(step, high["step_id"][:count])
        or not np.array_equal(absolute, expected)
    ):
        raise ValueError("Interior observation clock differs from the original source writer")
    dt = float(group.attrs["dt"])
    if (
        not np.isfinite(dt)
        or dt <= 0
        or np.any(~np.isin(phase, [0, 1, 2]))
        or np.any(np.diff(phase) < 0)
        or np.any(step < 0)
        or np.any((np.diff(phase) == 0) & (np.diff(step) <= 0))
    ):
        raise ValueError("Invalid original FEM clock")
    for name, expected_time in [
        ("time_s", absolute.astype(float) * dt),
        ("velocity_time_s", (absolute.astype(float) + 0.5) * dt),
    ]:
        if group[name].dtype != np.float64 or not np.array_equal(
            group[name][:count], expected_time
        ):
            raise ValueError("Interior endpoint/half-step clocks differ from integer source steps")
    result, stored_plan = {}, {}
    for body in ["moving", "stationary"]:
        saved = group[body]
        points = saved["points"][:]
        n = len(points)
        if (
            points.dtype != np.float64
            or points.shape != (n, 2)
            or n < 1
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("Invalid true float64 physical probe positions")
        for name, tail in [
            ("displacement", (n, 2)),
            ("velocity_half_step", (n, 2)),
            ("gradient", (n, 2, 2)),
            ("stress", (n, 4)),
        ]:
            array = saved[name]
            if (
                array.dtype != np.float64
                or array.shape[1:] != tail
                or len(array) < count
                or not np.all(np.isfinite(array[:count]))
            ):
                raise ValueError("Require complete true float64 interior probe fields")
        result[body] = n
        stored_plan[body] = points.tolist()
    if canonical_plan(stored_plan)[1] != group.attrs["probe_plan_sha256"]:
        raise ValueError("Stored physical probe coordinates differ from the recorded plan")
    return dict(
        written_frames=written,
        validated_frames=count,
        complete=complete,
        probes_per_body=result,
        actual_endpoint_and_half_step_clock_verified=True,
    )

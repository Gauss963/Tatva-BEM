"""Float64 endpoint observations added only to the pinned FEM output helper.

The original source file and module bindings are untouched. One observer call
is appended to its high-rate writer; integration/contact equations retain the
exact original AST. This is an explicit diagnostic adapter.
"""

import ast
import hashlib
import inspect
import textwrap
from pathlib import Path

import numpy as np

SOURCE_SHA256 = "5f549e3cbd017030cac3db1cbc957ab1998583f868134fe8b69a718c32c13309"
FIELDS = (
    "normal_jump",
    "tangential_jump",
    "pressure",
    "elastic_shear",
    "plastic_slip",
    "cumulative_slip",
    "friction_coefficient",
    "friction_strength",
)


def _instrument_writer(function, observer):
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    original = ast.dump(tree, include_attributes=False)
    matches = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "save_high_rate_interface"
    ]
    if len(matches) != 1:
        raise ValueError("Require exactly one original high-rate writer")
    target = matches[0]
    names = ["h5", "frame_idx", "carry", "phase_id", "step_id", "model", "interface_plot_mask"]
    target.body.append(
        ast.Expr(
            ast.Call(
                func=ast.Name(id="_tatva_bem_contact_observer", ctx=ast.Load()),
                args=[ast.Name(id=name, ctx=ast.Load()) for name in names],
                keywords=[],
            )
        )
    )
    inserted = target.body.pop()
    if ast.dump(tree, include_attributes=False) != original:
        raise RuntimeError("Observation transformation changed original source statements")
    target.body.append(inserted)
    tree.body.insert(
        0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    )
    namespace = {**function.__globals__, "_tatva_bem_contact_observer": observer}
    code = compile(
        ast.fix_missing_locations(tree),
        function.__code__.co_filename + "|contact-observations",
        "exec",
    )
    exec(code, namespace)
    return namespace[function.__name__]


class ContactObserver:
    def __init__(self):
        self.seen_files = set()

    def __call__(self, h5, index, carry, phase, step, model, mask):
        if (
            int(model["dimension"]) != 2
            or model["friction_law"] != "slip-weakening"
            or not np.all(mask)
        ):
            raise ValueError("Endpoint observations require the whole 2D LSW contact interface")
        u, z, c, mu, strength = (np.asarray(carry[k]) for k in [0, 2, 3, 6, 8])
        if any(
            a.dtype != np.float64 or not np.all(np.isfinite(a)) for a in [u, z, c, mu, strength]
        ):
            raise ValueError("Contact observations require finite true float64 source state")
        master, slave = (np.asarray(model[k], dtype=int) for k in ["master_nodes", "slave_nodes"])
        n = len(master)
        if slave.shape != (n,) or any(a.shape != (n,) for a in [z, c, mu, strength]):
            raise ValueError("Source contact state and node pairs differ")
        offset = int(model["moving_offset"])
        gap = np.asarray(model["interface_initial_gap"], dtype=float)
        kn = float(model["penalty_n"])
        kt = np.broadcast_to(np.asarray(model["penalty_t"], dtype=float), (n,))
        dn = u[2 * master] - u[offset + 2 * slave] - gap
        ds = u[2 * master + 1] - u[offset + 2 * slave + 1]
        pressure = kn * np.maximum(dn, 0)
        shear = np.where(dn > 0, kt * (ds - z), 0)
        values = dict(zip(FIELDS, [dn, ds, pressure, shear, z, c, mu, strength], strict=True))
        if any(not np.all(np.isfinite(a)) for a in values.values()):
            raise ValueError("Nonfinite endpoint contact observation")
        high = h5["interface_high_rate"]
        count = high["history"].shape[0]
        coords = np.asarray(model["moving"].mesh.coords)
        y = np.asarray(coords[master, 1], dtype=float)
        if "contact_observations" not in h5:
            if index != 0:
                raise ValueError("Cannot add missing observations after a checkpoint")
            group = h5.create_group("contact_observations")
            group.attrs.update(
                schema_version=1,
                written_frames=0,
                normal_penalty=kn,
                shear_definition="kt*(tangential jump - committed plastic offset), zero when open",
                time_definition="Integer absolute step times original model dt",
                scope="Endpoint state observations, not an extra force evaluation in the integrator",
            )
            for name, value in dict(
                y=y,
                initial_gap=gap,
                weights=np.asarray(model["interface_weights"]),
                tangential_penalty=kt,
            ).items():
                group.create_dataset(name, data=value, dtype="f8")
            for name in FIELDS:
                group.create_dataset(
                    name, shape=(count, n), dtype="f8", chunks=(1, n), compression="lzf"
                )
            for name, dtype in [
                ("phase_id", "i4"),
                ("step_id", "i8"),
                ("absolute_step", "i8"),
                ("time_s", "f8"),
            ]:
                group.create_dataset(name, shape=(count,), dtype=dtype)
        group = h5["contact_observations"]
        if (
            group.attrs["schema_version"] != 1
            or group["pressure"].shape != (count, n)
            or not np.array_equal(group["y"][:], y)
            or not np.array_equal(group["initial_gap"][:], gap)
            or not np.array_equal(group["tangential_penalty"][:], kt)
            or group.attrs["normal_penalty"] != kn
        ):
            raise ValueError("Contact observation schema or physical mapping changed")
        if h5.filename not in self.seen_files:
            if h5.attrs.get("resumed_from_checkpoint", 0):
                checkpoint_frames = int(h5.attrs["saved_interface_frames"])
                if index != checkpoint_frames or checkpoint_frames > int(
                    group.attrs["written_frames"]
                ):
                    raise ValueError(
                        "Contact observations do not contain the saved checkpoint prefix"
                    )
                # The original source rewinds to the last checkpoint. Rows
                # written after that checkpoint must be overwritten on resume.
                group.attrs["written_frames"] = checkpoint_frames
            self.seen_files.add(h5.filename)
        if index != int(group.attrs["written_frames"]):
            raise ValueError("Contact observation frame is duplicated or out of order")
        absolute = int(step) + (int(model["pressure_steps"]) if phase == 2 else 0)
        for name, value in values.items():
            group[name][index] = value
        for name, value in dict(
            phase_id=phase,
            step_id=step,
            absolute_step=absolute,
            time_s=absolute * float(model["dt"]),
        ).items():
            group[name][index] = value
        group.attrs["written_frames"] = index + 1


def make_observed_runner(function):
    path = Path(function.__code__.co_filename)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != SOURCE_SHA256 or function.__name__ != "run_simulation_dumped":
        raise ValueError("Contact observation adapter requires the pinned 933ae1f FEM source")
    return _instrument_writer(function, ContactObserver())

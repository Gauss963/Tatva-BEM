"""Read completed archived FEM traces without depending on bulk frame cadence."""

from pathlib import Path

import numpy as np


def read_fem_trace(path, *, fields=()):
    import h5py

    path = Path(path)
    before = path.stat()
    with h5py.File(path, "r") as stream:
        selected = "interface_high_rate" if "interface_high_rate" in stream else "/"
        group = stream[selected]
        count_key = "saved_interface_frames" if selected != "/" else "saved_frames"
        count = int(stream.attrs.get(count_key, -1))
        if count < 1 or count != len(group["history"]):
            raise ValueError("FEM history is not marked complete")
        names = [
            item.decode() if isinstance(item, bytes) else str(item)
            for item in group["history_columns"][:]
        ]
        values = np.asarray(group["history"][:], dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] != len(names)
            or len(set(names)) != len(names)
            or "time" not in names
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Invalid FEM history columns or values")
        history = {name: values[:, index] for index, name in enumerate(names)}
        phase = np.asarray(group["phase_id"][:])
        step = np.asarray(group["step_id"][:])
        if (
            phase.shape != (count,)
            or step.shape != (count,)
            or np.any(np.diff(history["time"]) < 0)
        ):
            raise ValueError("FEM history phases, steps or time are inconsistent")
        interface = stream["interface"] if selected == "/" else group
        arrays, y = {}, None
        if fields:
            y = np.asarray(interface["contact_line_y"][:], dtype=float)
            if y.ndim != 1 or not np.all(np.isfinite(y)) or np.any(np.diff(y) <= 0):
                raise ValueError("FEM interface nodes must be finite and increasing")
            for name in fields:
                array = np.asarray(interface[name][:], dtype=float)
                if array.shape != (count, len(y)) or not np.all(np.isfinite(array)):
                    raise ValueError(f"Invalid FEM interface field: {name}")
                arrays[name] = array
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("FEM source file changed while reading")
    return dict(
        history=history,
        phase_id=phase,
        step_id=step,
        y=y,
        interface_fields=arrays,
        provenance=dict(
            path=str(path.resolve()),
            bytes=before.st_size,
            mtime_ns=before.st_mtime_ns,
            history_group=selected,
            saved_frames=count,
            columns=names,
        ),
    )

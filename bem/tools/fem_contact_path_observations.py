"""Observe original accepted FEM contact paths and audit temporal projection.

The observer only adds outputs to a scan of the supplied original step.
Projection is a diagnostic on a saved path, not a modified FEM simulation
or a claim about the error of a computed BEM solution.
"""

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from finiteblock_bem.jax_backend import contact_values

TOOLS = Path(__file__).resolve().parent
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


def contact_observer(model):
    """Whole-interface post-state fields, without another contact force call."""
    master, slave = (jnp.asarray(model[k]) for k in ["master_nodes", "slave_nodes"])
    offset = int(model["moving_offset"])
    gap = jnp.asarray(model["interface_initial_gap"])
    kn, kt = jnp.asarray(model["penalty_n"]), jnp.asarray(model["penalty_t"])

    def observe(carry):
        u, z, c, mu, strength = (carry[i] for i in [0, 2, 3, 6, 8])
        dn = u[2 * master] - u[offset + 2 * slave] - gap
        ds = u[2 * master + 1] - u[offset + 2 * slave + 1]
        pressure = kn * jnp.maximum(dn, 0)
        shear = jnp.where(dn > 0, kt * (ds - z), 0)
        return jnp.stack((dn, ds, pressure, shear, z, c, mu, strength))

    return jax.jit(observe)


def observed_scan(step, model, zero_dofs, prescribed_dofs):
    """Call the unchanged original shear step once and observe its accepted state."""
    observe = contact_observer(model)

    def update(carry, loading):
        new, row = step(carry, loading, zero_dofs, prescribed_dofs, True, False)
        return new, (row, observe(new))

    return jax.jit(lambda carry, loading: jax.lax.scan(update, carry, loading))


def radau_path_clock(start, stop, dt):
    """Global shear-time stage grid with two explicit window boundary anchors."""
    if not all(np.isfinite(a) for a in [start, stop, dt]) or not 0 <= start < stop or dt <= 0:
        raise ValueError("Require ordered finite shear times and positive projection step")
    first, last = max(0, int(np.floor(start / dt)) - 1), int(np.ceil(stop / dt)) + 1
    if last - first > 100000:
        raise ValueError("Projection clock exceeds the bounded diagnostic budget")
    interval = np.arange(first, last + 1)
    stage = ((interval[:, None] + np.array([1 / 3, 1.0])) * dt).ravel()
    return np.unique(np.r_[start, stage[(stage > start) & (stage < stop)], stop])


def interpolate_time(time, values, target):
    """P1 time evaluation, retaining the original spatial coordinates."""
    time, values, target = map(np.asarray, [time, values, target])
    if (
        time.ndim != 1
        or len(time) < 2
        or values.shape[0] != len(time)
        or target.ndim != 1
        or not len(target)
        or np.any(np.diff(time) <= 0)
        or np.any(np.diff(target) <= 0)
        or target[0] < time[0]
        or target[-1] > time[-1]
        or not all(np.all(np.isfinite(a)) for a in [time, values, target])
    ):
        raise ValueError("Require finite ordered saved-time interpolation without extrapolation")
    left = np.clip(np.searchsorted(time, target, side="right") - 1, 0, len(time) - 2)
    fraction = ((target - time[left]) / (time[left + 1] - time[left])).reshape(
        (-1,) + (1,) * (values.ndim - 1)
    )
    return (1 - fraction) * values[left] + fraction * values[left + 1]


@jax.jit
def replay_local_path(jumps, plastic, cumulative, parameters):
    """Same previous-state LSW map at every supplied jump endpoint."""

    def update(carry, jump):
        response = contact_values(jump, *carry, parameters)
        return (response[:, 2], response[:, 3]), response

    return jax.lax.scan(update, (plastic, cumulative), jumps)


@jax.jit
def replay_source_order_path(jumps, plastic, cumulative, parameters):
    """Same LSW equations in the archived FEM's scalar operation order.

    Diagnostic only. This does not change the production BEM law or rerun
    the coupled FEM. Inputs are the already rounded observed jumps.
    """
    kn, kt, mus, muk, dc = parameters

    def update(carry, jump):
        old_z, old_c = carry
        dn, ds = jump[:, 0], jump[:, 1]
        p = kn * jnp.maximum(dn, 0.0)
        active = dn > 0
        trial = kt * (ds - old_z)
        mu = jnp.maximum(muk, mus - (mus - muk) * jnp.minimum(old_c / dc, 1.0))
        strength = jnp.where(active, mu * p, 0.0)
        sliding = active & (jnp.abs(trial) > strength)
        correction = jnp.sign(trial) * jnp.maximum(jnp.abs(trial) - strength, 0.0) / kt
        increment = jnp.where(sliding, correction, 0.0)
        z, c = old_z + increment, old_c + jnp.abs(increment)
        tau = jnp.where(active, trial - kt * increment, 0.0)
        return (z, c), jnp.stack((p, tau, z, c, mu), axis=1)

    return jax.lax.scan(update, (plastic, cumulative), jumps)


def native_law_diagnostics(time, y, actual, fields):
    """Retain each failed quantity and its location instead of a generic error."""
    names = ["pressure", "shear", "plastic", "cumulative", "mu"]
    tolerances = [1e-7, 1e-7, 1e-9, 1e-9, 1e-12]
    records = {}
    for i, (name, index, tolerance) in enumerate(
        zip(names, [2, 3, 4, 5, 6], tolerances, strict=True)
    ):
        expected = fields[1:, index]
        error = abs(actual[:, :, i] - expected)
        row, column = np.unravel_index(np.argmax(error), error.shape)
        maximum = float(error[row, column])
        records[name] = dict(
            maximum_absolute_error=maximum,
            unchanged_absolute_tolerance=tolerance,
            actual_shear_time_s=float(time[row + 1]),
            contact_y_mm=float(y[column]),
            actual=float(actual[row, column, i]),
            recorded=float(expected[row, column]),
            passed=bool(np.isfinite(maximum) and maximum <= tolerance),
        )
    return dict(passed=all(v["passed"] for v in records.values()), quantities=records)


def projection_report(time, y, fields, parameters, *, steps=(2e-6, 1e-6)):
    """Measure stage-path loss at the native constitutive sampling density."""
    time, y, fields, parameters = map(np.asarray, [time, y, fields, parameters])
    if (
        fields.shape != (len(time), len(FIELDS), len(y))
        or len(y) < 2
        or np.any(np.diff(y) <= 0)
        or parameters.shape != (5,)
        or not np.all(np.isfinite(fields))
        or not np.all(np.isfinite(parameters))
        or any(not np.isfinite(h) or h <= 0 for h in steps)
    ):
        raise ValueError("Require complete finite native contact paths and five uniform parameters")
    compare = runpy.run_path(str(TOOLS / "compare_refined_fem_history.py"))[
        "compare_saved_p1_fields"
    ]
    jump = fields[:, :2].transpose(0, 2, 1)
    z, c = fields[0, 4], fields[0, 5]
    _, native = replay_local_path(jump[1:], z, c, parameters)
    native = np.asarray(native)
    native_check = native_law_diagnostics(time, y, native, fields)
    _, source_order = replay_source_order_path(jump[1:], z, c, parameters)
    source_check = native_law_diagnostics(time, y, np.asarray(source_order), fields)
    del source_order
    reports = {}
    for h in steps:
        clock = radau_path_clock(time[0], time[-1], h)
        sampled = interpolate_time(time, fields, clock)
        values = None
        if native_check["passed"]:
            projected = interpolate_time(clock, sampled[:, :2].transpose(0, 2, 1), time)
            _, values = replay_local_path(projected[1:], z, c, parameters)
            values = np.asarray(values)
            if not np.all(np.isfinite(values)):
                raise ValueError("Nonfinite projected-path local response")
        reports[str(h)] = dict(
            dt_s=h,
            global_shear_time_stage_clock_s=clock.tolist(),
            window_boundaries_added=True,
            saved_field_projection_errors={
                name: compare((time, y, fields[:, i]), (clock, y, sampled[:, i]))
                for i, name in enumerate(FIELDS)
            },
            replayed_constitutive_path_errors=None
            if values is None
            else {
                name: compare(
                    (time[1:], y, native[:, :, index]), (time[1:], y, values[:, :, index])
                )
                for name, index in [("plastic_slip", 2), ("cumulative_slip", 3)]
            },
            maximum_cumulative_slip_mm=None if values is None else float(values[-1, :, 3].max()),
            maximum_pointwise_cumulative_difference_mm=None
            if values is None
            else float(np.max(abs(values[-1, :, 3] - native[-1, :, 3]))),
            constitutive_clock="Both paths evaluated at every original native endpoint",
        )
    return dict(
        production_bem_native_path_replay=native_check,
        source_order_native_path_replay=source_check,
        constitutive_projection_accepted=native_check["passed"],
        native_maximum_cumulative_slip_mm=float(native[-1, :, 3].max()),
        projections=reports,
        actual_bem_solution_error_or_bound=False,
        scope="Projection of the observed FEM path on global Radau stage knots, retaining all spatial endpoints and native constitutive sampling. Window boundary values are explicit anchors. Diagnostic only; original full-event gates unchanged.",
    )

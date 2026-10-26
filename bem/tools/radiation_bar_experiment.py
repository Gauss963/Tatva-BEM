"""Finite shear-bar control for microstep contact and macrostep delayed waves.

Two identical bars, fixed at their remote ends, have relative-jump impedance
Z(s) = C*s*coth(s*T), C=rho*cs/2 and T=L/cs (unit area). Its complete reflected
traction is 2*C*sum(v(t-2*m*T), m>=1). A delay recurrence retains every return;
coarsening its input to interval-average velocity is an explicit approximation.
This is an anti-plane 1D control, not the finite 2D event-16 backend.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.jax_backend import contact_point


def drive(time, peak, rise):
    phase = jnp.clip(time / rise, 0, 1)
    return peak * (1 - jnp.cos(jnp.pi * phase)) / 2


def local_radiation_step(
    previous_q, plastic, cumulative, force, dt, radiation, pressure, parameters
):
    """Exact unique root of a backward-Euler step with previous-state friction."""
    kn, kt, mus, muk, dc = parameters
    mu = mus - (mus - muk) * jnp.minimum(cumulative / dc, 1)
    a = radiation / dt
    stick = (force + a * previous_q + kt * plastic) / (a + kt)
    trial = kt * (stick - plastic)
    slide = jnp.abs(trial) > pressure * mu
    sliding = previous_q + (force - jnp.sign(trial) * pressure * mu) / a
    q = jnp.where(slide, sliding, stick)
    response = contact_point(jnp.array([pressure / kn, q]), plastic, cumulative, parameters)
    residual = a * (q - previous_q) + response[1] - force
    return q, response, residual


def delay_rollout(*, micro_dt, macro_dt, duration, round_trip, radiation, pressure, peak, rise):
    substeps = round(macro_dt / micro_dt)
    steps = round(duration / macro_dt)
    delay = round(round_trip / macro_dt)
    if min(substeps, steps, delay) < 1 or not np.allclose(
        [substeps * micro_dt, steps * macro_dt, delay * macro_dt],
        [macro_dt, duration, round_trip],
        rtol=1e-12,
        atol=1e-16,
    ):
        raise ValueError("Macro/micro grid, horizon and complete round-trip delay must align")

    def rollout(parameters):
        def macro(carry, index):
            q, z, c, ring = carry
            incoming = ring[index % delay]

            def micro(state, inner):
                old_q, old_z, old_c = state
                time = (index * substeps + inner + 1) * micro_dt
                force = drive(time, peak, rise)
                new_q, response, residual = local_radiation_step(
                    old_q, old_z, old_c, force - incoming, micro_dt, radiation, pressure, parameters
                )
                values = jnp.array(
                    [
                        new_q,
                        response[2],
                        response[3],
                        response[1],
                        (new_q - old_q) / micro_dt,
                        force,
                        incoming,
                        residual,
                    ]
                )
                return (new_q, response[2], response[3]), values

            state, values = jax.lax.scan(micro, (q, z, c), jnp.arange(substeps))
            average_velocity = (state[0] - q) / macro_dt
            ring = ring.at[index % delay].set(incoming + 2 * radiation * average_velocity)
            return (*state, ring), values

        initial = (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.zeros(delay))
        _, values = jax.lax.scan(jax.checkpoint(macro), initial, jnp.arange(steps))
        return values.reshape(steps * substeps, 8)

    return jax.jit(rollout)


def fem_rollout(*, elements, dt, duration, length, shear_modulus, density, pressure, peak, rise):
    """Independent lumped-P1 volume chain, central differences, original return map."""
    dx = length / elements
    mass = np.full(elements, density * dx / 2)
    mass[0] /= 2
    stiffness = shear_modulus / (2 * dx)
    steps = round(duration / dt)
    if elements < 2 or steps < 1 or abs(steps * dt - duration) > 1e-15:
        raise ValueError("Require an aligned finite volume grid")

    def rollout(parameters):
        def step(carry, index):
            u, velocity_half, z, c, tau = carry
            differences = u - jnp.r_[u[1:], 0.0]
            internal = stiffness * (differences - jnp.r_[0.0, differences[:-1]])
            force = -internal
            force = force.at[0].add(drive(index * dt, peak, rise) - tau)
            velocity_half = velocity_half + dt * force / jnp.asarray(mass)
            u = u + dt * velocity_half
            response = contact_point(jnp.array([pressure / parameters[0], u[0]]), z, c, parameters)
            values = jnp.array([u[0], response[2], response[3], response[1]])
            return (u, velocity_half, response[2], response[3], response[1]), values

        initial = (
            jnp.zeros(elements),
            jnp.zeros(elements),
            jnp.array(0.0),
            jnp.array(0.0),
            jnp.array(0.0),
        )
        final, values = jax.lax.scan(step, initial, jnp.arange(steps))
        return values, final[0], final[1]

    return jax.jit(rollout), mass, stiffness


def reflected_traction(velocity, delay, radiation):
    """Direct image recurrence on every saved microstep, including all returns."""
    velocity = np.asarray(velocity)
    answer = np.zeros_like(velocity)
    for start in range(delay, len(velocity), delay):
        end = min(start + delay, len(velocity))
        answer[start:end] = (
            answer[start - delay : end - delay]
            + 2 * radiation * velocity[start - delay : end - delay]
        )
    return answer


def fine_delay_adjoint(parameters, states, forcing, incoming, *, dt, delay, radiation, pressure):
    """Reverse all reflected images with O(N+delay) storage and local JAX VJPs.

    g[n+delay] = g[n] + 2*C*(q[n]-q[n-1])/dt. The cotangent of this recurrence
    is propagated explicitly; no tape of every complete forward ring is needed.
    The scalar objective is the final cumulative slip on the recorded branches.
    """
    count = len(states)

    def backward(carry, index):
        state_bar, history_bar, gradient = carry
        future = history_bar[index + delay]
        current_bar = state_bar.at[0].add(2 * radiation / dt * future)
        previous = jnp.where(index > 0, states[jnp.maximum(index - 1, 0)], jnp.zeros(3))

        def point(old, history, theta):
            q, response, _ = local_radiation_step(
                old[0],
                old[1],
                old[2],
                forcing[index] - history,
                dt,
                radiation,
                pressure,
                theta,
            )
            return jnp.array([q, response[2], response[3]])

        _, pullback = jax.vjp(point, previous, incoming[index], parameters)
        old_bar, local_history_bar, parameter_bar = pullback(current_bar)
        history_bar = history_bar.at[index].set(future + local_history_bar)
        old_bar = old_bar.at[0].add(-2 * radiation / dt * future)
        return (old_bar, history_bar, gradient + parameter_bar), None

    initial = (jnp.array([0.0, 0.0, 1.0]), jnp.zeros(count + delay), jnp.zeros_like(parameters))
    result, _ = jax.lax.scan(backward, initial, jnp.arange(count - 1, -1, -1))
    return result[2]


def fine_delay_loss(**options):
    """Custom VJP for the complete fine-history terminal cumulative-slip loss."""
    rollout = delay_rollout(macro_dt=options["micro_dt"], **options)
    dt = options["micro_dt"]
    delay = round(options["round_trip"] / dt)

    @jax.custom_vjp
    def loss(parameters):
        return rollout(parameters)[-1, 2]

    def forward(parameters):
        values = rollout(parameters)
        return values[-1, 2], (parameters, values[:, :3], values[:, 5], values[:, 6])

    def reverse(saved, cotangent):
        parameters, states, forcing, incoming = saved
        gradient = fine_delay_adjoint(
            parameters,
            states,
            forcing,
            incoming,
            dt=dt,
            delay=delay,
            radiation=options["radiation"],
            pressure=options["pressure"],
        )
        return (cotangent * gradient,)

    loss.defvjp(forward, reverse)
    return jax.jit(loss), rollout


def reconstructed_bulk_energy(velocity, *, dt, round_trip, density, speed):
    """Exact cell integration for piecewise-constant boundary velocity images."""
    delay = round(round_trip / dt)
    if delay % 2:
        raise ValueError("Energy reconstruction needs the one-way transit on the microgrid")
    cells = delay // 2
    count = len(velocity)
    left, right = np.zeros(cells), np.zeros(cells)
    for image in range(count // delay + 1):
        il = count - np.arange(cells) - 1 - image * delay
        ir = count - delay + np.arange(cells) - image * delay
        for ids, values in [(il, left), (ir, right)]:
            keep = (ids >= 0) & (ids < count)
            values[keep] += velocity[ids[keep]]
    return float(density / 2 * speed * dt * np.sum(left**2 + right**2))


def timed(function, parameters):
    start = perf_counter()
    value = function(parameters)
    jax.block_until_ready(value)
    compiled = perf_counter() - start
    start = perf_counter()
    value = function(parameters)
    jax.block_until_ready(value)
    return value, dict(first_including_jit=compiled, warm_seconds=perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-dt", type=float, default=12.5e-9)
    parser.add_argument(
        "--macro-dts", type=float, nargs="+", default=[12.5e-9, 0.125e-6, 0.5e-6, 2e-6]
    )
    parser.add_argument("--fem-elements", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--duration", type=float, default=0.001)
    parser.add_argument("--ad-macro-dt", type=float, default=0.5e-6)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new finite-bar diagnostic directory")
    if any(
        not np.isfinite(v) or v <= 0
        for v in [args.micro_dt, args.duration, *args.macro_dts, args.ad_macro_dt]
    ) or any(n < 2 for n in args.fem_elements):
        raise ValueError("Require positive finite time settings and at least two volume elements")
    theta = jnp.array([76620.0, 7662.0, 0.8, 0.6, 0.0125959])
    density, shear_modulus, pressure = 1.148e-9, 7662 / 2.4, 50 / 3
    speed = (shear_modulus / density) ** 0.5
    transit = 50e-6
    length = speed * transit
    radiation = density * speed / 2
    peak, rise = 14.0, 125e-6
    options = dict(
        micro_dt=args.micro_dt,
        duration=args.duration,
        round_trip=2 * transit,
        radiation=radiation,
        pressure=pressure,
        peak=peak,
        rise=rise,
    )
    source = fingerprint(Path(__file__))
    environment = environment_record()
    args.output.mkdir(parents=True)
    traces, records = {}, []
    for h in args.macro_dts:
        fun = delay_rollout(macro_dt=h, **options)
        values, timing = timed(fun, theta)
        data = np.asarray(values)
        actual = reflected_traction(data[:, 4], round(2 * transit / args.micro_dt), radiation)
        q, z, c, tau, velocity, force, history, residual = data.T
        work = float(np.dot((force + np.r_[0, force[:-1]]) / 2, np.diff(np.r_[0, q])))
        friction = float(np.dot(tau, np.diff(np.r_[0, z])))
        interface = float(tau[-1] ** 2 / (2 * float(theta[1])))
        bulk = reconstructed_bulk_energy(
            velocity, dt=args.micro_dt, round_trip=2 * transit, density=density, speed=speed
        )
        record = dict(
            kind="exact_delay" if h == args.micro_dt else "macro_average_history",
            macro_dt_s=h,
            final_cumulative_slip_mm=float(c[-1]),
            max_jump_mm=float(np.max(q)),
            maximum_microstep_residual_mpa=float(np.max(abs(residual))),
            maximum_reflected_traction_defect_mpa=float(np.max(abs(history - actual))),
            external_work=work,
            friction_work=friction,
            interface_energy=interface,
            bulk_energy=bulk,
            mechanical_remainder=work - friction - interface - bulk,
            all_finite=bool(np.all(np.isfinite(data))),
            cumulative_monotone=bool(np.all(np.diff(c) >= -1e-14)),
            timing_seconds=timing,
        )
        if (
            not record["all_finite"]
            or not record["cumulative_monotone"]
            or record["maximum_microstep_residual_mpa"] >= 1e-7
        ):
            raise ValueError("Finite-bar contact solve failed its unchanged residual/state guard")
        key = f"delay-{h:.9g}"
        traces[key] = data
        records.append(dict(name=key, **record))
        write_json(args.output / "progress.json", dict(records=records))
    for elements in args.fem_elements:
        fun, mass, stiffness = fem_rollout(
            elements=elements,
            dt=args.micro_dt,
            duration=args.duration,
            length=length,
            shear_modulus=shear_modulus,
            density=density,
            pressure=pressure,
            peak=peak,
            rise=rise,
        )
        diagonal = np.full(elements, 2 * stiffness)
        diagonal[0] = stiffness + float(theta[1])
        bound = diagonal / mass
        off = stiffness / np.sqrt(mass[:-1] * mass[1:])
        bound[:-1] += off
        bound[1:] += off
        if args.micro_dt**2 * max(bound) >= 4:
            raise ValueError(
                "Volume reference fails the conservative central-difference stability bound"
            )
        (values, u, vhalf), timing = timed(fun, theta)
        data = np.asarray(values)
        u, vhalf = np.asarray(u), np.asarray(vhalf)
        if (
            not np.all(np.isfinite(data))
            or not np.all(np.isfinite(u))
            or not np.all(np.isfinite(vhalf))
            or np.min(np.diff(data[:, 2])) < -1e-14
        ):
            raise ValueError("Independent volume reference failed its finite/monotone-state guard")
        force = np.asarray(drive((np.arange(len(data)) + 1) * args.micro_dt, peak, rise))
        q, z, c, tau = data.T
        work = float(np.dot((force + np.r_[0, force[:-1]]) / 2, np.diff(np.r_[0, q])))
        friction = float(np.dot(tau, np.diff(np.r_[0, z])))
        differences = u - np.r_[u[1:], 0]
        internal = stiffness * (differences - np.r_[0, differences[:-1]])
        acceleration = -internal
        acceleration[0] += force[-1] - tau[-1]
        acceleration /= mass
        velocity = vhalf + 0.5 * args.micro_dt * acceleration
        bulk = float(
            0.5 * np.dot(mass, velocity**2) + 0.5 * stiffness * np.dot(differences, differences)
        )
        interface = float(tau[-1] ** 2 / (2 * float(theta[1])))
        key = f"fem-{elements}"
        traces[key] = data
        records.append(
            dict(
                name=key,
                kind="volume_reference",
                elements=elements,
                final_cumulative_slip_mm=float(c[-1]),
                max_jump_mm=float(np.max(q)),
                external_work=work,
                friction_work=friction,
                bulk_energy=bulk,
                interface_energy=interface,
                mechanical_remainder=work - friction - bulk - interface,
                all_finite=bool(np.all(np.isfinite(data))),
                cumulative_monotone=bool(np.all(np.diff(c) >= -1e-14)),
                timing_seconds=timing,
            )
        )
    target = traces[f"fem-{max(args.fem_elements)}"]
    for record in records:
        value = traces[record["name"]]
        record["jump_history_relative_l2_vs_finest_volume"] = float(
            np.linalg.norm(value[:, 0] - target[:, 0]) / np.linalg.norm(target[:, 0])
        )
        record["cumulative_history_relative_l2_vs_finest_volume"] = float(
            np.linalg.norm(value[:, 2] - target[:, 2]) / np.linalg.norm(target[:, 2])
        )
    fun = delay_rollout(macro_dt=args.ad_macro_dt, **options)
    loss = jax.jit(lambda parameters: fun(parameters)[-1, 2])
    (value, gradient), ad_time = timed(jax.jit(jax.value_and_grad(loss)), theta)
    start = perf_counter()
    forward = jax.jacfwd(loss)(theta)
    jax.block_until_ready(forward)
    forward_seconds = perf_counter() - start
    passed = bool(np.allclose(gradient, forward, rtol=1e-5, atol=1e-12))
    if not passed:
        raise ValueError("Finite-bar reverse/forward parameter check failed")
    fine_loss, fine_rollout = fine_delay_loss(**options)
    (fine_value, fine_gradient), fine_time = timed(jax.jit(jax.value_and_grad(fine_loss)), theta)

    def fine_raw_loss(p):
        return fine_rollout(p)[-1, 2]

    start = perf_counter()
    fine_forward = jax.jit(jax.jacfwd(fine_raw_loss))(theta)
    jax.block_until_ready(fine_forward)
    fine_forward_seconds = perf_counter() - start
    fine_passed = bool(np.allclose(fine_gradient, fine_forward, rtol=1e-5, atol=1e-12))
    if not fine_passed:
        raise ValueError("Complete fine-history custom VJP differs from independent forward AD")
    count = round(args.duration / args.micro_dt)
    delay = round(2 * transit / args.micro_dt)
    np.savez(
        args.output / "traces.npz", time=(np.arange(len(target)) + 1) * args.micro_dt, **traces
    )
    if fingerprint(Path(__file__)) != source:
        raise ValueError("Finite-bar diagnostic changed during simulation")
    result = dict(
        records=records,
        parameters=np.asarray(theta).tolist(),
        pressure_mpa=pressure,
        micro_dt_s=args.micro_dt,
        duration_s=args.duration,
        length_mm=length,
        one_way_transit_s=transit,
        radiation_mpa_s_per_mm=radiation,
        peak_applied_traction_mpa=peak,
        loading_rise_s=rise,
        ad=dict(
            macro_dt_s=args.ad_macro_dt,
            loss=float(value),
            reverse=np.asarray(gradient).tolist(),
            forward=np.asarray(forward).tolist(),
            max_abs_difference=float(np.max(abs(gradient - forward))),
            passed=passed,
            timing_seconds=ad_time,
            forward_check_seconds=forward_seconds,
        ),
        fine_history_ad=dict(
            loss=float(fine_value),
            reverse=np.asarray(fine_gradient).tolist(),
            forward=np.asarray(fine_forward).tolist(),
            max_abs_difference=float(np.max(abs(fine_gradient - fine_forward))),
            passed=fine_passed,
            timing_seconds=fine_time,
            forward_check_seconds=fine_forward_seconds,
            estimated_saved_primal_trace_bytes=count * 5 * 8,
            reverse_history_cotangent_bytes=(count + delay) * 8,
            naive_ring_tape_bytes=count * delay * 8,
            scope="Full fine reflected history and all five contact parameters; terminal cumulative slip at fixed pressure, material, loading and recorded branches. Array sizes are not total process RSS.",
        ),
        environment=environment,
        source_sha256=source,
        scope="Two symmetric anti-plane 1D bars with fixed remote ends, constant normal pressure and original previous-state LSW. Macro-history averaging is an approximation with its physical reflected-traction defect reported; not event-16 or a validated 2D multirate backend.",
        full_original_event_reproduced=False,
        same_accuracy_fem_advantage_verified=False,
    )
    write_json(args.output / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "environment"}, indent=2))


if __name__ == "__main__":
    main()

"""Experimental adjoint of elastic-resolvent correction histories.

Closed/sticking startup has zero primal correction, but its kn/kt derivatives
are NOT zero. A frequency-block vector transpose action supplies those prefix
cotangents without storing full-horizon matrix weights. Bulk/lift/loading and
selected branches remain fixed, as for the original conditional adjoint.
"""

import argparse
import json
import runpy
from dataclasses import replace
from functools import partial
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import irfft, rfft

from finiteblock_bem.cq import BoundaryHistory, CQPlan, OnlineConvolution, invert_spectrum, plan_cq
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.jax_backend import JaxContactEquilibrium
from finiteblock_bem.nodal import load_nodal_model
from finiteblock_bem.nodal_dynamics import nodal_cache_key
from finiteblock_bem.sampling import SamplingOptions

ROOT = Path(__file__).resolve().parents[1]


def prepared_impedance(folder, expected_sha256, dofs):
    """Use the forward solver's actual instantaneous operator in its adjoint."""
    path = Path(folder) / "initial_impedance.npy"
    if fingerprint(path) != expected_sha256:
        raise ValueError("The prepared instantaneous impedance changed")
    matrix = np.load(path, allow_pickle=False)
    if (
        matrix.shape != (dofs, dofs)
        or matrix.dtype != np.float64
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("Invalid prepared instantaneous impedance")
    return matrix


def primal_field_differences(original, candidate):
    """Check the full time grid and all seven fields before comparing gradients."""
    fields = [
        "pressure",
        "shear",
        "plastic_slip",
        "cumulative_slip",
        "actuator",
        "platen",
        "contact_jumps",
    ]
    differences = {}
    with (
        np.load(Path(original) / "trajectory.npz", allow_pickle=False) as source,
        np.load(Path(candidate) / "trajectory.npz", allow_pickle=False) as target,
    ):
        np.testing.assert_array_equal(source["time"], target["time"])
        if not np.all(np.isfinite(source["time"])):
            raise ValueError("Nonfinite trajectory time grid")
        for name in fields:
            expected, actual = source[name], target[name]
            if (
                expected.shape != actual.shape
                or not np.all(np.isfinite(expected))
                or not np.all(np.isfinite(actual))
            ):
                raise ValueError(f"Invalid trajectory field: {name}")
            differences[name] = float(np.max(abs(actual - expected)))
    return differences


def load_gradient_reference(path, original, environment, *, steps, value, nodes):
    """Reject a different objective/run before comparing saved cotangents."""
    target = json.loads(Path(path).read_text())
    for name in ["source_sha256", "case_sha256", "restart_data_sha256"]:
        if target.get("environment", {}).get(name) != environment[name]:
            raise ValueError(f"Reference gradient differs in {name}")
    if "source_run" in target:
        if Path(target["source_run"]).resolve() != Path(original).resolve():
            raise ValueError("Reference gradient belongs to a different source run")
    elif Path(path).resolve().parent != Path(original).resolve():
        raise ValueError("Reference gradient must identify its source run")
    if (
        target.get("parameter_order")
        != ["normal_penalty", "tangential_penalty", "mu_s", "mu_k", "Dc"]
        or target.get("objective") != "final contact-weighted mean cumulative slip"
    ):
        raise ValueError("Reference gradient uses different parameters or objective")
    if "steps" in target and target["steps"] != steps:
        raise ValueError("Reference gradient has a different trajectory length")
    if "value_mm" in target and not np.isclose(target["value_mm"], value, rtol=1e-12, atol=1e-12):
        raise ValueError("Reference gradient has a different objective value")
    target["gradient"] = np.asarray(target.get("gradient", target.get("reverse_gradient")))
    if target["gradient"].shape != (5,) or not np.all(np.isfinite(target["gradient"])):
        raise ValueError("Reference gradient must contain five finite parameters")
    arrays_path = Path(path).parent / "adjoint.npz"
    if arrays_path.is_file() and Path(path).parent.resolve() == Path(original).resolve():
        with np.load(arrays_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["gradient"], target["gradient"]):
                raise ValueError("Reference JSON and adjoint arrays contain different gradients")
            for name in [
                "actuator_gradient",
                "initial_plastic_slip_gradient",
                "initial_cumulative_slip_gradient",
            ]:
                target[name] = saved[name]
        target["arrays_sha256"] = fingerprint(arrays_path)
    for name, shape in [
        ("actuator_gradient", (steps + 1,)),
        ("initial_plastic_slip_gradient", (nodes,)),
        ("initial_cumulative_slip_gradient", (nodes,)),
    ]:
        if name in target:
            target[name] = np.asarray(target[name])
            if target[name].shape != shape or not np.all(np.isfinite(target[name])):
                raise ValueError(f"Invalid reference array: {name}")
    return target


def inactive_prefix_transpose(spectrum, plan, hbar, start, *, coordinate_scale=None):
    """Exact damped-DFT coefficient action on the INACTIVE prefix only.

    In reverse time all nonzero inputs precede the requested outputs. Hence
    the frequency product has no future-input circular wrap on this prefix.
    Returning just the prefix prevents using the contaminated early reverse
    outputs as substitutes for the causal post-activation adjoint.
    """
    steps, d = len(hbar) - 1, hbar.shape[1]
    if not 1 <= start <= steps or spectrum.shape != (plan.fft_size // 2 + 1, d, d):
        raise ValueError("Invalid inactive-prefix dimensions")
    if np.any(hbar[:start] != 0):
        raise ValueError("Prefix transpose requires zero pre-activation h cotangents")
    scale = np.ones(d) if coordinate_scale is None else np.asarray(coordinate_scale)
    if scale.shape != (d,) or np.any(scale <= 0):
        raise ValueError("Positive coordinate scales required")
    damping = plan.radius ** np.arange(steps + 1)
    transformed = rfft(hbar[::-1] * damping[:, None] / scale, n=plan.fft_size, axis=0)
    output = np.empty_like(transformed)
    for first in range(0, len(output), 64):
        last = min(first + 64, len(output))
        # Full frequency blocks are contiguous reads even when the desired
        # operation is transposed. No strided all-frequency column scans.
        output[first:last] = np.einsum(
            "fji,fj->fi", spectrum[first:last], transformed[first:last], optimize=False
        )
    result = irfft(output, n=plan.fft_size, axis=0)[: steps + 1]
    result /= damping[:, None] * scale
    return result[::-1][:start].copy()


def resolvent_adjoint(
    reference, history, result, spectrum, start, *, coordinate_scale=None, convolution_factory=None
):
    """Final mean-c adjoint with complete post-source and prefix derivatives."""
    tick = perf_counter()
    steps = len(result.time) - 1
    n, d = result.plastic_slip.shape[1], history.plan.dofs
    if d != 2 * n + 1 or not 1 <= start <= steps or steps - start > history.plan.steps:
        raise ValueError("Correction horizon does not cover the activated source")
    if np.any(result.plastic_slip[:start] != result.plastic_slip[0]) or np.any(
        result.cumulative_slip[:start] != result.cumulative_slip[0]
    ):
        raise ValueError("The omitted primal prefix must have unchanged contact state")
    if np.any(result.contact_jumps[:start, :n] <= 0):
        raise ValueError("The omitted primal prefix must stay closed")
    system = JaxContactEquilibrium(reference)
    wc = jnp.asarray(reference.constraints.contact_weights)
    wr = jnp.asarray(reference.constraints.reduced_weights)
    law = reference.case.contact
    trial = law.tangential_penalty * (
        result.contact_jumps[:start, n : 2 * n] - result.plastic_slip[0]
    )
    strength = (
        law.coefficient(result.cumulative_slip[:start])
        * law.normal_penalty
        * result.contact_jumps[:start, :n]
    )
    if np.any(abs(trial) > strength):
        raise ValueError("The omitted prefix must use the exact sticking branch")
    cdiag = jnp.r_[wc * law.normal_penalty, wc * law.tangential_penalty, 0]
    a0 = jnp.asarray(reference.matrix) + jnp.diag(cdiag)
    q = np.c_[
        result.contact_jumps - np.r_[reference.base_dn, reference.base_ds],
        result.platen - reference.base_platen,
    ]

    @jax.jit
    def step(q, old_z, old_c, parameters, eta, rbar):
        def local(q, z, c, p):
            response = system.local_response(q, z, c, p)
            force = jnp.r_[wc * response[:, 0], wc * response[:, 1], 0]
            # Constant Fc0 and recorded hold forcing have zero derivatives
            # with respect to these arguments.
            return cdiag * q - force, response[:, 2:4]

        _, pull = jax.vjp(local, q, old_z, old_c, parameters)
        qbar, zbar, cbar, direct = pull((rbar, eta))
        jacobian = system.tangent(q, old_z, old_c, parameters)
        multiplier = jnp.linalg.solve(jacobian.T, qbar)

        def equation(z, c, p):
            return system.residual(q, z, c, p, jnp.zeros_like(q))

        _, root_pull = jax.vjp(equation, old_z, old_c, parameters)
        dz, dc, dp = root_pull(-multiplier)
        raw = multiplier / wr
        return a0.T @ raw, jnp.stack((zbar + dz, cbar + dc), axis=1), direct + dp, raw

    factory = OnlineConvolution if convolution_factory is None else convolution_factory
    reverse = factory(BoundaryHistory(history.plan, history.weights.transpose(0, 2, 1)))
    eta = np.c_[np.zeros(n), np.asarray(wc) / np.sum(wc)]
    gradient = np.zeros(5)
    actuator_gradient = np.zeros(steps + 1)
    hbar = np.zeros((steps + 1, d))
    for offset, index in enumerate(range(steps, start - 1, -1)):
        rbar = reverse.past(offset)
        h, eta, increment, raw = step(
            jnp.asarray(q[index]),
            jnp.asarray(result.plastic_slip[index - 1]),
            jnp.asarray(result.cumulative_slip[index - 1]),
            system.parameters,
            jnp.asarray(eta),
            jnp.asarray(rbar),
        )
        h, eta, increment, raw = (np.asarray(value) for value in [h, eta, increment, raw])
        if not all(np.all(np.isfinite(value)) for value in [h, eta, increment, raw]):
            raise ValueError(f"Nonfinite resolvent adjoint at {index}")
        hbar[index] = h
        gradient += increment
        actuator_gradient[index] = reference.spring * (rbar[-1] + raw[-1])
        reverse.append(h)
    cache_statistics = reverse.statistics() if hasattr(reverse, "statistics") else None
    del reverse
    eta = eta.copy()
    prefix = inactive_prefix_transpose(
        spectrum, history.plan, hbar, start, coordinate_scale=coordinate_scale
    )
    prior = prefix[1:start]
    old_z = result.plastic_slip[0]
    prefix_gradient = np.zeros(5)
    prefix_gradient[0] = -np.sum(prior[:, :n] * np.asarray(wc) * result.contact_jumps[1:start, :n])
    prefix_gradient[1] = -np.sum(
        prior[:, n : 2 * n] * np.asarray(wc) * (result.contact_jumps[1:start, n : 2 * n] - old_z)
    )
    gradient += prefix_gradient
    eta[:, 0] += law.tangential_penalty * np.asarray(wc) * np.sum(prior[:, n : 2 * n], axis=0)
    actuator_gradient[1:start] = reference.spring * prior[:, -1]
    return dict(
        objective="final contact-weighted mean cumulative slip",
        value_mm=float(np.average(result.cumulative_slip[-1], weights=np.asarray(wc))),
        parameter_order=["normal_penalty", "tangential_penalty", "mu_s", "mu_k", "Dc"],
        gradient=gradient,
        prefix_parameter_gradient=prefix_gradient,
        actuator_gradient=actuator_gradient,
        initial_plastic_slip_gradient=eta[:, 0],
        initial_cumulative_slip_gradient=eta[:, 1],
        seconds_including_jit=perf_counter() - tick,
        correction_start=start,
        scope="Fixed bulk/lift/recorded loading and active branches; complete correction transpose plus affine startup cotangents",
        event_time_derivative_included=False,
        online_fft_cache=cache_statistics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path, help="Original Z run, or standalone resolvent run")
    parser.add_argument("replay", type=Path, nargs="?", help="Omit for a standalone resolvent run")
    parser.add_argument("--reference-gradient", type=Path)
    parser.add_argument("--max-memory-mb", type=float, default=12288)
    parser.add_argument("--fft-cache-mb", type=float, help="Cap retained transpose kernel FFTs")
    parser.add_argument("--fft-rows", type=int, default=4)
    args = parser.parse_args()
    if not np.isfinite(args.max_memory_mb) or args.max_memory_mb <= 0:
        raise ValueError("A finite positive memory budget is required")
    validator_sha = fingerprint(Path(__file__))
    standalone = args.replay is None
    if standalone:
        args.replay = args.original
    original = json.loads((args.original / "summary.json").read_text())
    replay = json.loads((args.replay / "summary.json").read_text())
    options = json.loads((args.original / "run_manifest.json").read_text())["arguments"]
    environment = environment_record(options["case"])
    for record in [original, replay]:
        for name in ["source_sha256", "case_sha256", "restart_data_sha256"]:
            if record["environment"][name] != environment[name]:
                raise ValueError(f"Saved run differs in {name}")
    if (
        original["status"] != "completed_requested_horizon"
        or original["source_changed_during_run"]
        or replay.get(
            "maximum_residual_mpa", replay.get("max_boundary_contact_residual_mpa", float("inf"))
        )
        >= 1e-7
    ):
        raise ValueError("A complete converged source/replay is required")
    if standalone and original.get("history_representation") != "elastic_resolvent":
        raise ValueError("Standalone adjoint requires a saved elastic-resolvent trajectory")
    differences = None if standalone else primal_field_differences(args.original, args.replay)
    steps, d, dt = original["completed_steps"], original["history_dofs"][0], original["dt_s"]
    correction_steps = replay["correction_history_steps"]
    plan = plan_cq(
        d,
        dt,
        correction_steps,
        max_memory_mb=args.max_memory_mb if args.fft_cache_mb is None else 1e9,
        disk_spectrum=True,
    )
    # Additional source/output vectors and frequency-block copies.
    extra = 8 * (steps + 1) * d * 8 + 2 * 64 * d * d * 16 + 512 * 1024**2
    estimate = plan.estimated_peak_bytes + extra
    if (
        args.fft_cache_mb is None
        and plan.estimated_peak_bytes + extra > args.max_memory_mb * 1024**2
    ):
        raise ValueError("Resolvent adjoint arrays exceed requested memory budget")
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    helper = runpy.run_path(str(Path(__file__).with_name("replay_resolvent.py")))
    prepared_path = replay.get("prepared_kernel")
    coordinate_scale = None
    if prepared_path is not None:
        prepared_path = Path(prepared_path)
        metadata = json.loads((prepared_path / "summary.json").read_text())
        if metadata["status"] != "completed_resolvent_kernel_only":
            raise ValueError("A completed prepared kernel is required")
        if (metadata["dt_s"], metadata["full_steps"], metadata["history_dofs"]) != (dt, steps, d):
            raise ValueError("Prepared kernel uses a different time grid or contact trace")
        for name in ["source_sha256", "case_sha256", "restart_data_sha256"]:
            if metadata["environment"][name] != environment[name]:
                raise ValueError(f"Prepared kernel differs in {name}")
        plan = CQPlan(**metadata["plan"])
        if helper["digest_file"](prepared_path / "weights.npy") != metadata["weights_sha256"]:
            raise ValueError("Prepared correction weights differ")
        history = BoundaryHistory(
            plan, np.load(prepared_path / "weights.npy", mmap_mode="r", allow_pickle=False)
        )
        coordinate_scale = np.sqrt(
            np.r_[
                model.constraints.contact_weights,
                model.constraints.contact_weights,
                model.constraints.platen_width,
            ]
        )
    else:
        source_plan = CQPlan(**original["spectrum_storage"]["cq_plan"])
        plan = replace(plan, fft_size=source_plan.fft_size, radius=source_plan.radius)
    bounded_path = Path(__file__).with_name("bounded_history.py")
    bounded_sha, convolution_factory = None, None
    if args.fft_cache_mb is not None:
        bounded_sha = fingerprint(bounded_path)
        bounded = runpy.run_path(str(bounded_path))
        estimate = bounded["estimate_online_arrays"](
            plan, steps, args.fft_cache_mb, fft_rows=args.fft_rows
        )
        estimate += 3 * plan.fft_size * d * 16 + 2 * 64 * d * d * 16
        if prepared_path is None:
            # Legacy replays generate their complete prefix in RAM. Only
            # prepared kernels load time weights from a read-only file map.
            estimate += plan.weight_bytes
        if estimate > args.max_memory_mb * 1024**2:
            raise ValueError(f"Bounded adjoint arrays need about {estimate / 1024**2:.1f} MiB")
        convolution_factory = partial(
            bounded["BoundedOnlineConvolution"], cache_mb=args.fft_cache_mb, fft_rows=args.fft_rows
        )
    # Both prepared standalone runs and replays can be checked against an
    # independent original-Z trajectory; they use different saved field names.
    storage = (
        replay["resolvent_storage"] if "resolvent_storage" in replay else replay["spectrum_storage"]
    )
    if storage["backend"] == "polynomial_half_spectrum":
        folder = Path(storage["path"])
        if (
            not storage["all_frequencies_represented"]
            or fingerprint(folder / "summary.json") != storage["sha256"]
        ):
            raise ValueError("The complete compressed spectrum is required")
        compressed = runpy.run_path(str(Path(__file__).with_name("polynomial_spectrum.py")))
        spectrum = compressed["PolynomialSpectrum"](folder)
        if spectrum.shape != (plan.fft_size // 2 + 1, d, d):
            raise ValueError("Compressed spectrum contour dimensions differ")
        saved_plan = spectrum.metadata["provenance"]["plan"]
        if any(
            saved_plan[name] != getattr(plan, name) for name in ["dt", "dofs", "fft_size", "radius"]
        ):
            raise ValueError("Compressed spectrum uses a different CQ contour")
    else:
        if (
            not storage["all_frequency_matrices_written"]
            or helper["digest_file"](storage["path"]) != storage["sha256"]
        ):
            raise ValueError("The complete resolvent spectrum is required")
        spectrum = helper["SavedDiskSpectrum"](storage["path"], plan)
    if prepared_path is None:
        history = BoundaryHistory(plan, invert_spectrum(spectrum, plan))
    if prepared_path is not None:
        matrix = prepared_impedance(prepared_path, replay["initial_impedance_sha256"], d)
    else:
        if standalone:
            raise ValueError("Standalone adjoint requires its prepared kernel")
        original_plan = plan_cq(d, dt, steps, max_memory_mb=65536, disk_spectrum=True)
        sampling = (
            SamplingOptions(relative_tolerance=options["sampling_rtol"])
            if options["sampling"] == "adaptive"
            else None
        )
        cache = Path(options["cache"])
        if not cache.is_absolute():
            cache = ROOT.parent / cache
        cache /= (
            f"galerkin-{nodal_cache_key(model, original_plan, options['quadrature'], sampling)}.npy"
        )
        if not cache.is_file() or not cache.with_suffix(".json").is_file():
            raise ValueError("Matching original instantaneous Z coefficient required")
        mapped = np.load(cache, mmap_mode="r", allow_pickle=False)
        matrix = np.array(mapped[0], copy=True)
        del mapped
    spring = model.reference.spring_stiffness * model.constraints.platen_width
    matrix[-1, -1] += spring
    fields = ["time", "contact_jumps", "platen", "plastic_slip", "cumulative_slip"]
    with np.load(args.original / "trajectory.npz", allow_pickle=False) as data:
        source = SimpleNamespace(**{name: data[name] for name in fields})
    with np.load(args.replay / "trajectory.npz", allow_pickle=False) as data:
        result = SimpleNamespace(**{name: data[name] for name in fields})
    np.testing.assert_array_equal(source.time, result.time)
    n = (d - 1) // 2
    eye = np.eye(d)
    wc = model.constraints.contact_weights
    constraints = SimpleNamespace(
        size=d,
        normal_jump=eye[:n],
        tangential_jump=eye[n : 2 * n],
        contact_weights=wc,
        reduced_weights=np.r_[wc, wc, model.constraints.platen_width],
    )
    reference = SimpleNamespace(
        case=model.contact_case(),
        constraints=constraints,
        base_dn=source.contact_jumps[0, :n],
        base_ds=source.contact_jumps[0, n:],
        base_platen=source.platen[0],
        matrix=matrix,
        spring=spring,
    )
    reference_path = args.reference_gradient
    if not standalone and reference_path is None:
        reference_path = args.original / "adjoint.json"
    target = (
        None
        if reference_path is None
        else load_gradient_reference(
            reference_path,
            args.original,
            environment,
            steps=steps,
            value=float(np.average(source.cumulative_slip[-1], weights=wc)),
            nodes=n,
        )
    )
    branch_function = runpy.run_path(str(Path(__file__).with_name("run_adjoint.py")))[
        "exact_contact_branches"
    ]
    source_branches = branch_function(model.reference, source)
    branch_changes = (
        None
        if standalone
        else {
            name: int(np.count_nonzero(value != source_branches[name]))
            for name, value in branch_function(model.reference, result).items()
        }
    )
    answer = resolvent_adjoint(
        reference,
        history,
        result,
        spectrum,
        replay["first_correction_step"],
        coordinate_scale=coordinate_scale,
        convolution_factory=convolution_factory,
    )
    if bounded_sha is not None and bounded_sha != fingerprint(bounded_path):
        raise ValueError("Bounded history changed during adjoint evaluation")
    if validator_sha != fingerprint(Path(__file__)):
        raise ValueError("Resolvent adjoint implementation changed during evaluation")
    comparisons, checks = {}, []
    if target is not None:
        target_gradient = target["gradient"]
        delta = answer["gradient"] - target_gradient
        comparisons.update(
            gradient_max_abs_difference=float(np.max(abs(delta))),
            gradient_relative_difference=(
                abs(delta) / np.maximum(abs(target_gradient), 1e-15)
            ).tolist(),
        )
        checks.append((answer["gradient"], target_gradient))
    for name in [
        "actuator_gradient",
        "initial_plastic_slip_gradient",
        "initial_cumulative_slip_gradient",
    ]:
        if target is not None and name in target:
            target_value = np.asarray(target[name])
            comparisons[f"{name}_max_abs_difference"] = float(
                np.max(abs(answer[name] - target_value))
            )
            checks.append((answer[name], target_value))
    serializable = {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in answer.items()
    }
    record = dict(
        **serializable,
        comparisons=comparisons,
        maximum_absolute_field_difference=differences,
        strict_field_comparison_passed=None
        if differences is None
        else max(differences.values()) <= 1e-5,
        field_absolute_tolerance=1e-5,
        instantaneous_operator_source="prepared_kernel"
        if prepared_path is not None
        else "original_Z",
        exact_branch_changes=branch_changes,
        environment=environment,
        validator_sha256=validator_sha,
        bounded_history_sha256=bounded_sha,
        fft_cache_mb=args.fft_cache_mb,
        estimated_solver_array_peak_mib=estimate / 1024**2,
        reference_gradient_path=None if reference_path is None else str(reference_path.resolve()),
        reference_gradient_sha256=None if reference_path is None else fingerprint(reference_path),
        reference_arrays_sha256=None if target is None else target.get("arrays_sha256"),
        independent_reference_compared=target is not None,
        source_run=str(args.replay.resolve()),
        source_trajectory_sha256=fingerprint(args.replay / "trajectory.npz"),
        steps=steps,
    )
    write_json(args.replay / "resolvent_adjoint.json", record)
    print(
        json.dumps(
            {
                name: value
                for name, value in record.items()
                if name not in ["actuator_gradient", "environment"]
            },
            indent=2,
        )
    )
    if branch_changes is not None and any(branch_changes.values()):
        raise ValueError(
            "Replay changed a contact branch; conditional gradients cannot be directly compared"
        )
    for actual, expected in checks:
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-12)
    if differences is not None and max(differences.values()) > 1e-5:
        raise ValueError("Resolvent candidate differs from original fields by more than 1e-5")


if __name__ == "__main__":
    main()

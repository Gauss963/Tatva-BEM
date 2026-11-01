"""Audit frozen conservative FEM modes against saved native contact spectra.

Reuses the original setup and its sparse colored-AD bulk stiffness assembly.
No FEM trajectory is replayed, filtered, or substituted for acceptance data.
The frozen closed-stick control is not the sliding friction Jacobian.
"""

import argparse
import json
import runpy
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from finiteblock_bem.io import environment_record, write_json

TOOLS = Path(__file__).resolve().parent
F = runpy.run_path(str(TOOLS / "fem_modal_resolution.py"))
REPLAY = runpy.run_path(str(TOOLS / "replay_fem_native_contact.py"))
EXT, WINDOW, DIGEST = (REPLAY[k] for k in ["EXT", "WINDOW", "DIGEST"])


def capture_setup(manifest, output, count):
    """Observe matrices already assembled by setup; return original objects unchanged."""
    files = manifest["frozen_inputs_sha256"]
    fem = next(Path(p).parent for p in files if p.endswith("reference_manifest.json"))
    checkpoint = next(Path(p) for p in files if p.endswith("checkpoint.npz"))
    reference = json.loads((fem / "reference_manifest.json").read_text())
    sys.path.insert(0, reference["source_snapshot_path"])
    import jax
    from tatva.pmma import static_normal

    jax.config.update("jax_enable_x64", True)
    original, captured = static_normal._block_stiffness, []

    def observed(block, material):
        matrix = original(block, material)
        captured.append(matrix.copy())
        return matrix

    static_normal._block_stiffness = observed
    try:
        context = EXT["setup_prefix"](
            WINDOW["main"],
            SimpleNamespace(
                fem=fem,
                checkpoint=checkpoint,
                output=output / "setup",
                original_steps=count,
                factors=[1, 2, 4],
            ),
        )
    finally:
        static_normal._block_stiffness = original
    if len(captured) != 2 or DIGEST(output / "setup/manifest.json") != manifest["context_sha256"]:
        raise ValueError("Original two-block assembly/setup identity changed")
    return context, sparse.block_diag(captured, format="csr")


def audit_hessian(local, model, transform, stiffness, closed, spring, directions=3):
    """Independent forward-over-reverse products of original bulk energy plus control energy."""
    import jax
    import jax.numpy as jnp

    coordinate = transform.tocoo()
    rows, columns = jnp.asarray(coordinate.row), jnp.asarray(coordinate.col)
    master, slave = (jnp.asarray(model[k]) for k in ["master_nodes", "slave_nodes"])
    offset, total = (int(model[k]) for k in ["moving_offset", "total_dofs"])
    weights = jnp.asarray(np.asarray(model["interface_weights"]) * closed)
    kn, kt = model["penalty_n"], model["penalty_t"]

    def energy(q):
        u = jnp.zeros(total, dtype=q.dtype).at[rows].add(q[columns])
        normal = u[2 * master] - u[offset + 2 * slave]
        tangent = u[2 * master + 1] - u[offset + 2 * slave + 1]
        return (
            local["elastic_energy_with_extension"](u)[0]
            + 0.5 * jnp.sum(weights * (kn * normal**2 + kt * tangent**2))
            + 0.5 * spring * q[-1] ** 2
        )

    zero = jnp.zeros(transform.shape[1], dtype=jnp.float64)
    product = jax.jit(lambda v: jax.jvp(jax.grad(energy), (zero,), (v,))[1])
    errors = []
    rng = np.random.default_rng(811)
    for _ in range(directions):
        v = rng.normal(size=len(zero))
        actual = np.asarray(product(jnp.asarray(v)))
        expected = stiffness @ v
        errors.append(float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)))
    if not np.all(np.isfinite(errors)) or max(errors) > 1e-10:
        raise ValueError("Sparse frozen Hessian disagrees with independent original-energy AD")
    return dict(
        directions=directions,
        relative_l2_errors=errors,
        tolerance=1e-10,
        full_dense_hessian_constructed=False,
    )


def diagnose_state(context, bulk, transform, mass, initial, native_path, output):
    model, local = context["model"], context["base"]
    with np.load(native_path, allow_pickle=False) as data:
        time, y, normal = data["time_s"], data["contact_y"], data["fields"][:, 0, :].copy()
    frequency, power, spectrum = F["native_normal_jump_spectrum"](time, y, normal)
    jump_n, jump_t = F["interface_jump_matrices"](
        int(model["total_dofs"]),
        model["master_nodes"],
        model["slave_nodes"],
        int(model["moving_offset"]),
    )
    u = initial["carry_0"]
    dn = jump_n @ u - np.asarray(model["interface_initial_gap"])
    error = float(np.max(abs(dn - normal[0])))
    if error > 1e-16:
        raise ValueError("Accepted frozen state differs from native path initial normal jump")
    closed = dn > 0
    weights = np.asarray(model["interface_weights"])
    contact = (
        jump_n.T @ sparse.diags(weights * np.asarray(model["penalty_n"]) * closed) @ jump_n
        + jump_t.T @ sparse.diags(weights * np.asarray(model["penalty_t"]) * closed) @ jump_t
    )
    face = np.asarray(model["moving_shear_loading_dofs"], dtype=int)
    spring_gap = float(initial["carry_11"] - np.mean(u[face]))
    spring = float(model["shear_loading_stiffness"]) * float(np.sum(model["force_shear_unit"]))
    spring = spring if spring_gap > 0 else 0.0
    stiffness = (transform.T @ (bulk + contact) @ transform).tocsr()
    stiffness[-1, -1] += spring
    audit = audit_hessian(local, model, transform, stiffness, closed, spring)
    values, vectors, modes = F["largest_elastic_modes"](stiffness, mass)
    omega = np.sqrt(values)
    duration = (context["last"] - context["first"]) * context["old_dt"]
    factors = [1, 2, 4, 8, 16]

    def phase(w):
        return [
            dict(
                factor=factor,
                dt_s=context["old_dt"] / factor,
                **F["leapfrog_phase_control"](w, context["old_dt"] / factor, duration),
            )
            for factor in factors
        ]

    modal_phase = [
        dict(
            frequency_hz=float(w / (2 * np.pi)),
            scalar_leapfrog_critical_dt_s=float(2 / w),
            steps=phase(w),
        )
        for w in omega
    ]
    peak_phase = phase(2 * np.pi * spectrum["peak_frequency_hz"])
    cumulative = np.cumsum(power) / np.sum(power)
    spectrum["power_quantile_frequency_hz"] = {
        str(q): float(frequency[np.searchsorted(cumulative, q)]) for q in [0.5, 0.9, 0.99]
    }
    spectrum["fraction_sampled_power_above_hz"] = {
        str(f): float(np.sum(power[frequency > f]) / np.sum(power))
        for f in [250000, 500000, 1000000, 10000000]
    }
    # Save full spectrum and six modal contact traces for independent inspection.
    full_modes = transform @ vectors
    np.savez_compressed(
        output,
        frequency_hz=frequency,
        sampled_normal_jump_power=power,
        eigenvalues_rad2_per_s2=values,
        contact_y=y,
        normal_modal_trace=jump_n @ full_modes,
        tangent_modal_trace=jump_t @ full_modes,
    )
    return dict(
        native_input=str(native_path),
        native_input_sha256=DIGEST(native_path),
        initial_normal_jump_maximum_absolute_error_mm=error,
        closed_contact_nodes=int(np.sum(closed)),
        total_contact_nodes=len(closed),
        spring_gap_mm=spring_gap,
        original_length_unit="mm",
        time_unit="s",
        frozen_spring_stiffness_per_thickness=spring,
        full_dofs=len(u),
        constrained_dofs=transform.shape[1],
        frozen_stiffness_nnz=stiffness.nnz,
        energy_hessian_product_audit=audit,
        modes=modes,
        modal_phase_controls=modal_phase,
        native_spectrum=spectrum,
        native_peak_phase_control=peak_phase,
        phase_control_duration_s=duration,
        arrays=str(output),
        arrays_sha256=DIGEST(output),
        actual_sliding_jacobian_or_nonlinear_stability_proof=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extension, native, output = (p.resolve() for p in [args.extension, args.native, args.output])
    if (
        output.exists()
        or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve())
        or output.is_relative_to(extension)
        or output.is_relative_to(native)
    ):
        raise ValueError("Require fresh owned diagnostic output outside protected/input folders")
    begun = perf_counter()
    compare = runpy.run_path(str(TOOLS / "compare_extended_fem_windows.py"))
    summary, manifest, files = compare["verified_extension"](extension)
    recorded = json.loads((native / "summary.json").read_text())
    if recorded["status"] != "completed_bitwise_fem_native_contact_observations":
        raise ValueError("Require completed original native contact observations")
    records = [r for r in recorded["results"] if r["index"] in [55, 70]]
    if [r["index"] for r in records] != [55, 70]:
        raise ValueError("Keep both complete saved before/after hold windows")
    frozen = dict(manifest["frozen_inputs_sha256"])
    for path in [
        Path(__file__),
        TOOLS / "fem_modal_resolution.py",
        native / "summary.json",
        *files,
    ]:
        frozen[str(path.resolve())] = DIGEST(path)
    for record in records:
        path = native / f"native-chunk-{record['index']:05}.npz"
        if (
            DIGEST(path) != record["output_sha256"]
            or not record["every_native_scalar_row_bitwise"]
            or not all(record["all_final_carry_arrays_bitwise"])
            or not all(record["canonical_endpoint_fields_bitwise"].values())
        ):
            raise ValueError("Native observation provenance or bitwise control failed")
        frozen[str(path)] = record["output_sha256"]
    environment = environment_record()
    output.mkdir(parents=True)
    write_json(
        output / "manifest.json",
        dict(
            frozen_inputs_sha256=frozen,
            environment=environment,
            frozen_closed_stick_control_only=True,
            acceptance_fields_filtered=False,
            replayed_fine_steps=0,
        ),
    )
    context = capture_setup(
        manifest, output, summary["last_shear_step"] - summary["first_shear_step"]
    )
    context, bulk = context
    model, local = context["model"], context["base"]
    if (
        model["friction_law"] != "slip-weakening"
        or not model["shear_spring_rigid_face"]
        or model["shear_loading_mode"] != "spring-displacement"
    ):
        raise ValueError("Require original LSW rigid spring-loaded model")
    fixed = np.union1d(local["shear_phase_zero_dofs"], local["prescribed_dofs"])
    transform, mass = F["constrained_mass_map"](
        model["mass_flat"], fixed, model["moving_shear_loading_dofs"]
    )
    results = []
    for record in records:
        index = record["index"]
        prior = extension / "factor16-chunks" / f"chunk-{index - 1:05}.npz"
        current = prior.with_name(f"chunk-{index:05}.npz")
        initial, initial_meta = REPLAY["read_chunk"](prior)
        _, meta = REPLAY["read_chunk"](current)
        if (
            initial_meta["last"] != meta["first"]
            or meta["previous_sha256"] != DIGEST(prior)
            or meta["context_sha256"] != DIGEST(extension / "extension_manifest.json")
            or initial_meta["context_sha256"] != meta["context_sha256"]
        ):
            raise ValueError("Frozen accepted states do not share original native context")
        write_json(output / "progress.json", dict(phase="frozen_modal_diagnostic", index=index))
        result = diagnose_state(
            context,
            bulk,
            transform,
            mass,
            initial,
            native / f"native-chunk-{index:05}.npz",
            output / f"spectrum-modes-{index:05}.npz",
        )
        results.append(dict(index=index, **result))
        write_json(output / "completed_states.json", dict(results=results))
    if any(DIGEST(p) != value for p, value in frozen.items()):
        raise ValueError("Frozen sources or native observations changed")
    write_json(
        output / "summary.json",
        dict(
            status="completed_frozen_elastic_and_native_spectral_controls",
            results=results,
            source_inputs_unchanged=True,
            full_event_convergence_or_bem_advantage=False,
            seconds=perf_counter() - begun,
            scope="Frozen conservative closed-stick elastic control; not a bound on sliding LSW stability, a causal explanation, or altered acceptance fields.",
        ),
    )


if __name__ == "__main__":
    main()

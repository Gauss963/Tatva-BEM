"""Reuse completed original/mixed kernels for full coarse sliding and initialized AD.

This is the existing 128-step accelerated control, with its assembly prefix
replaced by verified saved arrays. The actual saved time weights drive both
primals and derivatives. No physical contour modes are evaluated anew.
"""

import argparse
import ast
import inspect
import json
import runpy
import textwrap
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import CQPlan, cq_plan_signature
from finiteblock_bem.io import environment_record, fingerprint, write_json

TOOLS = Path(__file__).resolve().parent
CONTROL = runpy.run_path(str(TOOLS / "check_paired_ray_nonlinear.py"))
RADAU = runpy.run_path(str(TOOLS / "radau_experiment.py"))


def verified_arrays(folder, plan, line, environment):
    """Resolve only owned local names, check hashes and independently invert spectra."""
    folder = Path(folder)
    metadata = json.loads((folder / "summary.json").read_text())
    if (
        metadata["status"] != "completed_compact_radau_resolvent_kernel_only"
        or cq_plan_signature(CQPlan(**metadata["plan"])) != cq_plan_signature(plan)
        or metadata["full_steps"] != plan.steps
        or metadata["correction_steps"] != plan.steps
        or metadata["family_spectra_coordinates"] != "diag(sqrt(w)) * L * diag(sqrt(w))"
        or any(
            metadata["environment"][k] != environment[k]
            for k in ["source_sha256", "case_sha256", "restart_data_sha256", "numpy", "scipy"]
        )
    ):
        raise ValueError("Completed cached physical problem, plan or runtime differs")
    records = metadata["family_spectra"], metadata["family_weights"]
    if any(len(group) != 2 for group in records):
        raise ValueError("Both complete compact families are required")
    filenames = []
    for group, names in zip(
        records,
        [[f"compact-family{i}.npy" for i in range(2)], [f"weights{i}.npy" for i in range(2)]],
        strict=True,
    ):
        for record, name in zip(group, names, strict=True):
            if Path(record["path"]).name != name or fingerprint(folder / name) != record["sha256"]:
                raise ValueError("Saved compact spectrum or time-weight bytes changed")
            filenames.append(name)
    for name, key in [
        ("shift_wrap.npy", "shift_wrap_sha256"),
        ("initial_impedance.npy", "initial_impedance_sha256"),
        ("elastic.npz", "elastic_sha256"),
    ]:
        if fingerprint(folder / name) != metadata[key]:
            raise ValueError("Saved startup or lag-zero bytes changed")
    arrays = {name: np.load(folder / name, allow_pickle=False) for name in filenames}
    spectra, weights = (
        [arrays[f"compact-family{i}.npy"] for i in range(2)],
        [arrays[f"weights{i}.npy"] for i in range(2)],
    )
    for values in spectra:
        if (
            values.shape != (plan.fft_size // 2 + 1, plan.dofs, plan.dofs)
            or values.dtype != np.complex128
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Require finite complete complex128 cached spectra")
    for values in weights:
        if (
            values.shape != (plan.steps + 1, plan.dofs, plan.dofs)
            or values.dtype != np.float64
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Require finite complete float64 cached weights")
    scale = np.sqrt(line[:, None] * line[None, :])
    physical = [values / scale for values in spectra]
    wrap = np.load(folder / "shift_wrap.npy", allow_pickle=False)
    impedance = np.load(folder / "initial_impedance.npy", allow_pickle=False)
    if (
        wrap.shape != (plan.dofs, plan.dofs)
        or impedance.shape != (2 * plan.dofs, 2 * plan.dofs)
        or not np.all(np.isfinite(wrap))
        or not np.all(np.isfinite(impedance))
    ):
        raise ValueError("Invalid cached physical lag-zero arrays")
    independent = RADAU["invert_compact_spectrum"](*physical, plan)
    for actual, expected in zip([*weights, wrap], independent, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
    return metadata, physical, (*weights, wrap), impedance


class CachedInputs:
    def __init__(self, original, mixed, environment):
        self.folders = dict(original=original, mixed=mixed)
        self.environment = environment
        self.metadata = {}

    def load(self, transfer, plan):
        spectra, impedances, weights, records = {}, {}, {}, {}
        line = np.r_[
            transfer.constraints.contact_weights,
            transfer.constraints.contact_weights,
            transfer.constraints.platen_width,
        ]
        for name, folder in self.folders.items():
            metadata, spectra[name], weights[name], impedances[name] = verified_arrays(
                folder, plan, line, self.environment
            )
            sources = metadata["frozen_inputs_sha256"]
            required = str((TOOLS / "prepare_raw_radau_resolvent.py").resolve())
            numerical_sources = {
                path: digest
                for path, digest in sources.items()
                if Path(path).is_relative_to(TOOLS)
                or Path(path).is_relative_to(TOOLS.parent / "src")
            }
            if (
                required not in numerical_sources
                or any(fingerprint(path) != digest for path, digest in numerical_sources.items())
                or metadata["durable_construction"]["attempts"][-1]["status"] != "completed"
                or metadata["numerical_discretization_overrides"]
            ):
                raise ValueError(
                    "Cached numerical source, completed attempt or discretization changed"
                )
            if metadata["source_mesh_arguments"] != dict(
                case=str(CONTROL["CASE"]),
                h=160,
                contact_h=160,
                contact_end_h=160,
                corner_h=160,
                quadrature=12,
                restart_knot_window=None,
                restart_knot_subdivisions=1,
                diagnostic_actuator_increment=0.5,
            ):
                raise ValueError("Reuse exactly the saved original 128-step coarse control")
            self.metadata[name] = metadata
            records[name] = dict(
                folder=str(folder),
                summary_sha256=fingerprint(folder / "summary.json"),
                actual_saved_time_weights_used=True,
                unchanged_cached_numerical_repo_sources=len(numerical_sources),
                independent_spectral_inversion_tolerance=dict(rtol=1e-8, atol=1e-12),
            )
        np.testing.assert_array_equal(impedances["original"], impedances["mixed"])
        backend = runpy.run_path(str(TOOLS / "prepare_jax_radau_resolvent.py"))
        backend["verify_context"](self.folders["mixed"])
        if not self.metadata["mixed"]["jax_backend"]["original_startup_impedance_uses_scipy"]:
            raise ValueError("Require the verified original lag-zero startup policy")

        # Any accidentally retained frequency-assembly call fails. Static
        # preload assembly uses its original, separate geometry/energy route.
        def forbidden(*args, **kwargs):
            raise AssertionError("Cached nonlinear control attempted frequency assembly")

        transfer.evaluate = forbidden
        for prepared in transfer.prepared:
            prepared.assemble = forbidden
        return spectra, impedances, weights, records

    def verify_startup(self, name, elastic, clock, ramp):
        with np.load(self.folders[name] / "elastic.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(clock, data["stage_time"])
            np.testing.assert_allclose(elastic, 4 * data["elastic"], rtol=1e-8, atol=1e-12)
            np.testing.assert_allclose(
                ramp - ramp[0],
                4 * (data["uninterrupted_actuator"] - data["uninterrupted_actuator"][0]),
                rtol=0,
                atol=1e-12,
            )


def cached_experiment(inputs):
    """Keep original static/primal/implicit AD code; replace only kernel acquisition."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(CONTROL["experiment"])))
    body = tree.body[0].body
    assignments = {
        ast.unparse(n.targets[0]): i
        for i, n in enumerate(body)
        if isinstance(n, ast.Assign) and len(n.targets) == 1
    }
    first, last = assignments["factory"], assignments["y"]
    if not first < assignments["spectra"] < assignments["impedances0"] < last:
        raise ValueError("Original assembly/control boundary changed")
    body[first:last] = ast.parse(
        "spectra, impedances0, cached_weights, cached_report = _load_cached(transfer, plan)\n"
        "contour_seconds = perf_counter() - begun\nraw_max = schur_max = None"
    ).body
    replaced = startup = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if ast.unparse(node.targets[0]) == "(w0, w1, wrap)":
                if ast.unparse(node.value) != "rk['invert_compact_spectrum'](*families, plan)":
                    raise ValueError("Original weight acquisition changed")
                node.value = ast.parse("cached_weights[name]", mode="eval").body
                replaced += 1
        if isinstance(node, ast.For) and ast.unparse(node.target) == "(name, families)":
            index = next(
                i
                for i, n in enumerate(node.body)
                if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "elastic"
            )
            node.body.insert(
                index + 1, ast.parse("_verify_startup(name, elastic, clock, ramp)").body[0]
            )
            startup += 1
    if replaced != 1 or startup != 1:
        raise ValueError("Original causal loop boundary changed")
    result = body[-1].value
    if not isinstance(result, ast.Call) or ast.unparse(result.func) != "dict":
        raise ValueError("Original result boundary changed")
    overrides = dict(
        status="'passed_complete_coarse_mixed_cache_sliding_and_initialized_ad'",
        all_physical_contour_modes_checked="False",
        physical_contour_mode_evaluations="0",
        additional_unique_lag_zero_modes="0",
    )
    for keyword in result.keywords:
        if keyword.arg in overrides:
            keyword.value = ast.parse(overrides.pop(keyword.arg), mode="eval").body
        if keyword.arg == "contour_seconds_including_initial_preparation":
            keyword.arg = "cached_input_validation_seconds"
    if overrides:
        raise ValueError("Original result flags changed")
    result.keywords.append(
        ast.keyword(
            arg="cached_kernel_validation", value=ast.Name(id="cached_report", ctx=ast.Load())
        )
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "paired":
            node.value = "mixed"
    namespace = {**CONTROL, "_load_cached": inputs.load, "_verify_startup": inputs.verify_startup}
    exec(compile(ast.fix_missing_locations(tree), __file__ + "|cached-control", "exec"), namespace)
    return namespace["experiment"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("mixed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original, mixed, output = (p.resolve() for p in [args.original, args.mixed, args.output])
    if (
        output.exists()
        or output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve())
        or any(output.is_relative_to(p) for p in [original, mixed])
        or original == mixed
    ):
        raise ValueError("Require distinct read-only kernels and a fresh owned output")
    import jax

    jax.config.update("jax_enable_x64", True)
    begun = perf_counter()
    frozen = {
        str(p.resolve()): fingerprint(p)
        for p in [
            *TOOLS.glob("*.py"),
            CONTROL["CASE"],
            CONTROL["SEED"],
            *[p for folder in [original, mixed] for p in folder.iterdir() if p.is_file()],
        ]
    }
    environment = environment_record(CONTROL["CASE"])
    output.mkdir(parents=True)
    write_json(
        output / "manifest.json",
        dict(
            frozen_inputs_sha256=frozen,
            environment=environment,
            original=str(original),
            mixed=str(mixed),
            extra_frequency_assemblies=0,
            diagnostic_actuator_rate_mm_per_s=7812.5,
            original_actuator_rate_mm_per_s=20,
        ),
    )
    try:
        experiment = cached_experiment(CachedInputs(original, mixed, environment))
        with threadpool_limits(limits=1, user_api="blas"):
            result = experiment(output, frozen, environment, begun)
        if (
            any(fingerprint(p) != digest for p, digest in frozen.items())
            or environment_record(CONTROL["CASE"]) != environment
        ):
            raise ValueError("Cached inputs, numerical source or runtime changed")
        result.update(
            total_seconds_including_controls=perf_counter() - begun,
            all_cached_inputs_and_sources_unchanged=True,
            full_event_or_fem_advantage=False,
            scope="Complete coarse accelerated 128-step sliding with saved mixed kernels; selected initialized fixed-bulk AD and recorded loading. No material, geometry, stopping-time derivative or full-event accuracy/speed claim.",
        )
        write_json(output / "summary.json", result)
    except BaseException as error:
        write_json(output / "failure.json", dict(error=repr(error), seconds=perf_counter() - begun))
        raise


if __name__ == "__main__":
    main()

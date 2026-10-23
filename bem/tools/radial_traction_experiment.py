"""Exact radial double-layer contraction, isolated from running production jobs.

For G=a*I+b*e*e.T, contract the analytic gradient with the source normal
before materializing the third-order gradient tensor. This changes arithmetic
and temporary arrays, not the Green function, quadrature, or constitutive law.
The prepared assembler is cloned with two local bindings; module globals and
the installed package are never replaced.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from types import FunctionType, MethodType

import numpy as np
from threadpoolctl import threadpool_limits

from finiteblock_bem.cq import bdf2_symbol, plan_cq
from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.io import environment_record, fingerprint, write_json
from finiteblock_bem.kernels import _radial, source_traction
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model


@dataclass(frozen=True)
class RadialGradient:
    e: np.ndarray
    da: np.ndarray
    db: np.ndarray
    b_over_r: np.ndarray
    sign: int = 1

    def reshape(self, *shape):
        if shape[-3:] != (2, 2, 2):
            raise ValueError("Expected a 2D third-order Green gradient")
        return replace(
            self,
            e=self.e.reshape(*shape[:-3], 2),
            da=self.da.reshape(shape[:-3]),
            db=self.db.reshape(shape[:-3]),
            b_over_r=self.b_over_r.reshape(shape[:-3]),
        )

    def __neg__(self):
        return replace(self, sign=-self.sign)


@lru_cache(maxsize=32)
def series_coefficients(ks, kp):
    """The SAME nine coefficient pairs, shared across geometry batches."""
    harmonic, coefficients = 0.0, []
    for k in range(1, 10):
        harmonic += 1 / k
        n = 2 * k
        cs = (ks / 2) ** n / math.factorial(k) ** 2
        cp = (kp / 2) ** n / math.factorial(k) ** 2
        a = cs * (harmonic - np.euler_gamma - np.log(ks / 2))
        a -= cp * (harmonic - np.euler_gamma - np.log(kp / 2))
        coefficients.append((n, a, -(cs - cp)))
    return tuple(coefficients)


def horner_difference_series(r, ks, kp):
    """Evaluate the existing finite derivative series as polynomials in r^2."""
    t, log_r = r * r, np.log(r)
    p1, p2, p3 = [np.zeros_like(r, dtype=complex) for _ in range(3)]
    for n, a, b in reversed(series_coefficients(complex(ks), complex(kp))):
        value = a + b * log_r
        p1 = t * p1 + n * value + b
        p2 = t * p2 + n * (n - 1) * value + (2 * n - 1) * b
        p3 = t * p3 + n * (n - 1) * (n - 2) * value + (3 * n * n - 6 * n + 2) * b
    return r * p1, p2, p3 / r


def radial_green(vector, radius, a, b, da, db):
    e = vector / radius[..., None]
    green = a[..., None, None] * np.eye(2) + np.einsum("...,...i,...j->...ij", b, e, e)
    return green, RadialGradient(e, da, db, b / radius)


def radial_source_traction(gradient, normals, material):
    if not isinstance(gradient, RadialGradient):
        return source_traction(gradient, normals, material)
    e, c, da, db = gradient.e, gradient.b_over_r, gradient.da, gradient.db
    normals = np.asarray(normals)
    en = np.sum(e * normals, axis=-1)
    alpha = material.mu * (da + c)
    beta = 2 * material.mu * (db - 2 * c)
    gamma = 2 * material.mu * c + material.lam * (da + db + c)
    answer = (alpha * en)[..., None, None] * np.eye(2)
    answer += np.einsum("...,...i,...j->...ij", beta * en, e, e)
    answer += np.einsum("...,...i,...j->...ij", alpha, normals, e)
    answer += np.einsum("...,...i,...j->...ij", gamma, e, normals)
    return gradient.sign * answer


class RadialTractionPrepared(PreparedGalerkinOperators):
    def __init__(self, *args, series="reference", **kwargs):
        super().__init__(*args, **kwargs)
        if series not in ["reference", "horner"]:
            raise ValueError("Unknown small-argument series evaluator")
        original = PreparedGalerkinOperators.assemble
        bindings = {
            **original.__globals__,
            "_tensor_from_radial": radial_green,
            "source_traction": radial_source_traction,
        }
        if series == "horner":
            radial = FunctionType(
                _radial.__code__,
                {**_radial.__globals__, "_difference_derivatives_series": horner_difference_series},
                _radial.__name__,
                _radial.__defaults__,
            )
            radial.__kwdefaults__ = dict(_radial.__kwdefaults__)
            bindings["_radial"] = radial
        clone = FunctionType(original.__code__, bindings, original.__name__, original.__defaults__)
        self.assemble = MethodType(clone, self)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Saved run specifying geometry/time grid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 0.1, 0.5, 2, np.pi])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--series", choices=["reference", "horner"], default="horner")
    args = parser.parse_args()
    tool_sha = fingerprint(Path(__file__))
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.repeats < 1 or any(not 0 <= angle <= np.pi for angle in args.angles):
        raise ValueError("Positive repeats and contour angles in [0,pi] required")
    options = json.loads((args.run / "run_manifest.json").read_text())["arguments"]
    summary = json.loads((args.run / "summary.json").read_text())
    model = load_nodal_model(
        options["case"],
        max_panel_length=options["h"],
        contact_panel_length=options["contact_h"],
        contact_end_length=options["contact_end_h"],
        corner_panel_length=options["corner_h"],
        restart_knot_window=options.get("restart_knot_window"),
        restart_knot_subdivisions=options.get("restart_knot_subdivisions", 1),
    )
    plan = plan_cq(
        2 * len(model.moving_contact) + 1,
        summary["dt_s"],
        summary["completed_steps"],
        max_memory_mb=65536,
    )
    frequencies = bdf2_symbol(plan.radius * np.exp(-1j * np.asarray(args.angles))) / plan.dt
    environment = environment_record(options["case"])
    with threadpool_limits(limits=1, user_api="blas"):
        original = NodalInterfaceTransfer(model, quadrature=options["quadrature"])
        variant = NodalInterfaceTransfer(model, quadrature=options["quadrature"], prepared=False)
        variant.prepared = [
            RadialTractionPrepared(
                mesh, model.reference.material, quadrature=options["quadrature"], series=args.series
            )
            for mesh in [model.reference.moving, model.reference.stationary]
        ]
        samples = []
        for s in frequencies:
            timings = {"original": [], "radial": []}
            matrices = {}
            for iteration in range(args.repeats):
                order = [("original", original), ("radial", variant)]
                if iteration % 2:
                    order.reverse()
                for name, transfer in order:
                    started = perf_counter()
                    matrices[name] = transfer.evaluate(s)
                    timings[name].append(perf_counter() - started)
            delta = matrices["radial"] - matrices["original"]
            relative = float(np.linalg.norm(delta) / np.linalg.norm(matrices["original"]))
            np.testing.assert_allclose(
                matrices["radial"], matrices["original"], rtol=2e-10, atol=1e-6
            )
            samples.append(
                dict(
                    s_real=float(s.real),
                    s_imag=float(s.imag),
                    relative_frobenius_difference=relative,
                    maximum_absolute_difference=float(np.max(abs(delta))),
                    timing_seconds=timings,
                )
            )
            print(json.dumps(samples[-1]), flush=True)
    if (
        tool_sha != fingerprint(Path(__file__))
        or environment["source_sha256"] != environment_record(options["case"])["source_sha256"]
    ):
        raise ValueError("Numerical source changed during the experiment")
    write_json(
        args.output / "summary.json",
        dict(
            samples=samples,
            source_run=str(args.run.resolve()),
            environment=environment,
            validator_sha256=tool_sha,
            assembler_sha256=fingerprint(
                Path(PreparedGalerkinOperators.assemble.__code__.co_filename)
            ),
            kernel_workers=int(os.environ.get("FDBEM_CORES", "1")),
            series_evaluation=args.series,
            scope="Single-frequency exact contraction experiment; alternating local timings may be contended; no full-trajectory, AD, passivity or whole-solver speed claim",
        ),
    )


if __name__ == "__main__":
    main()

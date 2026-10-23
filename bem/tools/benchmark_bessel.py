"""Isolated Bessel acceleration experiment; production numerical sources stay fixed.

Uses DLMF 10.40.10--12 in the right half plane, with coefficients from
10.17.1. The truncation bound covers the analytic expansion, not roundoff,
assembled-operator error or the complete nonlinear trajectory.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import kv, kve

from finiteblock_bem import galerkin_fast
from finiteblock_bem.cq import bdf2_symbol, plan_cq
from finiteblock_bem.io import environment_record, write_json
from finiteblock_bem.kernels import _difference_derivatives_series, _radial
from finiteblock_bem.nodal import NodalInterfaceTransfer, load_nodal_model

ROOT = Path(__file__).resolve().parents[1]
TERMS = 16
CUTOFF = 25.0


def coefficients(order):
    result = [1.0]
    for k in range(1, TERMS + 1):
        result.append(result[-1] * (4 * order**2 - (2 * k - 1) ** 2) / (8 * k))
    return np.array(result)


COEFFICIENTS = [coefficients(order) for order in [0, 1]]


def truncation_bound(order, radius):
    """Bound the remainder inside sqrt(pi/(2*z))*exp(-z), Re(z)>=0."""
    return (
        2
        * abs(COEFFICIENTS[order][-1])
        * np.asarray(radius) ** (-TERMS)
        * np.exp(abs(order**2 - 0.25) / np.asarray(radius))
    )


def hybrid_k01(z, *, scaled=False):
    z = np.asarray(z, complex)
    shape = z.shape
    z = z.reshape(-1)
    use = (z.real >= 0) & (abs(z) >= CUTOFF) & np.isfinite(z)
    result = np.empty((2, len(z)), complex)
    reference = kve if scaled else kv
    for order in [0, 1]:
        result[order, ~use] = reference(order, z[~use])
    if np.any(use):
        w = 1 / z[use]
        factor = np.sqrt(np.pi * w / 2)
        if not scaled:
            factor *= np.exp(-z[use])
        for order in [0, 1]:
            result[order, use] = factor * np.polynomial.polynomial.polyval(
                w, COEFFICIENTS[order][:-1]
            )
    return tuple(row.reshape(shape) for row in result)


def hybrid_radial(r, s, material, *, scalars=False):
    """Production radial formulas with only the K0/K1 evaluator substituted."""
    ks, kp = complex(s) / material.cs, complex(s) / material.cp
    zs, zp = ks * r, kp * r
    k0s, k1s = hybrid_k01(zs)
    k0p, k1p = hybrid_k01(zp)
    f1 = -ks * k1s + kp * k1p
    f2 = ks * ks * k0s - kp * kp * k0p - f1 / r
    f3 = -(ks**3) * k1s + kp**3 * k1p - f2 / r + f1 / r**2
    small = np.maximum(np.abs(zs), np.abs(zp)) < 0.2
    if np.any(small):
        d1, d2, d3 = _difference_derivatives_series(r[small], ks, kp)
        f1[small], f2[small], f3[small] = d1, d2, d3
    a = k0s - f1 / (ks * ks * r)
    b = -(f2 - f1 / r) / (ks * ks)
    da = -ks * k1s - (f2 / r - f1 / r**2) / (ks * ks)
    db = -(f3 - f2 / r + f1 / r**2) / (ks * ks)
    scale = 1 / (2 * np.pi * material.mu)
    result = (a * scale, b * scale, da * scale, db * scale)
    if scalars:
        return (*result, k0s / (2 * np.pi), k0p / (2 * np.pi), f1 / (2 * np.pi))
    return result


def scalar_checks():
    radii = np.unique(np.r_[np.geomspace(1e-5, 1000, 1001), np.nextafter(CUTOFF, 0), CUTOFF])
    angles = np.linspace(-np.pi / 2, np.pi / 2, 101)
    z = radii[:, None] * np.exp(1j * angles)
    actual = hybrid_k01(z, scaled=True)
    record = []
    for order in [0, 1]:
        exact = kve(order, z)
        relative = abs(actual[order] - exact) / abs(exact)
        np.testing.assert_allclose(actual[order], exact, rtol=2e-14, atol=0)
        record.append(
            dict(
                order=order,
                max_scaled_relative_error=float(np.max(relative)),
                truncation_bound_at_cutoff=float(truncation_bound(order, CUTOFF)),
                checked_points=z.size,
            )
        )
    # The fallback must also preserve arguments outside the right half plane.
    left = np.array([-3 + 2j, -30 + 10j, -3 - 2j, -30 - 10j])
    for order, actual in enumerate(hybrid_k01(left)):
        np.testing.assert_array_equal(actual, kv(order, left))
    return record


def matrix_checks(run):
    args = json.loads((run / "run_manifest.json").read_text())["arguments"]
    model = load_nodal_model(
        args["case"],
        max_panel_length=args["h"],
        contact_panel_length=args["contact_h"],
        contact_end_length=args["contact_end_h"],
        corner_panel_length=args["corner_h"],
        restart_knot_window=args.get("restart_knot_window"),
        restart_knot_subdivisions=args.get("restart_knot_subdivisions", 1),
    )
    steps = args["steps"] or round(args["duration"] / args["dt"])
    plan = plan_cq(2 * len(model.moving_contact) + 1, args["dt"], steps, max_memory_mb=65536)
    transfer = NodalInterfaceTransfer(model, quadrature=args["quadrature"])
    wc = model.constraints.contact_weights
    scale = 1 / np.r_[wc, wc, model.constraints.platen_width]
    records = []
    original = galerkin_fast._radial
    try:
        for k in [0, 1, 7, 127, 511, plan.fft_size // 8, plan.fft_size // 4, plan.fft_size // 2]:
            s = bdf2_symbol(plan.radius * np.exp(-2j * np.pi * k / plan.fft_size)) / plan.dt
            galerkin_fast._radial = original
            start = perf_counter()
            reference = transfer.evaluate(s)
            reference_seconds = perf_counter() - start
            galerkin_fast._radial = hybrid_radial
            start = perf_counter()
            actual = transfer.evaluate(s)
            prototype_seconds = perf_counter() - start
            error = np.linalg.norm((actual - reference) * scale[:, None]) / np.linalg.norm(
                reference * scale[:, None]
            )
            if error > 1e-11:
                raise AssertionError(f"Matrix error {error} at contour index {k}")
            records.append(
                dict(
                    contour_index=k,
                    s_real=float(s.real),
                    s_imag=float(s.imag),
                    relative_scaled_matrix_error=float(error),
                    reference_seconds=reference_seconds,
                    prototype_seconds=prototype_seconds,
                )
            )
            print(json.dumps(records[-1]), flush=True)
    finally:
        galerkin_fast._radial = original
    # The imported immutable production function remains available unchanged.
    assert original is _radial
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-run", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/bessel-experiment.json")
    args = parser.parse_args()
    checks = scalar_checks()
    matrices = [] if args.case_run is None else matrix_checks(args.case_run)
    record = dict(
        cutoff=CUTOFF,
        terms=TERMS,
        scalar_checks=checks,
        matrix_checks=matrices,
        environment=environment_record(),
        prototype_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources=["https://dlmf.nist.gov/10.17.E1", "https://dlmf.nist.gov/10.40.iii"],
        scope="Isolated prototype only; timings overlap an active four-worker simulation; no speed ratio claim",
    )
    write_json(args.output, record)
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()

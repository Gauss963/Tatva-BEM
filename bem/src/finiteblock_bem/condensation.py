"""Exact Laplace-domain elimination of linear exterior/common boundary DOFs.

K(s) z + B.T f = 0, q=B z. With homogeneous perturbation tractions on
eliminated boundaries, the retained dynamic impedance is
Z(s)=[B K(s)^-1 B.T]^-1. Elimination is performed BEFORE CQ sampling;
eliminating only W[0] would discard the eliminated boundaries' memory.
"""

import hashlib
import inspect
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from .constraints import BoundaryConstraints, build_constraints
from .cq import BoundaryHistory, ResourceBudgetError, convolution_weights, plan_cq
from .nonlinear import ContactEquilibrium
from .operators import PreparedBoundaryOperators


class InterfaceTransfer:
    def __init__(self, case, *, quadrature=20):
        self.case = case
        self.constraints = con = build_constraints(case)
        platen = np.zeros((1, con.size))
        platen[0, -1] = 1
        self.jump = np.vstack((con.normal_jump, con.tangential_jump, platen))
        self.dofs = len(self.jump)
        self.blocks = []
        first = 0
        for mesh in (case.moving, case.stationary):
            p = con.prolongation[first : first + mesh.dofs]
            columns = np.flatnonzero(np.any(p != 0, axis=0))
            self.blocks.append(
                (
                    PreparedBoundaryOperators(mesh, case.material, quadrature=quadrature),
                    p[:, columns],
                    con.full_weights[first : first + mesh.dofs],
                    self.jump[:, columns],
                    columns,
                )
            )
            first += mesh.dofs

    def evaluate(self, s, *, reconstruction=False):
        compliance = np.zeros((self.dofs, self.dofs), dtype=complex)
        responses = []
        for prepared, p, weights, b, columns in self.blocks:
            w = prepared.evaluate(s).impedance()
            k = p.T @ (weights[:, None] * (w @ p))
            response = solve(k, b.T, check_finite=False)
            compliance += b @ response
            if reconstruction:
                responses.append((columns, response))
        impedance = solve(compliance, np.eye(self.dofs), check_finite=False)
        if not reconstruction:
            return impedance
        prolongation = np.zeros((self.constraints.size, self.dofs), dtype=complex)
        for columns, response in responses:
            prolongation[columns] = response @ impedance
        return impedance, self.constraints.prolongation @ prolongation


class InterfaceContactEquilibrium(ContactEquilibrium):
    """Shared contact residual/Newton on q=(normal jump, tangent jump, platen).

    Only a static lift and constant external normal traction are supported.
    General time-dependent exterior loading needs its own condensed transfer.
    """

    def __init__(self, case, lift, impedance):
        self.case = case
        self.impedance = np.asarray(impedance, dtype=float)
        n = len(case.moving_contact)
        size = 2 * n + 1
        eye = np.eye(size)
        wc = case.moving.lengths[case.moving_contact]
        width = float(np.sum(case.moving.lengths[case.moving.indices("platen")]))
        self.constraints = BoundaryConstraints(
            eye,
            np.ones(size),
            np.r_[wc, wc, width],
            eye[:n],
            eye[n : 2 * n],
            wc,
            width,
            np.array([], int),
        )
        im, js = case.moving_contact, case.stationary_contact
        offset = case.moving.dofs
        self.base_dn = lift.displacement[2 * im] - lift.displacement[offset + 2 * js]
        self.base_ds = lift.displacement[2 * im + 1] - lift.displacement[offset + 2 * js + 1]
        self.base_platen = float(np.mean(lift.displacement[2 * case.moving.indices("platen") + 1]))
        # Preserve the supplied bulk lift when contact parameters change. Rebuilding
        # its contact forces from the new law would silently change the initial
        # stress and invalidate fixed-lift parameter derivatives.
        base_pressure = -lift.traction[2 * im]
        base_shear = -lift.traction[2 * im + 1]
        platen = case.moving.indices("platen")
        base_platen_force = np.dot(case.moving.lengths[platen], lift.traction[2 * platen + 1])
        self.spring = case.spring_stiffness * width
        self.base_u = np.zeros(size)
        self.base_t = -np.r_[
            wc * base_pressure,
            wc * base_shear,
            -base_platen_force,
        ]
        self.matrix = self.impedance.copy()
        self.matrix[-1, -1] += self.spring

    def external_traction(self, normal_pressure):
        if normal_pressure != self.case.normal_pressure:
            raise ValueError(
                "Interface formulation requires constant normal traction at the lift value"
            )
        return np.zeros(len(self.base_u))


def plan_interface(case, dt, steps, *, quadrature=20, workers=1, max_memory_mb=512):
    """Include full spatial assembly even though time histories are condensed."""
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    plan = plan_cq(2 * len(case.moving_contact) + 1, dt, steps, max_memory_mb=max_memory_mb)
    counts = [case.moving.count, case.stationary.count]
    geometry = sum(160 * n * n * quadrature for n in counts)
    frequency = workers * max(
        224 * n * n * quadrature + 240 * n * min(n, 16) * quadrature + 128 * (2 * n) ** 2
        for n in counts
    )
    peak = plan.estimated_peak_bytes + geometry + frequency
    if peak > max_memory_mb * 1024**2:
        raise ResourceBudgetError(
            f"Interface CQ plus full spatial assembly needs about "
            f"{peak / 1024**2:.1f} MiB; budget is {max_memory_mb:g} MiB"
        )
    return [plan], peak


def build_condensed_history(
    case,
    dt,
    steps,
    *,
    quadrature=20,
    workers=1,
    max_memory_mb=512,
    cache_directory=None,
    progress=None,
):
    plans, _ = plan_interface(
        case, dt, steps, quadrature=quadrature, workers=workers, max_memory_mb=max_memory_mb
    )
    plan, size = plans[0], plans[0].dofs
    cached = None
    if cache_directory is not None:
        digest = hashlib.sha256(repr((plan, case.material, quadrature)).encode())
        # Only the frequency transfer determines these weights. Contact/Newton
        # changes must not force rebuilding an identical linear wave kernel.
        digest.update(inspect.getsource(InterfaceTransfer).encode())
        for mesh in (case.moving, case.stationary):
            for data in (mesh.starts, mesh.ends, mesh.labels):
                digest.update(data.tobytes())
        for name in (
            "cq.py",
            "operators.py",
            "kernels.py",
            "material.py",
            "mesh.py",
            "constraints.py",
        ):
            digest.update(Path(__file__).with_name(name).read_bytes())
        directory = Path(cache_directory)
        directory.mkdir(parents=True, exist_ok=True)
        cached = directory / f"interface-{digest.hexdigest()}.npy"
        if cached.exists():
            weights = np.load(cached, mmap_mode="r", allow_pickle=False)
            if weights.shape != (steps + 1, size, size) or weights.dtype != np.float64:
                raise ValueError("Invalid condensed CQ cache")
            return BoundaryHistory(plan, weights)
    transfer = InterfaceTransfer(case, quadrature=quadrature)
    weights = convolution_weights(transfer.evaluate, plan, workers=workers, progress=progress)
    if cached is not None:
        import os

        temporary = cached.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, weights, allow_pickle=False)
        temporary.replace(cached)
    return BoundaryHistory(plan, weights)

"""Unilateral penalty contact with cumulative linear slip weakening.

The default evaluates strength from the PREVIOUS cumulative slip, matching
Tatva's archived contact update. Current displacement, pressure, and return
mapping are solved implicitly; cumulative weakening is committed once per
accepted step. An optional end-of-step softening formulation is retained for
comparisons and explicitly rejects nonunique local return maps.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactState:
    plastic_slip: np.ndarray
    cumulative_slip: np.ndarray

    def __post_init__(self):
        z = np.array(self.plastic_slip, dtype=float, copy=True)
        c = np.array(self.cumulative_slip, dtype=float, copy=True)
        if (
            z.ndim != 1
            or c.shape != z.shape
            or not np.all(np.isfinite(z))
            or not np.all(np.isfinite(c))
        ):
            raise ValueError("Contact state must contain matching finite vectors")
        if np.any(c < 0):
            raise ValueError("Negative cumulative slip")
        z.setflags(write=False)
        c.setflags(write=False)
        object.__setattr__(self, "plastic_slip", z)
        object.__setattr__(self, "cumulative_slip", c)


@dataclass(frozen=True)
class ContactResponse:
    pressure: np.ndarray
    shear: np.ndarray
    dp_dn: np.ndarray
    dtau_dn: np.ndarray
    dtau_ds: np.ndarray
    friction_coefficient: np.ndarray
    state: ContactState


@dataclass(frozen=True)
class SlipWeakeningContact:
    normal_penalty: float
    tangential_penalty: float
    static_friction: float
    kinetic_friction: float
    characteristic_slip: float
    weakening_evaluation: str = "previous"

    def __post_init__(self):
        values = (
            self.normal_penalty,
            self.tangential_penalty,
            self.static_friction,
            self.kinetic_friction,
            self.characteristic_slip,
        )
        if not all(np.isfinite(values)):
            raise ValueError("Contact parameters must be finite")
        if min(self.normal_penalty, self.tangential_penalty, self.characteristic_slip) <= 0:
            raise ValueError("Penalties and Dc must be positive")
        if not 0 <= self.kinetic_friction <= self.static_friction:
            raise ValueError("Require 0 <= mu_k <= mu_s")
        if self.weakening_evaluation not in {"previous", "updated"}:
            raise ValueError("weakening_evaluation must be previous or updated")

    def coefficient(self, cumulative):
        return self.static_friction - (self.static_friction - self.kinetic_friction) * np.minimum(
            np.asarray(cumulative) / self.characteristic_slip, 1
        )

    def evaluate(self, normal_closure, relative_slip, old: ContactState, *, frozen=False):
        dn = np.asarray(normal_closure, dtype=float)
        ds = np.asarray(relative_slip, dtype=float)
        if dn.shape != old.plastic_slip.shape or ds.shape != dn.shape:
            raise ValueError("Contact kinematics and state have different shapes")
        if not np.all(np.isfinite(dn)) or not np.all(np.isfinite(ds)):
            raise ValueError("Nonfinite contact kinematics")
        kn, kt = self.normal_penalty, self.tangential_penalty
        p = kn * np.maximum(dn, 0)
        active = dn > 0
        dp = np.where(active, kn, 0.0)
        trial = kt * (ds - old.plastic_slip)
        tau = np.where(active, trial, 0.0)
        ts = np.where(active, kt, 0.0)
        tp = np.zeros_like(dn)
        z = old.plastic_slip.copy()
        c = old.cumulative_slip.copy()
        applied_mu = self.coefficient(c)
        if not frozen:
            muold = applied_mu.copy()
            sliding = active & (np.abs(trial) > p * muold)
            if np.any(sliding) and self.weakening_evaluation == "previous":
                direction = np.sign(trial)
                increment = np.maximum(np.abs(trial) - p * muold, 0) / kt
                z[sliding] += direction[sliding] * increment[sliding]
                c[sliding] += increment[sliding]
                tau[sliding] = direction[sliding] * p[sliding] * muold[sliding]
                ts[sliding] = 0
                tp[sliding] = direction[sliding] * muold[sliding]
            elif np.any(sliding):
                slope = np.where(
                    c < self.characteristic_slip,
                    (self.static_friction - self.kinetic_friction) / self.characteristic_slip,
                    0.0,
                )
                denominator = kt - p * slope
                if np.any(denominator[sliding] <= 0):
                    raise FloatingPointError(
                        "The local softening return map is nonunique at this pressure/penalty"
                    )
                increment = np.zeros_like(c)
                increment[sliding] = (
                    np.abs(trial[sliding]) - p[sliding] * muold[sliding]
                ) / denominator[sliding]
                kinetic = sliding & (c + increment >= self.characteristic_slip)
                soft = sliding & ~kinetic
                direction = np.sign(trial)
                increment[kinetic] = (
                    np.abs(trial[kinetic]) - p[kinetic] * self.kinetic_friction
                ) / kt
                z[sliding] += direction[sliding] * increment[sliding]
                c[sliding] += increment[sliding]
                munew = self.coefficient(c)
                applied_mu = munew
                tau[sliding] = direction[sliding] * p[sliding] * munew[sliding]
                ts[soft] = -kt * p[soft] * slope[soft] / denominator[soft]
                tp[soft] = direction[soft] * kt * munew[soft] / denominator[soft]
                ts[kinetic] = 0
                tp[kinetic] = direction[kinetic] * self.kinetic_friction
        return ContactResponse(p, tau, dp, tp * dp, ts, applied_mu, ContactState(z, c))

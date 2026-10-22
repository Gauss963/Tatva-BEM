"""Work-conjugate support elimination and a rigid massless loading platen."""

from dataclasses import dataclass

import numpy as np

from .case import TwoBlockCase


@dataclass(frozen=True)
class BoundaryConstraints:
    prolongation: np.ndarray
    full_weights: np.ndarray
    reduced_weights: np.ndarray
    normal_jump: np.ndarray
    tangential_jump: np.ndarray
    contact_weights: np.ndarray
    platen_width: float
    fixed_dofs: np.ndarray

    @property
    def size(self):
        return self.prolongation.shape[1]

    def reduce_force(self, traction):
        return self.prolongation.T @ (self.full_weights * traction)


def build_constraints(case: TwoBlockCase):
    moving, stationary = case.moving, case.stationary
    offset = moving.dofs
    size = offset + stationary.dofs
    fixed = np.concatenate(
        (
            offset + 2 * stationary.indices("support_x"),
            offset + 2 * stationary.indices("support_y") + 1,
        )
    )
    platen = moving.indices("platen")
    shared = 2 * platen + 1
    free = np.setdiff1d(np.arange(size), np.concatenate((fixed, shared)))
    p = np.zeros((size, len(free) + 1))
    p[free, np.arange(len(free))] = 1
    p[shared, -1] = 1
    weights = np.repeat(np.concatenate((moving.lengths, stationary.lengths)), 2)
    im = case.moving_contact
    js = case.stationary_contact
    bn = p[2 * im] - p[offset + 2 * js]
    bt = p[2 * im + 1] - p[offset + 2 * js + 1]
    return BoundaryConstraints(
        p,
        weights,
        p.T @ weights,
        bn,
        bt,
        moving.lengths[im],
        float(np.sum(moving.lengths[platen])),
        fixed,
    )

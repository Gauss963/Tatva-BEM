"""Isotropic, small-strain, plane-strain elasticity in consistent units."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ElasticMaterial:
    young: float
    poisson: float
    density: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.young, self.poisson, self.density)):
            raise ValueError("Material parameters must be finite")
        if self.young <= 0 or self.density <= 0 or not -1 < self.poisson < 0.5:
            raise ValueError("Require E > 0, rho > 0 and -1 < nu < 0.5")

    @property
    def mu(self):
        return self.young / (2 * (1 + self.poisson))

    @property
    def lam(self):
        return self.young * self.poisson / ((1 + self.poisson) * (1 - 2 * self.poisson))

    @property
    def cp(self):
        return math.sqrt((self.lam + 2 * self.mu) / self.density)

    @property
    def cs(self):
        return math.sqrt(self.mu / self.density)

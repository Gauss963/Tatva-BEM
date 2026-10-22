"""Finite-domain, plane-strain elastodynamic boundary elements.

This package is independent of Tatva and Bem2d.jl. It uses physical boundary
displacements and tractions, not fictitious displacement discontinuities.
"""

from .material import ElasticMaterial
from .mesh import BoundaryMesh, polygon_boundary

__all__ = ["BoundaryMesh", "ElasticMaterial", "polygon_boundary"]
__version__ = "0.1.0"

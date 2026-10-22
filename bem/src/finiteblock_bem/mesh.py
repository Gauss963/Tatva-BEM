"""Straight, constant boundary panels on counterclockwise simple polygons."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundaryMesh:
    starts: np.ndarray
    ends: np.ndarray
    labels: np.ndarray

    def __post_init__(self):
        starts = np.asarray(self.starts, dtype=float)
        ends = np.asarray(self.ends, dtype=float)
        labels = np.asarray(self.labels, dtype=str)
        if starts.ndim != 2 or starts.shape[1] != 2 or starts.shape != ends.shape:
            raise ValueError("Boundary endpoints must have matching (panels, 2) shape")
        if not len(starts) or labels.shape != (len(starts),):
            raise ValueError("One label per panel is required")
        if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
            raise ValueError("Nonfinite boundary coordinates")
        if np.any(np.linalg.norm(ends - starts, axis=1) <= 0):
            raise ValueError("Zero-length boundary panel")
        if not np.allclose(ends, np.roll(starts, -1, axis=0), rtol=0, atol=1e-10):
            raise ValueError("Panels must form one connected closed loop")
        area = np.sum(starts[:, 0] * ends[:, 1] - starts[:, 1] * ends[:, 0]) / 2
        if area <= 0:
            raise ValueError("Boundary must be counterclockwise")
        for name, array in (("starts", starts), ("ends", ends), ("labels", labels)):
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def count(self):
        return len(self.starts)

    @property
    def dofs(self):
        return 2 * self.count

    @property
    def centers(self):
        return (self.starts + self.ends) / 2

    @property
    def lengths(self):
        return np.linalg.norm(self.ends - self.starts, axis=1)

    @property
    def tangents(self):
        return (self.ends - self.starts) / self.lengths[:, None]

    @property
    def normals(self):
        tangent = self.tangents
        return np.column_stack((tangent[:, 1], -tangent[:, 0]))

    def indices(self, label):
        return np.flatnonzero(self.labels == label)


def polygon_boundary(vertices, labels, max_length):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) != len(labels):
        raise ValueError("Expected polygon vertices and one label per edge")
    if not np.isfinite(max_length) or max_length <= 0:
        raise ValueError("max_length must be finite and positive")
    starts, ends, panel_labels = [], [], []
    for a, b, label in zip(vertices, np.roll(vertices, -1, axis=0), labels):
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / max_length - 1e-12)))
        points = np.linspace(a, b, count + 1)
        starts.extend(points[:-1])
        ends.extend(points[1:])
        panel_labels.extend([label] * count)
    return BoundaryMesh(starts, ends, panel_labels)

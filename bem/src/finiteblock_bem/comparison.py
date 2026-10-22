"""Compare P0 panel values with integrated, linearly interpolated FEM nodes."""

import numpy as np
from scipy.integrate import trapezoid


def panel_average(node_y, nodal_value, panel_y, panel_lengths):
    node_y, value = np.asarray(node_y), np.asarray(nodal_value)
    if np.any(np.diff(node_y) <= 0) or value.shape != node_y.shape:
        raise ValueError("Reference nodes must be strictly increasing and match the field")
    result = []
    for center, length in zip(panel_y, panel_lengths, strict=True):
        a, b = center - length / 2, center + length / 2
        if a < node_y[0] - 1e-8 or b > node_y[-1] + 1e-8 or length <= 0:
            raise ValueError("Panel lies outside the reference interval")
        points = np.concatenate(([a], node_y[(node_y > a) & (node_y < b)], [b]))
        result.append(trapezoid(np.interp(points, node_y, value), points) / length)
    return np.asarray(result)


def relative_l2(actual, reference, weights):
    denominator = np.sum(np.asarray(weights) * np.asarray(reference) ** 2)
    if denominator <= 0:
        raise ValueError("Relative error requires a nonzero reference")
    return float(
        np.sqrt(np.sum(np.asarray(weights) * (np.asarray(actual) - reference) ** 2) / denominator)
    )

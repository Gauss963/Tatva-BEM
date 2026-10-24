"""Fixed geometric near/far quadrature for the original Galerkin assembler.

Self, adjacent and geometrically close panel pairs retain the reference order.
Well-separated pairs use a lower order, chosen once before any frequency is
evaluated. No frequency-dependent switching, kernel truncation or changes to
the weak formulation, reciprocity or source traction formula are introduced.
This is an isolated experiment, not a replacement for the production backend.
"""

import numpy as np

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.nodal import NodalInterfaceTransfer


def segment_distances(a, b, c, d):
    """Minimum distance between corresponding 2D closed line segments."""
    u, v = b - a, d - c

    def endpoint(p, start, direction):
        coordinate = np.sum((p - start) * direction, axis=1) / np.sum(direction**2, axis=1)
        projection = start + np.clip(coordinate, 0, 1)[:, None] * direction
        return np.linalg.norm(p - projection, axis=1)

    distance = np.minimum.reduce(
        [endpoint(a, c, v), endpoint(b, c, v), endpoint(c, a, u), endpoint(d, a, u)]
    )

    def cross(x, y):
        return x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]

    denominator = cross(u, v)
    regular = denominator != 0
    t, s = np.full(len(a), np.inf), np.full(len(a), np.inf)
    np.divide(cross(c - a, v), denominator, out=t, where=regular)
    np.divide(cross(c - a, u), denominator, out=s, where=regular)
    intersect = regular & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)
    distance[intersect] = 0
    return distance


class MixedQuadraturePrepared(PreparedGalerkinOperators):
    def __init__(
        self, mesh, material, *, near_order=12, far_order=6, separation=2.0, batch_pairs=32
    ):
        if (
            not isinstance(near_order, int)
            or near_order < 4
            or near_order % 2
            or not isinstance(far_order, int)
            or not 4 <= far_order <= near_order
            or far_order % 2
        ):
            raise ValueError("Require even quadrature orders with 4 <= far_order <= near_order")
        if not np.isfinite(separation) or separation <= 0:
            raise ValueError("Geometric separation must be finite and positive")
        self.near_order, self.far_order, self.separation = near_order, far_order, separation
        super().__init__(mesh, material, quadrature=far_order, batch_pairs=batch_pairs)
        original, batches, orders = self.batches, [], []
        self.near_pairs, self.far_pairs = 0, 0
        length = mesh.lengths
        for batch in original:
            i, j = batch.i, batch.j
            distance = segment_distances(mesh.starts[i], mesh.ends[i], mesh.starts[j], mesh.ends[j])
            adjacent = ((i + 1) % mesh.count == j) | ((j + 1) % mesh.count == i)
            near = (i == j) | adjacent | (distance <= separation * np.maximum(length[i], length[j]))
            self.near_pairs += int(np.count_nonzero(near))
            self.far_pairs += int(np.count_nonzero(~near))
            if near_order == far_order:
                batches.append(batch)
                orders.append(near_order)
                continue
            for selected, order in [(near, near_order), (~near, far_order)]:
                if not np.any(selected):
                    continue
                prepared = (
                    batch
                    if order == far_order and np.all(selected)
                    else self._prepare(np.column_stack((i[selected], j[selected])), order)
                )
                for value in vars(prepared).values():
                    if isinstance(value, np.ndarray):
                        value.flags.writeable = False
                batches.append(prepared)
                orders.append(order)
        self.batches, self.orders = batches, tuple(orders)

    def statistics(self):
        points = sum(batch.measure.size for batch in self.batches)
        reference_points = self.mesh.count * (self.mesh.count + 1) // 2 + self.mesh.count
        reference_points *= self.near_order**2
        arrays = self.mass.nbytes + sum(
            value.nbytes
            for batch in self.batches
            for value in vars(batch).values()
            if isinstance(value, np.ndarray)
        )
        return dict(
            near_order=self.near_order,
            far_order=self.far_order,
            separation=self.separation,
            near_pairs=self.near_pairs,
            far_pairs=self.far_pairs,
            quadrature_points=points,
            uniform_near_order_points=reference_points,
            point_count_ratio=reference_points / points,
            retained_geometry_array_bytes=arrays,
            pair_selection_depends_on_frequency=False,
        )


def mixed_transfer(model, *, near_order=12, far_order=6, separation=2.0):
    transfer = NodalInterfaceTransfer(model, quadrature=near_order, prepared=False)
    transfer.prepared = [
        MixedQuadraturePrepared(
            mesh,
            model.reference.material,
            near_order=near_order,
            far_order=far_order,
            separation=separation,
        )
        for mesh in [model.reference.moving, model.reference.stationary]
    ]
    return transfer

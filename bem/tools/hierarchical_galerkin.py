"""Empirical far-field ACA assembly of the original V/H/D operators.

A shared tree encloses the UNION of P0-panel and P1-node supports, keeping
both vector components together. Near blocks are exact. Paired directions
share quadrature, preserving V/D bilinear reciprocity and BOTH distinct H
orientations. Dense output keeps the original global solves unchanged.

Row checks/ACA stopping are empirical, not certified matrix-error bounds.
Independent complete original-operator audits are required before use.
"""

import runpy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from finiteblock_bem.galerkin import GalerkinOperators

BLOCK = runpy.run_path(str(Path(__file__).with_name("galerkin_blocks.py")))


@dataclass(frozen=True)
class Cluster:
    labels: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    children: tuple

    @property
    def dofs(self):
        return (2 * self.labels[:, None] + np.arange(2)).ravel()

    @property
    def diameter(self):
        return np.linalg.norm(self.upper - self.lower)


def cluster_tree(mesh, leaf_panels):
    if not isinstance(leaf_panels, int) or leaf_panels < 2:
        raise ValueError("Require at least two basis labels per leaf")
    support = np.stack(
        [mesh.starts, mesh.ends, np.roll(mesh.starts, 1, axis=0), np.roll(mesh.ends, 1, axis=0)],
        axis=1,
    )
    lower, upper = support.min(axis=1), support.max(axis=1)
    centers = (lower + upper) / 2

    def build(labels):
        lo, hi = lower[labels].min(axis=0), upper[labels].max(axis=0)
        children = ()
        if len(labels) > leaf_panels:
            axis = np.argmax(np.ptp(centers[labels], axis=0))
            ordered = labels[np.argsort(centers[labels, axis], kind="stable")]
            children = (build(ordered[: len(labels) // 2]), build(ordered[len(labels) // 2 :]))
        labels = labels.copy()
        labels.flags.writeable = False
        lo.flags.writeable = hi.flags.writeable = False
        return Cluster(labels, lo, hi, children)

    return build(np.arange(mesh.count))


def partition(root, eta):
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("Require a positive geometric admissibility parameter")
    blocks, seen = [], set()

    def walk(a, b):
        key = tuple(sorted([id(a), id(b)]))
        if key in seen:
            return
        seen.add(key)
        gap = np.maximum(np.maximum(a.lower - b.upper, b.lower - a.upper), 0)
        distance = np.linalg.norm(gap)
        far = bool(distance > 0 and max(a.diameter, b.diameter) <= eta * distance)
        if far or not (a.children or b.children):
            blocks.append((a, b, far))
            return
        for x in a.children or (a,):
            for y in b.children or (b,):
                walk(x, y)

    walk(root, root)
    return tuple(blocks)


class PairedOracle:
    """Cache V_rc, H_rc, D_rc, H_cr.T, using a single pair evaluation."""

    def __init__(self, provider, rows, columns, s, prepared=None):
        self.provider, self.rows, self.columns, self.s = provider, rows, columns, s
        self.prepared = prepared
        self.values = np.empty((4, len(rows), len(columns)), dtype=float if s is None else complex)
        self.known = np.zeros(self.values.shape[1:], bool)
        self.calls = self.pairs = self.points = self.entries = 0

    def get(self, rows, columns):
        rows, columns = np.asarray(rows, int), np.asarray(columns, int)
        missing = ~self.known[np.ix_(rows, columns)]
        if np.any(missing):
            # Pair components are obtained by the same kernel evaluation.
            rr = rows[np.any(missing, axis=1)]
            cc = columns[np.any(missing, axis=0)]
            rr = np.unique(np.r_[rr // 2 * 2, rr // 2 * 2 + 1])
            cc = np.unique(np.r_[cc // 2 * 2, cc // 2 * 2 + 1])
            request = self.prepared
            if request is None or len(rr) != len(self.rows) or len(cc) != len(self.columns):
                request = self.provider.request(self.rows[rr], self.columns[cc])
            out = [np.zeros((len(rr), len(cc)), dtype=self.values.dtype) for _ in range(3)]
            reverse_h = np.zeros((len(cc), len(rr)), dtype=self.values.dtype)
            n = self.provider.mesh.dofs
            rlookup, clookup = np.full(n, -1), np.full(n, -1)
            rlookup[self.columns[cc]], clookup[self.rows[rr]] = (
                np.arange(len(cc)),
                np.arange(len(rr)),
            )
            for batch, maps in zip(request.batches, request.maps, strict=True):
                values = BLOCK["_PAIR"](self.provider.geometry, batch, self.s)
                for family, source, destination, positions in maps:
                    np.add.at(out[family].ravel(), destination, values[source].ravel()[positions])
                for source, indices in [(1, batch.hindices)] + (
                    [] if batch.self_pair else [(3, batch.reverse_hindices)]
                ):
                    r, c = rlookup[indices // n], clookup[indices % n]
                    valid = (r >= 0) & (c >= 0)
                    np.add.at(
                        reverse_h.ravel(),
                        (r[valid] * len(rr) + c[valid]).ravel(),
                        values[source].ravel()[np.flatnonzero(valid.ravel())],
                    )
            out[1] += 0.5 * request.mass
            i, j = self.columns[cc, None] // 2, self.rows[None, rr] // 2
            mass = (
                self.provider.mesh.lengths[i]
                / 2
                * (self.columns[cc, None] % 2 == self.rows[None, rr] % 2)
                * ((j == i) | (j == (i + 1) % self.provider.mesh.count))
            )
            reverse_h += 0.5 * mass
            fetched = np.stack([*out, reverse_h.T])
            if not np.all(np.isfinite(fetched)):
                raise ValueError("Nonfinite exact paired raw block")
            ix = np.ix_(np.arange(4), rr, cc)
            needed = ~self.known[np.ix_(rr, cc)]
            # Already-used values retain their bit patterns across later queries.
            self.values[ix] = np.where(needed[None], fetched, self.values[ix])
            self.known[np.ix_(rr, cc)] = True
            self.calls += 1
            self.pairs += sum(len(b.i) for b in request.batches)
            self.points += sum(b.measure.size for b in request.batches)
            self.entries += len(rr) * len(cc)
        return self.values[np.ix_(np.arange(4), rows, columns)]

    def full(self):
        return self.get(np.arange(len(self.rows)), np.arange(len(self.columns)))


@dataclass(frozen=True)
class CrossFactors:
    left: np.ndarray
    right: np.ndarray
    checked_rows: tuple

    def dense(self):
        return self.left @ self.right

    def transpose_apply(self, cotangent):
        return self.right.T @ (self.left.T @ cotangent)


def adaptive_cross(oracle, family, tolerance, *, max_rank=32, entry_fraction=0.75):
    """Scalar complex ACA with checked-row continuation and dense fallback.

    A converged update plus deterministic full-row checks is an empirical
    stop only. It is explicitly NOT a guarantee for every unqueried entry.
    """
    m, n = oracle.known.shape
    if not np.isfinite(tolerance) or tolerance <= 0 or max_rank < 1:
        raise ValueError("Require positive ACA tolerance/rank")
    left = np.empty((m, 0), dtype=oracle.values.dtype)
    right = np.empty((0, n), dtype=oracle.values.dtype)
    used_rows, used_columns, checked = set(), set(), set()
    row, small_update = 0, False
    iterations = 0
    while True:
        iterations += 1
        if iterations > 2 * (m + n + max_rank):
            return None
        if len(used_columns) >= min(max_rank, m, n) or np.mean(oracle.known) >= entry_fraction:
            return None
        row_value = oracle.get([row], np.arange(n))[family, 0] - left[row] @ right
        candidates = abs(row_value).copy()
        if used_columns:
            candidates[list(used_columns)] = 0
        col = int(np.argmax(candidates))
        if np.linalg.norm(candidates) * np.sqrt(m) <= 0.1 * tolerance:
            small_update = True
        else:
            column = oracle.get(np.arange(m), [col])[family, :, 0] - left @ right[:, col]
            pivot = row_value[col]
            u = column / pivot
            v = row_value
            if not np.all(np.isfinite(u)):
                return None
            left = np.column_stack([left, u])
            right = np.vstack([right, v])
            used_columns.add(col)
            used_rows.add(row)
            small_update = np.linalg.norm(u) * np.linalg.norm(v) <= 0.1 * tolerance
            scores = abs(u).copy()
            scores[list(used_rows)] = -1
            row = int(np.argmax(scores))
        if small_update:
            # Include both polarizations and separated parts of the row cluster.
            probes = np.unique(np.r_[0, 1, m // 3, m // 2, max(0, m - 2), m - 1])
            exact = oracle.get(probes, np.arange(n))[family]
            residual = exact - left[probes] @ right
            row_errors = np.linalg.norm(residual, axis=1)
            checked.update(probes.tolist())
            if np.max(row_errors) <= 0.25 * tolerance / np.sqrt(m):
                return CrossFactors(left, right, tuple(sorted(checked)))
            row = int(probes[np.argmax(row_errors)])
            if row in used_rows:
                return None
            small_update = False


class HierarchicalGalerkin:
    def __init__(
        self,
        mesh,
        material,
        *,
        quadrature=12,
        leaf_panels=24,
        eta=1.0,
        rtol=1e-11,
        atol=1e-14,
        max_rank=32,
        aca_safety=0.05,
    ):
        if not 0 < rtol < 1 or not 0 < atol < 1 or max_rank < 1:
            raise ValueError("Require explicit positive compression tolerances/rank")
        self.provider = BLOCK["GalerkinBlockProvider"](mesh, material, quadrature=quadrature)
        self.blocks = partition(cluster_tree(mesh, leaf_panels), eta)
        if not np.isfinite(aca_safety) or not 0 < aca_safety <= 1:
            raise ValueError("Require a conservative ACA internal tolerance factor")
        self.rtol, self.atol, self.max_rank = rtol, atol, max_rank
        self.aca_safety = aca_safety
        started = perf_counter()
        self.near_requests, self.pinned_geometry_bytes = {}, 0
        for a, b, far in self.blocks:
            if far:
                continue
            request = self.provider.request(a.dofs, b.dofs)
            info = request.statistics()
            size = info["prepared_geometry_bytes"] + info["map_bytes"] + request.mass.nbytes
            if self.pinned_geometry_bytes + size <= 512 * 1024**2:
                self.near_requests[id(a), id(b)] = request
                self.pinned_geometry_bytes += size
        self.geometry_preparation_seconds = perf_counter() - started

    def evaluate(self, s=None):
        n = self.provider.mesh.dofs
        matrices = [np.zeros((n, n), dtype=float if s is None else complex) for _ in range(3)]
        near_squared = np.zeros(3)
        records = []
        far_entries = sum(2 * len(a.dofs) * len(b.dofs) for a, b, far in self.blocks if far)
        total_pairs = total_calls = total_points = 0

        def publish(a, b, values):
            r, c = a.dofs, b.dofs
            for k in range(3):
                matrices[k][np.ix_(r, c)] = values[k]
            if a is not b:
                matrices[0][np.ix_(c, r)] = values[0].T
                matrices[2][np.ix_(c, r)] = values[2].T
                matrices[1][np.ix_(c, r)] = values[3].T

        for a, b, far in self.blocks:
            if far:
                continue
            oracle = PairedOracle(
                self.provider, a.dofs, b.dofs, s, self.near_requests.get((id(a), id(b)))
            )
            values = oracle.full()
            publish(a, b, values)
            for k in range(3):
                near_squared[k] += np.linalg.norm(values[k]) ** 2
            if a is not b:
                near_squared[0] += np.linalg.norm(values[0]) ** 2
                near_squared[2] += np.linalg.norm(values[2]) ** 2
                near_squared[1] += np.linalg.norm(values[3]) ** 2
            total_pairs += oracle.pairs
            total_calls += oracle.calls
            total_points += oracle.points
            records.append(
                dict(
                    shape=list(oracle.known.shape),
                    kind="exact_near",
                    calls=oracle.calls,
                    pair_evaluations=oracle.pairs,
                )
            )
        near_norm = np.sqrt(near_squared)
        for a, b, far in self.blocks:
            if not far:
                continue
            oracle = PairedOracle(self.provider, a.dofs, b.dofs, s)
            weight = np.sqrt(len(a.dofs) * len(b.dofs) / far_entries)
            tolerances = self.aca_safety * (self.atol + self.rtol * near_norm) * weight
            values, factors = [], []
            for family in range(4):
                tolerance = tolerances[family if family < 3 else 1]
                factor = adaptive_cross(oracle, family, tolerance, max_rank=self.max_rank)
                if factor is None:
                    values = oracle.full()
                    factors = []
                    break
                factors.append(factor)
                values.append(factor.dense())
            values = np.asarray(values)
            publish(a, b, values)
            total_pairs += oracle.pairs
            total_calls += oracle.calls
            total_points += oracle.points
            records.append(
                dict(
                    shape=list(oracle.known.shape),
                    kind="empirical_aca" if factors else "exact_fallback",
                    ranks=[f.left.shape[1] for f in factors],
                    calls=oracle.calls,
                    pair_evaluations=oracle.pairs,
                    exact_entry_fraction=float(np.mean(oracle.known)),
                )
            )
        mass = np.zeros((n, n))
        for i in range(self.provider.mesh.count):
            for j in [i, (i + 1) % self.provider.mesh.count]:
                mass[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] += (
                    self.provider.mesh.lengths[i] / 2 * np.eye(2)
                )
        return GalerkinOperators(*matrices, mass), dict(
            blocks=records,
            near_norm_lower_bounds=near_norm.tolist(),
            raw_rtol=self.rtol,
            raw_atol=self.atol,
            internal_aca_safety_factor=self.aca_safety,
            source_pair_evaluations=total_pairs,
            source_batch_requests=total_calls,
            source_quadrature_points=total_points,
            original_complete_panel_pairs=self.provider.mesh.count
            * (self.provider.mesh.count + 1)
            // 2,
            pinned_near_geometry_bytes=self.pinned_geometry_bytes,
            geometry_preparation_seconds=self.geometry_preparation_seconds,
            certified_error_bound=False,
            frequency_compression=False,
            dense_global_output_and_solve=True,
            v_d_bilinear_reciprocity_preserved=True,
        )

"""Pack selected exact panel pairs into full batches of one quadrature kind.

The original sparse requests preserve global batch boundaries for bitwise
controls. ACA fibers otherwise visit many one-pair batches. This adapter
reuses their geometry and maps, changing floating summation order only.
Independent complete-operator audits remain required.
"""

import runpy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import FunctionType

import numpy as np

TOOLS = Path(__file__).resolve().parent
BASE = runpy.run_path(str(TOOLS / "galerkin_blocks.py"))
HIERARCHY = runpy.run_path(str(TOOLS / "hierarchical_galerkin.py"))


def pack_request(request, provider=None):
    groups = {}
    for batch in request.batches:
        groups.setdefault((batch.self_pair, batch.measure.shape[1]), []).append(batch)
    packed = []
    for batches in groups.values():
        sample = batches[0]
        arrays = {
            name: np.concatenate([getattr(b, name) for b in batches], axis=0)
            for name, value in vars(sample).items()
            if isinstance(value, np.ndarray)
        }
        for array in arrays.values():
            array.flags.writeable = False
        for first in range(0, len(arrays["i"]), request.provider.batch_pairs):
            last = first + request.provider.batch_pairs
            packed.append(
                replace(sample, **{key: value[first:last] for key, value in arrays.items()})
            )
    n = request.provider.mesh.dofs
    rows, columns = request.rows, request.columns
    rlookup, clookup = np.full(n, -1), np.full(n, -1)
    rlookup[rows], clookup[columns] = np.arange(len(rows)), np.arange(len(columns))
    maps = []
    for b in packed:
        indices = [(0, 0, b.vindices), (1, 1, b.hindices), (2, 2, b.dindices)]
        if not b.self_pair:
            indices += [
                (0, 0, (b.vindices % n) * n + b.vindices // n),
                (2, 2, (b.dindices % n) * n + b.dindices // n),
                (1, 3, b.reverse_hindices),
            ]
        block_maps = []
        for family, source, index in indices:
            r, c = rlookup[index // n], clookup[index % n]
            valid = (r >= 0) & (c >= 0)
            destination = (r[valid] * len(columns) + c[valid]).ravel()
            positions = np.flatnonzero(valid.ravel())
            destination.flags.writeable = positions.flags.writeable = False
            block_maps.append((family, source, destination, positions))
        maps.append(tuple(block_maps))
    result = type(request).__new__(type(request))
    result.__dict__.update(request.__dict__)
    result.batches, result.maps = tuple(packed), tuple(maps)
    if provider is not None:
        result.provider = provider
    return result


class PackedBlockProvider(BASE["GalerkinBlockProvider"]):
    def request(self, rows, columns, *, max_pairs=1000000):
        return pack_request(super().request(rows, columns, max_pairs=max_pairs))


class PackedHierarchicalGalerkin(HIERARCHY["HierarchicalGalerkin"]):
    def __init__(self, mesh, material, **kwargs):
        super().__init__(mesh, material, **kwargs)
        start = perf_counter()
        self.provider = PackedBlockProvider(
            mesh,
            material,
            quadrature=self.provider.quadrature,
            batch_pairs=self.provider.batch_pairs,
        )
        for key, request in self.near_requests.items():
            self.near_requests[key] = pack_request(request, self.provider)
        self.packing_seconds = perf_counter() - start
        self.geometry_preparation_seconds += self.packing_seconds

    def evaluate(self, s=None):
        cache = PairValueCache(self.provider, s)
        original_get = HIERARCHY["PairedOracle"].get
        get_globals = {
            **original_get.__globals__,
            "BLOCK": {**HIERARCHY["BLOCK"], "_PAIR": cache.evaluate},
        }
        cached_get = FunctionType(
            original_get.__code__, get_globals, original_get.__name__, original_get.__defaults__
        )
        cached_oracle = type(
            "FrequencyCachedOracle", (HIERARCHY["PairedOracle"],), {"get": cached_get}
        )
        original_evaluate = HIERARCHY["HierarchicalGalerkin"].evaluate
        evaluate = FunctionType(
            original_evaluate.__code__,
            {**original_evaluate.__globals__, "PairedOracle": cached_oracle},
            original_evaluate.__name__,
            original_evaluate.__defaults__,
        )
        operators, record = evaluate(self, s)
        return operators, record | dict(
            selected_pair_batching="packed_by_quadrature_kind",
            packing_preparation_seconds=self.packing_seconds,
            floating_summation_order_changed=True,
            requested_panel_pair_visits=record["source_pair_evaluations"],
            requested_quadrature_point_visits=record["source_quadrature_points"],
            source_pair_evaluations=cache.evaluations,
            source_quadrature_points=cache.points,
            exact_kernel_batch_calls=cache.calls,
            frequency_local_pair_cache_bytes=cache.nbytes,
            pair_cache_enabled=cache.enabled,
        )


class PairValueCache:
    """Bounded per-frequency exact local contributions, without cached geometry.

    Dense output is already quadratic; this prototype allocates at most
    128 MiB extra. Larger meshes automatically use the uncached evaluator.
    It is private to ONE evaluation and never shares values across frequencies.
    """

    def __init__(self, provider, s, max_bytes=128 * 1024**2):
        self.provider, self.s = provider, s
        self.n = provider.mesh.count
        count = self.n * (self.n + 1) // 2
        dtype = np.dtype(float if s is None else complex)
        self.nbytes = count * (36 * dtype.itemsize + 1)
        self.enabled = self.nbytes <= max_bytes
        self.known = np.zeros(count, bool) if self.enabled else None
        self.data = np.empty((count, 36), dtype=dtype) if self.enabled else None
        if not self.enabled:
            self.nbytes = 0
        self.evaluations = self.points = self.calls = 0

    def evaluate(self, geometry, batch, s=None):
        if (
            geometry.mesh is not self.provider.mesh
            or geometry.material is not self.provider.material
            or geometry.reference != self.provider.geometry.reference
            or s != self.s
        ):
            raise ValueError("A local pair cache cannot cross meshes or frequencies")
        if not self.enabled:
            self.evaluations += len(batch.i)
            self.points += batch.measure.size
            self.calls += 1
            return BASE["_PAIR"](geometry, batch, s)
        indices = batch.i * self.n - batch.i * (batch.i - 1) // 2 + batch.j - batch.i
        missing = ~self.known[indices]
        if np.any(missing):
            subset = replace(
                batch,
                **{
                    name: value[missing]
                    for name, value in vars(batch).items()
                    if isinstance(value, np.ndarray)
                },
            )
            values = BASE["_PAIR"](geometry, subset, s)
            count = int(np.count_nonzero(missing))
            reverse = (
                np.zeros((count, 2, 4), dtype=self.data.dtype) if values[3] is None else values[3]
            )
            self.data[indices[missing]] = np.concatenate(
                [a.reshape(count, -1) for a in [*values[:3], reverse]], axis=1
            )
            self.known[indices[missing]] = True
            self.evaluations += count
            self.points += subset.measure.size
            self.calls += 1
        values = self.data[indices]
        return (
            values[:, :4].reshape(-1, 2, 2),
            values[:, 4:12].reshape(-1, 2, 4),
            values[:, 12:28].reshape(-1, 4, 4),
            None if batch.self_pair else values[:, 28:].reshape(-1, 2, 4),
        )

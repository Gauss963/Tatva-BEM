"""Selected exact P0/P1 Galerkin blocks without full matrices or pair geometry.

The original batched quadrature/Green/traction/singular-correction expressions
are extracted verbatim. Only requested supporting panel pairs are prepared.
Original pair-group/batch order and bilinear reciprocal scatters are retained.
No low-rank, kernel, quadrature or frequency approximation is introduced.
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass

import numpy as np

from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.mesh import BoundaryMesh

_SCATTER = """
np.add.at(vmat.ravel(), b.vindices, local_v)
np.add.at(hmat.ravel(), b.hindices, local_k)
np.add.at(dmat.ravel(), b.dindices, local_d.reshape(p, 4, 4))
if not b.self_pair:
    np.add.at(vmat.ravel(), (b.vindices % n) * n + b.vindices // n, local_v)
    np.add.at(dmat.ravel(), (b.dindices % n) * n + b.dindices // n, local_d.reshape(p, 4, 4))
    local_reverse = np.einsum("pq,pqab,pqi->paib", b.measure, reverse, b.px).reshape(p, 2, 4)
    np.add.at(hmat.ravel(), b.reverse_hindices, local_reverse)
"""


def _dump(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


def _extract_pair_evaluator():
    original = PreparedGalerkinOperators.assemble
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    loops = [i for i, n in enumerate(function.body) if isinstance(n, ast.For)]
    if len(loops) != 1:
        raise ValueError("Original batch loop changed")
    k = loops[0]
    loop = function.body[k]
    allocations = ast.parse(
        "vmat = np.zeros((n, n), dtype=float if s is None else complex)\nhmat, dmat = vmat.copy(), vmat.copy()"
    ).body
    tail = ast.parse(
        "hmat += 0.5 * self.mass\nreturn GalerkinOperators(vmat, hmat, dmat, self.mass)"
    ).body
    if (
        _dump(function.body[k - 2 : k]) != _dump(allocations)
        or _dump(function.body[k + 1 :]) != _dump(tail)
        or ast.dump(loop.target) != ast.dump(ast.Name(id="b", ctx=ast.Store()))
        or ast.dump(loop.iter) != ast.dump(ast.parse("self.batches", mode="eval").body)
    ):
        raise ValueError("Original allocation/scatter contract changed")
    expected = ast.parse(_SCATTER).body
    if _dump(loop.body[-4:]) != _dump(expected):
        raise ValueError("Original reciprocal scatter changed")
    numerical = function.body[: k - 2] + loop.body[:-4]
    reverse = loop.body[-1].body[2]
    # Every numerical statement is the original AST; outputs replace allocation
    # and accumulation only. The reverse-H expression is copied from its branch.
    function.name = "pair_contributions"
    function.args.args.insert(1, ast.arg(arg="b"))
    function.body = (
        numerical
        + ast.parse("local_reverse = None").body
        + [
            ast.If(test=loop.body[-1].test, body=[reverse], orelse=[]),
            ast.Return(
                value=ast.parse(
                    "(local_v, local_k, local_d.reshape(p, 4, 4), local_reverse)", mode="eval"
                ).body
            ),
        ]
    )
    if _dump(function.body[: len(numerical)]) != _dump(numerical):
        raise RuntimeError("Original local quadrature expressions changed")
    namespace = dict(original.__globals__)
    exec(
        compile(ast.fix_missing_locations(tree), original.__code__.co_filename + "|blocks", "exec"),
        namespace,
    )
    return namespace[function.name]


_PAIR = _extract_pair_evaluator()


def pair_order(i, j, count):
    """Original self/adjacent/disjoint group and ordinal, without an O(n²) list."""
    if i == j:
        return 0, i
    if j == i + 1 or (i == 0 and j == count - 1):
        return 1, (0 if j == 1 else 1) if i == 0 else i + 1
    upper = i * count - i * (i - 1) // 2 + j - i
    return 2, upper - (2 if i == 0 else 2 * i + 3)


def _indices(values, size):
    values = np.asarray(values)
    if (
        values.ndim != 1
        or values.dtype.kind not in "iu"
        or not len(values)
        or np.any(values < 0)
        or np.any(values >= size)
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError("Require distinct nonempty integer DOF indices inside the mesh")
    result = values.astype(np.int64, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class GalerkinBlocks:
    """V(P0,P0), H(P0,P1), D(P1,P1); equal labels have different basis roles."""

    rows: np.ndarray
    columns: np.ndarray
    single_layer: np.ndarray
    trace_double_layer: np.ndarray
    hypersingular: np.ndarray
    mixed_mass: np.ndarray


class GalerkinBlockProvider:
    def __init__(self, mesh, material, *, quadrature=12, batch_pairs=32):
        if (
            not isinstance(mesh, BoundaryMesh)
            or mesh.count < 3
            or not isinstance(quadrature, int)
            or quadrature < 4
            or quadrature % 2
            or not isinstance(batch_pairs, int)
            or batch_pairs < 1
        ):
            raise ValueError("Require a closed mesh, even quadrature >=4 and positive batch size")
        self.mesh, self.material = mesh, material
        self.quadrature, self.batch_pairs = quadrature, batch_pairs
        # _prepare needs only these three fields. Neither the dense mixed mass
        # nor the full O(n²) pair list/cache is constructed.
        self.geometry = PreparedGalerkinOperators.__new__(PreparedGalerkinOperators)
        self.geometry.mesh, self.geometry.material = mesh, material
        self.geometry.reference = max(np.ptp(mesh.starts, axis=0).max(), 1.0)

    def request(self, rows, columns, *, max_pairs=1000000):
        return GalerkinBlockRequest(self, rows, columns, max_pairs=max_pairs)


class GalerkinBlockRequest:
    def __init__(self, provider, rows, columns, *, max_pairs):
        mesh = provider.mesh
        self.rows, self.columns = _indices(rows, mesh.dofs), _indices(columns, mesh.dofs)
        self.provider = provider
        r0, c0 = np.unique(self.rows // 2), np.unique(self.columns // 2)
        r1, c1 = [np.union1d(a, (a - 1) % mesh.count) for a in [r0, c0]]
        if not isinstance(max_pairs, int) or max_pairs < 1:
            raise ValueError("Require a positive panel-pair preparation budget")
        pairs = set()
        for rr, cc in [(r0, c0), (r0, c1), (r1, c1)]:
            for i in rr:
                for j in cc:
                    pairs.add((int(min(i, j)), int(max(i, j))))
                    if len(pairs) > max_pairs:
                        raise ValueError("Requested block exceeds its panel-pair budget")
        ordered = sorted(pairs, key=lambda pair: pair_order(*pair, mesh.count))
        groups = {}
        for pair in ordered:
            kind, ordinal = pair_order(*pair, mesh.count)
            groups.setdefault((kind, ordinal // provider.batch_pairs), []).append(pair)
        rlookup, clookup = np.full(mesh.dofs, -1), np.full(mesh.dofs, -1)
        rlookup[self.rows], clookup[self.columns] = (
            np.arange(len(self.rows)),
            np.arange(len(self.columns)),
        )
        self.batches, self.maps = [], []
        for pair_list in groups.values():
            batch = provider.geometry._prepare(pair_list, provider.quadrature)
            for value in vars(batch).values():
                if isinstance(value, np.ndarray):
                    value.flags.writeable = False
            self.batches.append(batch)
            indices = [(0, 0, batch.vindices), (1, 1, batch.hindices), (2, 2, batch.dindices)]
            if not batch.self_pair:
                indices += [
                    (0, 0, (batch.vindices % mesh.dofs) * mesh.dofs + batch.vindices // mesh.dofs),
                    (2, 2, (batch.dindices % mesh.dofs) * mesh.dofs + batch.dindices // mesh.dofs),
                    (1, 3, batch.reverse_hindices),
                ]
            mappings = []
            for family, source, index in indices:
                r, c = rlookup[index // mesh.dofs], clookup[index % mesh.dofs]
                valid = (r >= 0) & (c >= 0)
                destination = (r[valid] * len(self.columns) + c[valid]).ravel()
                source_indices = np.flatnonzero(valid.ravel())
                destination.flags.writeable = source_indices.flags.writeable = False
                mappings.append((family, source, destination, source_indices))
            self.maps.append(tuple(mappings))
        # M maps P1 nodal displacement to P0 panel test functions.
        i, j = self.rows[:, None] // 2, self.columns[None, :] // 2
        same_component = self.rows[:, None] % 2 == self.columns[None, :] % 2
        self.mass = (
            (mesh.lengths[i] / 2) * same_component * ((j == i) | (j == (i + 1) % mesh.count))
        )
        self.mass.flags.writeable = False
        self.batches, self.maps = tuple(self.batches), tuple(self.maps)

    def evaluate(self, s=None):
        if s is not None and (not np.isfinite(s) or np.real(s) <= 0):
            raise ValueError("Laplace parameter must have positive real part")
        matrices = [
            np.zeros((len(self.rows), len(self.columns)), dtype=float if s is None else complex)
            for _ in range(3)
        ]
        for batch, maps in zip(self.batches, self.maps, strict=True):
            values = _PAIR(self.provider.geometry, batch, s)
            for family, source, destination, positions in maps:
                np.add.at(matrices[family].ravel(), destination, values[source].ravel()[positions])
        matrices[1] += 0.5 * self.mass
        return GalerkinBlocks(self.rows, self.columns, *matrices, self.mass)

    def statistics(self):
        n = self.provider.mesh.count
        return dict(
            total_panel_pairs=n * (n + 1) // 2,
            prepared_panel_pairs=sum(len(b.i) for b in self.batches),
            prepared_quadrature_points=sum(b.measure.size for b in self.batches),
            prepared_geometry_bytes=sum(
                a.nbytes
                for b in self.batches
                for a in vars(b).values()
                if isinstance(a, np.ndarray)
            ),
            map_bytes=sum(a.nbytes for maps in self.maps for _, _, *arrays in maps for a in arrays),
            output_entries_per_matrix=len(self.rows) * len(self.columns),
            full_entries_per_matrix=self.provider.mesh.dofs**2,
            requested_full_output=(
                len(self.rows) == self.provider.mesh.dofs
                and len(self.columns) == self.provider.mesh.dofs
            ),
            prepared_complete_pair_set=(sum(len(b.i) for b in self.batches) == n * (n + 1) // 2),
            unrequested_dense_output_allocated=False,
            approximation=False,
        )

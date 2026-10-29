"""Shared body/probe raw-layer sampling with stable fixed-trace recovery.

The original Chebyshev/tail algorithm is unchanged. Its output shape becomes
rectangular and its condensed check is strengthened to every observation row.
Optional interval bookkeeping restores the identical original partition.
"""

import ast
import inspect
import runpy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from finiteblock_bem.galerkin import GalerkinOperators, assemble_galerkin_operators

TOOLS = Path(__file__).resolve().parent
RAW = runpy.run_path(str(TOOLS / "raw_layer_sampling.py"))
CHECKPOINT = runpy.run_path(str(TOOLS / "radau_sampling_checkpoint.py"))
FIELD = runpy.run_path(str(TOOLS / "interior_fields.py"))
SCHUR = runpy.run_path(str(TOOLS / "trace_schur_experiment.py"))["TraceSchurTransfer"]
KEYS = ["single", "double", "single_gradient", "double_gradient"]


@dataclass(frozen=True)
class InteriorPacket:
    families: tuple
    masses: tuple

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.families)

    def validate(self, reference=None):
        if len(self.masses) != 2 or len(self.families) != 14:
            raise ValueError("Require six body layers and eight probe maps")
        if any(
            a.ndim != 2 or min(a.shape) < 1 or not np.all(np.isfinite(a))
            for a in [*self.families, *self.masses]
        ):
            raise ValueError("Invalid finite raw body/probe matrices")
        for i, mass in enumerate(self.masses):
            if mass.shape[0] != mass.shape[1] or any(
                a.shape != mass.shape for a in self.families[3 * i : 3 * i + 3]
            ):
                raise ValueError("Body layers must match the fixed mixed mass")
            maps = self.families[6 + 4 * i : 10 + 4 * i]
            if (
                any(a.shape[1] != len(mass) for a in maps)
                or maps[0].shape != maps[1].shape
                or maps[2].shape != maps[3].shape
                or maps[2].shape[0] != 2 * maps[0].shape[0]
            ):
                raise ValueError("Probe potential/gradient shapes changed")
        if reference is not None:
            if (
                len(reference.families) != len(self.families)
                or any(
                    a.shape != b.shape
                    for a, b in zip(self.families, reference.families, strict=True)
                )
                or any(
                    not np.array_equal(a, b)
                    for a, b in zip(self.masses, reference.masses, strict=True)
                )
            ):
                raise ValueError("Fixed body/probe geometry changed")

    def operators(self):
        return [
            GalerkinOperators(*self.families[3 * i : 3 * i + 3], mass)
            for i, mass in enumerate(self.masses)
        ]

    def probe_maps(self):
        return [
            dict(zip(KEYS, self.families[6 + 4 * i : 10 + 4 * i], strict=True)) for i in range(2)
        ]


class StableInteriorTransfer:
    def __init__(self, model, points, *, quadrature=12, probe_quadrature=16, prepared=True):
        if len(points) != 2:
            raise ValueError("Provide both bodies probe coordinates")
        self.model, self.quadrature = model, quadrature
        self.bulk = SCHUR(model, quadrature=quadrature, prepared=prepared)
        self.potentials = [
            FIELD["InteriorPotentials"](
                mesh, model.reference.material, p, quadrature=probe_quadrature
            )
            for mesh, p in zip(
                [model.reference.moving, model.reference.stationary], points, strict=True
            )
        ]

    def raw(self, s):
        if not np.isfinite(s) or s.real <= 0:
            raise ValueError("Require Re(s)>0")
        meshes = [self.model.reference.moving, self.model.reference.stationary]
        operators = (
            [prepared.assemble(s) for prepared in self.bulk.prepared]
            if self.bulk.prepared is not None
            else [
                assemble_galerkin_operators(
                    mesh, self.model.reference.material, s, quadrature=self.quadrature
                )
                for mesh in meshes
            ]
        )
        maps = [p.matrices(s) for p in self.potentials]
        return InteriorPacket(
            tuple(
                a
                for op in operators
                for a in [op.single_layer, op.trace_double_layer, op.hypersingular]
            )
            + tuple(m[k] for m in maps for k in KEYS),
            tuple(op.mixed_mass for op in operators),
        )

    def evaluate_packet(self, packet, s):
        operators = packet.operators()
        impedance, boundary = self.bulk.evaluate_operators(operators, reconstruction=True)
        displacement, gradient = [], []
        offset = 0
        for mesh, op, maps in zip(
            [self.model.reference.moving, self.model.reference.stationary],
            operators,
            packet.probe_maps(),
            strict=True,
        ):
            trace = boundary[offset : offset + mesh.dofs]
            offset += mesh.dofs
            density = solve(op.single_layer, op.trace_double_layer @ trace, check_finite=False)
            displacement.append(maps["single"] @ density - maps["double"] @ trace)
            gradient.append(maps["single_gradient"] @ density - maps["double_gradient"] @ trace)
        displacement, gradient = np.vstack(displacement), np.vstack(gradient)
        d = impedance.shape[0]
        tensor = gradient.reshape(-1, 2, 2, d).transpose(3, 0, 1, 2)
        stress = (
            FIELD["plane_strain_stress"](tensor, self.model.reference.material)
            .transpose(1, 2, 0)
            .reshape(-1, d)
        )
        return dict(
            impedance=impedance,
            boundary_displacement=boundary,
            displacement=displacement,
            gradient=gradient,
            stress=stress,
            velocity=s * displacement,
        )

    def evaluate(self, s):
        return self.evaluate_packet(self.raw(s), s)


def maximum_row_error(actual, predicted, options):
    errors = []
    for a, b in zip(actual, predicted, strict=True):
        errors.append(
            float(
                np.linalg.norm(a - b)
                / max(
                    options.output_atol + options.output_rtol * np.linalg.norm(a),
                    np.finfo(float).tiny,
                )
            )
        )
        errors.append(
            float(
                np.max(
                    np.linalg.norm(a - b, axis=1)
                    / np.maximum(
                        options.output_atol + options.output_rtol * np.linalg.norm(a, axis=1),
                        np.finfo(float).tiny,
                    )
                )
            )
        )
    return max(errors)


def rectangular_sampler(outputs, inputs, *, stack=None, start=0, interval=None):
    """Adapt only output shape/check and optional restart bookkeeping; prove AST identity."""
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in [outputs, inputs]):
        raise ValueError("Positive rectangular channel counts required")
    tree = ast.parse(inspect.getsource(RAW["sample_raw_layers"]))
    baseline = ast.dump(tree, include_attributes=False)
    restores = []
    shapes = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Compare)
        and ast.dump(n.left) == ast.dump(ast.parse("value.shape", mode="eval").body)
    ]
    errors = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "output_error" for t in n.targets)
        and isinstance(n.value, ast.Call)
    ]
    if (
        len(shapes) != 1
        or len(errors) != 1
        or ast.dump(shapes[0].comparators[0])
        != ast.dump(ast.parse("(plan.dofs,plan.dofs)", mode="eval").body)
    ):
        raise ValueError("Original raw sampler output guards changed")
    restores.append((shapes[0], "comparators", shapes[0].comparators))
    shapes[0].comparators = [ast.Name(id="_rectangular_shape", ctx=ast.Load())]
    restores.append((errors[0], "value", errors[0].value))
    errors[0].value = ast.parse(
        "_maximum_row_error(actual_values,predicted_values,options)", mode="eval"
    ).body
    namespace = {
        **RAW["sample_raw_layers"].__globals__,
        "RawLayerPacket": InteriorPacket,
        "_rectangular_shape": (outputs, inputs),
        "_maximum_row_error": maximum_row_error,
    }
    callback = None
    loop = None
    if stack is not None:
        if interval is None:
            raise ValueError("Restored sampler needs its interval callback")
        assignments = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "stack"
        ]
        counters = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.keyword) and n.arg == "transformed_grid_points"
        ]
        loops = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.While) and isinstance(n.test, ast.Name) and n.test.id == "stack"
        ]
        if len(assignments) != 1 or len(counters) != 1 or len(loops) != 1:
            raise ValueError("Original raw sampler restart boundaries changed")
        if (
            ast.dump(assignments[0].value) != ast.dump(ast.parse("[(0,half+1)]", mode="eval").body)
            or not isinstance(counters[0].value, ast.Constant)
            or counters[0].value.value != 0
        ):
            raise ValueError("Original raw sampler initial state changed")
        restores.extend(
            [
                (assignments[0], "value", assignments[0].value),
                (counters[0], "value", counters[0].value),
            ]
        )
        assignments[0].value = ast.parse("list(_restored_stack)", mode="eval").body
        counters[0].value = ast.Name(id="_restored_start", ctx=ast.Load())
        loop = loops[0]
        if ast.dump(loop.body[0]) != ast.dump(ast.parse("left,right=stack.pop()").body[0]):
            raise ValueError("Original raw sampler interval entry changed")
        callback = ast.parse("_interval(left,right,stack)").body[0]
        loop.body.insert(1, callback)
        namespace.update(
            _restored_stack=list(map(tuple, stack)), _restored_start=start, _interval=interval
        )
    # Restore every allowed edit; no interpolation/tail/acceptance statement changes.
    changed = [(node, key, getattr(node, key)) for node, key, _ in restores]
    for node, key, old in restores:
        setattr(node, key, old)
    if callback is not None:
        loop.body.pop(1)
    if ast.dump(tree, include_attributes=False) != baseline:
        raise RuntimeError("Rectangular adapter changed another raw numerical statement")
    for node, key, new in changed:
        setattr(node, key, new)
    if callback is not None:
        loop.body.insert(1, callback)
    adapted = {}
    exec(
        compile(
            ast.fix_missing_locations(tree),
            str(TOOLS / "raw_layer_sampling.py") + "|rectangular",
            "exec",
        ),
        namespace,
        adapted,
    )
    original = adapted["sample_raw_layers"]

    def sample(raw, transform, plan, consume, **kwargs):
        if stack is not None:
            CHECKPOINT["validate_stack"](stack, plan.fft_size // 2 + 1, start)
        return original(raw, transform, plan, consume, **kwargs)

    return sample

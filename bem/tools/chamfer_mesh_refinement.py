"""Mesh-only adapter for the pinned Tatva chamfered quad4 reference model.

The archived generator rejects refinement plus a chamfer. This adapter adds
the actual contact endpoint to both bodies' graded y axes, then reuses its
original quad construction and exact polygon mapping. Copied function globals
select the adapted generator: original bindings and source files are unchanged.
Constitutive, contact, static and dynamic kernels are not replaced.
"""

from types import FunctionType


def rebind(function, replacements):
    """Copy a function while changing only explicitly named global bindings."""
    if not isinstance(function, FunctionType):
        raise TypeError("A Python reference function is required")
    result = FunctionType(
        function.__code__,
        {**function.__globals__, **replacements},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    result.__kwdefaults__ = (
        None if function.__kwdefaults__ is None else function.__kwdefaults__.copy()
    )
    result.__annotations__ = function.__annotations__.copy()
    return result


def make_chamfer_refined_builder(module):
    """Return a build_case_model replacement for explicit mesh-control studies.

    A null fault_end_refinement_size calls the original builder directly.
    The original RunConfig still controls material/loading and validates the
    physical time-step limit, including its existing graded pair-mass bound.
    """
    original = {
        name: getattr(module, name)
        for name in [
            "build_case_model",
            "build_block_model",
            "create_structured_quad_block",
            "_structured_2d_grid",
        ]
    }

    def build(case, config):
        if config.fault_end_refinement_size is None:
            return original["build_case_model"](case, config)
        if (
            config.dimension != 2
            or config.element_type != "quad4"
            or config.moving_leading_chamfer_along_fault <= 0
            or config.moving_leading_chamfer_perpendicular <= 0
        ):
            raise ValueError("This adapter requires the explicit 2D quad4 chamfer refinement case")
        contact_end = (
            float(case.moving.origin[1])
            + float(case.moving.dimensions[1])
            - float(config.moving_leading_chamfer_along_fault)
        )

        def quad(spec, mesh_size, dtype, **kwargs):
            refinement = kwargs.pop("refinement", None)
            if refinement is None:
                raise ValueError("The refined model must pass its explicit grading specification")
            grading = {**refinement, "y_points": sorted({*refinement["y_points"], contact_end})}

            def grid(grid_spec, grid_size, grid_dtype, refinement=None):
                return original["_structured_2d_grid"](grid_spec, grid_size, grid_dtype, grading)

            # The original generator sees refinement=None only for its unsupported
            # combination guard; its actual grid comes from the explicit closure.
            generator = rebind(
                original["create_structured_quad_block"], {"_structured_2d_grid": grid}
            )
            return generator(spec, mesh_size, dtype, refinement=None, **kwargs)

        block = rebind(original["build_block_model"], {"create_structured_quad_block": quad})
        builder = rebind(original["build_case_model"], {"build_block_model": block})
        result = builder(case, config)
        if any(getattr(module, name) is not value for name, value in original.items()):
            raise RuntimeError("Original FEM function bindings changed during adapted construction")
        return result

    return build

"""Guarded O(n) solves for the equal-material straight-contact radiation limit.

This is a research primitive for Cinf, not an approximation of a CQ lag-zero
matrix. The full reflected remainder Z(s)-s*Cinf is still required by a dynamic
solver. Coordinates retain the production order (all normal jumps, all
tangential jumps, platen). A dense constrained-energy reference is mandatory.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, solve_banded, solve_triangular


@dataclass(frozen=True)
class RadiationChain:
    diagonal: np.ndarray  # (normal/tangent, contact node)
    off_diagonal: np.ndarray
    platen: float
    border: np.ndarray  # Five endpoint/platen coordinates, not a truncated SVD.
    diagnostics: dict

    @property
    def nodes(self):
        return self.diagonal.shape[1]

    @property
    def border_indices(self):
        n = self.nodes
        return np.array([0, n - 1, n, 2 * n - 1, 2 * n])

    @property
    def stored_bytes(self):
        return self.diagonal.nbytes + self.off_diagonal.nbytes + self.border.nbytes + 8

    @classmethod
    def from_dense(cls, model, reference, *, tolerance=1e-10):
        """Verify geometry, SPD and reconstruction before using the sparse form.

        The local jump energy follows from u_m=a+q/2, u_s=a-q/2:
        u_m.T F u_m + u_s.T F u_s = 2*a.T F*a + q.T F*q/2.
        Only contact endpoints touch other boundary faces or platen constraints;
        eliminating a and the remaining traces changes those five coordinates.
        """
        moving, stationary = model.reference.moving, model.reference.stationary
        xy = moving.starts[model.moving_contact]
        yy = stationary.starts[model.stationary_contact]
        n = len(xy)
        if (
            n < 2
            or xy.shape != (n, 2)
            or yy.shape != xy.shape
            or not np.allclose(xy, yy, rtol=0, atol=1e-10)
            or not np.allclose(xy[:, 0], xy[0, 0], rtol=0, atol=1e-10)
            or np.any(np.diff(xy[:, 1]) <= 0)
            or not np.allclose(
                abs(moving.normals[moving.indices("platen"), 1]), 1, rtol=0, atol=1e-12
            )
        ):
            raise ValueError("Require matched vertical contact and a horizontal platen")
        if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-6:
            raise ValueError("Require a finite reconstruction tolerance in (0, 1e-6]")
        reference = np.asarray(reference)
        if (
            reference.shape != (2 * n + 1, 2 * n + 1)
            or np.iscomplexobj(reference)
            or not np.all(np.isfinite(reference))
        ):
            raise ValueError("Require a finite real dense radiation reference in jump/platen order")
        reference = np.array(reference, dtype=float, copy=True)
        if np.any(np.diag(reference) <= 0):
            raise ValueError("Dense radiation reference must be positive definite")
        scale = np.sqrt(np.diag(reference))
        symmetry = np.max(abs((reference - reference.T) / scale[:, None] / scale[None, :]))
        if symmetry > tolerance:
            raise ValueError("Dense radiation reference is not symmetric at the declared tolerance")
        symmetric = (reference + reference.T) / 2
        root = cholesky(symmetric, lower=True)
        material = model.reference.material
        length = np.diff(xy[:, 1])
        coefficients = material.density * np.array([material.cp, material.cs])
        diagonal = coefficients[:, None] * (np.r_[length, 0] + np.r_[0, length])[None] * 7 / 48
        off = coefficients[:, None] * length[None] * 5 / 48
        platen = material.density * material.cp * model.constraints.platen_width
        provisional = cls(diagonal, off, platen, np.zeros((5, 5)), {})
        baseline = provisional.dense()
        indices = provisional.border_indices
        border = (symmetric - baseline)[np.ix_(indices, indices)]
        chain = cls(diagonal, off, platen, border, {})
        difference = chain.dense() - reference
        left = solve_triangular(root, difference, lower=True)
        energy = solve_triangular(root, left.T, lower=True).T
        # Frobenius bounds the spectral norm in the reference Cholesky energy.
        energy_bound = float(np.linalg.norm(energy))
        if energy_bound > tolerance:
            raise ValueError(
                "Radiation matrix does not have the verified contact-chain plus five-coordinate structure"
            )
        chain.diagnostics.update(
            reference_relative_energy_frobenius=energy_bound,
            reference_scaled_entry_max=float(
                np.max(abs(difference / scale[:, None] / scale[None, :]))
            ),
            reference_scaled_symmetry_error=float(symmetry),
            reconstruction_tolerance=tolerance,
            history_dofs=2 * n + 1,
            contact_nodes=n,
            dense_bytes=reference.nbytes,
            structured_bytes=chain.stored_bytes,
            scope="Discrete Cinf only; the retarded remainder is not removed or approximated",
        )
        for value in (chain.diagonal, chain.off_diagonal, chain.border):
            value.setflags(write=False)
        return chain

    def dense(self):
        n = self.nodes
        result = np.zeros((2 * n + 1, 2 * n + 1))
        for component in range(2):
            ids = component * n + np.arange(n)
            result[ids, ids] = self.diagonal[component]
            result[ids[:-1], ids[1:]] = self.off_diagonal[component]
            result[ids[1:], ids[:-1]] = self.off_diagonal[component]
        result[-1, -1] = self.platen
        result[np.ix_(self.border_indices, self.border_indices)] += self.border
        return result

    def validate_step(self, coefficient, point_tangent, spring):
        tangent = np.asarray(point_tangent, dtype=float)
        if (
            tangent.shape != (self.nodes, 2, 2)
            or not np.all(np.isfinite(tangent))
            or not np.isfinite(coefficient)
            or coefficient <= 0
            or not np.isfinite(spring)
            or spring < 0
        ):
            raise ValueError(
                "Require finite node tangents, positive radiation scale and nonnegative spring"
            )
        if np.any(tangent[:, 0, 1] != 0) or np.any(tangent[:, [0, 1], [0, 1]] < 0):
            raise ValueError("Fast solve requires dp/ds=0 and nonnegative diagonal point tangents")
        return tangent

    def dense_step(self, coefficient, point_tangent, spring):
        tangent = self.validate_step(coefficient, point_tangent, spring)
        result = coefficient * self.dense()
        for i in range(self.nodes):
            ids = np.array([i, self.nodes + i])
            result[np.ix_(ids, ids)] += tangent[i]
        result[-1, -1] += spring
        return result

    def solve(self, rhs, coefficient, point_tangent, spring, *, transpose=False):
        """Pivoted banded SciPy reference, including nonsymmetric friction coupling."""
        tangent = self.validate_step(coefficient, point_tangent, spring)
        rhs = np.asarray(rhs, dtype=float)
        if (
            rhs.ndim not in (1, 2)
            or rhs.shape[0] != 2 * self.nodes + 1
            or not np.all(np.isfinite(rhs))
        ):
            raise ValueError(
                "Require finite vector or matrix right-hand sides in jump/platen order"
            )
        vector = rhs.ndim == 1
        rhs = rhs[:, None] if vector else rhs
        n, ids = self.nodes, self.border_indices

        def base_solve(b):
            def component(index, r):
                off = coefficient * self.off_diagonal[index]
                band = np.array(
                    [
                        np.r_[0, off],
                        coefficient * self.diagonal[index] + tangent[:, index, index],
                        np.r_[off, 0],
                    ]
                )
                return solve_banded((1, 1), band, r, check_finite=False)

            coupling = tangent[:, 1, 0, None]
            if transpose:
                tangential = component(1, b[n : 2 * n])
                normal = component(0, b[:n] - coupling * tangential)
            else:
                normal = component(0, b[:n])
                tangential = component(1, b[n : 2 * n] - coupling * normal)
            return np.vstack((normal, tangential, b[-1:] / (coefficient * self.platen + spring)))

        # Solve all physical RHS and five selector columns together. No inverse
        # of the border is taken, so a singular/indefinite border is allowed.
        selector = np.zeros((2 * n + 1, 5))
        selector[ids, np.arange(5)] = 1
        all_solutions = base_solve(np.column_stack((rhs, selector)))
        y, green = all_solutions[:, :-5], all_solutions[:, -5:]
        border = coefficient * (self.border.T if transpose else self.border)
        correction = np.linalg.solve(np.eye(5) + border @ green[ids], border @ y[ids])
        answer = y - green @ correction
        return answer[:, 0] if vector else answer

    def jax_solver(self):
        """Implicit JVP/VJP through the full represented matrix, not its factor tape.

        Call validate_step outside jit before using a new tangent family. The
        production previous-state law has dp/ds=0 and nonnegative diagonal
        tangents. This API does not accept pressure coupled to tangential jump.
        Array arguments are (rhs, coefficient, point_tangent, spring).
        """
        import jax
        import jax.numpy as jnp

        if not jax.config.x64_enabled:
            raise ValueError("The guarded radiation solver requires JAX float64")
        n, ids = self.nodes, self.border_indices
        diagonal, off, border0 = (
            jnp.asarray(v) for v in (self.diagonal, self.off_diagonal, self.border)
        )

        def solve(rhs, coefficient, tangent, spring):
            def matvec(q):
                blocks = q[:-1].reshape(2, n)
                value = coefficient * diagonal * blocks
                value = value.at[:, :-1].add(coefficient * off * blocks[:, 1:])
                value = value.at[:, 1:].add(coefficient * off * blocks[:, :-1])
                value = value + jnp.einsum("nij,jn->in", tangent, blocks)
                result = jnp.r_[value.reshape(-1), (coefficient * self.platen + spring) * q[-1]]
                return result.at[ids].add(coefficient * border0 @ q[ids])

            def inverse(_, b, *, transpose):
                def component(index, r):
                    dd = coefficient * diagonal[index] + tangent[:, index, index]
                    ee = coefficient * off[index]
                    if n == 2:
                        return jnp.linalg.solve(
                            jnp.diag(dd) + jnp.diag(ee, 1) + jnp.diag(ee, -1), r
                        )
                    return jax.lax.linalg.tridiagonal_solve(jnp.r_[0, ee], dd, jnp.r_[ee, 0], r)

                def base_solve(r):
                    coupling = tangent[:, 1, 0, None]
                    if transpose:
                        tangential = component(1, r[n : 2 * n])
                        normal = component(0, r[:n] - coupling * tangential)
                    else:
                        normal = component(0, r[:n])
                        tangential = component(1, r[n : 2 * n] - coupling * normal)
                    return jnp.vstack(
                        (normal, tangential, r[-1:] / (coefficient * self.platen + spring))
                    )

                selector = jnp.zeros((2 * n + 1, 5)).at[ids, jnp.arange(5)].set(1)
                all_solutions = base_solve(jnp.column_stack((b, selector)))
                y, green = all_solutions[:, 0], all_solutions[:, 1:]
                border = coefficient * (border0.T if transpose else border0)
                correction = jnp.linalg.solve(jnp.eye(5) + border @ green[ids], border @ y[ids])
                return y - green @ correction

            answer = jax.lax.custom_linear_solve(
                matvec,
                rhs,
                solve=lambda f, b: inverse(f, b, transpose=False),
                transpose_solve=lambda f, b: inverse(f, b, transpose=True),
                symmetric=False,
            )
            valid = (
                jnp.isfinite(coefficient)
                & (coefficient > 0)
                & jnp.isfinite(spring)
                & (spring >= 0)
                & jnp.all(jnp.isfinite(tangent))
                & jnp.all(tangent[:, 0, 1] == 0)
                & jnp.all(tangent[:, jnp.array([0, 1]), jnp.array([0, 1])] >= 0)
            )
            # A traced unsupported law must not silently get the triangular
            # inverse. validate_step gives a descriptive exception before jit.
            return jnp.where(valid, answer, jnp.full_like(answer, jnp.nan))

        return jax.jit(solve)

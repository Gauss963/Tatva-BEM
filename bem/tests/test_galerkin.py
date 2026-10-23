from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from finiteblock_bem.galerkin import _scalar_kernels, _weak_kernels, assemble_galerkin_operators
from finiteblock_bem.galerkin_fast import PreparedGalerkinOperators
from finiteblock_bem.kernels import green_and_gradient, source_traction
from finiteblock_bem.material import ElasticMaterial
from finiteblock_bem.mesh import polygon_boundary
from finiteblock_bem.nodal import (
    NodalContactEquilibrium,
    NodalInterfaceEquilibrium,
    NodalInterfaceTransfer,
    initialize_nodal_restart,
    load_nodal_model,
    nodal_body_stiffness,
)

CASE = Path(__file__).resolve().parents[1] / "src/finiteblock_bem/resources/cases/pmma_event16.toml"


def test_prepared_galerkin_matches_independent_pair_assembly():
    mesh = polygon_boundary([[0, 0], [1.7, 0.1], [1.3, 1.1], [-0.1, 0.7]], ["free"] * 4, 0.4)
    material = ElasticMaterial(10, 0.25, 1)
    prepared = PreparedGalerkinOperators(mesh, material, batch_pairs=7)
    for s in [None, 0.01 + 0.02j, 1 + 2j, 100 + 300j]:
        reference = assemble_galerkin_operators(mesh, material, s)
        result = prepared.assemble(s)
        for name in ["single_layer", "trace_double_layer", "hypersingular", "mixed_mass"]:
            np.testing.assert_allclose(
                getattr(result, name), getattr(reference, name), rtol=1e-12, atol=1e-12
            )
    model = load_nodal_model(CASE, max_panel_length=80)
    a, b = NodalInterfaceTransfer(model), NodalInterfaceTransfer(model, prepared=False)
    s = 200 + 2j * np.pi * 5000
    np.testing.assert_allclose(a.evaluate(s), b.evaluate(s), rtol=1e-10, atol=1e-7)


def test_weak_hypersingular_form_matches_independent_traction_derivative():
    material, s = ElasticMaterial(10, 0.25, 1), 1 + 0.7j
    z, w = leggauss(24)
    z, w = (z + 1) / 2, w / 2
    weak, direct = np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    nx, ny, ty = np.array([0.0, -1.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0])
    # Compact P1 hats on disjoint supports remove endpoint terms in integration
    # by parts. A direct finite difference of source traction uses no weak-D formula.
    for i in [0, 1]:
        for j in [0, 1]:
            x = np.column_stack((i + z, np.zeros_like(z)))[:, None, :]
            y = np.column_stack((2 - j - z, np.ones_like(z)))[None, :, :]
            r = y - x
            g = green_and_gradient(r, s, material)[0]
            scalar = _scalar_kernels(r.reshape(-1, 2), s, material)
            terms = _weak_kernels(g.reshape(-1, 2, 2), *scalar, nx, ny, ty, s, material)
            mass, derivative, left, right = [a.reshape(len(z), len(z), 2, 2) for a in terms]
            px = (z if i == 0 else 1 - z)[:, None]
            py = (z if j == 0 else 1 - z)[None, :]
            dx, dy = (1 if i == 0 else -1), (1 if j == 0 else -1)
            measure = w[:, None] * w[None, :]
            weak += np.einsum(
                "ab,abij->ij",
                measure,
                mass * (px * py)[..., None, None]
                + derivative * dx * dy
                + left * (px * dy)[..., None, None]
                + right * (dx * py)[..., None, None],
            )
            diff = []
            h = 1e-5
            for k in [0, 1]:
                shift = np.eye(2)[k] * h
                a = source_traction(green_and_gradient(r - shift, s, material)[1], ny, material)
                b = source_traction(green_and_gradient(r + shift, s, material)[1], ny, material)
                diff.append((a - b) / (2 * h))
            grad = np.stack(diff, axis=-3)
            traction = material.lam * nx[:, None] * np.einsum("...iij->...j", grad)[..., None, :]
            traction += material.mu * (
                np.einsum("k,...kij->...ij", nx, grad) + np.einsum("k,...ikj->...ij", nx, grad)
            )
            direct -= np.einsum("ab,abij->ij", measure * px * py, traction)
    np.testing.assert_allclose(weak, direct, rtol=2e-8, atol=1e-9)


def test_galerkin_static_rigid_modes_and_affine_patch_on_nonorthogonal_polygon():
    material = ElasticMaterial(10, 0.25, 1)
    mesh = polygon_boundary([[0, 0], [1.7, 0.1], [1.3, 1.1], [-0.1, 0.7]], ["free"] * 4, 0.4)
    op = assemble_galerkin_operators(mesh, material, quadrature=16)
    k = op.force_stiffness()
    rotation = np.column_stack((-mesh.starts[:, 1], mesh.starts[:, 0])).ravel()
    rigid = np.column_stack((np.tile([1, 0], mesh.count), np.tile([0, 1], mesh.count), rotation))
    np.testing.assert_allclose(k @ rigid, 0, atol=1e-7)
    np.testing.assert_allclose(k, k.T, atol=1e-12)
    gradient = np.array([[0.3, 0.7], [-0.2, 0.4]])
    u = (mesh.starts @ gradient.T).ravel()
    stress = material.mu * (gradient + gradient.T) + material.lam * np.trace(gradient) * np.eye(2)
    traction = mesh.normals @ stress.T
    force = np.zeros((mesh.count, 2))
    for edge in range(mesh.count):
        force[edge] += mesh.lengths[edge] / 2 * traction[edge]
        force[(edge + 1) % mesh.count] += mesh.lengths[edge] / 2 * traction[edge]
    np.testing.assert_allclose(k @ u, force.ravel(), rtol=1e-7, atol=1e-8)


def test_galerkin_longitudinal_and_shear_manufactured_fields_converge():
    material, s = ElasticMaterial(10, 0.25, 1), 1 + 2j
    d = np.array([0.6, 0.8])
    for polarization, speed in [(d, material.cp), (np.array([-0.8, 0.6]), material.cs)]:
        errors = []
        for h in [0.5, 0.25, 0.125]:
            mesh = polygon_boundary([[0, 0], [1, 0], [1, 1], [0, 1]], ["free"] * 4, h)
            k = assemble_galerkin_operators(mesh, material, s, quadrature=12).force_stiffness()
            kap = s / speed
            u = (np.exp(kap * (mesh.starts @ d))[:, None] * polarization).ravel()
            force = np.zeros((mesh.count, 2), complex)
            z, w = leggauss(24)
            z, w = (z + 1) / 2, w / 2
            for edge in range(mesh.count):
                x = mesh.starts[edge] + z[:, None] * (mesh.ends[edge] - mesh.starts[edge])
                n = mesh.normals[edge]
                vector = material.lam * np.dot(polarization, d) * n
                vector += material.mu * (polarization * np.dot(d, n) + d * np.dot(polarization, n))
                traction = np.exp(kap * (x @ d))[:, None] * kap * vector
                force[edge] += mesh.lengths[edge] * np.einsum("q,qi->i", w * (1 - z), traction)
                force[(edge + 1) % mesh.count] += mesh.lengths[edge] * np.einsum(
                    "q,qi->i", w * z, traction
                )
            errors.append(np.linalg.norm(k @ u - force.ravel()) / np.linalg.norm(force))
        assert errors[-1] < 0.006, errors
        assert all(a > 1.5 * b for a, b in pairwise(errors)), errors


def test_nodal_contact_and_loading_share_endpoint_and_positive_line_weights():
    model = load_nodal_model(CASE, max_panel_length=80)
    ref, con = model.reference, model.constraints
    assert len(model.moving_contact) == len(ref.moving_contact) + 1
    assert con.prolongation[2 * model.moving_contact[0] + 1, -1] == 1
    np.testing.assert_allclose(con.contact_weights[[0, -1]], 40)
    np.testing.assert_allclose(con.contact_weights.sum(), 480)
    np.testing.assert_allclose(model.normal_force_shape.sum(), 500)
    np.testing.assert_array_equal(con.full_weights, 1)


def test_seed_knot_alignment_preserves_physical_pairing_and_piecewise_seed_values():
    model = load_nodal_model(
        CASE,
        max_panel_length=20,
        contact_panel_length=5,
        contact_end_length=0.5,
        corner_panel_length=0.5,
        restart_knot_window=8,
        restart_knot_subdivisions=4,
    )
    y = model.reference.moving.starts[model.moving_contact, 1]
    reference = np.genfromtxt(
        CASE.parent / "../data/ev16_restart_interface.csv", delimiter=",", names=True
    )
    for knot in [0, 0.5, 1, 1.5, 2, 4, 6, 8, 472, 474, 476, 478, 480]:
        index = np.flatnonzero(abs(y - knot) < 1e-10)
        assert len(index) == 1
        np.testing.assert_allclose(
            model.restart.plastic_slip[index[0]],
            np.interp(knot, reference["y"], reference["plastic_slip"]),
            atol=1e-13,
        )
    assert np.min(np.diff(y)) > 1e-8
    np.testing.assert_allclose(np.diff(y[y <= 8 + 1e-10]), 0.5, atol=1e-10)
    np.testing.assert_allclose(np.diff(y[y >= 472 - 1e-10]), 0.5, atol=1e-10)
    np.testing.assert_allclose(model.constraints.contact_weights.sum(), 480)


def test_galerkin_condensation_preserves_force_work_and_sampled_passivity():
    model = load_nodal_model(CASE, max_panel_length=80)
    transfer = NodalInterfaceTransfer(model)
    rng = np.random.default_rng(812)
    for s in [200 + 2j * np.pi * 5000, 2000 + 2j * np.pi * 10000]:
        z, reconstruct = transfer.evaluate(s, reconstruction=True)
        np.testing.assert_allclose(z, z.T, rtol=1e-10, atol=1e-6)
        weights = np.r_[
            model.constraints.contact_weights,
            model.constraints.contact_weights,
            model.constraints.platen_width,
        ]
        velocity = z / s / np.sqrt(weights[:, None] * weights[None, :])
        spectrum = np.linalg.eigvalsh((velocity + velocity.conj().T) / 2)
        assert spectrum[0] > 0, spectrum
        q = rng.normal(size=transfer.dofs) + 1j * rng.normal(size=transfer.dofs)
        u = reconstruct @ q
        stiffness = nodal_body_stiffness(model, s)
        np.testing.assert_allclose(np.vdot(u, stiffness @ u), np.vdot(q, z @ q), rtol=1e-9)
        np.testing.assert_array_equal(u[model.constraints.fixed_dofs], 0)


def test_nodal_restart_and_condensed_initial_force_are_consistent():
    model = load_nodal_model(CASE, max_panel_length=80)
    ref, con = model.reference, model.constraints
    lift = initialize_nodal_restart(model)
    assert lift.diagnostics["initial_max_strength_ratio"] < 1
    np.testing.assert_array_equal(lift.state.plastic_slip, model.restart.plastic_slip)
    np.testing.assert_array_equal(lift.state.cumulative_slip, model.restart.cumulative_slip)
    np.testing.assert_allclose(lift.diagnostics["mean_pressure_mpa"], 16 * 500 / 480, rtol=1e-8)
    transfer = NodalInterfaceTransfer(model)
    z = transfer.evaluate(1.5 / 8e-6).real
    system = NodalInterfaceEquilibrium(model, lift, z)
    r, _, _ = system.evaluate(
        np.zeros(transfer.dofs),
        np.zeros(transfer.dofs),
        ref.normal_pressure,
        ref.restart_actuator,
        lift.state,
    )
    np.testing.assert_allclose(r, 0, atol=1e-7)
    full = NodalContactEquilibrium(model, nodal_body_stiffness(model))
    guess = con.prolongation.T @ lift.displacement / np.sum(con.prolongation**2, axis=0)
    r, _, _ = full.evaluate(
        guess,
        np.zeros(len(lift.displacement)),
        ref.normal_pressure,
        ref.restart_actuator,
        lift.state,
        frozen=True,
    )
    np.testing.assert_allclose(r, 0, atol=1e-7)

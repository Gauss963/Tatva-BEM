# Roadmap to a validated finite-block event-16 reproduction

The first implementation is a fully elastodynamic **reference engine**. The
complete original event is still an open objective, with these acceptance
gates. A short successful diagnostic does not close the later gates.

- [x] Independent Laplace-domain plane-strain Green tensor and source traction.
- [x] Direct finite-interior boundary operators, jump term and singular integration.
- [x] BDF2-CQ weights and complete causal boundary history.
- [x] Analytic finite-bar arrival/reflection and paired mesh/time refinement.
- [x] Two finite blocks, chamfer, original supports, rigid platen and spring.
- [x] Unilateral contact and previous-state LSW update matching the source law.
- [x] Admissible static lift using the archived event-16 plastic slip.
- [x] Nonlinear short sliding diagnostic and original-rate short transient.
- [x] Standalone package structure, local artifacts, provenance and CI configuration.
- [x] Exact online dyadic FFT history, parallel frequency assembly and fingerprinted weights cache.
- [x] JAX local AD tangents, implicit root gradients and a bounded differentiable full-history scan.
- [x] Coarse 45 ms run with original loading schedule and archived FEM comparison.
- [x] Exact frequency-domain boundary condensation and paired contact/corner grading.
- [x] Prepared radial traction assembly, near-panel subdivision and streamed static initialization.
- [x] Reverse CQ discrete adjoint with JAX VJPs, checked against scan AD and actuator finite differences.
- [x] Literature-driven optional adaptive frequency sampling, dense-CQ trajectory comparison and sampled passivity audit.
- [x] Symmetric P1/P0 Galerkin bulk, continuous nodal contact/platen trace, independent weak-D and wave checks.
- [x] Shared prepared Galerkin geometry and same-matrix four-worker benchmark.
- [x] Native adaptive-interpolant storage and disk time weights, with real-BEM diagnostic field/cotangent checks.
- [x] Capacity-limited exact forward/transpose histories: full coarse 45 ms equality and fine4 adjoint cache equality; original-Z fine4 guard remains open.
- [x] Isolated fixed near/far quadrature: sampled finite-block contour, work passivity and short nonlinear/conditional-adjoint checks; full-horizon validation remains open.
- [ ] Quantified nonlinear stability and temporal/spatial convergence for Galerkin.
- [ ] Independent spatial refinement at the contact ends and process-zone resolution.
- [ ] Validate static-lift error against original FEM interior fields and quantify zero-initial-velocity assumptions.
- [ ] Oblivious CQ or space-time compression to remove dense weight storage; online FFT is implemented but retains it.
- [ ] Production memory/time benchmarks for a 45 ms run at a physically justified h and dt.
- [ ] Boundary-history-aware checkpoint/resume; no state-only CQ restart.
- [ ] Interior displacement, stress and velocity reconstruction with independent tests.
- [ ] Full 45 ms event-16 loading and rupture with temporal/spatial convergence.
- [ ] Compare nucleation time/location, rupture speed, stress drop, slip, loading reaction and energy budget with FEM.
- [ ] Integrate a tested backend interface into a separate Tatva checkout if requested.
- [ ] Checkpoint/oblivious adjoint, load-hold event-time sensitivity and material/geometry/lift derivatives.

## Suggested next implementation order

The [2024–2026 literature review](literature.md) led to the implemented
[P1/P0 Galerkin energy formulation](galerkin.md). It removes the previously
observed negative bulk directions at the sampled frequencies. Keep the
independent pair assembler and analytic benchmark in CI, quantify the
remaining endpoint/static-lift discrepancy, and resolve the process zone
before claiming an event reproduction or a same-accuracy speed advantage over FEM.

Increasing the dense memory budget or taking very large time steps is not
a substitute for resolving the relevant waves and weakening scale.

## Separation from the original repository

No code under `/Volumes/Gauss-T7/tatva` was changed. All model-specific data
used by this package are local snapshots under `resources/`. Future Tatva
integration should use a separate checkout and an explicit boundary-backend
interface; the friction law can be shared, while elastodynamics and history
remain backend-specific.

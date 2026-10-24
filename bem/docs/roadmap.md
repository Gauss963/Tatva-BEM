# Roadmap to a validated finite-block event-16 reproduction

The first implementation is a fully elastodynamic **reference engine**. The
complete original event is still an open objective, with these acceptance
gates. A short successful diagnostic does not close the later gates.

Latest completed result: [2 us / Q12, full 45 ms](fine2_event.md), maximum
slip 0.511082 mm versus archived FEM 0.644123 mm (20.65% low).
The earlier 4 us result uses Q6, so this pair does not establish temporal convergence.
The original-mesh four-core FEM rerun is still running. A serialized study will
check identical one-substep fields before comparing 4 and 16 constitutive substeps.

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
- [x] Isolated fixed near/far quadrature: sampled contour and short nonlinear/AD checks; full coarse 45 ms fields and original-equation balance pass, one actuator cotangent still fails its unchanged guard.
- [x] Constrained trace Schur formulation and sampled low-frequency conditioning checks; full-event validation remains open.
- [x] Local constitutive substeps with previous-displacement cotangents, checked against an independent small original-Z/JAX scan; full-event accuracy study remains open.
- [x] Guarded circle-reuse prototype and analytic kernel/startup regression; large FMM and nonlinear validation remain open.
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

The current order is to finish the isolated FEM baseline, inspect the contact
substep study, then use [guarded spectral reuse](cauchy_reuse.md) for the missing
same-Q12 time-grid control if its independent BEM checks pass. Keep residual,
physical error, cold setup and cached rollout costs separate. Material/geometry
AD and event-time derivatives remain open rather than being inferred from
successful contact-only derivatives.

## Separation from the original repository

No code under `/Volumes/Gauss-T7/tatva` was changed. All model-specific data
used by this package are local snapshots under `resources/`. Future Tatva
integration should use a separate checkout and an explicit boundary-backend
interface; the friction law can be shared, while elastodynamics and history
remain backend-specific.

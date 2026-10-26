# Roadmap to a validated finite-block event-16 reproduction

The first implementation is a fully elastodynamic **reference engine**. The
complete original event is still an open objective, with these acceptance
gates. A short successful diagnostic does not close the later gates.

Latest completed result: [2 us / Q12, full 45 ms](fine2_event.md), maximum
slip 0.511082 mm versus archived FEM 0.644123 mm (20.65% low).
The earlier 4 us result uses Q6, so this pair does not establish temporal convergence.
The first local h2 FEM rerun used the history-storage dtype as its model dtype
and is excluded from the original float64 reference. See the
[precision correction](reference_precision.md). Corrected initialization matches
the archived first fields exactly. The full one-substep BEM fields also match
exactly. The original four-substep Newton solve fails at 29.144 ms; residual-checked
hybrid/continuation recovery then completes both four- and sixteen-substep events.
The latter reaches 0.559055 mm (13.21% low), with pressure/shear profile errors
still near 20%. See the [substep results](substepped_contact.md).
The 64-substep study first fails at 29.132 ms. An optional constant-mu_k trial
seed, accepted only in the original full residual, recovers that root and
completes 45 ms at 0.563924 mm (12.45% low). Pressure/shear profile errors remain
near 20%. The complete 16-substep five-parameter reverse gradient now agrees
with independent forward JVP histories to a maximum absolute difference of
4.88e-12; material/geometry/lift/event-time derivatives remain outside that check.
The corrected full h2/float64 FEM rerun now matches all 9000 shear rows,
the checked initial fields and final state exactly. Its four-core simulation
including reduced bulk output takes 8032.347 s; preparation plus simulation
takes 8032.866 s. See [verified evidence](fem_verified_reference_results.json).
The [same-Q12 time-grid control](time_grid_comparison.md) is now complete:
at a matched 0.125 us local microstep and original float64 seed, dt2/N16 gives
0.559077 mm and dt4/N32 gives 0.536574 mm. The full 4 us reused kernel takes
4100.89 s plus its previously paid source construction. Both stock-seed N16
gradients pass independent forward JVP checks, but the gradients differ strongly
between grids. [Fixed-trace density controls](density_refinement.md) improve
static traction agreement with FEM h0.5 while worsening agreement with h2;
neither control establishes dynamic or spatial convergence.
The [compact Radau path](radau_experiment.md) now stores two exact coefficient
families, shares stage FFTs, preserves the finite-DFT lag-zero wrap and commits
the coupled contact path only at the endpoint. A real two-block 256 us
diagnostic agrees with freshly assembled original-Z RK-CQ to 4.27e-9 mm in
displacement and 6.24e-7 MPa in shear; its original-equation residual is
3.32e-8 MPa. The [full 45 ms Radau event](radau_event.md) now completes at
0.565404 mm, 12.22% below FEM, with final pressure/shear errors still near 20%.
Its cached forward takes 167.89 s versus the matched original-seed BDF2
control's 117.30 s; the new kernel additionally takes 4173.50 s and relies on
the previously paid source construction. Prescribing the true FEM actuator
plateau improves maximum slip only to 0.574861 mm and remains a diagnostic.
Accumulated-vs-net plastic slip rules out unresolved reversals as the main
0.0707 mm mean-slip deficit, but not other bulk time errors. Independent
continuous-time FEM/BEM closed-stick responses now separate spatial
differences from CQ; FEM h1 is closer to the same BEM than h2 at the three
sampled Laplace points. Nonlinear convergence and the Radau trajectory adjoint remain open.
The [discrete radiation-limit experiment](radiation_limit.md) now tests a
possible instantaneous/history split against fresh operators, including the
P0-projection contribution to the P1/P0 leading term. It is not a production
multirate solver and has not improved the full-event result.

- [x] Independent Laplace-domain plane-strain Green tensor and source traction.
- [x] Direct finite-interior boundary operators, jump term and singular integration.
- [x] BDF2-CQ weights and complete causal boundary history.
- [x] Analytic finite-bar arrival/reflection and paired mesh/time refinement.
- [x] Two finite blocks, chamfer, original supports, rigid platen and spring.
- [x] Unilateral contact and previous-state LSW update matching the source law.
- [x] Admissible static lift using the archived event-16 plastic slip.
- [x] Nonlinear short sliding diagnostic and original-rate short transient.
- [x] Standalone package structure, local artifacts, provenance and CI configuration.
- [x] Faithful original h2/float64 FEM rerun, exact saved-field comparison and serialized local timing.
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
- [x] Guarded circle reuse, independent sampled matrices, complete 4 us/Q12 kernel, full nonlinear time-grid controls and N16 reverse/forward parameter checks; physical convergence remains open.
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

The isolated FEM baseline is complete. The current order is to inspect the original
float64 seed and contact substep controls, then use [guarded spectral reuse](cauchy_reuse.md) for the missing
same-Q12 time-grid control if its independent BEM checks pass. Keep residual,
physical error, cold setup and cached rollout costs separate. Material/geometry
AD and event-time derivatives remain open rather than being inferred from
successful contact-only derivatives.
The [linear modal diagnostic](modal_time_resolution.md) also separates stable
time integration from wave accuracy. Contact substeps do not refine the bulk
time symbol; modal attenuation is a diagnostic, not an error bound for rupture.
The verified [FEM work/storage budget](energy_budget.md) now provides a separate
energy reference. Its remainder includes friction and numerical error; current
BEM micro friction work alone does not close the full energy/stability gate.

## Separation from the original repository

No code under `/Volumes/Gauss-T7/tatva` was changed. All model-specific data
used by this package are local snapshots under `resources/`. Future Tatva
integration should use a separate checkout and an explicit boundary-backend
interface; the friction law can be shared, while elastodynamics and history
remain backend-specific.

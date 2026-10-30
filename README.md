# Tatva-BEM

Fully elastodynamic boundary elements for finite-block frictional contact,
developed alongside the Tatva-based PMMA finite-element model.

**Research implementation: the full 45 ms event-16 FEM rupture has not yet
been reproduced or shown to converge.** The completed Radau 6 us / Q12 event
with the original float64 seed reaches 0.5654 mm, 12.22% below the archived FEM.
Final pressure/shear profile differences
remain about 20%; temporal and spatial convergence are still open.
The cached Radau forward solve is slower than its matched BDF2 control;
no same-accuracy speed advantage is established.
The completed same-dt FEM h1 control reaches 0.619884 mm; h2/h1 endpoint
pressure fields still differ by 17.62% in P1 L2. Radau differs from h1 by
8.79% in maximum slip and 10.04% in the pressure profile, so the remaining
accuracy gates are still open. See [the completed mesh control](bem/docs/fem_mesh_control.md).
See the [full Radau and error-isolation results](bem/docs/radau_event.md),
the [contact substep comparison](bem/docs/substepped_contact.md),
the [original one-step comparison](bem/docs/fine2_event.md) and the
[earlier collocation comparison](bem/docs/event16_comparison.md).

The latest [independent endpoint controls](bem/docs/chamfer_refinement.md)
reduce the initial whole-interface pressure/shear differences to 0.58%/0.51%
against FEM h1 at matching 0.0078125 mm endpoint sizes. Three continuous-time linear response
samples pass the unchanged 5% screen. These are static and linear controls;
they do not establish complete nonlinear event agreement or a speed advantage.
An explicit uncondensed static backend also resolves a fine-panel reconstruction
roundoff failure at the original equilibrium tolerance.

Fresh 2 us Radau construction has started on the graded fine mesh, following
the [measured time-symbol comparison](bem/docs/bulk_time_symbol.md).
The complete independent graded FEM reference is replaying its damaged-output
prefix before continuing from a verified checkpoint after an abrupt process interruption.
The BEM frequency prefix is preserved on disk; its
[verified disk continuation](bem/docs/radau_sampling_checkpoint.md) now passes
interrupted-kernel controls. Full fine construction still needs to resume.
See the [execution and measured CPU-budget record](bem/docs/reference_first.md).
The [complete causal RK audit and conditional adjoint](bem/docs/radau_history_validation.md)
pass real two-body short-run controls, including selected healed-preload
sensitivities. Fine event/FEM agreement and fine initialized Radau gradients
remain unverified.

This repository preserves the existing Tatva/PMMA Git history. Its upstream
is [smec-ethz/tatva](https://github.com/smec-ethz/tatva); the inherited PMMA
baseline comes from [Gauss963/PMMA-RSF](https://github.com/Gauss963/PMMA-RSF).

## BEM implementation

- Direct 2D plane-strain elastodynamic boundary integral operators.
- Laplace-domain P/S wave kernels and singular-panel integration.
- BDF2 convolution quadrature retaining the complete boundary history.
- Two finite blocks with chamfer, original supports, rigid platen and spring.
- Unilateral penalty contact and cumulative linear slip weakening.
- Static-lift initialization using a local snapshot of event-16 plastic slip.
- Explicit failure states, memory preflight, local artifacts and provenance.
- Exact online FFT history, parallel kernels and reusable fingerprinted weights.
- JAX local AD tangents, JIT Newton and verified short-trajectory implicit gradients.
- Exact contact-coordinate condensation, contact/corner grading and a full-history conditional adjoint.
- Symmetric P1/P0 Galerkin nodal-force backend with regularized hypersingular integrals and batched assembly.
- Exact last-use FFT row streaming, with identical full-event forward/adjoint results and lower peak memory.
- Optional disk-backed CQ half spectra with explicit storage, resource checks and unchanged sampling/IFFT formulas.
- Positive, work-conjugate refined contact states on a fixed bulk trace, with complete-history forward/reverse AD controls.

The BEM package is independent of the Tatva FEM time integrator. A shared
backend interface and memory-compressed production history remain
future work. No files in the external original Tatva checkout are modified.

The exact captured float64 restart is now included as
[`ev16_restart_float64.npz`](bem/src/finiteblock_bem/resources/data/ev16_restart_float64.npz),
with provenance and a verified wheel payload. Pass it explicitly to the research
tools' `--restart-seed` option; the existing default CSV remains unchanged.
It contains the 241-knot plastic offset, healed zero memory and actuator,
allowing the BEM input to be reproduced without the original large FEM HDF5.

## Quick start

Python 3.11 or newer:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e './bem[test,plot,jax]'
.venv/bin/python -m pytest bem/tests
.venv/bin/finiteblock-bem benchmark --name bar-reference
.venv/bin/finiteblock-bem smoke --name smoke-reference
.venv/bin/finiteblock-bem event16 --name event16-short
```

The helper `bem/run.sh` also runs directly from a checkout with an existing
compatible Python. Set `FDBEM_PYTHON` if needed. For scientific plots:

```sh
FDBEM_PYTHON=.venv/bin/python bem/run.sh plot --name bar-reference
```

`benchmark` checks an analytic finite-bar reflection problem. `smoke` uses
accelerated diagnostic loading, not the original event history. `event16`
starts from a BEM static lift and uses the original actuator speed, but the
default run is only 0.512 ms and does not reach rupture nucleation.

See [BEM usage and current results](bem/README.md),
[formulation](bem/docs/formulation.md),
[validation](bem/docs/validation.md), and
[remaining acceptance gates](bem/docs/roadmap.md).
The [AD design](bem/docs/autodiff.md) explains conditional contact gradients;
material/geometry and load-hold event-time derivatives remain open. A selected
static-active-set initialization derivative has passed full-event reverse/forward
checks for both seed interpolation modes. Gradients remain mesh dependent; see
[initialized full-history AD](bem/docs/initialized_contact_ad.md).
See the [latest refinement and adjoint results](bem/docs/refinement.md) for
45 ms conditional gradients and the remaining physical solution differences.
The [2024–2026 literature review](bem/docs/literature.md) records the next
accuracy/stability gates and measured adaptive-frequency experiments.
The [refined contact experiment](bem/docs/refined_contact.md) separates contact
integration from bulk resolution and requires a full-event identity control
before attributing a change to refinement. The complete identity control passed;
273 states reach 0.569536 mm. Finer integration exposes initially inadmissible
frozen offsets near an endpoint. The new [healed static contact control](bem/docs/fem_static_initialization.md)
matches Tatva's Coulomb preload policy. Six complete controls and original causal
equation audits have passed their algebraic checks; FEM agreement remains below target.
The [241-node static screen](bem/docs/fem_contact_grid_study.md) failed its original
accuracy gate; its large-kernel stages were not started. Independent endpoint
refinement is now resolving the spatial discrepancy before new transient runs.
The [batched AD history prototype](bem/docs/batched_ad_history.md) shares kernel FFTs
across derivative directions. The [complete-event layout control](bem/docs/layout_history.md)
preserves the primal, all five gradients and every actuator cotangent, with
substantially shorter measured AD time on this machine.
The [original-equation audit](bem/docs/original_history_audit.md) additionally
checks causal force histories and the initial P1 seed-field projection.
The [float64 contact output control](bem/docs/float64_contact_comparison.md)
adds signed FEM endpoint shear without changing the original numerical state;
all original datasets and checkpoint/resume outputs passed bitwise identity.
The [event timing and propagation diagnostics](bem/docs/event_observables.md)
retain saved-time uncertainty for nucleation, fixed-gauge propagation and load
hold. The completed older 6 µs event still fails these diagnostic tolerances;
the new graded 2 µs comparison remains pending.
The [interior-field reconstruction controls](bem/docs/interior_fields.md) add
displacement, velocity, stress and a fixed-kernel observation pullback, with
independent analytic P/S wave tests. Full-event interior reconstruction and
matching float64 FEM observations remain open.
The [FEM interior observation adapter](bem/docs/fem_interior_observations.md)
records fixed physical probes with separate endpoint and half-step clocks;
small pinned-source runs preserve all original datasets and 12 state arrays,
including checkpoint continuation. The full probe reference remains pending.
The [bounded interior CQ backend](bem/docs/rectangular_cq_history.md) now retains
all causal lags with a complete transpose and a fixed-kernel JAX VJP.
Small two-block probe reconstructions match the dense stage-map reference;
the [resumable body/probe frequency producer](bem/docs/interior_raw_sampling.md)
now feeds this backend directly. Small raw/compact interruptions recover the
same spectra. Full-event sampling, initialized observation gradients and a
same-accuracy performance advantage remain unverified.
The [storage-stage Radau producer](bem/docs/storage_radau_pipeline.md) preserves
the original sampling/audits while adding resumable ownership transfer,
in-place compact spectra and direct physical Fortran weights. Its full 1 µs
resource preflight has passed; the full refined event has not been run.
The [factored Galerkin contractions](bem/docs/factored_galerkin.md) reduce
assembly time by 5–11% at two measured fine-grid frequencies, with complete
small causal/transpose controls; full-event speed remains unverified.
The [completed graded FEM diagnostics](bem/docs/graded_fem_complete.md) reveal
strong endpoint sensitivity after rupture, requiring independent dynamic
refinement before treating that result as a converged reference.
The [checkpointed FEM time-step controls](bem/docs/fem_timestep_window.md)
preserve the original step/scan, rephase half-step velocities and verify
bitwise baseline contact output before comparing smaller time steps.
The short onset window passed, but the [extended postrupture window](bem/docs/fem_postrupture_timestep.md)
failed its time-refinement diagnostic; the completed FEM is not a converged dynamic reference.
The [selected Galerkin block/fiber provider](bem/docs/galerkin_blocks.md) avoids
constructing unrequested matrices or pair geometry. Actual fine-grid requested
entries match independent complete assembly; a complete compression solver
and end-to-end advantage remain open.
The [far-field ACA assembly controls](bem/docs/hierarchical_galerkin.md) now
pass sampled complete raw-operator, Schur and linear port checks. They are
slower at the tested low/complex frequencies and faster at one strongly
damped frequency; full production adoption remains deferred.
The [Galerkin implementation and validation](bem/docs/galerkin.md) addresses
the sampled passivity failures of the original collocation backend.

## Repository layout

```text
bem/                       Independent finiteblock-bem Python package
validation/static_bem2d/   Earlier static DDM comparison and small FEM snapshots
tatva/                     Inherited Tatva FEM core and PMMA model
cases/, scripts/, slurm/   Inherited FEM inputs and execution workflow
tests/                     Inherited Tatva/FEM tests
docs/PMMA-RSF.md            Preserved FEM usage documentation
docs/UPSTREAM.md            Repository ancestry and remote conventions
.github/workflows/bem.yml   BEM tests and style checks on Python 3.11/3.12
```

The legacy static comparison vendors the original MIT-licensed Bem2d.jl
kernel so its Julia environment does not point outside this repository.
It is a separate validation experiment, not the dynamic BEM backend.

Generated trajectories, caches, run logs and large HDF5 outputs are ignored.
Only small model snapshots needed to reproduce the diagnostics are included.

## FEM baseline

The inherited finite-element workflow is documented in
[docs/PMMA-RSF.md](docs/PMMA-RSF.md). Its dependencies and test suite are
separate from the minimal BEM package. The BEM workflow does not certify
the inherited FEM suite or the complete coupled event.

## Licensing

The inherited Tatva code retains LGPL-3.0-or-later and its existing notices.
The independent BEM package has its own [MIT license](bem/LICENSE).
The vendored Bem2d.jl kernel retains its original MIT notice.

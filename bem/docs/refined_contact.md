# Refined contact states on the same dynamic bulk trace

The full event still differs from the verified h2 FEM reference. The matched
float64-seed BDF2/N64 control reaches 0.563086 mm maximum cumulative slip,
versus 0.644123 mm in FEM. A completed cached forward is faster than FEM,
but the previously paid, contended 73048.67 s kernel construction prevents a
claim of a cold runtime advantage. Neither smaller root residuals nor local
AD agreement closes the remaining physical error.

This experiment changes contact integration while keeping the 137 contact
trace nodes, 275 retained bulk unknowns, original finite blocks, complete
dynamic operator, physical parameters and loading policy. It distinguishes
contact-state resolution from bulk spatial and time resolution.

## Discretization and static lift

`tools/refined_contact.py` defines a positive composite nodal trapezoid on
each original P1 segment. If Q interpolates coarse jumps to fine state points
and W contains fine line weights, fine contact forces pull back as

```text
dn_fine = Q dn_bulk,   ds_fine = Q ds_bulk
Fc = [Q.T W p; Q.T W tau; 0].
```

The interpolation and force restriction are virtual-work conjugates.
Q.T W 1 equals the original coarse nodal weights. One subdivision recovers
the original contact integration exactly. For an elastic P1 segment, the
quadrature energy error decreases as the inverse square of subdivision count.
This does not add bulk modes or establish nonlinear spatial convergence.

The default seed interpolates the **same coarse P1 seed field** to fine state
points. This isolates integration changes. `--seed-interpolation original-p1`
instead samples the original float64 source P1 field on the new state points;
it is a separate input-resolution control. It does not automatically include
every source knot.

The static reduction solves the original constrained saddle system for bulk
displacement, contact jumps and platen motion. There is no artificial spring
or inverse of an unsupported free body. Initialization must preserve z/c,
satisfy the frozen friction bound and balance the uncondensed discrete body
below 1e-7 MPa. The original Galerkin matrix has a small finite-quadrature
rigid-translation defect; the code records its force resultant rather than
replacing the resulting mean pressure by an idealized value. The actual
lifted bulk force and strictly past response to its small residual are retained.

## Complete history is required from step one

The reused reference diagonal Dref is the old coarse elastic penalty. A fine
sticking contact stiffness Q.T W k Q differs from that diagonal, so the
correction source can be nonzero well before the first slip. The previous
20 ms late-activation history is therefore insufficient for this experiment.

`tools/extend_resolvent_window.py` retains all 22501 coefficients through
45 ms from the same completed frequency circle. It does not change dt=2 us,
FFT size 65536, radius or frequency matrices. The original coefficient prefix
must remain bitwise equal, and lag-zero inverse composition must pass.
No source spectrum or existing kernel is overwritten.

The actual resource preflight requires 13,664,243,536 bytes of new disk and
estimates 2,582,316,000 bytes of IFFT array workspace. It is not a completed
extension or an OS RSS bound. Full-circle interpolation and late-lag numerical
error still need physical trajectory checks; prefix equality cannot certify
unseen later coefficients.

From the repository root, after the serialized FEM control and its comparison
have finished:

```sh
FDBEM_CORES=4 bem/run.sh resolvent-extend \
  bem/outputs/direct-resolvent-fine2-q12 \
  --workers 4 --fft-rows 4 --max-workspace-mb 3072 \
  --output /Volumes/Gauss-T7/Tatva-BEM-cache/direct-resolvent-fine2-q12-full45-20261026

FDBEM_CORES=4 bem/run.sh refined-contact-simulate \
  /Volumes/Gauss-T7/Tatva-BEM-cache/direct-resolvent-fine2-q12-full45-20261026 \
  --restart-seed bem/outputs/event16-original-f64-seed/restart_seed.npz \
  --subdivisions 1 --constitutive-substeps 64 --kinetic-root-seed \
  --output bem/outputs/refined-contact-full45-subdiv1-n64-original-seed
```

New outputs must use new paths. The identity-grid run is checked against the
existing full event before attributing a change to subdivisions 2/4/8.
NPZ outputs distinguish fine `contact_y`, weights and jumps from
`retained_contact_y` and the 275-component displacement perturbation. They
also retain the actual initial trace and bulk force for reproducible AD.

## Full-history automatic differentiation

`tools/refined_contact_adjoint.py` supports different numbers of contact
states and bulk trace nodes. JAX differentiates independent local constitutive
updates and the converged equilibrium equations; Newton iterates are bypassed
by the implicit root derivative. Fine contact cotangents use the same Q/W
pairing. Reverse CQ propagates every retained history source, and substepped
friction also propagates the previous displacement cotangent. Five independent
forward JVP histories provide a separate check.

The prepared resolvent and Dref are fixed algebraic preconditioners. The
physical kn/kt derivatives in the root residual remain active, including the
sticking prefix. The scope holds the bulk material/geometry, initial static
lift/state and recorded actuator schedule fixed and differentiates selected
discrete branches. Coverage event-time derivatives, full input sensitivity
through the static lift, and continuum gradient convergence are not included.

`tools/check_refined_contact_ad.py` checks a completed saved event and writes
its evidence in a separate output, without changing the primal result:

```sh
FDBEM_CORES=4 bem/run.sh refined-contact-ad-check \
  bem/outputs/refined-contact-full45-subdiv1-n64-original-seed \
  --output bem/outputs/refined-contact-full45-subdiv1-n64-ad
```

The new regression tests cover virtual work, static full-body balance,
original-Z versus resolvent sliding trajectories, fine-grid output mappings,
reverse/JVP agreement, and all five parameter plus actuator finite differences
against independent direct original-Z histories. Substepping is checked
separately. Short regression success is not a measured full-event gradient
or a runtime advantage.

Numerical source commit `3d58e98` passed the complete Python 3.11/3.12
[BEM verification workflow](https://github.com/Gauss963/Tatva-BEM/actions/runs/40308979926).
The actuator finite-difference control first caught a missing future-history
cotangent; the fix adds the correction-source contribution with the original
tolerances retained. [Machine-readable evidence](refined_contact_results.json)
records the source, preflight and waiting local study separately.
The additional [original-history and seed-field controls](original_history_audit.md)
measure original-equation residuals independently of the replay recurrence
and distinguish the original P1 seed field from its coarse nodal projection.

## Literature and limits of transfer

[Aimi, Di Credico and Gimperlein (2024)](https://arxiv.org/abs/2405.08566)
formulate dynamic two-body frictional contact using an energetic space-time
BEM. Their Tresca analysis and Coulomb numerical results motivate consistent
boundary work and energy checks; they do not prove stability for Tatva's
previous-state slip weakening.

[Ciardo and Romanet (2026)](https://arxiv.org/abs/2603.00383) separate dynamic
self effects and interactions, using selective hierarchical compression for
fault networks. Their anti-plane, unbounded setting does not justify dropping
finite-block reflected histories or transplanting their leading coefficient.

[Keshava and Schanz (2026)](https://arxiv.org/abs/2607.15292) investigate
3D adaptive-cross-approximation compression with generalized CQ. This is a
separate operator-acceleration direction. Its reported performance is not a
measured speedup of the present plane-strain nonlinear solver.

The same-dt h1/h2 FEM comparison is now complete: maximum slip is
0.644123/0.619884 mm and the final pressure P1 L2 difference is 17.62%.
The next physical evidence is the full-history identity and contact-grid
controls, including original-P1 seed sampling. Runtime conclusions must
include kernel construction/reuse costs and use comparable accuracy.

# Original equations and restart-field controls

The complete event is still unvalidated against FEM at comparable accuracy.
This experiment checks two possible error sources without changing the
material, friction parameters, finite-block geometry or loading policy:
causal time-domain algebra after the resolvent rewrite, and the initial
plastic-slip field between retained bulk nodes.

## Original causal equilibrium

The replay's small Newton residual is measured in a rewritten equation.
`tools/original_history_audit.py` independently reconstructs A=Z+Dref+spring
by inverting each completed work-coordinate frequency matrix. Its row IFFTs
recover every requested coefficient, and a zero-padded **linear** convolution
then acts on the saved displacement. Future input must not wrap into early
outputs. The code retains all lags through the requested horizon and zeros
only the padding beyond that horizon.

For each saved time, `tools/audit_refined_original_history.py` evaluates

```text
R = (Z*q) + lifted_bulk_force + Q.T*W*contact_traction
    + spring*(lifted_platen + q_platen - actuator)*e_platen.
```

The residual is divided by the original retained row weights to report MPa.
The audit shares the sampled Laplace operator with the primal, but computes
the time-domain force independently of the nonlinear correction recurrence.
Five freshly assembled full BEM matrices check the reconstruction at distinct
contour frequencies. They provide spot checks, not a uniform interpolation
error certificate.

The complete audit uses four frequency workers with one BLAS thread each,
then row FFTs and batched displacement histories. The reconstructed spectrum
needs about 39.65 GB of disk; it is retained separately from the source.
No original kernel or trajectory is changed. Six trajectories can be audited
with two FFT rows within the 4096 MiB explicit action-array budget; the
8192 MiB overall estimate additionally reserves trajectory/runtime arrays.
These estimates do not bound OS caches or actual RSS.

The requested original-equation tolerance is 1e-4 MPa, while the primal root
keeps its existing 1e-7 MPa tolerance. The looser independent check allows
frequency interpolation/inversion and finite-DFT effects to be measured
separately; passing it does not establish physical convergence. Fresh matrix
relative work-Frobenius errors must be below 1e-6. Failures preserve the
reconstructed operator, residuals and diagnostic records.

The audit also records signed BDF2 boundary work for the **perturbation** bulk
port, including radius-weighted and unweighted prefix sums. This is not a
reconstruction of total physical energy or a passivity theorem. A separate
mass/stiffness test checks the exact BDF2 G-energy identity and its numerical
dissipation terms, so the diagnostic's sign/derivative conventions have an
independent analytic control.

Short controls already cover a noncommuting matrix memory recurrence, work
coordinate transformations, batched histories, explicit future-wrap detection,
and freshly assembled finite-block sliding histories. Numerical source
`0833a01` passed Python 3.11/3.12
[CI](https://github.com/Gauss963/Tatva-BEM/actions/runs/40312542278).
The six full-event audits completed on 2026-10-27. Maximum original-equation
residuals are 7.4550e-7 to 7.4930e-7 MPa, below the unchanged 1e-4 MPa audit
tolerance. Every frequency inverse identity was checked, with maximum error
2.3831e-11; the five fresh BEM checks differ by at most 1.125e-15 in relative
work Frobenius norm. The complete zero-padded causal action includes all
22501 lags for all six trajectories. Total audit time including hashing and
output was 1731.25 s, peak process RSS 2487.70 MiB.

Full records, SHA-256 provenance and unchanged tolerances are preserved in
[fem_initialized_completed_results.json](fem_initialized_completed_results.json).
The earlier pending snapshot remains in `original_history_audit_results.json`.
This excludes a large residual in the tested discrete causal equation as the
explanation of the remaining FEM discrepancy. It does not certify continuous
accuracy, untested frequencies or nonlinear stability. Signed perturbation
port-work prefixes were nonnegative for these paths; they still provide no
total-energy or operator passivity proof.

## Initial seed values and their P1 field

Using the original float64 seed preserves its values at the retained nodes.
Its coarse P1 interpolant still differs from the original 241-knot P1 field
between those nodes. `tools/audit_restart_projection.py` measures this
difference exactly on the union of both knot sets; the L2 norm integrates
each linear error segment exactly and is checked against independent Gauss
integration.

For the actual 137-node bulk trace and the captured original seed:

|Quantity|Measured value|
|---|---:|
|Maximum plastic-slip field difference|0.0001777687226 mm|
|P1 L2 field difference|0.0007012595831 mm sqrt(mm)|
|Location of maximum difference|y=458 mm|
|Equivalent trial-shear change at fixed jump|1.362063953 MPa|

These are interpolation diagnostics, not a measured explanation of the final
12.22% slip difference. A separate probe on the old lift gives a maximum
original-field trial-strength ratio of 0.78694; it does not replace a new
full-body static balance solve.

The initial integration controls use `--seed-interpolation coarse-p1` to
isolate quadrature effects. Additional subdivisions 4/8 controls instead use
`--seed-interpolation original-p1`, directly sampling the original source
field at fine state points. The first frozen-offset attempts failed their
static strength check at the fine end points. The completed controls instead
use the explicitly labelled [Tatva healed static active set](fem_static_initialization.md),
which can change the initial plastic offset while keeping weakening memory
healed. The source seed file is read-only and the complete dynamic history is
retained. The fine grid does not automatically contain all source knots, so
this remains a contact integration refinement rather than exact integration.

All successful controls are compared to both completed FEM meshes and get
conditional reverse/JVP checks. An inadmissible original-field lift is kept
as a failure and must be resolved by a suitable discretization. Its failure
does not suppress independent equation audits of already completed controls.
`tools/plot_contact_refinement.py` labels contact-state counts and seed-field
modes explicitly so curves remain distinguishable.

## Literature behind the error separation

For this check, an older foundational result complements the current
[2024–2026 literature review](literature.md): Betcke, Salles and Smigaj,
[Overresolving in the Laplace domain for convolution quadrature methods](https://arxiv.org/pdf/1603.01761)
(2016), Section 4, especially Theorem 4.1 and Remark 4.6. The finite contour
inversion aliases later coefficients into the requested coefficient; reducing
the radius can improve this error while amplifying finite-precision effects.
Their acoustic exterior analysis motivates separating contour inversion from
time-stepping error. The finite-block nonlinear contact problem still needs
its own numerical and physical checks.

The current implementation does not assert that frequency-wise inverse
identities imply exact equality of separately truncated causal convolutions.
That possible discrepancy is precisely what the full original-equation
residual measures.

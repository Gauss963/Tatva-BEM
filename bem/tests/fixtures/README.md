# Contact switching regression

`opening_switch.npz` records the first failed step (3700, 29.600 ms) of the
original 40 mm / 8 us event-16 diagnostic before the line-search correction.
It stores the fixed instantaneous CQ operator, static lift, old interface
state, past traction and displacement guess. It contains no original FEM bulk
data. The archived material, geometry and contact inputs remain in `resources`.

The old infinity-norm merit rejected every trial direction at this state.
An L2 merit converges to the root in two Newton updates with the same
equations, tolerances and old contact state. Both analytic and AD tangents
must solve this case; no step may be silently skipped or regularized.

`graded_switch.npz` records the corner-graded interface case at step 3554
(28.432 ms), with 69 retained coordinates. Even an L2 line search rejected
the first direction at the active-set kink. A second provisional Newton
correction gives a combined direction that passes the same merit test;
the final residual is checked against the unchanged tolerance. An independent
SciPy hybrid root solve also reaches the same equilibrium. Both provisional
trials keep the old contact state frozen.

`contact4_event16_equilibrium.npz` records the 275-coordinate equilibrium at
step 14572 (29.144 ms) of the 2 us / Q12 four-substep event. The primary Newton
solve stops with a 17.3886 MPa residual. Powell hybrid recovery and independent
contact continuation reach the same root without changing the equation or
the 1e-7 MPa acceptance tolerance. Tests check that trials commit no state and
compare the recovered root's implicit mu_k gradient with central differences.
The adjacent JSON records provenance and the retained-weight correction:
the raw failure export accidentally saved full-boundary weights; this fixture
reconstructs the retained weights from the unchanged nodal port definition.
It is a single equilibrium fixture, not a trajectory or CQ restart checkpoint.

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

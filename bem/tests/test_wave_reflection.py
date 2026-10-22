import numpy as np

from finiteblock_bem.benchmarks import finite_bar_benchmark


def test_finite_bar_arrival_reflection_and_paired_refinement():
    coarse = finite_bar_benchmark(h=0.2, dt=0.04)
    fine = finite_bar_benchmark(h=0.1, dt=0.02)
    assert fine.loading_relative_l2 < coarse.loading_relative_l2
    assert fine.probe_relative_l2 < coarse.probe_relative_l2
    assert fine.loading_relative_l2 < 0.03
    assert fine.probe_relative_l2 < 0.03
    assert fine.precursor_relative_max < 0.001
    assert fine.boundary_residual < 1e-9
    # At t=2L/cp the first clamp reflection returns to the loading face.
    # The analytic and computed loading-face velocity must reverse afterwards.
    velocity = np.gradient(fine.loading_displacement, fine.time)
    assert np.mean(velocity[(fine.time > 2.5) & (fine.time < 3.3)]) < -0.8

import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import CQPlan, bdf2_symbol
from finiteblock_bem.galerkin import _scalar_kernels
from finiteblock_bem.material import ElasticMaterial

TOOLS = Path(__file__).resolve().parents[1] / "tools"
BASE = runpy.run_path(str(TOOLS / "raw_layer_sampling.py"))
SEEDED = runpy.run_path(str(TOOLS / "seeded_raw_sampling.py"))


def test_same_primitive_partition_skips_rejected_work_without_changing_any_emission():
    material = ElasticMaterial(7662, 0.2, 1.148e-9)
    distance = np.asarray(SEEDED["DEFAULT_DISTANCES"])
    n, count = len(distance), 2049
    plan = CQPlan(2e-6, 1, n, 2 * (count - 1), 0.999648715112665, 0, 0)
    mass = np.eye(n)
    vectors = np.column_stack([distance, np.zeros(n)])
    calls = [0]

    def raw(s):
        calls[0] += 1
        gs, gp, _, gradient = _scalar_kernels(vectors, s, material)
        return BASE["RawLayerPacket"]((np.diag(gs), np.diag(gp), np.diag(gradient[:, 0])), (mass,))

    options = BASE["RawSamplingOptions"](degree=16)
    output = []
    records = []
    counts = []
    code = BASE["sample_raw_layers"].__code__
    for seeded in [False, True]:
        values = np.empty((count, n, n), complex)
        calls[0] = 0

        def consume(index, s, value):
            values[index] = value

        if seeded:
            record = SEEDED["sample_with_primitive_partition"](
                BASE["sample_raw_layers"],
                raw,
                lambda packet, s: packet.families[0],
                plan,
                consume,
                material=material,
                options=options,
                workers=1,
                max_workspace_mb=32,
            )
        else:
            record = BASE["sample_raw_layers"](
                raw,
                lambda packet, s: packet.families[0],
                plan,
                consume,
                options=options,
                workers=1,
                max_workspace_mb=32,
            )
        counts.append(calls[0])
        records.append(record)
        output.append(values)
    assert counts == [record["source_evaluations"] for record in records]
    assert counts[1] < counts[0]
    np.testing.assert_array_equal(output[0], output[1])
    assert records[1]["all_fft_grid_points_emitted"]
    assert records[1]["original_acceptance_checks_retained"]
    assert not records[1]["primitive_partition_screen"]["body_operator_values_reused"]
    assert BASE["sample_raw_layers"].__code__ is code
    assert "_raw_initial_stack" not in BASE["sample_raw_layers"].__globals__


def test_bad_advisory_interval_still_refines_and_checks_the_actual_kernel():
    plan = CQPlan(1e-4, 1, 1, 256, 0.99, 0, 0)
    mass = np.eye(1)

    def raw(s):
        value = np.array([[np.exp(-0.025 * s)]])
        return BASE["RawLayerPacket"]((value, value, value), (mass,))

    output = np.empty((129, 1, 1), complex)
    record = SEEDED["sample_with_partition"](
        BASE["sample_raw_layers"],
        raw,
        lambda packet, s: packet.families[0],
        plan,
        lambda index, s, value: output.__setitem__(index, value),
        initial_intervals=[(0, 129)],
        options=BASE["RawSamplingOptions"](degree=8),
        max_workspace_mb=32,
    )
    assert record["rejected_intervals"] > 0
    assert record["all_fft_grid_points_emitted"]
    s = bdf2_symbol(plan.radius * np.exp(-1j * np.linspace(0, np.pi, 129))) / plan.dt
    np.testing.assert_allclose(output[:, 0, 0], np.exp(-0.025 * s), rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize(
    "intervals", [[], [(0, 3)], [(1, 4)], [(0, 2), (3, 4)], [(0, 3), (2, 4)], [(0, True), (1, 4)]]
)
def test_initial_partition_cannot_skip_duplicate_or_reorder_fft_indices(intervals):
    with pytest.raises(ValueError):
        SEEDED["sampler_with_partition"](BASE["sample_raw_layers"], intervals, 4)

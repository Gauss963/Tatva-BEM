import json
import runpy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import plan_cq
from finiteblock_bem.nodal import load_nodal_model

TOOLS = Path(__file__).resolve().parents[1] / "tools"
METHOD = runpy.run_path(str(TOOLS / "interior_raw_sampling.py"))
IO = runpy.run_path(str(TOOLS / "rectangular_sampling_checkpoint.py"))


def packet(s):
    families = [np.eye(3) * (i + 1) for i in range(6)]
    families += [np.ones((2 if k < 2 else 4, 3)) * (i + k + 1) for i in range(2) for k in range(4)]
    return METHOD["InteriorPacket"](tuple(families), (np.eye(3), np.eye(3)))


@pytest.mark.parametrize("s", [200.0 + 0j, 200.0 + 6283j, 1e5 + 2e5j])
def test_stable_interior_trace_and_all_fields_match_original_free_body_algebra(s):
    case = TOOLS.parent / "src/finiteblock_bem/resources/cases/pmma_event16.toml"
    model = load_nodal_model(case, max_panel_length=160)
    points = [[[100.25, 240.25]], [[270.25, 240.25]]]
    stable = METHOD["StableInteriorTransfer"](model, points, quadrature=8)
    original = METHOD["FIELD"]["InteriorRetainedTransfer"](model, points, quadrature=8)
    value = stable.evaluate(s)
    reference = original.evaluate(s)
    for key in [
        "impedance",
        "boundary_displacement",
        "displacement",
        "gradient",
        "stress",
        "velocity",
    ]:
        error = np.linalg.norm(value[key] - reference[key]) / max(
            np.linalg.norm(reference[key]), 1e-30
        )
        assert error < 1e-8, (key, error)


def test_rectangular_sampler_retains_empirical_tail_checks_and_transforms_every_grid_point():
    plan = plan_cq(3, 0.1, 257, max_memory_mb=128)
    emitted = []
    result = METHOD["rectangular_sampler"](2, 3)(
        packet,
        lambda p, s: p.families[6] + p.families[10],
        plan,
        lambda i, s, a: emitted.append((i, a.copy())),
        workers=1,
        max_workspace_mb=128,
    )
    assert [i for i, a in emitted] == list(range(plan.fft_size // 2 + 1))
    assert all(a.shape == (2, 3) for i, a in emitted)
    assert result["accepted_intervals"] > 0 and result["interpolated_grid_points"] > 0
    assert result["source_evaluations"] < len(emitted)
    assert result["certified_error_bound"] is False


def test_row_check_catches_small_displacement_row_even_when_large_stress_dominates():
    options = METHOD["RAW"]["RawSamplingOptions"](output_rtol=1e-6)
    exact = np.array([[1e6, 1e6], [1e-6, 1e-6]])
    bad = exact.copy()
    bad[1] *= 1.01
    global_error = np.linalg.norm(exact - bad) / (
        options.output_atol + options.output_rtol * np.linalg.norm(exact)
    )
    assert global_error < 1
    assert METHOD["maximum_row_error"]([exact], [bad], options) > 1
    # The old absolute Frobenius guard is retained too.
    exact = np.zeros((100, 3))
    bad = np.full_like(exact, options.output_atol / 2)
    assert METHOD["maximum_row_error"]([exact], [bad], options) > 1


def test_rectangular_checkpoint_replays_same_interpolated_interval_and_verifies_prefix_bits(
    tmp_path,
):
    plan = plan_cq(3, 0.1, 257, max_memory_mb=128)
    shape = (plan.fft_size // 2 + 1, 2, 3)
    (tmp_path / "preflight.json").write_text(
        json.dumps(
            dict(kind="rectangular_interior_raw_sampling", plan=asdict(plan), shape=list(shape))
        )
    )
    journal = IO["RectangularJournal"](tmp_path, plan, 2, 3)
    journal.begin()
    spectrum = journal.spectrum("work-family0.npy")
    writer = IO["Writer"](journal, spectrum, matrices=64)
    stack, start = journal.sampler_stack("work-family0.npy", [(0, shape[0])])

    def sample(writer, stack, start, stop=False):
        sampler = METHOD["rectangular_sampler"](
            2,
            3,
            stack=stack,
            start=start,
            interval=lambda left, right, rest: writer.journal.save_sampler_state(
                "work-family0.npy", left, right, rest, writer
            ),
        )

        def consume(i, s, a):
            if stop and i == 130:
                raise InterruptedError("inside accepted interpolation interval")
            writer.accept(i, a)

        return sampler(
            packet,
            lambda p, s: p.families[6] + p.families[10],
            plan,
            consume,
            workers=1,
            max_workspace_mb=128,
        )

    with pytest.raises(InterruptedError):
        sample(writer, stack, start, True)
    journal.end("interrupted_or_failed")
    assert journal.prefix("work-family0.npy") == 128
    restored = IO["RectangularJournal"](tmp_path, plan, 2, 3, resume=True)
    restored.begin()
    spectrum = restored.spectrum("work-family0.npy")
    writer = IO["Writer"](restored, spectrum, matrices=64)
    stack, start = restored.sampler_stack("work-family0.npy", [(0, shape[0])])
    assert start == 0
    sample(writer, stack, start)
    writer.finish()
    assert restored.prefix("work-family0.npy") == shape[0]
    values = spectrum[:]
    np.testing.assert_allclose(values, np.full(shape, 3.0), atol=1e-12)


def test_rectangular_committed_corruption_is_rejected_without_rewriting_it(tmp_path):
    plan = plan_cq(3, 0.1, 9, max_memory_mb=128)
    shape = (plan.fft_size // 2 + 1, 2, 3)
    (tmp_path / "preflight.json").write_text(
        json.dumps(
            dict(kind="rectangular_interior_raw_sampling", plan=asdict(plan), shape=list(shape))
        )
    )
    journal = IO["RectangularJournal"](tmp_path, plan, 2, 3)
    journal.begin()
    spectrum = journal.spectrum("work-family0.npy")
    writer = IO["Writer"](journal, spectrum, matrices=4)
    for i in range(4):
        writer.accept(i, np.ones((2, 3)))
    path = tmp_path / "work-family0.npy"
    data = np.load(path, mmap_mode="r+")
    data[0, 0, 0] += 1
    data.flush()
    data._mmap.close()
    before = path.read_bytes()
    state = (tmp_path / "sampling-journal.json").read_bytes()
    with pytest.raises(ValueError, match="changed"):
        IO["RectangularJournal"](tmp_path, plan, 2, 3, resume=True)
    assert (
        path.read_bytes() == before and (tmp_path / "sampling-journal.json").read_bytes() == state
    )

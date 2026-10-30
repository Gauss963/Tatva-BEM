import argparse
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
EXTEND = runpy.run_path(str(TOOLS / "extend_fem_timestep_window.py"))
CHUNKS = runpy.run_path(str(TOOLS / "fem_refinement_chunks.py"))


def toy_setup():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("number")
    args = parser.parse_args()
    value = args.number * 3
    records = []
    raise RuntimeError(f"The original branch loop must not run: {value} {records}")


def test_exact_setup_prefix_returns_before_original_branch_loop():
    result = EXTEND["setup_prefix"](toy_setup, SimpleNamespace(number=7))
    assert result["value"] == 21
    assert "records" not in result


def test_refined_schedule_preserves_all_original_and_nested_endpoint_bits():
    rng = np.random.default_rng(782)
    model = {k: rng.normal(size=20) for k in EXTEND["WINDOW"]["SCHEDULES"]}
    for factor in [1, 2, 4]:
        expected = EXTEND["WINDOW"]["window_schedule"](model, 2, 18, factor)
        np.testing.assert_array_equal(EXTEND["schedule"](model, 2, 18, factor), expected)
    for factor in [8, 16]:
        actual = EXTEND["schedule"](model, 2, 18, factor)
        expected = EXTEND["schedule"](model, 2, 18, factor // 2)
        np.testing.assert_array_equal(actual[1::2], expected)
        np.testing.assert_array_equal(
            actual[factor - 1 :: factor], EXTEND["schedule"](model, 2, 18, 1)
        )
    with pytest.raises(ValueError):
        EXTEND["schedule"](model, 2, 18, 3)


def carry(value=0):
    return tuple(
        np.asarray(value > 0) if i == 10 else np.full((3,) if i < 9 else (), value, dtype=float)
        for i in range(12)
    )


def store(path, factor=8, context="a" * 64):
    return CHUNKS["WindowChunks"](
        path, context, factor, 10, [13, 17], [13, 17], carry(), {"pressure": np.zeros(3)}
    )


def append_pair(target):
    target.append(
        carry(1), np.arange(24 * 17, dtype=float).reshape(24, 17), {"pressure": np.ones(3)}
    )
    target.append(
        carry(2), np.arange(32 * 17, dtype=float).reshape(32, 17), {"pressure": np.full(3, 2.0)}
    )


def test_real_files_resume_all_twelve_states_and_every_fine_scalar_bitwise(tmp_path):
    uninterrupted = store(tmp_path / "uninterrupted")
    append_pair(uninterrupted)
    interrupted = store(tmp_path / "interrupted")
    interrupted.append(
        carry(1), np.arange(24 * 17, dtype=float).reshape(24, 17), {"pressure": np.ones(3)}
    )
    (interrupted.folder / ".pending-incomplete").write_bytes(b"not committed")
    resumed = store(interrupted.folder)
    restored = resumed.restore()
    for a, b in zip(restored, carry(1), strict=True):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype
    resumed.append(
        carry(2), np.arange(32 * 17, dtype=float).reshape(32, 17), {"pressure": np.full(3, 2.0)}
    )
    a, b = [s.combine(0.01, [0, 1, 2]) for s in [uninterrupted, resumed]]
    assert set(a) == set(b)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert a["all_scalar_history"].shape == (56, 17)
    assert len([k for k in a if k.startswith("final_carry_")]) == 12
    with pytest.raises(ValueError, match="already complete"):
        resumed.append(carry(), np.zeros((1, 17)), {})


def test_missing_or_changed_context_and_payload_are_rejected(tmp_path):
    target = store(tmp_path / "chunks")
    append_pair(target)
    with pytest.raises(ValueError, match="context"):
        store(target.folder, context="b" * 64).restore()
    first = target.paths[0]
    original = first.read_bytes()
    with np.load(first, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["carry_0"][0] += 1
    with first.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    with pytest.raises(ValueError, match="context"):
        store(target.folder).restore()
    first.write_bytes(original)
    first.unlink()
    with pytest.raises(ValueError, match="noncontiguous"):
        store(target.folder).restore()


@pytest.mark.parametrize("bad", ["missing_row", "dtype", "nonfinite", "carry_shape", "field_shape"])
def test_invalid_accepted_chunk_is_rejected_before_commit(tmp_path, bad):
    target = store(tmp_path / "chunks")
    state, scalar, fields = carry(1), np.zeros((24, 17)), {"pressure": np.zeros(3)}
    if bad == "missing_row":
        scalar = scalar[:-1]
    elif bad == "dtype":
        scalar = scalar.astype(np.float32)
    elif bad == "nonfinite":
        scalar[0, 2] = np.nan
    elif bad == "carry_shape":
        state = (np.zeros(2), *state[1:])
    else:
        fields = {"pressure": np.zeros(2)}
    with pytest.raises(ValueError):
        target.append(state, scalar, fields)
    assert not list(target.folder.glob("chunk-*.npz"))


def test_incomplete_export_and_overwrite_cannot_claim_completion(tmp_path):
    target = store(tmp_path / "chunks")
    with pytest.raises(ValueError, match="incomplete"):
        target.combine(0.01, [0, 1, 2])
    path = tmp_path / "existing.npz"
    CHUNKS["atomic_npz"](path, {"values": np.ones(3)})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        CHUNKS["atomic_npz"](path, {"values": np.zeros(3)})
    assert path.read_bytes() == before


def test_failed_attempt_retains_time_and_error_without_claiming_complete(tmp_path):
    from time import perf_counter

    with pytest.raises(RuntimeError, match="interrupted"):
        with EXTEND["record_attempt"](tmp_path, perf_counter()):
            raise RuntimeError("interrupted")
    record = json.loads(next(tmp_path.glob("attempt-*.json")).read_text())
    assert record["status"] == "failed" and record["seconds_this_invocation"] > 0
    assert "interrupted" in record["error"]

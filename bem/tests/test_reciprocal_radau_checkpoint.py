"""Interruption and corruption controls for tile-major raw Radau sampling."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from finiteblock_bem.cq import plan_cq

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def test_reciprocal_checkpoint_replays_and_rewrites_uncommitted_suffix(tmp_path):
    module = runpy.run_path(str(TOOLS / "reciprocal_radau_checkpoint.py"))
    journal_type, writer_type = (
        module["ReciprocalSamplingJournal"],
        module["JournaledReciprocalWriter"],
    )
    plan = plan_cq(5, 0.009, 10)
    count = plan.fft_size // 2 + 1
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(count, 5, 5)) + 1j * rng.normal(size=(count, 5, 5))
    matrix += matrix.transpose(0, 2, 1).copy()
    name = "branch0.bin"
    (tmp_path / "preflight.json").write_text("{}")
    initial = journal_type(tmp_path, plan, 2)
    with pytest.raises(RuntimeError, match="interrupt"), initial.open_branch(name) as branch:
        writer = writer_type(initial, branch, 0, matrices=4)
        for index in range(4):
            writer.accept(index, matrix[index])
        # Simulate a crash after writing some bytes of the next batch.
        branch.append(2 * matrix[4:6])
        branch.sync()
        raise RuntimeError("interrupt")
    assert not (tmp_path / name).exists()

    restored = journal_type(tmp_path, plan, 2, resume=True)
    assert restored.prefix(name) == 4
    with restored.open_branch(name) as branch:
        writer = writer_type(restored, branch, 2, matrices=5)
        for index in range(2, count):
            writer.accept(index, matrix[index])
        writer.finish()

    storage = module["STORAGE"]
    reference = tmp_path / "reference.bin"
    with storage["ReciprocalBranch"](reference, plan, block=2) as branch:
        branch.append(matrix)
    assert (tmp_path / name).read_bytes() == reference.read_bytes()
    assert journal_type(tmp_path, plan, 2, resume=True).prefix(name) == count

    with (tmp_path / name).open("r+b") as stream:
        byte = stream.read(1)
        stream.seek(0)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="bytes or intervals changed"):
        journal_type(tmp_path, plan, 2, resume=True)


def test_reciprocal_checkpoint_rejects_different_preserved_matrix(tmp_path):
    module = runpy.run_path(str(TOOLS / "reciprocal_radau_checkpoint.py"))
    journal_type, writer_type = (
        module["ReciprocalSamplingJournal"],
        module["JournaledReciprocalWriter"],
    )
    plan = plan_cq(3, 0.009, 6)
    (tmp_path / "preflight.json").write_text("{}")
    name = "branch1.bin"
    initial = journal_type(tmp_path, plan, 2)
    with pytest.raises(RuntimeError, match="interrupt"), initial.open_branch(name) as branch:
        writer = writer_type(initial, branch, 0, matrices=1)
        writer.accept(0, np.eye(3))
        raise RuntimeError("interrupt")
    resumed = journal_type(tmp_path, plan, 2, resume=True)
    with (
        pytest.raises(ValueError, match="not bitwise identical"),
        resumed.open_branch(name) as branch,
    ):
        writer = writer_type(resumed, branch, 0, matrices=1)
        writer.accept(0, 2 * np.eye(3))
    assert not (tmp_path / name).exists()


def test_inversion_retry_cannot_remove_a_different_builds_weights(tmp_path):
    from types import SimpleNamespace

    tool = runpy.run_path(str(TOOLS / "prepare_reciprocal_radau_correction.py"))
    journal = SimpleNamespace(
        data={"phase": "inverting"}, folder=tmp_path / "spectra", preflight="candidate"
    )
    weights = tmp_path / "weights"
    weights.mkdir()
    marker = weights / tool["WEIGHTS_OWNER"]
    file = weights / "weights0.npy"
    file.write_bytes(b"physical weights")
    actual = tool["inversion_owner"](journal)
    tool["CK"]["atomic_json"](marker, {**actual, "preflight_sha256": "other"})
    with pytest.raises(ValueError, match="belongs to a different build"):
        tool["discard_incomplete_inversion"](journal, weights)
    assert file.read_bytes() == b"physical weights"

    tool["CK"]["atomic_json"](marker, actual)
    tool["discard_incomplete_inversion"](journal, weights)
    assert not weights.exists()

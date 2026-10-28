import runpy
import signal
from pathlib import Path

import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/limit_process_cpu.py"))


@pytest.mark.parametrize(
    "value,seconds",
    [("109:45.83", 6585.83), ("2:03:04.50", 7384.5), ("1-02:03:04.50", 93784.5)],
)
def test_cpu_clock_preserves_all_accumulated_minutes(value, seconds):
    assert TOOL["cpu_seconds"](value) == pytest.approx(seconds)


@pytest.mark.parametrize("value", ["nan:00", "-1:00", "1:60.0", "1.5:00", "1"])
def test_invalid_cpu_clock_is_rejected(value):
    with pytest.raises(ValueError):
        TOOL["cpu_seconds"](value)


def test_quota_carries_debt_without_banking_idle_time():
    update = TOOL["update_credit"]
    assert update(0, 100, 0, 4, 0.25) == 1
    assert update(0, 0.25, 2, 4, 0.25) == -1
    wall = used = credit = 0.0
    for speed in [8, 0.5, 6, 10, 2] * 100:
        wall += 0.25
        used += 0.25 * speed
        credit = update(credit, 0.25, 0.25 * speed, 4, 0.25)
        if credit < 0:
            wait = -credit / 4
            wall += wait
            credit = update(credit, wait, 0, 4, 0.25)
        assert used <= 4 * wall + 1e-9
    with pytest.raises(ValueError, match="backwards"):
        update(0, 1, -1, 4, 0.25)


def test_only_owned_stops_are_resumed_and_pid_reuse_is_rejected(monkeypatch):
    resume = TOOL["resume_owned"]
    namespace = resume.__globals__
    signals = []
    sample = dict(start="Wed Oct 28 17:36:14 2026", state="T", command="python owned-job.py")
    monkeypatch.setitem(namespace, "sample_process", lambda pid: sample)
    monkeypatch.setattr(namespace["os"], "kill", lambda pid, sig: signals.append((pid, sig)))
    resume(10, False, sample["start"], "owned-job.py")
    assert not signals
    resume(10, True, sample["start"], "owned-job.py")
    assert signals == [(10, signal.SIGCONT)]
    sample["command"] = "python another-job.py"
    with pytest.raises(RuntimeError, match="identity changed"):
        resume(10, True, sample["start"], "owned-job.py")
    assert len(signals) == 1


def test_interrupt_during_quota_wait_resumes_target(tmp_path, monkeypatch):
    control = TOOL["control"]
    namespace = control.__globals__
    clock = [0.0]
    signals = []
    target = dict(
        start="Wed Oct 28 17:36:14 2026",
        state="R",
        cpu_seconds=0.0,
        command="python owned-job.py",
    )

    def sleep(seconds):
        clock[0] += seconds
        if signals:
            raise InterruptedError("test interrupt during owned stop")
        target["cpu_seconds"] += 8 * seconds

    monkeypatch.setitem(namespace, "sample_process", lambda pid: target.copy())
    monkeypatch.setattr(namespace["time"], "monotonic", lambda: clock[0])
    monkeypatch.setattr(namespace["time"], "sleep", sleep)
    monkeypatch.setattr(namespace["os"], "kill", lambda pid, sig: signals.append(sig))
    with pytest.raises(InterruptedError):
        control(10, target["start"], "owned-job.py", 4, 0.25, tmp_path / "quota")
    assert signals == [signal.SIGSTOP, signal.SIGCONT]


def test_already_paused_target_is_never_resumed_or_claimed(tmp_path, monkeypatch):
    control = TOOL["control"]
    namespace = control.__globals__
    target = dict(start="Wed Oct 28 17:36:14 2026", state="T", command="python owned-job.py")
    signals = []
    monkeypatch.setitem(namespace, "sample_process", lambda pid: target)
    monkeypatch.setattr(namespace["os"], "kill", lambda pid, sig: signals.append(sig))
    output = tmp_path / "unclaimed"
    with pytest.raises(ValueError, match="already stopped"):
        control(10, target["start"], "owned-job.py", 4, 0.25, output)
    assert not signals
    assert not output.exists()


def test_matching_zombie_is_terminal_without_a_command_but_reused_pid_is_rejected():
    sample = dict(start="Wed Oct 28 17:36:14 2026", state="Z", command="<defunct>")
    assert not TOOL["validate_identity"](sample, sample["start"], "owned-job.py")
    with pytest.raises(RuntimeError, match="identity changed"):
        TOOL["validate_identity"](sample, "Wed Oct 28 17:36:15 2026", "owned-job.py")

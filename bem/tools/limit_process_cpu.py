"""Bound an exclusively owned process's average CPU time, including live threads.

This is a portable POSIX duty-cycle controller, not CPU affinity. Stop this
controller before intentionally pausing its target. SIGTERM/SIGINT resume a
target stopped by this controller; SIGKILL cannot provide that cleanup.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def cpu_seconds(value):
    """Parse ps's [days-][hours:]minutes:seconds clock without losing live threads."""
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
    fields = value.split(":")
    if len(fields) not in (2, 3) or days < 0:
        raise ValueError("Unexpected ps CPU clock")
    values = [float(item) for item in fields]
    if not all(math.isfinite(item) and item >= 0 for item in values):
        raise ValueError("Nonfinite or negative ps CPU clock")
    if values[-1] >= 60 or (len(values) == 3 and values[-2] >= 60):
        raise ValueError("Invalid ps CPU clock components")
    if any(item != int(item) for item in values[:-1]):
        raise ValueError("Fractional ps hours/minutes")
    return days * 86400 + sum(item * 60**i for i, item in enumerate(reversed(values)))


def sample_process(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=,stat=,time=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 1 and not result.stdout.strip():
        return None
    if result.returncode:
        raise RuntimeError(f"Cannot sample target process: {result.stderr.strip()}")
    fields = result.stdout.strip().split(maxsplit=7)
    if len(fields) != 8:
        raise ValueError("Unexpected ps process identity record")
    return dict(
        start=" ".join(fields[:5]),
        state=fields[5],
        cpu_seconds=cpu_seconds(fields[6]),
        command=fields[7],
    )


def validate_identity(sample, expected_start, command_fragment):
    if sample is None:
        return False
    if sample["start"] != " ".join(expected_start.split()):
        raise RuntimeError("Target PID identity changed; no signal sent")
    if sample["state"].startswith("Z"):
        return False
    if command_fragment not in sample["command"]:
        raise RuntimeError("Target PID identity changed; no signal sent")
    return True


def update_credit(credit, elapsed, used_cpu, cores, interval):
    """Carry CPU debt forward, but never bank more than one work interval."""
    if elapsed < 0 or used_cpu < 0:
        raise ValueError("Process CPU or monotonic wall clock moved backwards")
    return min(credit + cores * elapsed - used_cpu, cores * interval)


def resume_owned(pid, owned_stop, expected_start, command_fragment):
    if not owned_stop:
        return
    current = sample_process(pid)
    if validate_identity(current, expected_start, command_fragment):
        os.kill(pid, signal.SIGCONT)


def write_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def utc():
    return datetime.now(UTC).isoformat()


def control(pid, expected_start, command_fragment, cores, interval, output):
    first = sample_process(pid)
    if not validate_identity(first, expected_start, command_fragment):
        raise ValueError("Target is no longer running")
    if "T" in first["state"]:
        raise ValueError("Refuse to take ownership of an already stopped target")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    previous_wall = started
    previous_cpu = first["cpu_seconds"]
    credit = 0.0
    owned_stop = False
    record = dict(
        pid=pid,
        controller_pid=os.getpid(),
        utc_start=utc(),
        target_identity=first,
        requested_average_cores=cores,
        interval_seconds=interval,
        method="ps cumulative process CPU counter with bounded-credit SIGSTOP/SIGCONT",
        hard_affinity=False,
        counts_live_thread_cpu=True,
        pre_attachment_cpu_time_excluded=True,
        status="controlling",
        stop_cycles=0,
        stopped_wall_seconds=0.0,
    )
    last_report = started
    write_json(output / "status.json", record)
    try:
        while True:
            time.sleep(interval)
            current = sample_process(pid)
            if not validate_identity(current, expected_start, command_fragment):
                record["status"] = "target_exited"
                break
            if "T" in current["state"]:
                raise RuntimeError("Target was stopped externally; leave it stopped")
            now = time.monotonic()
            credit = update_credit(
                credit, now - previous_wall, current["cpu_seconds"] - previous_cpu, cores, interval
            )
            previous_wall, previous_cpu = now, current["cpu_seconds"]
            elapsed = now - started
            record.update(
                utc_sample=utc(),
                last_target=current,
                wall_seconds=elapsed,
                cpu_seconds=current["cpu_seconds"] - first["cpu_seconds"],
                measured_average_cores=(current["cpu_seconds"] - first["cpu_seconds"]) / elapsed,
            )
            if credit < 0:
                os.kill(pid, signal.SIGSTOP)
                owned_stop = True
                stopped_at = time.monotonic()
                try:
                    time.sleep(-credit / cores)
                finally:
                    resume_owned(pid, owned_stop, expected_start, command_fragment)
                    owned_stop = False
                    record["stopped_wall_seconds"] += time.monotonic() - stopped_at
                    record["stop_cycles"] += 1
            if now - last_report >= 30:
                write_json(output / "status.json", record)
                last_report = now
    except BaseException as error:
        record.update(status="controller_stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        resume_owned(pid, owned_stop, expected_start, command_fragment)
        record["utc_end"] = utc()
        write_json(output / "status.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--expected-start", required=True, help="Exact ps lstart identity")
    parser.add_argument("--expected-command-fragment", required=True)
    parser.add_argument("--cores", type=float, default=3.85)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.pid <= 1
        or args.pid == os.getpid()
        or not args.expected_command_fragment
        or not math.isfinite(args.cores)
        or not 0 < args.cores <= (os.cpu_count() or 1)
        or not math.isfinite(args.interval)
        or not 0.01 <= args.interval <= 10
    ):
        raise ValueError("Require a valid owned PID, core budget, and sampling interval")
    output = args.output.resolve()
    if output.is_relative_to(Path("/Volumes/Gauss-T7/tatva").resolve()):
        raise ValueError("Output must be outside protected Tatva")

    def interrupt(signum, frame):
        raise InterruptedError(f"Controller received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupt)
    control(
        args.pid,
        args.expected_start,
        args.expected_command_fragment,
        args.cores,
        args.interval,
        output,
    )


if __name__ == "__main__":
    main()

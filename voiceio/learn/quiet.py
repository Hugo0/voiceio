"""Background learning that never competes with the person at the computer.

Labeling, evaluating and training are CPU-heavy for an hour or more. Idle CPU
priority is not enough on its own (see voiceio.presence), so a scheduled run
only works while nobody is using the machine:

  * it starts once you have been away `[learn] away_minutes` (no input, or
    the screen locked), on mains power, with the machine otherwise quiet;
  * the work runs in a child process, and the parent checks every couple of
    seconds; when you come back (or unplug) the child is stopped at once;
  * every stage resumes where it stopped: labels are saved per clip,
    training checkpoints its progress (voiceio.learn.train), finished evals
    are reused. The next time you are away it carries on, within the same
    run while it waits, else at the next scheduled run.

Load is checked only before starting: while the work runs, the load is
mostly the work itself, and a check that its own evidence can trip would
stop every run.
"""
from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from voiceio import presence
from voiceio.power import on_battery

# Set in the child that does the work; its absence makes `maintain
# --when-idle` the watcher instead.
WORKER_ENV = "VOICEIO_LEARN_WORKER"
# A later child after an interruption: carry on quietly (no second "learning" notice).
RESUMED_ENV = "VOICEIO_LEARN_RESUMED"
# Path of a file the child holds while it must not be cut off (switching models).
HOLD_ENV = "VOICEIO_LEARN_HOLD"


def busy_reason(max_load: float, away_after: float | None = None) -> str | None:
    """Why learning should wait right now, or None when it may start."""
    if on_battery():
        return "on battery"
    if away_after is not None:
        p = presence.check(away_after)
        if p.here:
            return p.why
    load = os.getloadavg()[0] / (os.cpu_count() or 1)
    if load > max_load:
        return f"busy (load {load:.2f} per CPU, waiting for {max_load:g})"
    return None


def stop_reason(away_after: float) -> str | None:
    """Why running work must stop now (you're back, or on battery), else None."""
    if on_battery():
        return "on battery"
    p = presence.check(away_after)
    return p.why if p.here else None


def wait_until_quiet(max_load: float, wait_hours: float, *, away_after: float | None = None,
                     poll: float = 300,
                     sleep: Callable[[float], None] = time.sleep,
                     clock: Callable[[], float] = time.monotonic,
                     say: Callable[[str], None] = print) -> bool:
    """Block until learning may start. False when it couldn't for wait_hours."""
    deadline = clock() + wait_hours * 3600
    last = None
    while (reason := busy_reason(max_load, away_after)) is not None:
        if (kind := re.sub(r"[\d.]+", "#", reason)) != last:   # once per reason, not per reading
            say(f"waiting: {reason}")
            last = kind
        if clock() + poll > deadline:
            say(f"still {reason} after {wait_hours:g} h; trying again at the next scheduled run")
            return False
        sleep(poll)
    return True


def worker_threads(threads: int) -> int:
    """`[learn] threads`, with 0 meaning half the logical CPUs (the physical
    cores on an SMT machine): fast on an idle machine, never all of it."""
    return threads if threads > 0 else max(1, (os.cpu_count() or 2) // 2)


def worker_argv() -> list[str]:
    """This same command, for the child. orig_argv keeps "-m voiceio": re-
    running __main__.py as a script would put voiceio/ first on sys.path."""
    return [sys.executable, *sys.orig_argv[1:]]


def _spawn(argv: list[str], env: dict[str, str]) -> subprocess.Popen:
    # Own process group: stopping it takes the decoder subprocesses along.
    return subprocess.Popen(argv, env=env, start_new_session=True)


def _stop(proc: subprocess.Popen, grace: float = 5.0) -> None:
    """End the work now. Nothing is lost that a later run needs: every stage
    saves its progress as it goes, so there is no shutdown to wait for."""
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            continue
    with contextlib.suppress(ProcessLookupError):   # stragglers left in the group
        os.killpg(proc.pid, signal.SIGKILL)


def run_while_away(argv: list[str], *, max_load: float, away_minutes: float,
                   wait_hours: float, threads: int, hold_file: Path,
                   poll: float = 2.0, wait_poll: float = 30.0,
                   spawn: Callable[[list[str], dict[str, str]], subprocess.Popen] = _spawn,
                   stop: Callable[[subprocess.Popen], None] = _stop,
                   sleep: Callable[[float], None] = time.sleep,
                   clock: Callable[[], float] = time.monotonic,
                   say: Callable[[str], None] = print) -> int:
    """Run `argv` (the work) only while nobody is using the machine: stop it
    within `poll` seconds when that changes, and start it again (it resumes)
    once they are away again. Returns its exit status; 0 when it gave up
    waiting (the next scheduled run continues)."""
    away_after = away_minutes * 60
    n = str(worker_threads(threads))
    # torch, CTranslate2 and numpy size their thread pools from these.
    env = {**os.environ, WORKER_ENV: "1", HOLD_ENV: str(hold_file),
           "OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n}
    resumed = False
    while True:
        if not wait_until_quiet(max_load, wait_hours, away_after=away_after, poll=wait_poll,
                                sleep=sleep, clock=clock, say=say):
            return 0
        hold_file.unlink(missing_ok=True)
        proc = spawn(argv, {**env, RESUMED_ENV: "1"} if resumed else env)
        reason = None
        while proc.poll() is None:
            sleep(poll)
            if hold_file.exists():  # switching models: seconds, and must not be torn
                continue
            reason = stop_reason(away_after)
            if reason is not None and proc.poll() is None:
                stop(proc)
                break
        else:
            hold_file.unlink(missing_ok=True)
            return proc.returncode
        say(f"paused: {reason}; it resumes where it stopped once you've been away "
            f"{away_minutes:g} min")
        resumed = True


@contextlib.contextmanager
def hold() -> Iterator[None]:
    """Don't let the watcher stop the work inside this block (a model switch:
    config, history and the daemon restart belong together)."""
    path = os.environ.get(HOLD_ENV)
    if not path:
        yield
        return
    Path(path).touch()
    try:
        yield
    finally:
        Path(path).unlink(missing_ok=True)
